r"""Add the three DockQ sub-scores to caches built before the composite loss existed.

The confidence caches and the pooled-latent shards were written when the head
predicted a single scalar, so they record only ``dockq``.  The composite head is
supervised on ``fnat``, ``irmsd`` and ``lrmsd`` as well, and those numbers already
exist -- DockQ was computed from them -- in the DockQ manifest CSVs.  This is
therefore an **exact join**, not a recompute: no structures are re-read and no
DockQ is re-derived, which is what makes rewriting the caches in place safe.

The join key is the structure path (``structure_path`` in the cache,
``pdb_path`` in the manifest).  Verified before writing anything:
``max |manifest.dockq - cache.dockq| = 0`` and
``max |mean(scaled components) - dockq| ~ 7e-07`` -- so this is a join, not a
recompute.  On the train cache, of 108,041 score rows:

* 107,261 get their sub-scores;
* 672 have no ``dockq`` either -- the manifest leaves all four fields blank for
  them -- so the NaN filter already dropped them and nothing changes;
* 108 (6 entries, 18 decoys each) have a DockQ but no manifest row at all.

Those last 108 are left without sub-scores rather than guessed at.
``EntryGroupedCacheDataset`` drops rows whose components are absent and reports
the count, so they disappear from training with a log line instead of silently
training on a fabricated label.  The val cache matches 100%.

Two artifacts, both rewritten in place (atomically, via a temp file + rename):

1. ``{cache_dir}/scores/<entry>.jsonl`` from the manifest CSVs.
2. ``{latent_dir}/latents/<entry>.npz`` from the *already backfilled* JSONLs.
   The sub-scores are labels, not features, so the pooled latents themselves are
   still valid -- there is no need to re-extract them on a GPU, and
   ``POOLING_VERSION`` does not change.

Usage -- train cache and its latents::

    poetry run python workflows/scripts/data/backfill_dockq_components.py \
        --cache_dir /data/confidock/sabdab_train/confidence_cache \
        --manifest data/dockq_predictor/dockq_training_manifest_full_plus_perturbed.csv \
        --latent_dir /data/confidock/sabdab_train/dockq_latents

Validation cache (two manifests cover it)::

    poetry run python workflows/scripts/data/backfill_dockq_components.py \
        --cache_dir /data/confidock/sabdab_val_easy/confidence_cache \
        --manifest data/dockq_predictor/val_easy_manifest.csv \
        --manifest data/dockq_predictor/val_easy_manifest_shared.csv \
        --latent_dir /data/confidock/sabdab_val_easy/dockq_latents

Pass ``--dry_run`` first: it reports the match rate and the DockQ agreement
without touching a byte.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from confidock.boltz.finetune.latent_cache import COMPONENT_KEYS

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", type=Path, required=True, help="confidence cache holding scores/<entry>.jsonl")
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="DockQ manifest CSV; repeatable, since val_easy is split across two",
    )
    parser.add_argument(
        "--latent_dir",
        type=Path,
        default=None,
        help="also upgrade this dir's latents/<entry>.npz from the backfilled JSONLs",
    )
    parser.add_argument(
        "--latent_only",
        action="store_true",
        help="skip the JSONLs and only upgrade the latents (they are already backfilled)",
    )
    parser.add_argument("--dry_run", action="store_true", help="report what would change, write nothing")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args(argv)


def _is_number(value: object) -> bool:
    """Return whether ``value`` is a finite float, i.e. usable as a label."""
    try:
        return bool(np.isfinite(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def load_manifest_components(paths: list[Path]) -> dict[str, dict[str, float]]:
    """Map structure path -> the three sub-scores plus ``dockq``, from the manifests.

    A path present in two manifests must agree, so the first entry wins and a
    conflict is reported: silently overwriting would make the result depend on the
    order the manifests happened to be passed in.
    """
    table: dict[str, dict[str, float]] = {}
    n_rows = 0
    n_conflicts = 0
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = row.get("pdb_path") or ""
                if not key or not all(_is_number(row.get(k)) for k in (*COMPONENT_KEYS, "dockq")):
                    continue
                n_rows += 1
                values = {k: float(row[k]) for k in (*COMPONENT_KEYS, "dockq")}
                previous = table.get(key)
                if previous is None:
                    table[key] = values
                elif any(abs(previous[k] - values[k]) > 1e-4 for k in values):
                    n_conflicts += 1
        LOGGER.info("manifest %s -> %d usable rows so far", path.name, n_rows)
    if n_conflicts:
        LOGGER.warning("%d structure paths disagreed between manifests; kept the first", n_conflicts)
    LOGGER.info("manifest index: %d distinct structure paths", len(table))
    return table


def _write_atomically(path: Path, write) -> None:
    """Run ``write(tmp_path)`` then rename over ``path``.

    In-place rewriting without a backup was the chosen trade-off, so the one thing
    that must not happen is a truncated file: ``os.replace`` is atomic, so an
    interrupted run leaves the original intact.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _fill_row(row: dict[str, object], manifest: dict[str, dict[str, float]]) -> tuple[str, float]:
    """Add one score row's sub-scores from the manifest, in place.

    Returns the outcome to tally the row under, and its DockQ gap against the manifest
    -- zero unless the row was filled.  That gap is the evidence for the claim in this
    module's docstring: a nonzero one would mean the two sides are not describing the
    same structure and this is not the exact join it is supposed to be.
    """
    if all(k in row for k in COMPONENT_KEYS):
        return "already_present", 0.0
    if not _is_number(row.get("dockq")):
        # These 672 rows carry no DockQ either -- the manifest has all four fields blank
        # for them -- so they were already dropped by the NaN filter and the composite
        # loss costs nothing extra here.  Counted apart from the real misses so the two
        # numbers are not conflated.
        return "no_dockq", 0.0
    found = manifest.get(str(row.get("structure_path", "")))
    if found is None:
        return "unmatched", 0.0
    for key in COMPONENT_KEYS:
        row[key] = found[key]
    return "filled", abs(found["dockq"] - float(row["dockq"]))  # type: ignore[arg-type]


def _backfill_rows(rows: list[dict[str, object]], manifest: dict[str, dict[str, float]]) -> tuple[Counter, float]:
    """Fill in every row the manifest covers, in place; returns tallies and the worst gap."""
    tally: Counter = Counter()
    worst_dockq_gap = 0.0
    for row in rows:
        tally["rows"] += 1
        outcome, gap = _fill_row(row, manifest)
        tally[outcome] += 1
        worst_dockq_gap = max(worst_dockq_gap, gap)
    return tally, worst_dockq_gap


def _backfill_score_file(path: Path, manifest: dict[str, dict[str, float]], *, dry_run: bool) -> tuple[Counter, float]:
    """Rewrite one ``scores/<entry>.jsonl`` with the sub-scores; returns tallies and the worst gap.

    A file with nothing to fill is left untouched rather than rewritten identically,
    so a re-run of a finished backfill is a no-op on disk.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tally, worst_dockq_gap = _backfill_rows(rows, manifest)
    if tally["filled"] and not dry_run:
        payload = "".join(json.dumps(r) + "\n" for r in rows)
        _write_atomically(path, lambda tmp, payload=payload: tmp.write_text(payload))
        tally["files_written"] += 1
    return tally, worst_dockq_gap


def _log_score_summary(tally: Counter, n_files: int, worst_dockq_gap: float, unmatched_entries: Counter) -> None:
    """Report what the score pass did, and which entries came away short."""
    LOGGER.info(
        "scores: %d rows across %d files -- %d filled, %d already present, %d unmatched, %d had no DockQ to begin with",
        tally["rows"],
        n_files,
        tally["filled"],
        tally["already_present"],
        tally["unmatched"],
        tally["no_dockq"],
    )
    LOGGER.info("max |manifest.dockq - cache.dockq| = %.3g (a real join, not a recompute)", worst_dockq_gap)
    if unmatched_entries:
        # Name them: the dataset will drop these decoys, and knowing which complexes
        # lost decoys is the difference between an accepted 0.1% and an unnoticed bias.
        LOGGER.warning(
            "%d entries have unmatched decoys, dropped at training time: %s",
            len(unmatched_entries),
            ", ".join(f"{e}({n})" for e, n in sorted(unmatched_entries.items())),
        )


def backfill_scores(cache_dir: Path, manifest: dict[str, dict[str, float]], *, dry_run: bool) -> Counter:
    """Add the sub-scores to every ``scores/<entry>.jsonl``; returns tallies."""
    score_paths = sorted((cache_dir / "scores").glob("*.jsonl"))
    if not score_paths:
        msg = f"no score files under {cache_dir / 'scores'}"
        raise ValueError(msg)

    tally: Counter = Counter()
    worst_dockq_gap = 0.0
    unmatched_entries: Counter = Counter()
    for path in score_paths:
        file_tally, gap = _backfill_score_file(path, manifest, dry_run=dry_run)
        tally.update(file_tally)
        worst_dockq_gap = max(worst_dockq_gap, gap)
        if file_tally["unmatched"]:
            unmatched_entries[path.stem] = file_tally["unmatched"]

    _log_score_summary(tally, len(score_paths), worst_dockq_gap, unmatched_entries)
    return tally


def load_score_components(cache_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Map ``(entry, decoy_id)`` -> sub-scores, read back from the JSONLs."""
    table: dict[tuple[str, str], dict[str, float]] = {}
    for path in sorted((cache_dir / "scores").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(_is_number(row.get(k)) for k in COMPONENT_KEYS):
                continue
            table[(path.stem, str(row["decoy_id"]))] = {k: float(row[k]) for k in COMPONENT_KEYS}
    LOGGER.info("score index: %d decoys with sub-scores", len(table))
    return table


def _save_npz(path: Path, blob: dict[str, object]) -> None:
    """Write an npz to exactly ``path``.

    Through a file handle rather than a name: given a name that does not end in
    ``.npz``, ``savez_compressed`` appends the extension, so the temp file it
    actually created was not the one the atomic rename then looked for.
    """
    with path.open("wb") as handle:
        np.savez_compressed(handle, **blob)


def _quarantine_shard(path: Path, *, dry_run: bool) -> str:
    """Move a shard none of whose decoys matched out of the way; returns the outcome tag.

    The 6 unmatched train entries are unmatched in *every* decoy, so there is nothing to
    upgrade and nothing left to train on.  Moved aside rather than deleted (they cost a
    GPU pass to make) and rather than left in place, where they would trip
    ``load_latents``' components guard on every run.
    """
    quarantine = path.parent.parent / f"{path.parent.name}_no_components"
    LOGGER.warning("%s: no decoy matched the scores; moving to %s/", path.name, quarantine.name)
    if not dry_run:
        quarantine.mkdir(parents=True, exist_ok=True)
        path.replace(quarantine / path.name)
    return "no_match"


def _is_per_decoy(key: str, value: object, keep: np.ndarray) -> bool:
    """Whether ``value`` is an array indexed by decoy, so it has to lose the dropped rows.

    ``metadata`` is a JSON scalar and the sub-scores were just built at the kept length,
    so both are excluded by name; everything else is judged by its leading axis.
    """
    if key == "metadata" or key in COMPONENT_KEYS:
        return False
    return isinstance(value, np.ndarray) and value.shape[:1] == keep.shape


def _drop_missing_decoys(blob: dict[str, object], keep: np.ndarray) -> None:
    """Keep only the matched decoys' rows, in place, and correct ``n_decoys``.

    Every per-decoy array has to lose the same rows, or the labels end up paired with
    another decoy's latents.
    """
    for key, value in list(blob.items()):
        if _is_per_decoy(key, value, keep):
            blob[key] = value[keep]  # type: ignore[index]
    meta = json.loads(str(blob["metadata"]))
    meta["n_decoys"] = int(keep.sum())
    blob["metadata"] = json.dumps(meta)


def _upgrade_shard(path: Path, scores: dict[tuple[str, str], dict[str, float]], *, dry_run: bool) -> str:
    """Add or repair one shard's sub-score arrays; returns the outcome tag."""
    blob = dict(np.load(path, allow_pickle=True).items())
    if all(k in blob for k in COMPONENT_KEYS):
        return "already_present"

    decoy_ids = [str(d) for d in blob["decoy_ids"]]
    found = [scores.get((path.stem, d)) for d in decoy_ids]
    keep = np.array([f is not None for f in found], dtype=bool)
    if not keep.any():
        return _quarantine_shard(path, dry_run=dry_run)

    for key in COMPONENT_KEYS:
        blob[key] = np.array([f[key] for f in found if f is not None], dtype=np.float32)
    if not keep.all():
        _drop_missing_decoys(blob, keep)

    if not dry_run:
        _write_atomically(path, lambda tmp, blob=blob: _save_npz(tmp, blob))
    return "upgraded" if keep.all() else "upgraded_with_drops"


def backfill_latents(latent_dir: Path, scores: dict[tuple[str, str], dict[str, float]], *, dry_run: bool) -> Counter:
    """Add the sub-score arrays to every ``latents/<entry>.npz``; returns tallies."""
    root = latent_dir / "latents" if (latent_dir / "latents").is_dir() else latent_dir
    paths = sorted(root.glob("*.npz"))
    if not paths:
        msg = f"no latent shards under {root}"
        raise ValueError(msg)

    tally: Counter = Counter()
    for path in paths:
        tally[_upgrade_shard(path, scores, dry_run=dry_run)] += 1
    LOGGER.info("latents: %d shards -- %s", len(paths), dict(tally))
    return tally


def main(argv: list[str] | None = None) -> int:
    """Backfill the sub-scores into the score JSONLs and, optionally, the latents."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    if args.dry_run:
        LOGGER.info("dry run: nothing will be written")

    if not args.latent_only:
        if not args.manifest:
            LOGGER.error("--manifest is required unless --latent_only is given")
            return 1
        manifest = load_manifest_components(args.manifest)
        backfill_scores(args.cache_dir, manifest, dry_run=args.dry_run)

    if args.latent_dir is not None:
        if args.dry_run and not args.latent_only:
            # The latent pass reads the JSONLs the previous pass would have written,
            # so on a dry run it would see the old files and report nothing to do.
            LOGGER.warning("skipping the latent pass: a dry run leaves the JSONLs un-backfilled")
        else:
            backfill_latents(args.latent_dir, load_score_components(args.cache_dir), dry_run=args.dry_run)

    LOGGER.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
