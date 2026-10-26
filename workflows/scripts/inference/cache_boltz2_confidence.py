r"""Cache Boltz-2 confidence-head inputs for every labelled decoy in sabdab_train.

What this does
--------------
For each SAbDab entry it runs the Boltz-2 trunk **once**, then scores every decoy
of that entry with the confidence head, saving the head's inputs so the head can
later be fine-tuned without touching the trunk.

Decoys come from all four prediction sources under the sabdab_train root
(``boltz2_r1``, ``esmfold2_r1``, ``protenix_r1``, ``perturbed_structures``) and
are joined to their existing ``dockq_scores.csv`` labels.

Why the trunk only runs once per entry
--------------------------------------
The trunk never sees coordinates -- in ``boltz/model/modules/trunkv2.py`` the only
coordinate use is ``feats["template_cb"]``/``template_ca`` inside the template
modules, which are unused here.  So ``s``/``z`` depend on sequence + MSA alone
and are identical for all ~33 decoys of an entry.  Verified bit-exact against the
full ``model.forward`` (max |Δ| = 0.0) with a fixed seed; measured on an A100 at
38.4 s for the trunk versus 1.25 s per decoy.

Scoring semantics
-----------------
The model runs with ``skip_run_structure=True``, so the confidence head scores the
decoy we supply rather than a freshly diffused structure (boltz2.py:750).  That
keeps each cached record in 1:1 correspondence with the DockQ label of the
structure it was computed from.  Passing decoys as YAML ``templates`` would not:
Boltz would re-predict a structure and the label would describe a different one.

Output layout::

    {output_dir}/
        run_config.shard<N>.json         # full configuration of each shard
        entries/<entry>.pt               # s_inputs, s, z, feats  (per entry)
        decoys/<entry>/<decoy_id>.npz    # x_pred + presence mask (per decoy)
        scores/<entry>.jsonl             # confidence metrics + DockQ per decoy
        manifest.shard<N>.jsonl          # one line per completed entry
        failures.jsonl                   # per-decoy errors, if any

Usage::

    poetry run python workflows/scripts/inference/cache_boltz2_confidence.py \
        --root       /data/confidock/sabdab_train \
        --output_dir /data/confidock/sabdab_train/confidence_cache \
        --boltz_cache /data/boltz-cache

Shard across GPUs (see ``cache_boltz2_confidence_auto_restart.sh``)::

    CUDA_VISIBLE_DEVICES=0 ... --shard_index 0 --num_shards 6

The run is resumable: entries whose scores file is already complete are skipped
unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from boltz.data.types import Manifest, StructureV2

from confidock.boltz.inference.cache_runner import (
    CONFIDENCE_SCALARS,
    CacheRunConfig,
    build_entry_cache,
    build_x_pred,
    featurize_entry,
    load_cache_model,
    run_confidence_head,
    run_trunk,
)
from confidock.boltz.inference.confidence_cache import (
    save_decoy_coords,
    save_entry_cache,
)
from confidock.boltz.inference.decoy_coords import load_decoy_coords
from confidock.boltz.inference.decoy_sources import (
    SOURCES,
    collect_decoys,
    collect_decoys_from_manifest,
)
from confidock.paths import default_boltz_cache

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/data/confidock/sabdab_train")
DEFAULT_BOLTZ_CACHE = default_boltz_cache()
#: The processed dir from the boltz2_r1 grid run; supplies structures + MSAs.
DEFAULT_PROCESSED = "boltz2_r1/msa=full_msa__recycling=3__scale=1.5/boltz_results__staged_inputs/processed"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="sabdab_train directory")
    parser.add_argument(
        "--manifest_csv",
        type=Path,
        default=None,
        help="take decoys from this CSV instead of walking <root> (the validation "
        "sets use a different directory layout and have no perturbed source)",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="where to write the cache")
    parser.add_argument("--boltz_cache", type=Path, default=DEFAULT_BOLTZ_CACHE)
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=None,
        help=f"Boltz processed dir (default: <root>/{DEFAULT_PROCESSED})",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[s.name for s in SOURCES],
        choices=[s.name for s in SOURCES],
    )
    parser.add_argument("--recycling_steps", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="fixes MSA subsampling (trunkv2.py:632) so runs are reproducible",
    )
    parser.add_argument("--store_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--use_kernels",
        action="store_true",
        help="enable Boltz's Triton kernels (faster, needs a compatible GPU)",
    )
    parser.add_argument(
        "--skip_entry_cache",
        action="store_true",
        help="score decoys but do not write the trunk cache (~46 MiB/entry avg)",
    )
    parser.add_argument("--max_entries", type=int, default=None, help="process at most N entries")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="report the work plan and exit without loading the model",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def group_by_entry(refs: list) -> dict[str, list]:
    """Group decoy refs by entry, preserving the sorted order from collection."""
    grouped: dict[str, list] = {}
    for ref in refs:
        grouped.setdefault(ref.entry, []).append(ref)
    return grouped


def select_shard(entries: list[str], shard_index: int, num_shards: int) -> list[str]:
    """Deterministic round-robin shard so parallel workers never overlap."""
    if num_shards <= 1:
        return entries
    return [e for i, e in enumerate(entries) if i % num_shards == shard_index]


def entry_is_complete(scores_path: Path, expected: int) -> bool:
    """Check whether ``scores_path`` already holds ``expected`` decoy records."""
    if not scores_path.is_file():
        return False
    try:
        with scores_path.open() as handle:
            return sum(1 for line in handle if line.strip()) >= expected
    except OSError:
        return False


def write_run_config(path: Path, args: argparse.Namespace, n_entries: int, n_decoys: int) -> None:
    """Record this shard's full configuration next to its output."""
    payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    payload.update(
        {
            "n_entries": n_entries,
            "n_decoys": n_decoys,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def fmt_duration(seconds: float) -> str:
    """Format a duration as ``1h02m03s`` / ``2m03s`` / ``3s``."""
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------


def process_entry(
    *,
    model,
    config: CacheRunConfig,
    record,
    decoys: list,
    structure: StructureV2,
    output_dir: Path,
    skip_entry_cache: bool,
) -> tuple[int, int]:
    """Run the trunk once and score every decoy of one entry.

    Returns ``(n_scored, n_failed)``.
    """
    entry = record.id
    feats = featurize_entry(config, record)

    # Seed before the trunk: trunkv2.py:632 subsamples the MSA with randperm,
    # so an unseeded run would give a different z for every restart.
    torch.manual_seed(config.seed)
    s_inputs, s, z = run_trunk(model, feats, config.recycling_steps)

    if not skip_entry_cache:
        cache = build_entry_cache(config, record, feats, s_inputs, s, z, structure)
        save_entry_cache(
            cache,
            output_dir / "entries" / f"{entry}.pt",
            dtype=config.torch_store_dtype,
        )

    # Distogram logits are a single linear layer on z; compute once, reuse for
    # every decoy, and never store (they would add ~71 GiB across the dataset).
    pred_distogram_logits = model.distogram_module(z)[:, :, :, 0]

    scores_path = output_dir / "scores" / f"{entry}.jsonl"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failures.jsonl"

    n_scored = n_failed = 0
    with scores_path.open("w") as scores_handle:
        for ref in decoys:
            try:
                decoy = load_decoy_coords(ref.path, structure)
                coords = build_x_pred(decoy.coords, feats, device=config.device)
                out = run_confidence_head(
                    model,
                    s_inputs=s_inputs,
                    s=s,
                    z=z,
                    feats=feats,
                    coords=coords,
                    pred_distogram_logits=pred_distogram_logits,
                )

                save_decoy_coords(
                    output_dir / "decoys" / entry / f"{ref.decoy_id}.npz",
                    decoy.coords,
                    decoy.present,
                    {
                        "entry": entry,
                        "source": ref.source,
                        "condition": ref.condition,
                        "dockq": ref.dockq,
                        "coverage": decoy.coverage,
                        **{k: _jsonable(v) for k, v in ref.metadata.items()},
                    },
                )

                record_out = {
                    "entry": entry,
                    "decoy_id": ref.decoy_id,
                    "source": ref.source,
                    "condition": ref.condition,
                    "dockq": ref.dockq,
                    "coverage": decoy.coverage,
                    "structure_path": str(ref.path),
                }
                for key in CONFIDENCE_SCALARS:
                    if key in out:
                        record_out[key] = float(out[key].item())
                scores_handle.write(json.dumps(record_out, default=_jsonable) + "\n")
                n_scored += 1

            except Exception as exc:  # noqa: BLE001 - one bad decoy must not kill the run
                n_failed += 1
                LOGGER.warning("decoy %s failed: %s", ref.decoy_id, exc)
                with failures_path.open("a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "entry": entry,
                                "decoy_id": ref.decoy_id,
                                "path": str(ref.path),
                                "error": str(exc),
                                "traceback": traceback.format_exc(limit=5),
                            }
                        )
                        + "\n"
                    )

    del s_inputs, s, z, pred_distogram_logits, feats
    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()

    return n_scored, n_failed


def _jsonable(value):
    """Coerce numpy scalars from pandas into plain Python for json.dump."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return str(value)


def _append_jsonl(path: Path, payload: dict) -> None:
    """Append one JSON line; used for both the manifest and the failure log."""
    with path.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")


def _collect_refs(args: argparse.Namespace) -> list:
    """Gather decoy references, either from a manifest CSV or by walking the root."""
    if args.manifest_csv is None:
        LOGGER.info("Collecting decoys from %s", args.root)
        return collect_decoys(args.root, sources=tuple(args.sources))

    LOGGER.info("Collecting decoys from manifest %s", args.manifest_csv)
    # --sources defaults to all four; only narrow if the user asked for a
    # subset, since a validation manifest has no `perturbed` rows to match.
    explicit = set(args.sources) != {s.name for s in SOURCES}
    return collect_decoys_from_manifest(
        args.manifest_csv,
        root=args.root,
        sources=tuple(args.sources) if explicit else None,
    )


def _score_entry(
    model,
    config: CacheRunConfig,
    args: argparse.Namespace,
    *,
    entry: str,
    record,
    decoys: list,
    processed_dir: Path,
    label: str,
) -> tuple[int, int] | None:
    """Score one entry, returning ``(n_scored, n_failed)``, or ``None`` if it failed.

    A bad entry is logged to ``failures.jsonl`` and skipped rather than killing a
    shard that may already have hours of work behind it.
    """
    try:
        structure = StructureV2.load(processed_dir / "structures" / f"{entry}.npz")
        return process_entry(
            model=model,
            config=config,
            record=record,
            decoys=decoys,
            structure=structure,
            output_dir=args.output_dir,
            skip_entry_cache=args.skip_entry_cache,
        )
    except Exception as exc:  # noqa: BLE001 - keep going through bad entries
        LOGGER.exception("%s %s FAILED: %s", label, entry, exc)
        _append_jsonl(
            args.output_dir / "failures.jsonl",
            {"entry": entry, "error": str(exc), "traceback": traceback.format_exc(limit=10)},
        )
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the caching sweep for one shard."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    processed_dir = args.processed_dir or (args.root / DEFAULT_PROCESSED)
    if not processed_dir.is_dir():
        LOGGER.error("processed dir not found: %s", processed_dir)
        return 1

    grouped = group_by_entry(_collect_refs(args))

    manifest = Manifest.load(processed_dir / "manifest.json")
    records = {r.id: r for r in manifest.records}

    entries = [e for e in sorted(grouped) if e in records]
    skipped_no_record = sorted(set(grouped) - set(records))
    if skipped_no_record:
        LOGGER.warning(
            "%d entries have decoys but no processed record (skipped): %s",
            len(skipped_no_record),
            skipped_no_record[:5],
        )

    entries = select_shard(entries, args.shard_index, args.num_shards)
    if args.max_entries is not None:
        entries = entries[: args.max_entries]

    n_decoys = sum(len(grouped[e]) for e in entries)
    LOGGER.info(
        "Shard %d/%d: %d entries, %d decoys",
        args.shard_index,
        args.num_shards,
        len(entries),
        n_decoys,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(
        args.output_dir / f"run_config.shard{args.shard_index}.json",
        args,
        len(entries),
        n_decoys,
    )

    if args.dry_run:
        for entry in entries[:5]:
            LOGGER.info("  %s: %d decoys", entry, len(grouped[entry]))
        LOGGER.info("Dry run complete; no model loaded.")
        return 0

    config = CacheRunConfig(
        processed_dir=processed_dir,
        boltz_cache=args.boltz_cache,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        recycling_steps=args.recycling_steps,
        seed=args.seed,
        store_dtype=args.store_dtype,
        device=args.device,
        use_kernels=args.use_kernels,
    )

    LOGGER.info("Loading model from %s", config.checkpoint_path)
    t_load = time.time()
    model = load_cache_model(config)
    LOGGER.info("Model loaded in %s", fmt_duration(time.time() - t_load))

    manifest_path = args.output_dir / f"manifest.shard{args.shard_index}.jsonl"
    total_scored = total_failed = total_skipped = 0
    t_start = time.time()

    for index, entry in enumerate(entries, start=1):
        label = f"[{index}/{len(entries)}]"
        decoys = grouped[entry]
        scores_path = args.output_dir / "scores" / f"{entry}.jsonl"
        if not args.overwrite and entry_is_complete(scores_path, len(decoys)):
            total_skipped += 1
            LOGGER.info("%s %s already complete, skipping", label, entry)
            continue

        t_entry = time.time()
        result = _score_entry(
            model,
            config,
            args,
            entry=entry,
            record=records[entry],
            decoys=decoys,
            processed_dir=processed_dir,
            label=label,
        )
        if result is None:
            continue

        scored, failed = result
        total_scored += scored
        total_failed += failed
        elapsed = time.time() - t_entry
        _append_jsonl(
            manifest_path,
            {"entry": entry, "n_scored": scored, "n_failed": failed, "seconds": round(elapsed, 2)},
        )

        done = index - total_skipped
        rate = (time.time() - t_start) / max(done, 1)
        remaining = rate * (len(entries) - index)
        LOGGER.info(
            "%s %s: %d scored, %d failed in %s (eta %s)",
            label,
            entry,
            scored,
            failed,
            fmt_duration(elapsed),
            fmt_duration(remaining),
        )

    LOGGER.info(
        "Done: %d decoys scored, %d failed, %d entries skipped, total %s",
        total_scored,
        total_failed,
        total_skipped,
        fmt_duration(time.time() - t_start),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
