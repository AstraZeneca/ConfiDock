"""Tests for the DockQ sub-score backfill.

The script rewrites the confidence caches **in place and without a backup**, on the
strength of one claim: that adding ``fnat``/``irmsd``/``lrmsd`` is an exact join on the
structure path, not a recompute.  These tests pin the three things that claim rests on
-- that a filled row's own ``dockq`` is never touched, that a row the manifest does not
cover is left alone rather than guessed at, and that a file with nothing to fill is not
rewritten at all -- plus the atomicity that makes an interrupted run recoverable.

The tallies are checked per outcome because they are how a run is read: ``unmatched``
decoys silently disappear at training time, so the count and the entry names in the log
are the only warning that a complex lost decoys.

The latent pass has one failure mode worse than losing a decoy: a shard's arrays are
positional, so dropping the unmatched rows from some of them and not others pairs a
decoy's latents with another decoy's labels, and training would simply proceed.  Those
tests check every array's length together rather than one at a time.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1] / "workflows" / "scripts"
sys.path.insert(0, str(WORKFLOWS_ROOT / "data"))

import backfill_dockq_components as backfill  # noqa: E402

from confidock.boltz.finetune.latent_cache import COMPONENT_KEYS  # noqa: E402

MANIFEST_FIELDS = ["pdb_path", *COMPONENT_KEYS, "dockq"]


def _row(decoy: str, structure_path: str, dockq: float | None, **extra: object) -> dict[str, object]:
    """One line of a cache's ``scores/<entry>.jsonl``."""
    return {"decoy_id": decoy, "structure_path": structure_path, "dockq": dockq, **extra}


def _write_scores(cache_dir: Path, rows_by_entry: dict[str, list[dict[str, object]]]) -> Path:
    """Lay out ``{cache_dir}/scores/<entry>.jsonl`` and return the cache dir."""
    scores = cache_dir / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    for entry, rows in rows_by_entry.items():
        (scores / f"{entry}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return cache_dir


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a DockQ manifest CSV holding only the columns the join reads."""
    lines = [",".join(MANIFEST_FIELDS)]
    lines += [",".join("" if row.get(f) is None else str(row.get(f, "")) for f in MANIFEST_FIELDS) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _manifest(*rows: tuple[str, float, float, float, float]) -> dict[str, dict[str, float]]:
    """An in-memory manifest table, as :func:`load_manifest_components` would return it."""
    return {
        pdb_path: {"fnat": fnat, "irmsd": irmsd, "lrmsd": lrmsd, "dockq": dockq}
        for pdb_path, fnat, irmsd, lrmsd, dockq in rows
    }


def _read_rows(cache_dir: Path, entry: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (cache_dir / "scores" / f"{entry}.jsonl").read_text().splitlines()]


def _write_shard(latent_dir: Path, entry: str, decoys: list[str], *, components: bool = False, **extra: object) -> Path:
    """Write ``{latent_dir}/latents/<entry>.npz`` the way :func:`save_latent_shard` does.

    Only the keys the backfill has to reason about: the pooled latents, the labels, the
    decoy axis it joins on, one baseline metric standing in for the rest, and the JSON
    ``metadata`` blob holding ``n_decoys``.
    """
    n = len(decoys)
    payload: dict[str, object] = {
        "pooled": np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        "dockq": np.linspace(0.1, 0.9, n, dtype=np.float32),
        "decoy_ids": np.asarray(decoys, dtype=object),
        "sources": np.asarray(["boltz2"] * n, dtype=object),
        "baseline__confidence_score": np.linspace(0.5, 0.6, n, dtype=np.float32),
        "metadata": json.dumps({"entry": entry, "pooling_version": 1, "n_decoys": n}),
    }
    if components:
        payload.update({k: np.full(n, 0.25, dtype=np.float32) for k in COMPONENT_KEYS})
    payload.update(extra)
    path = latent_dir / "latents" / f"{entry}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return path


def _read_shard(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True).items())


def _scores(entry: str, decoys: list[str]) -> dict[tuple[str, str], dict[str, float]]:
    """A score index as :func:`load_score_components` returns it, one row per decoy."""
    return {
        (entry, decoy): {"fnat": 0.4 + 0.01 * i, "irmsd": 3.0 + i, "lrmsd": 9.0 + i} for i, decoy in enumerate(decoys)
    }


def test_fills_the_sub_scores_without_touching_dockq(tmp_path):
    cache = _write_scores(tmp_path, {"e0": [_row("d0", "/s/d0.cif", 0.5), _row("d1", "/s/d1.cif", 0.25)]})
    manifest = _manifest(("/s/d0.cif", 0.4, 3.0, 9.0, 0.5), ("/s/d1.cif", 0.1, 6.0, 15.0, 0.25))

    tally = backfill.backfill_scores(cache, manifest, dry_run=False)

    rows = _read_rows(cache, "e0")
    assert tally["filled"] == 2
    assert [r["decoy_id"] for r in rows] == ["d0", "d1"], "the rewrite must preserve row order"
    assert [r["dockq"] for r in rows] == pytest.approx([0.5, 0.25]), "dockq is the join key's witness, not an output"
    assert rows[0]["fnat"] == pytest.approx(0.4)
    assert rows[1]["lrmsd"] == pytest.approx(15.0)
    # The manifest's own dockq is a check, not a column: it must not leak into the cache.
    assert all(set(r) == {"decoy_id", "structure_path", "dockq", *COMPONENT_KEYS} for r in rows)


def test_every_row_is_tallied_under_exactly_one_outcome(tmp_path):
    cache = _write_scores(
        tmp_path,
        {
            "e0": [
                _row("d0", "/s/d0.cif", 0.5),
                _row("d1", "/s/d1.cif", 0.7, fnat=0.1, irmsd=1.0, lrmsd=2.0),
                _row("d2", "/s/d2.cif", None),
                _row("d3", "/s/gone.cif", 0.3),
            ],
            "e1": [_row("d0", "/s/e1d0.cif", 0.9)],
        },
    )
    manifest = _manifest(("/s/d0.cif", 0.4, 3.0, 9.0, 0.5), ("/s/e1d0.cif", 0.8, 1.0, 2.0, 0.9))

    tally = backfill.backfill_scores(cache, manifest, dry_run=False)

    assert tally["rows"] == 5
    assert tally["filled"] == 2
    assert tally["already_present"] == 1
    # A row with no DockQ is counted apart from a real miss: it was already dropped by
    # the NaN filter, so it costs the composite loss nothing.
    assert tally["no_dockq"] == 1
    assert tally["unmatched"] == 1
    assert tally["filled"] + tally["already_present"] + tally["no_dockq"] + tally["unmatched"] == tally["rows"]
    assert tally["files_written"] == 2


def test_an_unmatched_row_is_left_alone_and_its_entry_is_named(tmp_path, caplog):
    cache = _write_scores(
        tmp_path,
        {"e0": [_row("d0", "/s/gone.cif", 0.3)], "e1": [_row("d0", "/s/d0.cif", 0.5)]},
    )
    manifest = _manifest(("/s/d0.cif", 0.4, 3.0, 9.0, 0.5))

    with caplog.at_level(logging.WARNING, logger=backfill.LOGGER.name):
        backfill.backfill_scores(cache, manifest, dry_run=False)

    assert _read_rows(cache, "e0") == [{"decoy_id": "d0", "structure_path": "/s/gone.cif", "dockq": pytest.approx(0.3)}]
    # Naming the entries is the point of the warning: the dataset drops these decoys
    # later, so this is the only place the loss is attributable to a complex.
    assert "e0(1)" in caplog.text
    assert "e1" not in caplog.text


def test_the_dockq_agreement_of_the_join_is_reported(tmp_path, caplog):
    cache = _write_scores(tmp_path, {"e0": [_row("d0", "/s/d0.cif", 0.5), _row("d1", "/s/d1.cif", 0.5)]})
    # The second row's manifest DockQ is off by 0.02, which would mean the two sides are
    # not describing the same structure -- the number the operator has to see.
    manifest = _manifest(("/s/d0.cif", 0.4, 3.0, 9.0, 0.5), ("/s/d1.cif", 0.4, 3.0, 9.0, 0.52))

    with caplog.at_level(logging.INFO, logger=backfill.LOGGER.name):
        backfill.backfill_scores(cache, manifest, dry_run=False)

    assert "max |manifest.dockq - cache.dockq| = 0.02" in caplog.text


def test_a_file_with_nothing_to_fill_is_not_rewritten(tmp_path):
    rows = {"e0": [_row("d0", "/s/d0.cif", 0.5, fnat=0.1, irmsd=1.0, lrmsd=2.0)], "e1": [_row("d0", "/s/x.cif", None)]}
    cache = _write_scores(tmp_path, rows)
    before = {p: p.read_bytes() for p in sorted((cache / "scores").glob("*.jsonl"))}

    tally = backfill.backfill_scores(cache, _manifest(("/s/d0.cif", 0.4, 3.0, 9.0, 0.5)), dry_run=False)

    # Re-running a finished backfill has to be a no-op on disk, not an identical rewrite:
    # every rewrite is another chance to truncate a cache that has no backup.
    assert tally["files_written"] == 0
    assert {p: p.read_bytes() for p in sorted((cache / "scores").glob("*.jsonl"))} == before


def test_dry_run_reports_the_tallies_and_writes_nothing(tmp_path):
    cache = _write_scores(tmp_path, {"e0": [_row("d0", "/s/d0.cif", 0.5)]})
    before = (cache / "scores" / "e0.jsonl").read_bytes()

    tally = backfill.backfill_scores(cache, _manifest(("/s/d0.cif", 0.4, 3.0, 9.0, 0.5)), dry_run=True)

    assert tally["filled"] == 1
    assert tally["files_written"] == 0
    assert (cache / "scores" / "e0.jsonl").read_bytes() == before


def test_a_cache_with_no_score_files_is_an_error(tmp_path):
    (tmp_path / "scores").mkdir()

    with pytest.raises(ValueError, match="no score files under"):
        backfill.backfill_scores(tmp_path, {}, dry_run=False)


def test_a_failed_write_leaves_the_original_intact(tmp_path):
    path = tmp_path / "e0.jsonl"
    path.write_text("original\n")

    def failing(tmp: Path) -> None:
        tmp.write_text("half a file")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        backfill._write_atomically(path, failing)

    assert path.read_text() == "original\n"
    assert list(tmp_path.iterdir()) == [path], "the temp file must not be left behind"


def test_the_manifest_keeps_the_first_of_two_disagreeing_rows(tmp_path, caplog):
    agreed = {"pdb_path": "/s/same.cif", "fnat": 0.2, "irmsd": 2.0, "lrmsd": 4.0, "dockq": 0.3}
    first = _write_manifest(
        tmp_path / "a.csv", [{"pdb_path": "/s/d0.cif", "fnat": 0.4, "irmsd": 3.0, "lrmsd": 9.0, "dockq": 0.5}, agreed]
    )
    second = _write_manifest(
        tmp_path / "b.csv", [{"pdb_path": "/s/d0.cif", "fnat": 0.9, "irmsd": 3.0, "lrmsd": 9.0, "dockq": 0.5}, agreed]
    )

    with caplog.at_level(logging.WARNING, logger=backfill.LOGGER.name):
        table = backfill.load_manifest_components([first, second])

    # Order-independence is the point: whichever manifest was passed first would
    # otherwise decide the label, and the conflict would go unnoticed.
    assert table["/s/d0.cif"]["fnat"] == pytest.approx(0.4)
    # val_easy is covered by two overlapping manifests, so a path in both is normal; only
    # a *disagreement* is worth a warning.
    assert table["/s/same.cif"]["fnat"] == pytest.approx(0.2)
    assert "1 structure paths disagreed" in caplog.text


def test_the_manifest_skips_rows_missing_any_of_the_four_numbers(tmp_path):
    path = _write_manifest(
        tmp_path / "m.csv",
        [
            {"pdb_path": "/s/ok.cif", "fnat": 0.4, "irmsd": 3.0, "lrmsd": 9.0, "dockq": 0.5},
            {"pdb_path": "/s/no_dockq.cif", "fnat": 0.4, "irmsd": 3.0, "lrmsd": 9.0, "dockq": None},
            {"pdb_path": "/s/no_fnat.cif", "fnat": None, "irmsd": 3.0, "lrmsd": 9.0, "dockq": 0.5},
            {"pdb_path": None, "fnat": 0.4, "irmsd": 3.0, "lrmsd": 9.0, "dockq": 0.5},
        ],
    )

    assert list(backfill.load_manifest_components([path])) == ["/s/ok.cif"]


def test_backfilled_rows_read_back_as_components(tmp_path):
    cache = _write_scores(
        tmp_path,
        {"e0": [_row("d0", "/s/d0.cif", 0.5), _row("d1", "/s/gone.cif", 0.3)]},
    )
    # A trailing blank line: both passes read these files line by line, and a cache that
    # ends in a newline is the normal case rather than a corrupt one.
    with (cache / "scores" / "e0.jsonl").open("a") as handle:
        handle.write("\n")

    backfill.backfill_scores(cache, _manifest(("/s/d0.cif", 0.4, 3.0, 9.0, 0.5)), dry_run=False)
    table = backfill.load_score_components(cache)

    # The latent pass joins on (entry, decoy_id), so what the score pass wrote has to be
    # findable under the file's stem -- and the unmatched decoy must stay absent.
    assert list(table) == [("e0", "d0")]
    assert table[("e0", "d0")] == pytest.approx({"fnat": 0.4, "irmsd": 3.0, "lrmsd": 9.0})


def test_an_already_upgraded_shard_is_not_rewritten(tmp_path):
    path = _write_shard(tmp_path, "e0", ["d0", "d1"], components=True)
    before = path.read_bytes()

    outcome = backfill._upgrade_shard(path, _scores("e0", ["d0", "d1"]), dry_run=False)

    # Same reasoning as the score files: a re-run must not rewrite a shard that is
    # already complete, because the pooled latents cost a GPU pass and have no backup.
    assert outcome == "already_present"
    assert path.read_bytes() == before


def test_the_sub_score_arrays_follow_the_shard_s_own_decoy_order(tmp_path):
    path = _write_shard(tmp_path, "e0", ["d0", "d1", "d2"])
    # Deliberately indexed in another order than the shard's: the score index is a dict
    # keyed by decoy, so only the shard's ``decoy_ids`` may decide the row order.
    scores = _scores("e0", ["d2", "d1", "d0"])

    outcome = backfill._upgrade_shard(path, scores, dry_run=False)

    blob = _read_shard(path)
    assert outcome == "upgraded"
    assert [str(d) for d in blob["decoy_ids"]] == ["d0", "d1", "d2"]
    assert blob["fnat"].tolist() == pytest.approx([0.42, 0.41, 0.4])
    assert all(blob[k].dtype == np.float32 for k in COMPONENT_KEYS), "the labels are read as float32 tensors"
    # Everything that was already there survives the rewrite unchanged.
    assert blob["pooled"] == pytest.approx(np.arange(9, dtype=np.float32).reshape(3, 3))
    assert json.loads(str(blob["metadata"])) == {"entry": "e0", "pooling_version": 1, "n_decoys": 3}


def test_unmatched_decoys_are_dropped_from_every_per_decoy_array(tmp_path):
    path = _write_shard(
        tmp_path,
        "e0",
        ["d0", "d1", "d2", "d3"],
        # A per-decoy matrix and a same-dtype array that is *not* indexed by decoy: the
        # leading axis is the only thing that may decide which arrays get trimmed.
        pair_scores=np.arange(16, dtype=np.float32).reshape(4, 4),
        in_dim=np.asarray([3.0], dtype=np.float32),
    )
    kept = _read_shard(path)

    outcome = backfill._upgrade_shard(path, _scores("e0", ["d1", "d3"]), dry_run=False)

    blob = _read_shard(path)
    assert outcome == "upgraded_with_drops"
    # One length for every per-decoy array, or a decoy's latents end up carrying another
    # decoy's label and nothing downstream would notice.
    per_decoy = [
        "pooled",
        "dockq",
        "decoy_ids",
        "sources",
        "baseline__confidence_score",
        "pair_scores",
        *COMPONENT_KEYS,
    ]
    assert {k: len(blob[k]) for k in per_decoy} == dict.fromkeys(per_decoy, 2)
    assert [str(d) for d in blob["decoy_ids"]] == ["d1", "d3"]
    assert blob["dockq"] == pytest.approx(kept["dockq"][[1, 3]])
    assert blob["pooled"] == pytest.approx(kept["pooled"][[1, 3]])
    assert blob["fnat"].tolist() == pytest.approx(
        [0.4, 0.41]
    ), "the components pair with the kept decoys, not with the dropped ones"
    # ``n_decoys`` is what the loader reports, so a stale one would misdescribe the shard.
    assert json.loads(str(blob["metadata"]))["n_decoys"] == 2
    assert blob["in_dim"] == pytest.approx([3.0]), "an array that is not indexed by decoy must be left alone"


def test_a_shard_no_decoy_of_which_matched_is_moved_aside(tmp_path, caplog):
    path = _write_shard(tmp_path, "e0", ["d0", "d1"])

    with caplog.at_level(logging.WARNING, logger=backfill.LOGGER.name):
        outcome = backfill._upgrade_shard(path, _scores("other", ["d0", "d1"]), dry_run=False)

    assert outcome == "no_match"
    # Moved rather than deleted (a GPU pass made it) and rather than left in place, where
    # it would trip the loader's components guard on every run.
    assert not path.exists()
    assert (tmp_path / "latents_no_components" / "e0.npz").exists()
    assert "no decoy matched" in caplog.text


def test_dry_run_reports_the_outcomes_and_moves_and_writes_nothing(tmp_path):
    upgrade = _write_shard(tmp_path, "e0", ["d0", "d1"])
    unmatched = _write_shard(tmp_path, "e1", ["d0"])
    before = {p: p.read_bytes() for p in (upgrade, unmatched)}

    outcomes = [
        backfill._upgrade_shard(upgrade, _scores("e0", ["d0", "d1"]), dry_run=True),
        backfill._upgrade_shard(unmatched, _scores("e0", ["d0"]), dry_run=True),
    ]

    assert outcomes == ["upgraded", "no_match"]
    assert {p: p.read_bytes() for p in (upgrade, unmatched)} == before
    assert not (tmp_path / "latents_no_components").exists()


def test_backfill_latents_tallies_every_shard(tmp_path):
    _write_shard(tmp_path, "e0", ["d0", "d1"])
    _write_shard(tmp_path, "e1", ["d0", "d1"], components=True)
    _write_shard(tmp_path, "e2", ["d0", "d1"])
    _write_shard(tmp_path, "e3", ["d0", "d1"])
    scores = {**_scores("e0", ["d0", "d1"]), **_scores("e2", ["d0"])}

    tally = backfill.backfill_latents(tmp_path, scores, dry_run=False)

    assert dict(tally) == {"upgraded": 1, "already_present": 1, "upgraded_with_drops": 1, "no_match": 1}
    assert sum(tally.values()) == 4, "every shard is tallied under exactly one outcome"


def test_a_latent_dir_is_taken_either_with_or_without_the_latents_subdir(tmp_path):
    # The cache writes latents/<entry>.npz, but the quarantine dir and hand-assembled
    # directories are flat, so a dir of shards is accepted as-is.
    flat = tmp_path / "flat"
    flat.mkdir()
    nested = _write_shard(tmp_path / "nested", "e0", ["d0"])
    nested.replace(flat / "e0.npz")

    tally = backfill.backfill_latents(flat, _scores("e0", ["d0"]), dry_run=False)

    assert dict(tally) == {"upgraded": 1}
    assert _read_shard(flat / "e0.npz")["fnat"].tolist() == pytest.approx([0.4])


def test_a_latent_dir_with_no_shards_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no latent shards under"):
        backfill.backfill_latents(tmp_path, {}, dry_run=False)


def test_the_two_passes_join_on_entry_and_decoy_id(tmp_path):
    cache = _write_scores(tmp_path / "cache", {"e0": [_row("d0", "/s/d0.cif", 0.5), _row("d1", "/s/d1.cif", 0.25)]})
    path = _write_shard(tmp_path / "latents", "e0", ["d1", "d0"])
    manifest = _manifest(("/s/d0.cif", 0.4, 3.0, 9.0, 0.5), ("/s/d1.cif", 0.1, 6.0, 15.0, 0.25))

    backfill.backfill_scores(cache, manifest, dry_run=False)
    backfill.backfill_latents(tmp_path / "latents", backfill.load_score_components(cache), dry_run=False)

    # End to end the two passes are one join, keyed by the entry's file stem on one side
    # and its shard name on the other; the shard's order is what the labels must follow.
    blob = _read_shard(path)
    assert blob["fnat"].tolist() == pytest.approx([0.1, 0.4])
    assert blob["lrmsd"].tolist() == pytest.approx([15.0, 9.0])
