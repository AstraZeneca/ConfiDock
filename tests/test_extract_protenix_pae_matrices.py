"""Tests for the Protenix full_data extractor.

This script is the only thing that makes Protenix's confidence output usable: the raw
``*_full_data_*.json`` files are 16+ MB each and a large sweep writes thousands of them, so the
run is normally invoked with ``--delete_full_data`` and the JSON is gone by the time anyone looks. That
makes two contracts load-bearing beyond the usual. The ``.npz`` has to land where the featuriser globs for
it -- beside the JSON, ``_full_data_`` swapped for ``_confidence_`` -- because a right-content file under a
wrong name is indistinguishable from a fold that never wrote confidence at all. And the JSON may only be
deleted after a *successful* extraction, or a single malformed file silently costs the run its data.

The glob-per-level behaviour matters for the same reason: the script is pointed at whichever level of
``{condition}/{entry}/seed_*/predictions/`` the caller happens to have, and a pattern that matches nothing
reports "no full_data JSON files found" and exits 0, which looks exactly like an already-extracted tree.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "workflows" / "scripts" / "processing"))

import extract_protenix_pae_matrices as extract_mod  # noqa: E402
from extract_protenix_pae_matrices import (  # noqa: E402
    FAILED,
    PROCESSED,
    SKIPPED,
    confidence_npz_path,
    extract_pae_from_json,
    full_data_pattern,
    process_one_file,
    process_predictions_dir,
)

N_TOKENS = 6


def _payload(*, with_pae: bool = True, **extra) -> dict:
    """A full_data payload whose every array is distinguishable from every other.

    The values are unique per array and asymmetric across axes so that a transposed matrix, or two arrays
    saved under each other's keys, cannot pass.
    """
    data = {
        "token_pair_pde": (np.arange(N_TOKENS * N_TOKENS, dtype=float).reshape(N_TOKENS, N_TOKENS) / 4).tolist(),
        "contact_probs": (np.arange(N_TOKENS * N_TOKENS, dtype=float).reshape(N_TOKENS, N_TOKENS) / 100).tolist(),
        "atom_plddt": [30.5, 41.25, 55.0, 68.75, 90.5],
        "atom_to_token_idx": [0, 0, 1, 2, 2],
        "token_asym_id": [0, 0, 0, 1, 1, 1],
        "token_has_frame": [1, 1, 0, 1, 1, 1],
    }
    if with_pae:
        data["token_pair_pae"] = (np.arange(N_TOKENS * N_TOKENS, dtype=float).reshape(N_TOKENS, N_TOKENS) / 2).tolist()
    data.update(extra)
    return data


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _predictions_tree(root: Path) -> Path:
    """Build ``{condition}/{entry}/seed_*/predictions/`` the way Protenix writes it, and return the root.

    Two conditions, so a recursive sweep has more than one to find; two seeds and two samples under one
    entry, so the per-file loop has repeats; plus the companion ``.cif`` and ``_summary_confidence_`` files
    that live in the same directory and must never be picked up.
    """
    for entry in ("7xyz_A_B", "8abc_H_L"):
        for seed in (101, 202):
            predictions = root / "msa=full_msa__cycle=10" / entry / f"seed_{seed}" / "predictions"
            for sample in (0, 1):
                _write_json(predictions / f"{entry}_full_data_sample_{sample}.json", _payload())
            (predictions / f"{entry}_sample_0.cif").write_text("data_fake\n", encoding="utf-8")
            _write_json(predictions / f"{entry}_summary_confidence_sample_0.json", {"iptm": 0.71})
    _write_json(
        root / "msa=no_msa__cycle=4" / "6bad_A_B" / "seed_303" / "predictions" / "6bad_A_B_full_data_sample_0.json",
        _payload(with_pae=False),
    )
    return root


def _one_file(tmp_path: Path, name: str = "7xyz_A_B_full_data_sample_0.json", **payload_kwargs) -> tuple[Path, Path]:
    """A single full_data JSON at the depth the script expects, with its predictions dir."""
    predictions = tmp_path / "cond" / "7xyz_A_B" / "seed_101" / "predictions"
    return _write_json(predictions / name, _payload(**payload_kwargs)), predictions


class TestConfidenceNpzPath:
    def test_it_swaps_full_data_for_confidence(self, tmp_path):
        got = confidence_npz_path(tmp_path / "7xyz_A_B_full_data_sample_0.json")
        assert got.name == "7xyz_A_B_confidence_sample_0.npz"

    def test_it_keeps_the_arrays_beside_their_json(self, tmp_path):
        json_path = tmp_path / "seed_101" / "predictions" / "7xyz_A_B_full_data_sample_0.json"
        assert confidence_npz_path(json_path).parent == json_path.parent

    def test_it_preserves_the_sample_index(self, tmp_path):
        assert confidence_npz_path(tmp_path / "e_full_data_sample_4.json").name == "e_confidence_sample_4.npz"

    def test_it_does_not_confuse_two_samples_of_one_entry(self, tmp_path):
        first = confidence_npz_path(tmp_path / "7xyz_A_B_full_data_sample_0.json")
        second = confidence_npz_path(tmp_path / "7xyz_A_B_full_data_sample_1.json")
        assert first != second

    def test_it_replaces_only_the_extension_when_the_marker_is_absent(self, tmp_path):
        """A name Protenix would not write still becomes a .npz rather than silently overwriting the JSON."""
        assert confidence_npz_path(tmp_path / "oddly_named.json").name == "oddly_named.npz"


class TestFullDataPattern:
    def test_recursive_sweeps_the_whole_subtree(self, tmp_path):
        assert full_data_pattern(tmp_path, recursive=True) == "**/*_full_data_*.json"

    def test_recursive_ignores_where_in_the_tree_it_was_pointed(self, tmp_path):
        """The recursive pattern must not depend on the seed_* probe: it matches at any depth either way."""
        (tmp_path / "seed_101").mkdir()
        assert full_data_pattern(tmp_path, recursive=True) == "**/*_full_data_*.json"

    def test_an_entry_directory_is_recognised_by_its_seed_dirs(self, tmp_path):
        (tmp_path / "seed_101").mkdir()
        assert full_data_pattern(tmp_path, recursive=False) == "seed_*/predictions/*_full_data_*.json"

    def test_a_condition_directory_gets_the_extra_entry_level(self, tmp_path):
        (tmp_path / "7xyz_A_B" / "seed_101").mkdir(parents=True)
        assert full_data_pattern(tmp_path, recursive=False) == "*/seed_*/predictions/*_full_data_*.json"

    def test_the_entry_pattern_finds_every_seed_and_sample(self, tmp_path):
        root = _predictions_tree(tmp_path)
        entry = root / "msa=full_msa__cycle=10" / "7xyz_A_B"
        found = sorted(p.name for p in entry.glob(full_data_pattern(entry, recursive=False)))
        assert found == [
            "7xyz_A_B_full_data_sample_0.json",
            "7xyz_A_B_full_data_sample_0.json",
            "7xyz_A_B_full_data_sample_1.json",
            "7xyz_A_B_full_data_sample_1.json",
        ]

    def test_the_condition_pattern_finds_every_entry(self, tmp_path):
        condition = _predictions_tree(tmp_path) / "msa=full_msa__cycle=10"
        found = {p.parents[2].name for p in condition.glob(full_data_pattern(condition, recursive=False))}
        assert found == {"7xyz_A_B", "8abc_H_L"}

    def test_a_non_recursive_sweep_of_the_run_root_matches_nothing(self, tmp_path):
        """One level too high is the failure this pattern cannot see: it reports an empty tree, not an error."""
        root = _predictions_tree(tmp_path)
        assert list(root.glob(full_data_pattern(root, recursive=False))) == []

    def test_recursive_reaches_across_conditions(self, tmp_path):
        root = _predictions_tree(tmp_path)
        found = list(root.glob(full_data_pattern(root, recursive=True)))
        assert len(found) == 9

    def test_no_pattern_picks_up_the_companion_files(self, tmp_path):
        root = _predictions_tree(tmp_path)
        found = list(root.glob(full_data_pattern(root, recursive=True)))
        assert all(p.suffix == ".json" and "_full_data_" in p.name for p in found)


class TestProcessOneFile:
    def test_it_extracts_and_reports_processed(self, tmp_path):
        json_path, predictions = _one_file(tmp_path)
        assert process_one_file(json_path, predictions) == PROCESSED
        assert confidence_npz_path(json_path).exists()

    def test_it_keeps_the_json_by_default(self, tmp_path):
        json_path, predictions = _one_file(tmp_path)
        process_one_file(json_path, predictions)
        assert json_path.exists()

    def test_an_existing_npz_is_skipped(self, tmp_path):
        json_path, predictions = _one_file(tmp_path)
        confidence_npz_path(json_path).write_bytes(b"stale")
        assert process_one_file(json_path, predictions) == SKIPPED

    def test_a_skip_leaves_the_existing_arrays_alone(self, tmp_path):
        json_path, predictions = _one_file(tmp_path)
        npz_path = confidence_npz_path(json_path)
        npz_path.write_bytes(b"stale")
        process_one_file(json_path, predictions)
        assert npz_path.read_bytes() == b"stale"

    def test_overwrite_re_extracts_over_a_stale_npz(self, tmp_path):
        json_path, predictions = _one_file(tmp_path)
        npz_path = confidence_npz_path(json_path)
        npz_path.write_bytes(b"stale")
        assert process_one_file(json_path, predictions, overwrite=True) == PROCESSED
        with np.load(npz_path) as arrays:
            assert arrays["pae"].shape == (N_TOKENS, N_TOKENS)

    def test_a_dry_run_writes_nothing(self, tmp_path):
        json_path, predictions = _one_file(tmp_path)
        assert process_one_file(json_path, predictions, dry_run=True) == PROCESSED
        assert not confidence_npz_path(json_path).exists()

    def test_a_dry_run_does_not_delete_the_json(self, tmp_path):
        """--dry_run with --delete_full_data must be safe to run against a tree you care about."""
        json_path, predictions = _one_file(tmp_path)
        process_one_file(json_path, predictions, delete_full_data=True, dry_run=True)
        assert json_path.exists()

    def test_an_already_extracted_file_is_skipped_before_the_dry_run_report(self, tmp_path):
        """The skip decision comes first, so a dry run predicts what a real run would actually do."""
        json_path, predictions = _one_file(tmp_path)
        confidence_npz_path(json_path).write_bytes(b"stale")
        assert process_one_file(json_path, predictions, dry_run=True) == SKIPPED

    def test_delete_full_data_removes_the_json_after_a_success(self, tmp_path):
        json_path, predictions = _one_file(tmp_path)
        assert process_one_file(json_path, predictions, delete_full_data=True) == PROCESSED
        assert not json_path.exists()
        assert confidence_npz_path(json_path).exists()

    def test_delete_full_data_keeps_the_json_when_extraction_fails(self, tmp_path):
        """The only copy of the data is not thrown away on the strength of a failed read."""
        json_path, predictions = _one_file(tmp_path, with_pae=False)
        assert process_one_file(json_path, predictions, delete_full_data=True) == FAILED
        assert json_path.exists()

    def test_a_failed_delete_is_a_warning_not_a_failure(self, tmp_path, monkeypatch, caplog):
        """The arrays are already written, so an undeletable JSON costs disk, not data."""
        json_path, predictions = _one_file(tmp_path)

        def refuse(self, *args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "unlink", refuse)
        with caplog.at_level(logging.WARNING, logger=extract_mod.LOGGER.name):
            assert process_one_file(json_path, predictions, delete_full_data=True) == PROCESSED
        assert "Could not delete" in caplog.text

    def test_a_payload_without_pae_reports_failed(self, tmp_path):
        json_path, predictions = _one_file(tmp_path, with_pae=False)
        assert process_one_file(json_path, predictions) == FAILED
        assert not confidence_npz_path(json_path).exists()

    def test_unreadable_json_reports_failed(self, tmp_path):
        _, predictions = _one_file(tmp_path)
        broken = predictions / "7xyz_A_B_full_data_sample_9.json"
        broken.write_text("{not json at all", encoding="utf-8")
        assert process_one_file(broken, predictions) == FAILED

    def test_the_log_names_the_file_relative_to_the_run_root(self, tmp_path, caplog):
        """Absolute paths make the per-file log unreadable at sweep scale; the relative one is the anchor."""
        json_path, predictions = _one_file(tmp_path)
        with caplog.at_level(logging.INFO, logger=extract_mod.LOGGER.name):
            process_one_file(json_path, predictions)
        assert "Processing 7xyz_A_B_full_data_sample_0.json" in caplog.text


class TestProcessPredictionsDir:
    def test_a_missing_directory_is_reported_and_counted_as_nothing(self, tmp_path, caplog):
        with caplog.at_level(logging.ERROR, logger=extract_mod.LOGGER.name):
            assert process_predictions_dir(tmp_path / "absent") == (0, 0, 0)
        assert "Directory not found" in caplog.text

    def test_a_file_instead_of_a_directory_is_reported_too(self, tmp_path):
        target = tmp_path / "not_a_dir.json"
        target.write_text("{}", encoding="utf-8")
        assert process_predictions_dir(target) == (0, 0, 0)

    def test_an_empty_tree_is_not_an_error(self, tmp_path, caplog):
        (tmp_path / "seed_1" / "predictions").mkdir(parents=True)
        with caplog.at_level(logging.INFO, logger=extract_mod.LOGGER.name):
            assert process_predictions_dir(tmp_path) == (0, 0, 0)
        assert "No full_data JSON files found" in caplog.text

    def test_it_counts_the_three_outcomes_separately(self, tmp_path):
        """One already-extracted file and one unreadable payload, so no two counters can be confused."""
        root = _predictions_tree(tmp_path)
        already = root / "msa=full_msa__cycle=10" / "7xyz_A_B" / "seed_101" / "predictions"
        confidence_npz_path(already / "7xyz_A_B_full_data_sample_0.json").write_bytes(b"stale")
        assert process_predictions_dir(root, recursive=True) == (7, 1, 1)

    def test_a_condition_directory_needs_no_recursive_flag(self, tmp_path):
        condition = _predictions_tree(tmp_path) / "msa=full_msa__cycle=10"
        assert process_predictions_dir(condition) == (8, 0, 0)

    def test_an_entry_directory_needs_no_recursive_flag(self, tmp_path):
        entry = _predictions_tree(tmp_path) / "msa=full_msa__cycle=10" / "7xyz_A_B"
        assert process_predictions_dir(entry) == (4, 0, 0)

    def test_the_run_root_is_only_swept_with_recursive(self, tmp_path):
        root = _predictions_tree(tmp_path)
        assert process_predictions_dir(root) == (0, 0, 0)
        assert process_predictions_dir(root, recursive=True) == (8, 0, 1)

    def test_it_writes_one_npz_per_full_data_json(self, tmp_path):
        root = _predictions_tree(tmp_path)
        process_predictions_dir(root, recursive=True)
        assert len(list(root.rglob("*_confidence_*.npz"))) == 8

    def test_it_leaves_the_companion_files_untouched(self, tmp_path):
        root = _predictions_tree(tmp_path)
        before = {p: p.read_bytes() for p in root.rglob("*") if p.suffix == ".cif" or "_summary_" in p.name}
        process_predictions_dir(root, recursive=True, delete_full_data=True)
        assert {p: p.read_bytes() for p in before} == before

    def test_a_dry_run_counts_without_writing(self, tmp_path):
        root = _predictions_tree(tmp_path)
        assert process_predictions_dir(root, recursive=True, dry_run=True) == (9, 0, 0)
        assert list(root.rglob("*.npz")) == []

    def test_a_dry_run_does_not_notice_the_unreadable_payload(self, tmp_path):
        """It reports what it would attempt, not what would succeed -- the counts differ, deliberately."""
        root = _predictions_tree(tmp_path)
        assert process_predictions_dir(root, recursive=True, dry_run=True)[0] == 9
        assert process_predictions_dir(root, recursive=True)[0] == 8

    def test_delete_full_data_removes_only_what_it_extracted(self, tmp_path):
        root = _predictions_tree(tmp_path)
        process_predictions_dir(root, recursive=True, delete_full_data=True)
        left = [p.name for p in root.rglob("*_full_data_*.json")]
        assert left == ["6bad_A_B_full_data_sample_0.json"]

    def test_a_second_run_skips_everything_it_wrote(self, tmp_path):
        root = _predictions_tree(tmp_path)
        process_predictions_dir(root, recursive=True)
        assert process_predictions_dir(root, recursive=True) == (0, 8, 1)

    def test_overwrite_reprocesses_everything(self, tmp_path):
        root = _predictions_tree(tmp_path)
        process_predictions_dir(root, recursive=True)
        assert process_predictions_dir(root, recursive=True, overwrite=True) == (8, 0, 1)

    def test_files_are_processed_in_sorted_order(self, tmp_path, caplog):
        """Sorted order is what makes two runs' logs comparable when chasing a failure."""
        entry = _predictions_tree(tmp_path) / "msa=full_msa__cycle=10" / "7xyz_A_B"
        with caplog.at_level(logging.INFO, logger=extract_mod.LOGGER.name):
            process_predictions_dir(entry)
        seeds = [m for m in caplog.messages if m.startswith("Processing ")]
        assert seeds == sorted(seeds)

    def test_the_counts_add_up_to_the_files_found(self, tmp_path):
        root = _predictions_tree(tmp_path)
        found = len(list(root.rglob("*_full_data_*.json")))
        assert sum(process_predictions_dir(root, recursive=True)) == found


class TestExtractPaeFromJson:
    def test_it_round_trips_the_pae_values(self, tmp_path):
        json_path = _write_json(tmp_path / "e_full_data_sample_0.json", _payload())
        npz_path = tmp_path / "out.npz"
        assert extract_pae_from_json(json_path, npz_path) is True
        with np.load(npz_path) as arrays:
            expected = np.arange(N_TOKENS * N_TOKENS, dtype=np.float32).reshape(N_TOKENS, N_TOKENS) / 2
            assert np.array_equal(arrays["pae"], expected)

    def test_it_saves_every_optional_array_under_its_own_key(self, tmp_path):
        json_path = _write_json(tmp_path / "e_full_data_sample_0.json", _payload())
        npz_path = tmp_path / "out.npz"
        extract_pae_from_json(json_path, npz_path)
        with np.load(npz_path) as arrays:
            assert sorted(arrays.files) == [
                "atom_plddt",
                "atom_to_token_idx",
                "contact_probs",
                "pae",
                "pde",
                "token_asym_id",
                "token_has_frame",
            ]

    def test_the_index_arrays_stay_integral(self, tmp_path):
        """Saved as floats these become useless as indices downstream, silently."""
        json_path = _write_json(tmp_path / "e_full_data_sample_0.json", _payload())
        npz_path = tmp_path / "out.npz"
        extract_pae_from_json(json_path, npz_path)
        with np.load(npz_path) as arrays:
            assert arrays["atom_to_token_idx"].dtype == np.int32
            assert arrays["token_asym_id"].dtype == np.int32
            assert arrays["token_has_frame"].dtype == np.int8

    def test_the_confidence_matrices_stay_float32(self, tmp_path):
        json_path = _write_json(tmp_path / "e_full_data_sample_0.json", _payload())
        npz_path = tmp_path / "out.npz"
        extract_pae_from_json(json_path, npz_path)
        with np.load(npz_path) as arrays:
            assert {arrays[k].dtype for k in ("pae", "pde", "contact_probs", "atom_plddt")} == {np.dtype(np.float32)}

    def test_pde_and_pae_are_not_interchanged(self, tmp_path):
        json_path = _write_json(tmp_path / "e_full_data_sample_0.json", _payload())
        npz_path = tmp_path / "out.npz"
        extract_pae_from_json(json_path, npz_path)
        with np.load(npz_path) as arrays:
            assert np.allclose(arrays["pae"], arrays["pde"] * 2)

    def test_a_minimal_payload_saves_pae_alone(self, tmp_path):
        json_path = _write_json(tmp_path / "e_full_data_sample_0.json", {"token_pair_pae": [[0.0, 1.0], [2.0, 3.0]]})
        npz_path = tmp_path / "out.npz"
        assert extract_pae_from_json(json_path, npz_path) is True
        with np.load(npz_path) as arrays:
            assert arrays.files == ["pae"]

    def test_a_payload_without_pae_is_refused_and_named(self, tmp_path, caplog):
        json_path = _write_json(tmp_path / "e_full_data_sample_0.json", _payload(with_pae=False))
        npz_path = tmp_path / "out.npz"
        with caplog.at_level(logging.WARNING, logger=extract_mod.LOGGER.name):
            assert extract_pae_from_json(json_path, npz_path) is False
        assert "e_full_data_sample_0.json" in caplog.text
        assert not npz_path.exists()

    def test_unreadable_json_is_refused_and_logged(self, tmp_path, caplog):
        json_path = tmp_path / "e_full_data_sample_0.json"
        json_path.write_text("{not json at all", encoding="utf-8")
        with caplog.at_level(logging.ERROR, logger=extract_mod.LOGGER.name):
            assert extract_pae_from_json(json_path, tmp_path / "out.npz") is False
        assert "Failed to process" in caplog.text

    def test_a_missing_json_is_refused_rather_than_raised(self, tmp_path):
        assert extract_pae_from_json(tmp_path / "absent.json", tmp_path / "out.npz") is False

    def test_it_reports_the_size_reduction_it_achieved(self, tmp_path, caplog):
        """The saving is the whole point of the script, and this line is how an operator confirms it.

        The percentage is checked against the two files rather than against a fixed number: at fixture scale
        the zip overhead dominates and the "reduction" is negative, which is fine as long as it is honest.
        """
        json_path = _write_json(tmp_path / "e_full_data_sample_0.json", _payload())
        npz_path = tmp_path / "out.npz"
        with caplog.at_level(logging.INFO, logger=extract_mod.LOGGER.name):
            extract_pae_from_json(json_path, npz_path)
        expected = (1 - npz_path.stat().st_size / json_path.stat().st_size) * 100
        assert f"({expected:.1f}% reduction)" in caplog.text
        assert "out.npz" in caplog.text
