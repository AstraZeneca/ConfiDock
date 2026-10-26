"""Tests for confidock.protenix.evaluate module."""

import json

import numpy as np
import pytest

from confidock.protenix.evaluate import (
    chain_map_from_stem,
    find_confidence_file,
    find_pae_file,
    load_confidence,
    load_pae_matrix,
    prediction_samples,
)


class TestChainMapFromStem:
    """Tests for chain_map_from_stem function."""

    def test_chain_map_from_stem_basic(self):
        """Test basic chain mapping without native file (sequential A, B, C)."""
        result = chain_map_from_stem("8qf5_H_A")
        # Model uses sequential A, B; native from filename is H, A
        assert result == {"H": "A", "A": "B"}

    def test_chain_map_from_stem_three_chains(self):
        """Test three-chain mapping."""
        result = chain_map_from_stem("8qrg_H_L_A")
        assert result == {"H": "A", "L": "B", "A": "C"}

    def test_chain_map_from_stem_concatenated(self):
        """Test concatenated chain notation."""
        result = chain_map_from_stem("8jlj_HcLcA")
        assert result == {"Hc": "A", "Lc": "B", "A": "C"}

    def test_chain_map_from_stem_with_native_file(self, tmp_path):
        """Test with native PDB file."""
        pdb_content = """
ATOM      1  N   ALA H   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA H   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      3  N   ALA L   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      4  CA  ALA L   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      5  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      6  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C
"""
        native_path = tmp_path / "native.pdb"
        native_path.write_text(pdb_content)

        result = chain_map_from_stem("8qrg_H_L_A", native_path)
        # Native chains from file (H, L, A) -> model chains (A, B, C)
        assert result == {"H": "A", "L": "B", "A": "C"}

    def test_chain_map_from_stem_with_native_mismatch(self, tmp_path):
        """Test with native file that differs from filename."""
        # Filename says HLA, but native file has HLT
        pdb_content = """
ATOM      1  N   ALA H   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA H   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      3  N   ALA L   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      4  CA  ALA L   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      5  N   ALA T   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      6  CA  ALA T   1       2.000   3.000   4.000  1.00 10.00           C
"""
        native_path = tmp_path / "native.pdb"
        native_path.write_text(pdb_content)

        result = chain_map_from_stem("8qrg_H_L_A", native_path)
        # Native chains from file (H, L, T) -> model chains (A, B, C)
        assert result == {"H": "A", "L": "B", "T": "C"}

    def test_chain_map_from_stem_native_nonexistent(self, tmp_path):
        """Test with non-existent native file."""
        result = chain_map_from_stem("8qf5_H_A", tmp_path / "nonexistent.pdb")
        # Should fall back to filename parsing
        assert result == {"H": "A", "A": "B"}

    def test_chain_map_from_stem_native_empty(self, tmp_path):
        """Test with empty native file."""
        native_path = tmp_path / "empty.pdb"
        native_path.write_text("")

        result = chain_map_from_stem("8qf5_H_A", native_path)
        # No chains found in file, fall back to filename
        assert result == {"H": "A", "A": "B"}

    def test_chain_map_from_stem_single_chain(self):
        """Test single chain mapping."""
        result = chain_map_from_stem("1abc_A")
        assert result == {"A": "A"}

    def test_chain_map_from_stem_four_chains(self):
        """Test four-chain mapping."""
        result = chain_map_from_stem("1abc_H_L_A_B")
        assert result == {"H": "A", "L": "B", "A": "C", "B": "D"}


class TestPredictionSamples:
    """Tests for prediction_samples function."""

    def test_prediction_samples_basic(self, tmp_path):
        """Test finding Protenix prediction samples."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry

        # Create seed directories with predictions
        seed1_dir = entry_dir / "seed_1" / "predictions"
        seed1_dir.mkdir(parents=True)
        (seed1_dir / f"{entry}_sample_0.cif").touch()
        (seed1_dir / f"{entry}_sample_1.cif").touch()

        seed2_dir = entry_dir / "seed_2" / "predictions"
        seed2_dir.mkdir(parents=True)
        (seed2_dir / f"{entry}_sample_0.cif").touch()

        results = prediction_samples(predictions_dir, entry)

        assert len(results) == 3
        # Check structure: (seed, sample, path)
        seeds = [r[0] for r in results]
        samples = [r[1] for r in results]

        assert 1 in seeds and 2 in seeds
        assert all(s in [0, 1] for s in samples)

    def test_prediction_samples_nonexistent_entry(self, tmp_path):
        """Test with non-existent entry directory."""
        predictions_dir = tmp_path / "predictions"
        predictions_dir.mkdir()

        results = prediction_samples(predictions_dir, "nonexistent_entry")
        assert results == []

    def test_prediction_samples_no_seed_dirs(self, tmp_path):
        """Test with entry directory but no seed subdirectories."""
        predictions_dir = tmp_path / "predictions"
        entry_dir = predictions_dir / "8qf5_H_A"
        entry_dir.mkdir(parents=True)

        results = prediction_samples(predictions_dir, "8qf5_H_A")
        assert results == []

    def test_prediction_samples_no_predictions_subdir(self, tmp_path):
        """Test with seed directory but no predictions subdirectory."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        seed_dir = predictions_dir / entry / "seed_1"
        seed_dir.mkdir(parents=True)
        # No predictions subdirectory created

        results = prediction_samples(predictions_dir, entry)
        assert results == []

    def test_prediction_samples_other_files_ignored(self, tmp_path):
        """Test that non-matching files are ignored."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        pred_dir = predictions_dir / entry / "seed_1" / "predictions"
        pred_dir.mkdir(parents=True)

        # Create matching and non-matching files
        (pred_dir / f"{entry}_sample_0.cif").touch()
        (pred_dir / "other_file.cif").touch()
        (pred_dir / f"{entry}_summary_confidence_sample_0.json").touch()
        (pred_dir / "README.txt").touch()

        results = prediction_samples(predictions_dir, entry)

        # Should only find the one matching CIF
        assert len(results) == 1

    def test_prediction_samples_sorted_output(self, tmp_path):
        """Test that results are sorted by seed and sample."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"

        # Create in non-sorted order
        seed3_dir = predictions_dir / entry / "seed_3" / "predictions"
        seed3_dir.mkdir(parents=True)
        (seed3_dir / f"{entry}_sample_2.cif").touch()
        (seed3_dir / f"{entry}_sample_0.cif").touch()

        seed1_dir = predictions_dir / entry / "seed_1" / "predictions"
        seed1_dir.mkdir(parents=True)
        (seed1_dir / f"{entry}_sample_1.cif").touch()

        results = prediction_samples(predictions_dir, entry)

        # Should be sorted by seed first, then sample
        assert results[0][0] == 1  # seed_1 first
        assert results[1][0] == 3  # seed_3
        assert results[2][0] == 3  # seed_3
        assert results[1][1] == 0  # sample_0 before sample_2
        assert results[2][1] == 2

    def test_prediction_samples_invalid_seed_name(self, tmp_path):
        """Test that invalid seed directory names are ignored."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"

        # Create invalid seed directory name
        invalid_dir = predictions_dir / entry / "seed_abc" / "predictions"
        invalid_dir.mkdir(parents=True)
        (invalid_dir / f"{entry}_sample_0.cif").touch()

        results = prediction_samples(predictions_dir, entry)
        assert len(results) == 0


class TestFindConfidenceFile:
    """Tests for find_confidence_file function."""

    def test_find_confidence_file_basic(self, tmp_path):
        """Test finding confidence file path."""
        pred_dir = tmp_path / "8qf5_H_A" / "seed_1" / "predictions"
        pred_dir.mkdir(parents=True)
        cif_path = pred_dir / "8qf5_H_A_sample_0.cif"

        result = find_confidence_file(cif_path, "8qf5_H_A", 0)

        assert result == pred_dir / "8qf5_H_A_summary_confidence_sample_0.json"

    def test_find_confidence_file_different_sample(self, tmp_path):
        """Test finding confidence file for different sample index."""
        pred_dir = tmp_path / "8qf5_H_A" / "seed_2" / "predictions"
        pred_dir.mkdir(parents=True)
        cif_path = pred_dir / "8qf5_H_A_sample_5.cif"

        result = find_confidence_file(cif_path, "8qf5_H_A", 5)

        assert result == pred_dir / "8qf5_H_A_summary_confidence_sample_5.json"

    def test_find_confidence_file_path_construction(self, tmp_path):
        """Test that path is constructed correctly even if file doesn't exist."""
        pred_dir = tmp_path / "nonexistent"
        cif_path = pred_dir / "test.cif"

        result = find_confidence_file(cif_path, "test_entry", 0)

        assert result == pred_dir / "test_entry_summary_confidence_sample_0.json"
        assert not result.exists()


class TestLoadConfidence:
    """Tests for load_confidence function."""

    def test_load_confidence_valid_file(self, tmp_path):
        """Test loading valid confidence JSON."""
        conf_data = {
            "plddt": 85.5,
            "gpde": 0.75,
            "ptm": 0.80,
            "iptm": 0.82,
            "ranking_score": 0.88,
            "nested_dict": {"key": "value"},  # Should be skipped
            "list_field": [1, 2, 3],  # Should be skipped
        }
        conf_path = tmp_path / "summary.json"
        conf_path.write_text(json.dumps(conf_data))

        result = load_confidence(conf_path)

        # Should include scalar fields only
        assert result["plddt"] == pytest.approx(85.5)
        assert result["gpde"] == pytest.approx(0.75)
        assert result["ptm"] == pytest.approx(0.80)
        assert result["iptm"] == pytest.approx(0.82)
        assert result["ranking_score"] == pytest.approx(0.88)

        # Should exclude non-scalar fields
        assert "nested_dict" not in result
        assert "list_field" not in result

    def test_load_confidence_missing_file(self, tmp_path):
        """Test with non-existent confidence file."""
        result = load_confidence(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_confidence_invalid_json(self, tmp_path):
        """Test with invalid JSON."""
        conf_path = tmp_path / "invalid.json"
        conf_path.write_text("{ invalid json }")

        result = load_confidence(conf_path)
        assert result == {}

    def test_load_confidence_empty_file(self, tmp_path):
        """Test with empty JSON object."""
        conf_path = tmp_path / "empty.json"
        conf_path.write_text("{}")

        result = load_confidence(conf_path)
        assert result == {}

    def test_load_confidence_mixed_types(self, tmp_path):
        """Test with mixed scalar types."""
        conf_data = {
            "float_val": 0.85,
            "int_val": 42,
            "str_val": "high",
            "bool_val": True,
        }
        conf_path = tmp_path / "mixed.json"
        conf_path.write_text(json.dumps(conf_data))

        result = load_confidence(conf_path)

        assert result["float_val"] == pytest.approx(0.85)
        assert result["int_val"] == 42
        assert result["str_val"] == "high"
        assert result["bool_val"] is True


class TestFindPaeFile:
    """Tests for find_pae_file function."""

    def test_find_pae_file_npz_new_naming(self, tmp_path):
        """Test finding .npz file with new naming convention."""
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        cif_path = pred_dir / "8qf5_H_A_sample_0.cif"
        npz_path = pred_dir / "8qf5_H_A_confidence_sample_0.npz"
        npz_path.touch()

        result = find_pae_file(cif_path, "8qf5_H_A", 0)

        assert result == npz_path

    def test_find_pae_file_npz_old_naming(self, tmp_path):
        """Test finding .npz file with old naming convention."""
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        cif_path = pred_dir / "8qf5_H_A_sample_0.cif"
        npz_path = pred_dir / "8qf5_H_A_pae_sample_0.npz"
        npz_path.touch()

        result = find_pae_file(cif_path, "8qf5_H_A", 0)

        assert result == npz_path

    def test_find_pae_file_json_fallback(self, tmp_path):
        """Test fallback to JSON when no .npz exists."""
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        cif_path = pred_dir / "8qf5_H_A_sample_0.cif"
        json_path = pred_dir / "8qf5_H_A_full_data_sample_0.json"
        json_path.touch()

        result = find_pae_file(cif_path, "8qf5_H_A", 0)

        assert result == json_path

    def test_find_pae_file_prefers_new_npz(self, tmp_path):
        """Test that new .npz naming is preferred over old."""
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        cif_path = pred_dir / "8qf5_H_A_sample_0.cif"

        # Create both old and new naming
        new_npz = pred_dir / "8qf5_H_A_confidence_sample_0.npz"
        old_npz = pred_dir / "8qf5_H_A_pae_sample_0.npz"
        new_npz.touch()
        old_npz.touch()

        result = find_pae_file(cif_path, "8qf5_H_A", 0)

        # Should prefer new naming
        assert result == new_npz

    def test_find_pae_file_not_found(self, tmp_path):
        """Test when no PAE file exists."""
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        cif_path = pred_dir / "8qf5_H_A_sample_0.cif"

        result = find_pae_file(cif_path, "8qf5_H_A", 0)

        assert result is None


class TestLoadPaeMatrix:
    """Tests for load_pae_matrix function."""

    def test_load_pae_matrix_from_npz(self, tmp_path):
        """Test loading PAE from .npz file."""
        rng = np.random.default_rng(seed=42)
        pae_data = rng.random((10, 10), dtype=np.float32)
        pde_data = rng.random((10, 10), dtype=np.float32)

        npz_path = tmp_path / "test.npz"
        np.savez(npz_path, pae=pae_data, pde=pde_data)

        result = load_pae_matrix(npz_path)

        assert "pae" in result
        assert "pde" in result
        np.testing.assert_array_equal(result["pae"], pae_data)
        np.testing.assert_array_equal(result["pde"], pde_data)

    def test_load_pae_matrix_from_json(self, tmp_path):
        """Test loading PAE from full_data JSON."""
        pae_data = [[1.0, 2.0], [3.0, 4.0]]
        json_data = {
            "token_pair_pae": pae_data,
            "token_pair_pde": pae_data,
        }

        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(json_data))

        result = load_pae_matrix(json_path)

        assert "pae" in result
        assert "pde" in result
        assert result["pae"].shape == (2, 2)
        assert result["pde"].shape == (2, 2)

    def test_load_pae_matrix_partial_data(self, tmp_path):
        """Test loading with only some fields present."""
        rng = np.random.default_rng(seed=42)
        pae_data = rng.random((5, 5), dtype=np.float32)

        npz_path = tmp_path / "test.npz"
        np.savez(npz_path, pae=pae_data)  # Only PAE, no PDE

        result = load_pae_matrix(npz_path)

        assert "pae" in result
        assert "pde" not in result

    def test_load_pae_matrix_all_fields(self, tmp_path):
        """Test loading all supported fields."""
        rng = np.random.default_rng(seed=42)
        npz_path = tmp_path / "test.npz"
        np.savez(
            npz_path,
            pae=rng.random((5, 5), dtype=np.float32),
            pde=rng.random((5, 5), dtype=np.float32),
            contact_probs=rng.random((5, 5), dtype=np.float32),
            atom_plddt=rng.random(50, dtype=np.float32),
        )

        result = load_pae_matrix(npz_path)

        assert "pae" in result
        assert "pde" in result
        assert "contact_probs" in result
        assert "atom_plddt" in result

    def test_load_pae_matrix_missing_file(self, tmp_path):
        """Test with non-existent file."""
        result = load_pae_matrix(tmp_path / "nonexistent.npz")
        assert result == {}

    def test_load_pae_matrix_unsupported_format(self, tmp_path):
        """Test with unsupported file format."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("not a valid format")

        result = load_pae_matrix(txt_path)
        assert result == {}

    def test_load_pae_matrix_corrupted_file(self, tmp_path):
        """Test with corrupted data."""
        npz_path = tmp_path / "corrupted.npz"
        npz_path.write_text("corrupted data")

        result = load_pae_matrix(npz_path)
        assert result == {}


class TestIntegration:
    """Integration tests for protenix.evaluate module."""

    def test_full_evaluation_workflow(self, tmp_path):
        """Test complete workflow from prediction discovery to confidence loading."""
        # Setup directory structure
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"

        # Create predictions
        seed1_dir = predictions_dir / entry / "seed_1" / "predictions"
        seed1_dir.mkdir(parents=True)
        (seed1_dir / f"{entry}_sample_0.cif").touch()
        (seed1_dir / f"{entry}_sample_1.cif").touch()

        # Create confidence files
        conf0 = {"plddt": 85.5, "ptm": 0.80}
        conf1 = {"plddt": 88.2, "ptm": 0.85}

        (seed1_dir / f"{entry}_summary_confidence_sample_0.json").write_text(json.dumps(conf0))
        (seed1_dir / f"{entry}_summary_confidence_sample_1.json").write_text(json.dumps(conf1))

        # Create PAE files
        rng = np.random.default_rng(seed=42)
        pae0 = rng.random((10, 10), dtype=np.float32)
        pae1 = rng.random((10, 10), dtype=np.float32)

        np.savez(seed1_dir / f"{entry}_confidence_sample_0.npz", pae=pae0)
        np.savez(seed1_dir / f"{entry}_confidence_sample_1.npz", pae=pae1)

        # Discover predictions
        samples = prediction_samples(predictions_dir, entry)
        assert len(samples) == 2

        # Load confidence and PAE for each
        for seed, sample, cif_path in samples:
            conf_path = find_confidence_file(cif_path, entry, sample)
            conf = load_confidence(conf_path)
            assert conf["plddt"] in [85.5, 88.2]

            pae_path = find_pae_file(cif_path, entry, sample)
            assert pae_path is not None
            pae_data = load_pae_matrix(pae_path)
            assert "pae" in pae_data
            assert pae_data["pae"].shape == (10, 10)

    def test_chain_mapping_with_native(self, tmp_path):
        """Test chain mapping when native file has different chains."""
        # Create native PDB with chains H, L, T
        pdb_content = """
ATOM      1  N   ALA H   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA H   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      3  N   ALA L   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      4  CA  ALA L   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      5  N   ALA T   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      6  CA  ALA T   1       2.000   3.000   4.000  1.00 10.00           C
"""
        native_path = tmp_path / "native.pdb"
        native_path.write_text(pdb_content)

        # Entry filename says H_L_A but native has H_L_T
        chain_map = chain_map_from_stem("8qrg_H_L_A", native_path)

        # Should map native chains (H, L, T) to model chains (A, B, C)
        assert chain_map == {"H": "A", "L": "B", "T": "C"}
