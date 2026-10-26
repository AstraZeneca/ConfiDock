"""Tests for confidock.esmfold2.evaluate module."""

import json

import pytest

from confidock.esmfold2.evaluate import (
    chain_map_from_stem,
    find_confidence_file,
    load_confidence,
    prediction_samples,
)


class TestChainMapFromStem:
    """Tests for chain_map_from_stem function."""

    def test_chain_map_from_stem_basic(self):
        """Test basic chain mapping without native file."""
        result = chain_map_from_stem("8qf5_H_A")
        # Identity mapping when no native file provided
        assert result == {"H": "H", "A": "A"}

    def test_chain_map_from_stem_three_chains(self):
        """Test three-chain identity mapping."""
        result = chain_map_from_stem("8qrg_H_L_A")
        assert result == {"H": "H", "L": "L", "A": "A"}

    def test_chain_map_from_stem_concatenated(self):
        """Test concatenated chain notation."""
        result = chain_map_from_stem("8jlj_HcLcA")
        assert result == {"Hc": "Hc", "Lc": "Lc", "A": "A"}

    def test_chain_map_from_stem_with_native_matching(self, tmp_path):
        """Test with native file that matches filename."""
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
        # Native chains match filename, so identity mapping
        assert result == {"H": "H", "L": "L", "A": "A"}

    def test_chain_map_from_stem_with_native_mismatch(self, tmp_path):
        """Test with native file that differs from filename."""
        # Filename says HLN, but native file has HLT
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

        result = chain_map_from_stem("1a14_HLN", native_path)
        # Positional mapping: native H->model H, native L->model L, native T->model N
        assert result == {"H": "H", "L": "L", "T": "N"}

    def test_chain_map_from_stem_native_nonexistent(self, tmp_path):
        """Test with non-existent native file."""
        result = chain_map_from_stem("8qf5_H_A", tmp_path / "nonexistent.pdb")
        # Should fall back to identity mapping
        assert result == {"H": "H", "A": "A"}

    def test_chain_map_from_stem_native_empty(self, tmp_path):
        """Test with empty native file."""
        native_path = tmp_path / "empty.pdb"
        native_path.write_text("")

        result = chain_map_from_stem("8qf5_H_A", native_path)
        # No chains found, fall back to identity mapping
        assert result == {"H": "H", "A": "A"}

    def test_chain_map_from_stem_single_chain(self):
        """Test single chain mapping."""
        result = chain_map_from_stem("1abc_A")
        assert result == {"A": "A"}

    def test_chain_map_from_stem_two_char_chains(self):
        """Test two-character chain IDs."""
        result = chain_map_from_stem("8jlj_HcLc")
        assert result == {"Hc": "Hc", "Lc": "Lc"}

    def test_chain_map_from_stem_mixed_chains(self):
        """Test mixed single and two-character chains."""
        result = chain_map_from_stem("8abc_HcLcBxA")
        assert result == {"Hc": "Hc", "Lc": "Lc", "Bx": "Bx", "A": "A"}


class TestPredictionSamples:
    """Tests for prediction_samples function."""

    def test_prediction_samples_basic(self, tmp_path):
        """Test finding ESMFold2 prediction samples."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create test CIF files
        (entry_dir / f"{entry}_sample_0.cif").touch()
        (entry_dir / f"{entry}_sample_1.cif").touch()
        (entry_dir / f"{entry}_sample_2.cif").touch()

        results = prediction_samples(predictions_dir, entry)

        assert len(results) == 3
        # Check structure: (seed, sample, path)
        seeds = [r[0] for r in results]
        samples = [r[1] for r in results]

        # All seeds should be 0 for ESMFold2
        assert all(s == 0 for s in seeds)
        # Sample indices should be 0, 1, 2
        assert sorted(samples) == [0, 1, 2]

    def test_prediction_samples_nonexistent_entry(self, tmp_path):
        """Test with non-existent entry directory."""
        predictions_dir = tmp_path / "predictions"
        predictions_dir.mkdir()

        results = prediction_samples(predictions_dir, "nonexistent_entry")
        assert results == []

    def test_prediction_samples_empty_directory(self, tmp_path):
        """Test with empty entry directory."""
        predictions_dir = tmp_path / "predictions"
        entry_dir = predictions_dir / "8qf5_H_A"
        entry_dir.mkdir(parents=True)

        results = prediction_samples(predictions_dir, "8qf5_H_A")
        assert results == []

    def test_prediction_samples_other_files_ignored(self, tmp_path):
        """Test that non-matching files are ignored."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create matching and non-matching files
        (entry_dir / f"{entry}_sample_0.cif").touch()
        (entry_dir / "other_file.cif").touch()
        (entry_dir / f"summary_{entry}_sample_0.json").touch()
        (entry_dir / "README.txt").touch()

        results = prediction_samples(predictions_dir, entry)

        # Should only find the one matching CIF
        assert len(results) == 1

    def test_prediction_samples_sorted_output(self, tmp_path):
        """Test that results are sorted."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create in non-sorted order
        (entry_dir / f"{entry}_sample_5.cif").touch()
        (entry_dir / f"{entry}_sample_0.cif").touch()
        (entry_dir / f"{entry}_sample_3.cif").touch()

        results = prediction_samples(predictions_dir, entry)

        samples = [r[1] for r in results]
        assert samples == [0, 3, 5]  # Should be sorted

    def test_prediction_samples_pdb_not_cif(self, tmp_path):
        """Test that PDB files are not matched (ESMFold2 uses CIF)."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create PDB instead of CIF
        (entry_dir / f"{entry}_sample_0.pdb").touch()

        results = prediction_samples(predictions_dir, entry)

        # Should not find PDB files
        assert len(results) == 0

    def test_prediction_samples_wrong_entry_name(self, tmp_path):
        """Test that files with wrong entry name are not matched."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create file with different entry name
        (entry_dir / "other_entry_sample_0.cif").touch()

        results = prediction_samples(predictions_dir, entry)

        assert len(results) == 0

    def test_prediction_samples_direct_cif_only(self, tmp_path):
        """Test direct CIF pattern (no_msa mode) without sample files."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create only the direct CIF file (no_msa mode)
        direct_cif = entry_dir / f"{entry}.cif"
        direct_cif.touch()

        results = prediction_samples(predictions_dir, entry)

        # Should find the direct CIF file with sample_idx=0
        assert len(results) == 1
        seed, sample_idx, path = results[0]
        assert seed == 0
        assert sample_idx == 0
        assert path == direct_cif

    def test_prediction_samples_direct_cif_with_samples(self, tmp_path):
        """Test direct CIF pattern alongside _sample_ files."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create _sample_ files
        (entry_dir / f"{entry}_sample_0.cif").touch()
        (entry_dir / f"{entry}_sample_1.cif").touch()

        # Also create direct CIF file
        direct_cif = entry_dir / f"{entry}.cif"
        direct_cif.touch()

        results = prediction_samples(predictions_dir, entry)

        # Should find all 3 files
        assert len(results) == 3

        # Direct CIF should get sample_idx = max(existing) + 1 = 2
        samples = [r[1] for r in results]
        assert sorted(samples) == [0, 1, 2]

        # Verify the direct CIF is included
        paths = [r[2] for r in results]
        assert direct_cif in paths


class TestFindConfidenceFile:
    """Tests for find_confidence_file function."""

    def test_find_confidence_file_basic(self, tmp_path):
        """Test finding confidence file path."""
        entry_dir = tmp_path / "8qf5_H_A"
        entry_dir.mkdir()
        cif_path = entry_dir / "8qf5_H_A_sample_0.cif"

        result = find_confidence_file(cif_path, "8qf5_H_A", 0)

        assert result == entry_dir / "summary_8qf5_H_A_sample_0.json"

    def test_find_confidence_file_different_sample(self, tmp_path):
        """Test finding confidence file for different sample index."""
        entry_dir = tmp_path / "8qf5_H_A"
        entry_dir.mkdir()
        cif_path = entry_dir / "8qf5_H_A_sample_5.cif"

        result = find_confidence_file(cif_path, "8qf5_H_A", 5)

        assert result == entry_dir / "summary_8qf5_H_A_sample_5.json"

    def test_find_confidence_file_path_construction(self, tmp_path):
        """Test that path is constructed correctly even if file doesn't exist."""
        entry_dir = tmp_path / "nonexistent"
        cif_path = entry_dir / "test_entry_sample_0.cif"

        result = find_confidence_file(cif_path, "test_entry", 0)

        assert result == entry_dir / "summary_test_entry_sample_0.json"
        assert not result.exists()

    def test_find_confidence_file_direct_naming(self, tmp_path):
        """Test confidence file for direct CIF naming (no_msa mode)."""
        entry_dir = tmp_path / "8qf5_H_A"
        entry_dir.mkdir()
        # Direct CIF file without _sample_ in name
        cif_path = entry_dir / "8qf5_H_A.cif"

        result = find_confidence_file(cif_path, "8qf5_H_A", 0)

        # Should use direct naming pattern
        assert result == entry_dir / "summary_8qf5_H_A.json"
        assert "_sample_" not in result.name


class TestLoadConfidence:
    """Tests for load_confidence function."""

    def test_load_confidence_valid_file(self, tmp_path):
        """Test loading valid confidence JSON."""
        conf_data = {
            "plddt": 85.5,
            "gpde": 0.75,
            "ptm": 0.80,
            "iptm": 0.82,
            "other_field": "ignored",
        }
        conf_path = tmp_path / "summary_test.json"
        conf_path.write_text(json.dumps(conf_data))

        result = load_confidence(conf_path)

        assert result["plddt"] == pytest.approx(85.5)
        assert result["gpde"] == pytest.approx(0.75)
        assert result["ptm"] == pytest.approx(0.80)
        assert result["iptm"] == pytest.approx(0.82)
        # Only these 4 fields should be extracted
        assert len(result) == 4

    def test_load_confidence_missing_file(self, tmp_path):
        """Test with non-existent confidence file."""
        result = load_confidence(tmp_path / "nonexistent.json")

        # Should return dict with None values
        assert result == {
            "plddt": None,
            "gpde": None,
            "ptm": None,
            "iptm": None,
        }

    def test_load_confidence_invalid_json(self, tmp_path):
        """Test with invalid JSON."""
        conf_path = tmp_path / "invalid.json"
        conf_path.write_text("{ invalid json }")

        result = load_confidence(conf_path)

        # Should return dict with None values
        assert result == {
            "plddt": None,
            "gpde": None,
            "ptm": None,
            "iptm": None,
        }

    def test_load_confidence_empty_json(self, tmp_path):
        """Test with empty JSON object."""
        conf_path = tmp_path / "empty.json"
        conf_path.write_text("{}")

        result = load_confidence(conf_path)

        # Should return dict with None values (keys not present)
        assert result["plddt"] is None
        assert result["gpde"] is None
        assert result["ptm"] is None
        assert result["iptm"] is None

    def test_load_confidence_partial_data(self, tmp_path):
        """Test with partial confidence data."""
        conf_data = {
            "plddt": 85.5,
            "ptm": 0.80,
            # Missing gpde and iptm
        }
        conf_path = tmp_path / "partial.json"
        conf_path.write_text(json.dumps(conf_data))

        result = load_confidence(conf_path)

        assert result["plddt"] == pytest.approx(85.5)
        assert result["ptm"] == pytest.approx(0.80)
        assert result["gpde"] is None
        assert result["iptm"] is None

    def test_load_confidence_null_values(self, tmp_path):
        """Test with explicit null values."""
        conf_data = {
            "plddt": None,
            "gpde": None,
            "ptm": 0.80,
            "iptm": 0.82,
        }
        conf_path = tmp_path / "nulls.json"
        conf_path.write_text(json.dumps(conf_data))

        result = load_confidence(conf_path)

        assert result["plddt"] is None
        assert result["gpde"] is None
        assert result["ptm"] == pytest.approx(0.80)
        assert result["iptm"] == pytest.approx(0.82)

    def test_load_confidence_type_error(self, tmp_path):
        """Test with wrong type in JSON."""
        conf_path = tmp_path / "wrong_type.json"
        conf_path.write_text("[]")  # Array instead of object

        result = load_confidence(conf_path)

        # Should handle TypeError and return None values
        assert result == {
            "plddt": None,
            "gpde": None,
            "ptm": None,
            "iptm": None,
        }


class TestIntegration:
    """Integration tests for esmfold2.evaluate module."""

    def test_full_evaluation_workflow(self, tmp_path):
        """Test complete workflow from prediction discovery to confidence loading."""
        # Setup directory structure
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create CIF prediction files
        (entry_dir / f"{entry}_sample_0.cif").touch()
        (entry_dir / f"{entry}_sample_1.cif").touch()

        # Create confidence files
        conf0 = {"plddt": 85.5, "gpde": 0.75, "ptm": 0.80, "iptm": 0.82}
        conf1 = {"plddt": 88.2, "gpde": 0.78, "ptm": 0.85, "iptm": 0.87}

        (entry_dir / f"summary_{entry}_sample_0.json").write_text(json.dumps(conf0))
        (entry_dir / f"summary_{entry}_sample_1.json").write_text(json.dumps(conf1))

        # Discover predictions
        samples = prediction_samples(predictions_dir, entry)
        assert len(samples) == 2

        # Load confidence for each
        confidences = []
        for _, sample, cif_path in samples:
            conf_path = find_confidence_file(cif_path, entry, sample)
            conf = load_confidence(conf_path)
            confidences.append(conf)

        assert len(confidences) == 2
        assert confidences[0]["plddt"] in [85.5, 88.2]
        assert confidences[1]["plddt"] in [85.5, 88.2]
        assert all(c["ptm"] is not None for c in confidences)

    def test_chain_mapping_with_mismatched_native(self, tmp_path):
        """Test chain mapping when native file has different chains than filename."""
        # Create native PDB with chains H, L, T
        pdb_content = """
ATOM      1  N   ALA H   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA H   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      3  N   ALA L   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      4  CA  ALA L   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      5  N   ALA T   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      6  CA  ALA T   1       2.000   3.000   4.000  1.00 10.00           C
"""
        native_path = tmp_path / "1a14_HLN.pdb"
        native_path.write_text(pdb_content)

        # Filename says HLN but native has HLT
        chain_map = chain_map_from_stem("1a14_HLN", native_path)

        # Should map native T to model N (positional)
        assert chain_map["H"] == "H"
        assert chain_map["L"] == "L"
        assert chain_map["T"] == "N"  # Mismatch: native T -> model N

    def test_handles_missing_confidence_gracefully(self, tmp_path):
        """Test that missing confidence files are handled gracefully."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create CIF but no confidence file
        cif_path = entry_dir / f"{entry}_sample_0.cif"
        cif_path.touch()

        # Discover predictions
        samples = prediction_samples(predictions_dir, entry)
        assert len(samples) == 1

        # Try to load missing confidence
        _, sample, cif_path = samples[0]
        conf_path = find_confidence_file(cif_path, entry, sample)
        conf = load_confidence(conf_path)

        # Should return None values gracefully
        assert conf["plddt"] is None
        assert conf["gpde"] is None
        assert conf["ptm"] is None
        assert conf["iptm"] is None

    def test_direct_cif_workflow(self, tmp_path):
        """Test complete workflow with direct CIF naming (no_msa mode)."""
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create direct CIF file (no_msa mode)
        direct_cif = entry_dir / f"{entry}.cif"
        direct_cif.touch()

        # Create corresponding confidence file with direct naming
        conf_data = {"plddt": 90.0, "gpde": 0.85, "ptm": 0.88, "iptm": 0.90}
        conf_path = entry_dir / f"summary_{entry}.json"
        conf_path.write_text(json.dumps(conf_data))

        # Discover predictions
        samples = prediction_samples(predictions_dir, entry)
        assert len(samples) == 1

        # Load confidence using direct naming pattern
        _, sample, cif_path = samples[0]
        found_conf_path = find_confidence_file(cif_path, entry, sample)
        assert found_conf_path == conf_path
        assert "_sample_" not in found_conf_path.name

        conf = load_confidence(found_conf_path)
        assert conf["plddt"] == pytest.approx(90.0)
        assert conf["ptm"] == pytest.approx(0.88)
