"""Tests for confidock.boltz.evaluate module."""

import json

import pytest

from confidock.boltz.evaluate import (
    _parse_chain_token,
    chain_map_from_stem,
    chain_map_from_yaml,
    find_confidence_file,
    load_confidence,
    prediction_samples,
)


class TestParseChainToken:
    """Tests for _parse_chain_token function."""

    def test_parse_chain_token_two_char_chains(self):
        """Test parsing two-character chain IDs."""
        result = _parse_chain_token("HcLcA")
        assert result == ["Hc", "Lc", "A"]

    def test_parse_chain_token_single_char_only(self):
        """Test parsing single-character chain IDs."""
        result = _parse_chain_token("NMJ")
        assert result == ["N", "M", "J"]

    def test_parse_chain_token_mixed_case(self):
        """Test parsing mixed case chains."""
        result = _parse_chain_token("hlB")
        assert result == ["h", "l", "B"]

    def test_parse_chain_token_mixed_lengths(self):
        """Test parsing mixed one and two-char chains."""
        result = _parse_chain_token("BbA")
        assert result == ["Bb", "A"]

    def test_parse_chain_token_single_char_input(self):
        """Test single character input."""
        result = _parse_chain_token("H")
        assert result == ["H"]

    def test_parse_chain_token_empty(self):
        """Test empty string."""
        result = _parse_chain_token("")
        assert result == []

    def test_parse_chain_token_all_lowercase(self):
        """Test all lowercase chains."""
        result = _parse_chain_token("abc")
        assert result == ["a", "b", "c"]

    def test_parse_chain_token_all_uppercase(self):
        """Test all uppercase chains."""
        result = _parse_chain_token("ABC")
        assert result == ["A", "B", "C"]

    def test_parse_chain_token_complex_example(self):
        """Test complex realistic example."""
        result = _parse_chain_token("HcLcBxA")
        assert result == ["Hc", "Lc", "Bx", "A"]


class TestChainMapFromStem:
    """Tests for chain_map_from_stem function."""

    def test_chain_map_from_stem_two_chains(self):
        """Test two-chain mapping."""
        result = chain_map_from_stem("8qf5_H_A")
        assert result == {"A": "H", "B": "A"}

    def test_chain_map_from_stem_three_chains(self):
        """Test three-chain mapping."""
        result = chain_map_from_stem("8qrg_H_L_A")
        assert result == {"A": "H", "B": "L", "C": "A"}

    def test_chain_map_from_stem_concatenated_two_char(self):
        """Test concatenated two-character chain IDs."""
        result = chain_map_from_stem("8jlj_HcLcA")
        assert result == {"A": "Hc", "B": "Lc", "C": "A"}

    def test_chain_map_from_stem_concatenated_single_char(self):
        """Test concatenated single-character chain IDs."""
        result = chain_map_from_stem("8z6q_NMJ")
        assert result == {"A": "N", "B": "M", "C": "J"}

    def test_chain_map_from_stem_single_chain(self):
        """Test single chain mapping."""
        result = chain_map_from_stem("1abc_A")
        assert result == {"A": "A"}

    def test_chain_map_from_stem_four_chains(self):
        """Test four-chain mapping."""
        result = chain_map_from_stem("1abc_H_L_A_B")
        assert result == {"A": "H", "B": "L", "C": "A", "D": "B"}

    def test_chain_map_from_stem_mixed_notation(self):
        """Test mixed underscore and concatenated notation."""
        result = chain_map_from_stem("8abc_HcLc_A")
        assert result == {"A": "Hc", "B": "Lc", "C": "A"}


class TestChainMapFromYaml:
    """Tests for chain_map_from_yaml function."""

    def test_chain_map_from_yaml_sequential(self, tmp_path):
        """Test YAML chain mapping with sequential native IDs."""
        yaml_content = """
sequences:
  - protein:
      id: H
      sequence: MKLLV
  - protein:
      id: L
      sequence: DIQMT
  - protein:
      id: A
      sequence: EVQLV
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        result = chain_map_from_yaml(yaml_path)
        assert result == {"A": "H", "B": "L", "C": "A"}

    def test_chain_map_from_yaml_with_native_pdb(self, tmp_path):
        """Test YAML chain mapping with native PDB file."""
        yaml_content = """
sequences:
  - protein:
      id: Hc
      sequence: MKLLV
  - protein:
      id: Lc
      sequence: DIQMT
  - protein:
      id: T
      sequence: EVQLV
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        # Create native PDB with non-sequential chain IDs
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

        result = chain_map_from_yaml(yaml_path, native_path)
        assert result == {"H": "Hc", "L": "Lc", "T": "T"}

    def test_chain_map_from_yaml_missing_native(self, tmp_path):
        """Test YAML chain mapping with non-existent native PDB."""
        yaml_content = """
sequences:
  - protein:
      id: H
      sequence: MKLLV
  - protein:
      id: L
      sequence: DIQMT
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        result = chain_map_from_yaml(yaml_path, tmp_path / "nonexistent.pdb")
        # Should fall back to sequential mapping
        assert result == {"A": "H", "B": "L"}

    def test_chain_map_from_yaml_empty_sequences(self, tmp_path):
        """Test YAML with no protein sequences."""
        yaml_content = """
sequences: []
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        result = chain_map_from_yaml(yaml_path)
        assert result == {}

    def test_chain_map_from_yaml_mixed_sequence_types(self, tmp_path):
        """Test YAML with mixed sequence types (only protein should be extracted)."""
        yaml_content = """
sequences:
  - protein:
      id: H
      sequence: MKLLV
  - ligand:
      id: LIG
      smiles: CCO
  - protein:
      id: L
      sequence: DIQMT
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        result = chain_map_from_yaml(yaml_path)
        assert result == {"A": "H", "B": "L"}


class TestPredictionSamples:
    """Tests for prediction_samples function."""

    def test_prediction_samples_with_seeds(self, tmp_path):
        """Test finding predictions with seed and sample numbers."""
        predictions_dir = tmp_path / "predictions"
        entry_dir = predictions_dir / "8qf5_H_A"
        entry_dir.mkdir(parents=True)

        # Create test files
        (entry_dir / "8qf5_H_A_seed1_sample1_model.pdb").touch()
        (entry_dir / "8qf5_H_A_seed1_sample2_model.pdb").touch()
        (entry_dir / "8qf5_H_A_seed2_sample1_model.pdb").touch()

        results = prediction_samples(predictions_dir, "8qf5_H_A")

        assert len(results) == 3
        # Check structure: (seed, sample, path)
        seeds = [r[0] for r in results]
        samples = [r[1] for r in results]
        assert 1 in seeds and 2 in seeds
        assert all(s in [1, 2] for s in samples)

    def test_prediction_samples_without_seeds(self, tmp_path):
        """Test finding predictions without seed numbers."""
        predictions_dir = tmp_path / "predictions"
        entry_dir = predictions_dir / "8qf5_H_A"
        entry_dir.mkdir(parents=True)

        # Create test files without seed
        (entry_dir / "8qf5_H_A_sample1.pdb").touch()
        (entry_dir / "8qf5_H_A_sample2.pdb").touch()
        (entry_dir / "8qf5_H_A_sample3.pdb").touch()

        results = prediction_samples(predictions_dir, "8qf5_H_A")

        assert len(results) == 3
        # Seeds should be None
        assert all(r[0] is None for r in results)
        # Sample indices should be 1, 2, 3
        samples = sorted([r[1] for r in results])
        assert samples == [1, 2, 3]

    def test_prediction_samples_mixed_formats(self, tmp_path):
        """Test finding predictions with mixed naming formats."""
        predictions_dir = tmp_path / "predictions"
        entry_dir = predictions_dir / "8qf5_H_A"
        entry_dir.mkdir(parents=True)

        # Mix of formats
        (entry_dir / "8qf5_H_A_seed1_sample1_model.pdb").touch()
        (entry_dir / "8qf5_H_A_sample2.pdb").touch()

        results = prediction_samples(predictions_dir, "8qf5_H_A")

        assert len(results) == 2
        # One with seed, one without
        seeds = [r[0] for r in results]
        assert None in seeds and 1 in seeds

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
        entry_dir = predictions_dir / "8qf5_H_A"
        entry_dir.mkdir(parents=True)

        # Create matching and non-matching files
        (entry_dir / "8qf5_H_A_sample1.pdb").touch()
        (entry_dir / "other_file.pdb").touch()
        (entry_dir / "confidence_8qf5_H_A_sample1.json").touch()
        (entry_dir / "README.txt").touch()

        results = prediction_samples(predictions_dir, "8qf5_H_A")

        # Should only find the one matching PDB
        assert len(results) == 1

    def test_prediction_samples_sorted_output(self, tmp_path):
        """Test that results are sorted."""
        predictions_dir = tmp_path / "predictions"
        entry_dir = predictions_dir / "8qf5_H_A"
        entry_dir.mkdir(parents=True)

        # Create in non-sorted order
        (entry_dir / "8qf5_H_A_sample3.pdb").touch()
        (entry_dir / "8qf5_H_A_sample1.pdb").touch()
        (entry_dir / "8qf5_H_A_sample2.pdb").touch()

        results = prediction_samples(predictions_dir, "8qf5_H_A")

        samples = [r[1] for r in results]
        assert samples == [1, 2, 3]  # Should be sorted


class TestLoadConfidence:
    """Tests for load_confidence function."""

    def test_load_confidence_valid_file(self, tmp_path):
        """Test loading valid confidence JSON."""
        conf_data = {
            "confidence_score": 0.85,
            "ptm": 0.75,
            "iptm": 0.80,
            "complex_plddt": 85.5,
            "complex_iplddt": 82.3,
            "nested_dict": {"key": "value"},  # Should be skipped
            "list_field": [1, 2, 3],  # Should be skipped
        }
        conf_path = tmp_path / "confidence_test.json"
        conf_path.write_text(json.dumps(conf_data))

        result = load_confidence(conf_path)

        # Should include scalar fields only
        assert result["confidence_score"] == pytest.approx(0.85)
        assert result["ptm"] == pytest.approx(0.75)
        assert result["iptm"] == pytest.approx(0.80)
        assert result["complex_plddt"] == pytest.approx(85.5)
        assert result["complex_iplddt"] == pytest.approx(82.3)

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

    def test_load_confidence_all_nested(self, tmp_path):
        """Test with all nested/non-scalar values."""
        conf_data = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }
        conf_path = tmp_path / "nested.json"
        conf_path.write_text(json.dumps(conf_data))

        result = load_confidence(conf_path)
        assert result == {}

    def test_load_confidence_mixed_types(self, tmp_path):
        """Test with mixed scalar types."""
        conf_data = {
            "float_val": 0.85,
            "int_val": 42,
            "str_val": "high",
            "bool_val": True,
            "null_val": None,
        }
        conf_path = tmp_path / "mixed.json"
        conf_path.write_text(json.dumps(conf_data))

        result = load_confidence(conf_path)

        assert result["float_val"] == pytest.approx(0.85)
        assert result["int_val"] == 42
        assert result["str_val"] == "high"
        assert result["bool_val"] is True
        assert result["null_val"] is None


class TestFindConfidenceFile:
    """Tests for find_confidence_file function."""

    def test_find_confidence_file_basic(self, tmp_path):
        """Test finding confidence file path."""
        entry_dir = tmp_path / "8qf5_H_A"
        entry_dir.mkdir()

        result = find_confidence_file(entry_dir, "8qf5_H_A_sample1")

        assert result == entry_dir / "confidence_8qf5_H_A_sample1.json"

    def test_find_confidence_file_with_seed(self, tmp_path):
        """Test finding confidence file with seed in stem."""
        entry_dir = tmp_path / "8qf5_H_A"
        entry_dir.mkdir()

        result = find_confidence_file(entry_dir, "8qf5_H_A_seed1_sample2_model")

        assert result == entry_dir / "confidence_8qf5_H_A_seed1_sample2_model.json"

    def test_find_confidence_file_path_construction(self, tmp_path):
        """Test that path is constructed correctly even if file doesn't exist."""
        entry_dir = tmp_path / "nonexistent"

        result = find_confidence_file(entry_dir, "test_stem")

        # Should still construct the path correctly
        assert result == entry_dir / "confidence_test_stem.json"
        assert not result.exists()


class TestIntegration:
    """Integration tests for boltz.evaluate module."""

    def test_full_evaluation_workflow(self, tmp_path):
        """Test complete workflow from prediction discovery to confidence loading."""
        # Setup directory structure
        predictions_dir = tmp_path / "predictions"
        entry = "8qf5_H_A"
        entry_dir = predictions_dir / entry
        entry_dir.mkdir(parents=True)

        # Create prediction files
        (entry_dir / f"{entry}_seed1_sample1_model.pdb").touch()
        (entry_dir / f"{entry}_seed1_sample2_model.pdb").touch()

        # Create confidence files
        conf1 = {"confidence_score": 0.85, "ptm": 0.75}
        conf2 = {"confidence_score": 0.90, "ptm": 0.80}

        conf_path1 = find_confidence_file(entry_dir, f"{entry}_seed1_sample1_model")
        conf_path2 = find_confidence_file(entry_dir, f"{entry}_seed1_sample2_model")

        conf_path1.write_text(json.dumps(conf1))
        conf_path2.write_text(json.dumps(conf2))

        # Discover predictions
        samples = prediction_samples(predictions_dir, entry)
        assert len(samples) == 2

        # Load confidence for each
        confidences = []
        for seed, sample, path in samples:
            conf_path = find_confidence_file(entry_dir, path.stem)
            conf = load_confidence(conf_path)
            confidences.append(conf)

        assert len(confidences) == 2
        assert confidences[0]["confidence_score"] in [0.85, 0.90]
        assert confidences[1]["confidence_score"] in [0.85, 0.90]

    def test_chain_mapping_consistency(self):
        """Test that chain mapping methods are consistent."""
        # Old-style underscore format
        stem_map = chain_map_from_stem("8qf5_H_L_A")

        # Should map A->H, B->L, C->A
        assert stem_map["A"] == "H"
        assert stem_map["B"] == "L"
        assert stem_map["C"] == "A"

        # New-style concatenated format
        concat_map = chain_map_from_stem("8qf5_HLA")

        # Should produce same mapping
        assert concat_map == stem_map
