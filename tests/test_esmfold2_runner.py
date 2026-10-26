"""Tests for confidock.esmfold2.inference.runner module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from confidock.esmfold2.inference.runner import (
    MSA_PATH_KEY,
    coerce_modifications,
    load_entries,
    resolve_a3m,
    resolve_chain_a3m,
    save_pae_matrix,
    write_prediction_config,
)


class TestSavePaeMatrix:
    """Tests for save_pae_matrix function."""

    def test_no_pae_attribute(self, tmp_path):
        """Test when result has no PAE attribute."""
        result = MagicMock(spec=[])  # No pae attribute
        output_path = tmp_path / "pae.npz"

        saved, shape = save_pae_matrix(result, output_path)

        assert saved is False
        assert shape is None
        assert not output_path.exists()

    def test_pae_is_none(self, tmp_path):
        """Test when PAE attribute is None."""
        result = MagicMock()
        result.pae = None
        output_path = tmp_path / "pae.npz"

        saved, shape = save_pae_matrix(result, output_path)

        assert saved is False
        assert shape is None
        assert not output_path.exists()

    def test_save_numpy_array(self, tmp_path):
        """Test saving a numpy array PAE."""
        result = MagicMock()
        rng = np.random.default_rng(seed=42)
        pae_data = rng.random((100, 100))
        result.pae = pae_data
        output_path = tmp_path / "pae.npz"

        saved, shape = save_pae_matrix(result, output_path)

        assert saved is True
        assert shape == (100, 100)
        assert output_path.exists()

        # Verify saved data
        loaded = np.load(output_path)
        assert "pae" in loaded
        np.testing.assert_array_equal(loaded["pae"], pae_data)

    def test_save_torch_tensor(self, tmp_path):
        """Test saving a PyTorch tensor with detach."""
        result = MagicMock()
        # Mock a torch tensor with detach method
        mock_tensor = MagicMock()
        mock_cpu = MagicMock()
        rng = np.random.default_rng(seed=42)
        pae_data = rng.random((50, 50))
        mock_cpu.numpy.return_value = pae_data
        mock_tensor.detach.return_value = mock_cpu
        mock_tensor.detach.return_value.cpu.return_value = mock_cpu
        result.pae = mock_tensor
        output_path = tmp_path / "pae.npz"

        saved, shape = save_pae_matrix(result, output_path)

        assert saved is True
        assert shape == (50, 50)
        assert output_path.exists()

    def test_save_with_cpu_method(self, tmp_path):
        """Test saving a tensor with cpu() method but no detach."""
        result = MagicMock()
        mock_tensor = MagicMock(spec=["cpu", "numpy"])
        rng = np.random.default_rng(seed=42)
        pae_data = rng.random((30, 30))
        mock_tensor.cpu.return_value.numpy.return_value = pae_data
        result.pae = mock_tensor
        output_path = tmp_path / "pae.npz"

        saved, shape = save_pae_matrix(result, output_path)

        assert saved is True
        assert shape == (30, 30)
        assert output_path.exists()

    def test_exception_handling(self, tmp_path):
        """Test that exceptions are caught and logged."""
        result = MagicMock()
        # Create a mock that raises an exception
        result.pae = MagicMock()
        result.pae.detach.side_effect = RuntimeError("GPU error")
        output_path = tmp_path / "pae.npz"

        saved, shape = save_pae_matrix(result, output_path)

        assert saved is False
        assert shape is None
        assert not output_path.exists()

    def test_rectangular_pae(self, tmp_path):
        """Test saving a rectangular PAE matrix."""
        result = MagicMock()
        rng = np.random.default_rng(seed=42)
        pae_data = rng.random((100, 200))
        result.pae = pae_data
        output_path = tmp_path / "pae.npz"

        saved, shape = save_pae_matrix(result, output_path)

        assert saved is True
        assert shape == (100, 200)


class TestResolveChainA3m:
    """Tests for resolve_chain_a3m, which prefers an explicit msaPath over the hash lookup."""

    SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"

    @staticmethod
    def _hashed_a3m(msa_dir, seq, text=">seq1\nMKTAY\n>seq2\nMKTAYI\n"):
        """Write an .a3m under its hashed name and return the path."""
        import hashlib

        msa_dir.mkdir(exist_ok=True)
        path = msa_dir / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m"
        path.write_text(text)
        return path

    def test_explicit_path_wins_over_hash_lookup(self, tmp_path):
        """msaPath is used even when a hash-named .a3m also exists."""
        hashed = self._hashed_a3m(tmp_path / "msa", self.SEQ)
        explicit = tmp_path / "explicit.a3m"
        explicit.write_text(">q\nMKTAY\n>h\nMKTAYI\n")

        result = resolve_chain_a3m({"sequence": self.SEQ, MSA_PATH_KEY: str(explicit)}, tmp_path / "msa")

        assert result == explicit
        assert result != hashed

    def test_falls_back_to_hash_lookup(self, tmp_path):
        """Without msaPath the hash lookup is used, preserving existing behaviour."""
        hashed = self._hashed_a3m(tmp_path / "msa", self.SEQ)

        assert resolve_chain_a3m({"sequence": self.SEQ}, tmp_path / "msa") == hashed

    def test_explicit_path_works_without_msa_dir(self, tmp_path):
        """msaPath removes the need for a --msa_dir at all."""
        explicit = tmp_path / "explicit.a3m"
        explicit.write_text(">q\nMKTAY\n")

        assert resolve_chain_a3m({"sequence": self.SEQ, MSA_PATH_KEY: str(explicit)}, None) == explicit

    def test_dangling_explicit_path_raises(self, tmp_path):
        """A declared-but-missing msaPath fails loudly instead of folding MSA-free."""
        chain = {"id": "H", "sequence": self.SEQ, MSA_PATH_KEY: str(tmp_path / "gone.a3m")}

        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_chain_a3m(chain, tmp_path / "msa")

    def test_no_path_and_no_dir_returns_none(self):
        """Neither source configured means no MSA, not an error."""
        assert resolve_chain_a3m({"sequence": self.SEQ}, None) is None

    def test_missing_hash_lookup_returns_none(self, tmp_path):
        """An absent hash-named .a3m still returns None so the caller can warn."""
        (tmp_path / "msa").mkdir()

        assert resolve_chain_a3m({"sequence": self.SEQ}, tmp_path / "msa") is None


class TestResolveA3m:
    """Tests for resolve_a3m function."""

    def test_existing_a3m_file(self, tmp_path):
        """Test resolving an existing A3M file."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()

        # Create a sample A3M file with proper hash
        import hashlib

        seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
        digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
        a3m_path = msa_dir / f"{digest}.a3m"
        a3m_path.write_text(">seq1\nMKTAY\n>seq2\nMKTAYI\n")

        result = resolve_a3m(seq, msa_dir)

        assert result == a3m_path
        assert result.exists()

    def test_nonexistent_a3m_file(self, tmp_path):
        """Test when A3M file doesn't exist."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()

        seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"

        result = resolve_a3m(seq, msa_dir)

        assert result is None

    def test_empty_a3m_file(self, tmp_path):
        """Test when A3M file is too small."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()

        import hashlib

        seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
        digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
        a3m_path = msa_dir / f"{digest}.a3m"
        a3m_path.write_text("short")  # Less than 10 bytes

        result = resolve_a3m(seq, msa_dir)

        assert result is None

    def test_exact_10_byte_a3m_file(self, tmp_path):
        """Test A3M file with exactly 10 bytes."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()

        import hashlib

        seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
        digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
        a3m_path = msa_dir / f"{digest}.a3m"
        a3m_path.write_text("0123456789")  # Exactly 10 bytes

        result = resolve_a3m(seq, msa_dir)

        assert result == a3m_path

    def test_hash_consistency(self, tmp_path):
        """Test that hash is calculated consistently."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()

        import hashlib

        seq = "ACDEFGHIKLMNPQRSTVWY"
        digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
        a3m_path = msa_dir / f"{digest}.a3m"
        a3m_path.write_text(">seq\nACDEFGHIKLMNPQRSTVWY\n")

        result1 = resolve_a3m(seq, msa_dir)
        result2 = resolve_a3m(seq, msa_dir)

        assert result1 == result2
        assert result1 is not None


class TestWritePredictionConfig:
    """Tests for write_prediction_config function."""

    def test_write_basic_config(self, tmp_path):
        """Test writing a basic configuration."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        write_prediction_config(config_dir, model="esmfold2", batch_size=4)

        cfg_path = config_dir / "prediction_config.json"
        assert cfg_path.exists()

        with cfg_path.open() as fh:
            data = json.load(fh)

        assert data["batch_size"] == 4
        assert data["model"] == "esmfold2"

    def test_write_with_path_values(self, tmp_path):
        """Test that Path values are converted to strings."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        input_path = tmp_path / "input"
        output_path = tmp_path / "output"

        write_prediction_config(config_dir, input_dir=input_path, output_dir=output_path)

        cfg_path = config_dir / "prediction_config.json"
        with cfg_path.open() as fh:
            data = json.load(fh)

        assert isinstance(data["input_dir"], str)
        assert isinstance(data["output_dir"], str)
        assert data["input_dir"] == str(input_path)
        assert data["output_dir"] == str(output_path)

    def test_config_sorted_keys(self, tmp_path):
        """Test that config keys are sorted."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        write_prediction_config(config_dir, zebra=1, apple=2, middle=3)

        cfg_path = config_dir / "prediction_config.json"
        with cfg_path.open() as fh:
            content = fh.read()

        # Keys should appear in sorted order
        assert content.index("apple") < content.index("middle")
        assert content.index("middle") < content.index("zebra")

    def test_config_indented(self, tmp_path):
        """Test that config is properly indented."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        write_prediction_config(config_dir, key="value")

        cfg_path = config_dir / "prediction_config.json"
        with cfg_path.open() as fh:
            content = fh.read()

        # Check for indentation (json.dump with indent=2)
        assert "  " in content


class TestLoadEntries:
    """Tests for load_entries function."""

    def test_load_list_of_entries(self, tmp_path):
        """Test loading a list of entries."""
        input_json = tmp_path / "input.json"
        data = [
            {"name": "entry1", "sequence": "ACDEF"},
            {"name": "entry2", "sequence": "GHIKL"},
        ]
        input_json.write_text(json.dumps(data))

        result = load_entries(input_json, None)

        assert len(result) == 2
        assert result[0]["name"] == "entry1"
        assert result[1]["name"] == "entry2"

    def test_load_single_dict(self, tmp_path):
        """Test loading a single dict (converted to list)."""
        input_json = tmp_path / "input.json"
        data = {"name": "entry1", "sequence": "ACDEF"}
        input_json.write_text(json.dumps(data))

        result = load_entries(input_json, None)

        assert len(result) == 1
        assert result[0]["name"] == "entry1"

    def test_filter_requested_entries(self, tmp_path):
        """Test filtering by requested entry names."""
        input_json = tmp_path / "input.json"
        data = [
            {"name": "entry1", "sequence": "ACDEF"},
            {"name": "entry2", "sequence": "GHIKL"},
            {"name": "entry3", "sequence": "MNPQR"},
        ]
        input_json.write_text(json.dumps(data))

        result = load_entries(input_json, ["entry1", "entry3"])

        assert len(result) == 2
        assert result[0]["name"] == "entry1"
        assert result[1]["name"] == "entry3"

    def test_filter_nonexistent_entries(self, tmp_path):
        """Test filtering with non-existent entry names."""
        input_json = tmp_path / "input.json"
        data = [
            {"name": "entry1", "sequence": "ACDEF"},
            {"name": "entry2", "sequence": "GHIKL"},
        ]
        input_json.write_text(json.dumps(data))

        result = load_entries(input_json, ["entry3", "entry4"])

        assert len(result) == 0

    def test_invalid_json_type(self, tmp_path):
        """Test error when JSON is not a list or dict."""
        input_json = tmp_path / "input.json"
        input_json.write_text(json.dumps("not a list or dict"))

        with pytest.raises(TypeError, match="Expected a list of entries"):
            load_entries(input_json, None)

    def test_skip_non_dict_entries(self, tmp_path):
        """Test that non-dict entries are skipped."""
        input_json = tmp_path / "input.json"
        data = [
            {"name": "entry1", "sequence": "ACDEF"},
            "not a dict",
            {"name": "entry2", "sequence": "GHIKL"},
            123,
        ]
        input_json.write_text(json.dumps(data))

        result = load_entries(input_json, None)

        assert len(result) == 2
        assert result[0]["name"] == "entry1"
        assert result[1]["name"] == "entry2"

    def test_empty_list(self, tmp_path):
        """Test loading an empty list."""
        input_json = tmp_path / "input.json"
        input_json.write_text(json.dumps([]))

        result = load_entries(input_json, None)

        assert len(result) == 0


class TestCoerceModifications:
    """Tests for coerce_modifications function."""

    def test_none_modifications(self):
        """Test with None modifications."""
        result = coerce_modifications(None)
        assert result == []

    def test_empty_list(self):
        """Test with empty list."""
        result = coerce_modifications([])
        assert result == []

    def test_single_modification(self):
        """Test with a single modification."""
        raw = [{"position": "10", "ccd": "MSE"}]
        result = coerce_modifications(raw)

        assert len(result) == 1
        assert result[0]["position"] == 10
        assert result[0]["ccd"] == "MSE"
        assert isinstance(result[0]["position"], int)
        assert isinstance(result[0]["ccd"], str)

    def test_multiple_modifications(self):
        """Test with multiple modifications."""
        raw = [
            {"position": "5", "ccd": "MSE"},
            {"position": "10", "ccd": "SEP"},
            {"position": "15", "ccd": "TPO"},
        ]
        result = coerce_modifications(raw)

        assert len(result) == 3
        assert result[0]["position"] == 5
        assert result[1]["position"] == 10
        assert result[2]["position"] == 15
        assert result[0]["ccd"] == "MSE"
        assert result[1]["ccd"] == "SEP"
        assert result[2]["ccd"] == "TPO"

    def test_integer_position(self):
        """Test when position is already an integer."""
        raw = [{"position": 42, "ccd": "MSE"}]
        result = coerce_modifications(raw)

        assert result[0]["position"] == 42
        assert isinstance(result[0]["position"], int)

    def test_numeric_string_ccd(self):
        """Test when CCD is a numeric string."""
        raw = [{"position": "10", "ccd": 123}]
        result = coerce_modifications(raw)

        assert result[0]["ccd"] == "123"
        assert isinstance(result[0]["ccd"], str)

    def test_preserves_order(self):
        """Test that order is preserved."""
        raw = [
            {"position": "100", "ccd": "AAA"},
            {"position": "1", "ccd": "ZZZ"},
            {"position": "50", "ccd": "MMM"},
        ]
        result = coerce_modifications(raw)

        assert result[0]["position"] == 100
        assert result[1]["position"] == 1
        assert result[2]["position"] == 50
        assert result[0]["ccd"] == "AAA"
        assert result[1]["ccd"] == "ZZZ"
        assert result[2]["ccd"] == "MMM"
