"""Tests for confidock.msa module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from confidock.msa import (
    a3m_path,
    build_pdb_index,
    decode_query_seq,
    find_npz,
    is_valid_a3m,
    npz_to_a3m_text,
    resolve_from_local_npz,
)


class TestA3mPath:
    """Tests for a3m_path."""

    def test_generates_hash_filename(self, tmp_path):
        """Test that filename is based on SHA-256 hash."""
        # a3m_path is pure path arithmetic and never touches the filesystem, so the
        # directory is arbitrary; tmp_path keeps it out of a world-writable location.
        msa_dir = tmp_path / "msa"
        sequence = "ACDEFGHIKLMNPQRSTVWY"

        result = a3m_path(msa_dir, sequence)

        # Should be 16 hex characters
        assert result.parent == msa_dir
        assert result.suffix == ".a3m"
        assert len(result.stem) == 16
        assert all(c in "0123456789abcdef" for c in result.stem)

    def test_same_sequence_same_hash(self, tmp_path):
        """Test that same sequence produces same hash."""
        msa_dir = tmp_path / "msa"
        seq = "ACDEFGHIKLMNPQRSTVWY"

        path1 = a3m_path(msa_dir, seq)
        path2 = a3m_path(msa_dir, seq)

        assert path1 == path2

    def test_different_sequence_different_hash(self, tmp_path):
        """Test that different sequences produce different hashes."""
        msa_dir = tmp_path / "msa"

        path1 = a3m_path(msa_dir, "ACDEFG")
        path2 = a3m_path(msa_dir, "HIJKLM")

        assert path1 != path2


class TestIsValidA3m:
    """Tests for is_valid_a3m."""

    def test_valid_a3m(self):
        """Test valid a3m content."""
        content = """>seq_0
ACDEFGHIKLMNPQRSTVWY
>seq_1
ACDEFGHIKLMNPQRSTVWY
"""
        assert is_valid_a3m(content)

    def test_valid_a3m_with_hash_header(self):
        """Test valid a3m with # header."""
        content = """#header line
ACDEFGHIKLMNPQRSTVWY
"""
        assert is_valid_a3m(content)

    def test_empty_string(self):
        """Test empty string."""
        assert not is_valid_a3m("")

    def test_too_short(self):
        """Test content that's too short."""
        assert not is_valid_a3m(">s\nAC")

    def test_no_header(self):
        """Test content without header."""
        content = "ACDEFGHIKLMNPQRSTVWY"
        assert not is_valid_a3m(content)

    def test_header_but_no_sequence(self):
        """Test content with header but no sequence."""
        content = ">seq_0\n"
        assert not is_valid_a3m(content)

    def test_header_with_digits_in_sequence(self):
        """Test rejects sequences with digits."""
        content = """>seq_0
ACDEFG123HIKLM
"""
        assert not is_valid_a3m(content)

    def test_with_gaps(self):
        """Test valid a3m with gaps."""
        content = """>seq_0
ACDEFG-HIK--LMNPQRSTVWY
"""
        assert is_valid_a3m(content)


class TestDecodeQuerySeq:
    """Tests for decode_query_seq."""

    def test_decode_simple_sequence(self):
        """Test decoding a simple protein sequence."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:  # noqa: S108
            npz_path = Path(tmp.name)

            # Create mock npz with simple sequence
            sequences = np.array([(0, 5)], dtype=[("res_start", "i4"), ("res_end", "i4")])
            # Residue types for "ACDEF" (indices in boltz.data.const.tokens)
            residues = np.array([(0,), (2,), (3,), (4,), (5,)], dtype=[("res_type", "i4")])

            np.savez(npz_path, sequences=sequences, residues=residues)

            try:
                # Need to mock the token mapping
                token_map = {"A": "A", "C": "C", "D": "D", "E": "E", "F": "F"}
                with patch("confidock.msa._TOKENS", ["A", "-", "C", "D", "E", "F"]):
                    with patch("confidock.msa._PROT_TOKEN_TO_LETTER", token_map):
                        result = decode_query_seq(npz_path)
                        assert result == "ACDEF"
            finally:
                npz_path.unlink()

    def test_decode_empty_sequences(self):
        """Test decoding npz with no sequences."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:  # noqa: S108
            npz_path = Path(tmp.name)
            sequences = np.array([], dtype=[("res_start", "i4"), ("res_end", "i4")])
            residues = np.array([], dtype=[("res_type", "i4")])
            np.savez(npz_path, sequences=sequences, residues=residues)

            try:
                result = decode_query_seq(npz_path)
                assert result == ""
            finally:
                npz_path.unlink()


class TestNpzToA3mText:
    """Tests for npz_to_a3m_text."""

    def test_basic_conversion(self):
        """Test basic npz to a3m conversion."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:  # noqa: S108
            npz_path = Path(tmp.name)

            sequences = np.array(
                [(0, 3, 0, 0), (3, 6, 0, 0)],  # Need all 4 fields
                dtype=[("res_start", "i4"), ("res_end", "i4"), ("del_start", "i4"), ("del_end", "i4")],
            )
            residues = np.array([(0,), (1,), (2,), (0,), (1,), (2,)], dtype=[("res_type", "i4")])
            deletions = np.array([], dtype=[("res_idx", "i4"), ("deletion", "i4")])

            np.savez(npz_path, sequences=sequences, residues=residues, deletions=deletions)

            try:
                with patch("confidock.msa._TOKENS", ["A", "C", "D"]):
                    with patch("confidock.msa._PROT_TOKEN_TO_LETTER", {"A": "A", "C": "C", "D": "D"}):
                        result = npz_to_a3m_text(npz_path)

                        assert ">seq_0\n" in result
                        assert ">seq_1\n" in result
            finally:
                npz_path.unlink()


class TestBuildPdbIndex:
    """Tests for build_pdb_index."""

    def test_build_index_with_valid_npz(self):
        """Test building index from valid npz files."""
        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: S108
            npz_dir = Path(tmpdir)

            # Create mock npz file
            npz_path = npz_dir / "7anq_A.npz"
            sequences = np.array([(0, 3)], dtype=[("res_start", "i4"), ("res_end", "i4")])
            residues = np.array([(0,), (1,), (2,)], dtype=[("res_type", "i4")])
            np.savez(npz_path, sequences=sequences, residues=residues)

            with patch("confidock.msa.decode_query_seq", return_value="ACD"):
                index = build_pdb_index({"7anq"}, npz_dir)

                assert "7anq" in index
                assert len(index["7anq"]) == 1
                assert index["7anq"][0][0] == npz_path
                assert index["7anq"][0][1] == "ACD"

    def test_build_index_missing_pdb(self):
        """Test building index when PDB has no npz files."""
        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: S108
            npz_dir = Path(tmpdir)
            index = build_pdb_index({"7anq"}, npz_dir)
            assert "7anq" not in index

    def test_build_index_skip_corrupt_npz(self):
        """Test that corrupt npz files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: S108
            npz_dir = Path(tmpdir)
            npz_path = npz_dir / "7anq_A.npz"
            npz_path.write_text("corrupt data")

            index = build_pdb_index({"7anq"}, npz_dir)
            assert "7anq" not in index


class TestFindNpz:
    """Tests for find_npz."""

    def test_exact_match(self):
        """Test finding exact sequence match."""
        entries = [
            (Path("test1.npz"), "ACDEFG"),
            (Path("test2.npz"), "HIJKLM"),
        ]

        result = find_npz(entries, "ACDEFG")

        assert result is not None
        assert result[0] == Path("test1.npz")
        assert result[1] == 0

    def test_substring_match(self):
        """Test finding substring match."""
        entries = [
            (Path("test1.npz"), "ACDEFGHIJKLM"),
        ]

        result = find_npz(entries, "DEFGH")

        assert result is not None
        assert result[0] == Path("test1.npz")
        assert result[1] == 2  # "DEFGH" starts at position 2

    def test_no_match(self):
        """Test when no match is found."""
        entries = [
            (Path("test1.npz"), "ACDEFG"),
        ]

        result = find_npz(entries, "ZZZZZ")

        assert result is None

    def test_exact_match_preferred(self):
        """Test that exact match is preferred over substring."""
        entries = [
            (Path("test1.npz"), "ACDEFGHIJKLM"),
            (Path("test2.npz"), "DEFGH"),
        ]

        result = find_npz(entries, "DEFGH")

        # Should match test2.npz (exact) not test1.npz (substring)
        assert result[0] == Path("test2.npz")
        assert result[1] == 0


class TestResolveFromLocalNpz:
    """Tests for resolve_from_local_npz."""

    @patch("confidock.msa.build_pdb_index")
    @patch("confidock.msa.find_npz")
    @patch("confidock.msa.npz_to_a3m_text")
    def test_resolve_sequences(self, mock_npz_to_a3m, mock_find_npz, mock_build_index):
        """Test resolving sequences from local npz."""
        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: S108
            npz_msa_dir = Path(tmpdir) / "npz"
            msa_dir = Path(tmpdir) / "msa"
            npz_msa_dir.mkdir()
            msa_dir.mkdir()

            sequences = ["ACDEFG", "HIJKLM"]
            seq_to_pdb_id = {"ACDEFG": "7anq", "HIJKLM": "8qf5"}

            # Mock index
            mock_build_index.return_value = {
                "7anq": [(Path("7anq_A.npz"), "ACDEFG")],
                "8qf5": [(Path("8qf5_H.npz"), "HIJKLM")],
            }

            # Mock find_npz
            mock_find_npz.side_effect = [(Path("7anq_A.npz"), 0), (Path("8qf5_H.npz"), 0)]

            # Mock npz_to_a3m_text
            mock_npz_to_a3m.return_value = ">seq_0\nACDEFG\n"

            resolved = resolve_from_local_npz(sequences, seq_to_pdb_id, npz_msa_dir, msa_dir)

            assert len(resolved) == 2
            assert "ACDEFG" in resolved
            assert "HIJKLM" in resolved

    @patch("confidock.msa.build_pdb_index")
    def test_skip_existing_a3m(self, mock_build_index):
        """Test that existing valid a3m files are not recreated."""
        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: S108
            npz_msa_dir = Path(tmpdir) / "npz"
            msa_dir = Path(tmpdir) / "msa"
            npz_msa_dir.mkdir()
            msa_dir.mkdir()

            sequence = "ACDEFG"
            seq_to_pdb_id = {"ACDEFG": "7anq"}

            # Create existing a3m file
            a3m_file = a3m_path(msa_dir, sequence)
            a3m_file.write_text(">seq_0\nACDEFG\n")

            mock_build_index.return_value = {"7anq": [(Path("7anq_A.npz"), "ACDEFG")]}

            resolved = resolve_from_local_npz([sequence], seq_to_pdb_id, npz_msa_dir, msa_dir)

            # Should be resolved without calling find_npz
            assert sequence in resolved

    def test_no_pdb_mapping(self):
        """Test sequences without PDB ID mapping are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: S108
            npz_msa_dir = Path(tmpdir) / "npz"
            msa_dir = Path(tmpdir) / "msa"
            npz_msa_dir.mkdir()
            msa_dir.mkdir()

            sequences = ["ACDEFG"]
            seq_to_pdb_id = {}  # No mapping

            resolved = resolve_from_local_npz(sequences, seq_to_pdb_id, npz_msa_dir, msa_dir)

            assert len(resolved) == 0
