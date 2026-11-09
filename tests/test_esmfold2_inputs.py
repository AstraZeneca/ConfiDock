"""Tests for confidock.esmfold2.inputs module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from confidock.esmfold2.inference.inputs import (
    pdb_to_esmfold2_entry,
    pdbs_to_esmfold2_json,
)


class TestPdbToEsmfold2Entry:
    """Tests for pdb_to_esmfold2_entry."""

    def test_simple_structure(self):
        """Test conversion of simple structure."""
        pdb_text = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.389   0.000  1.00  0.00           O
ATOM      5  N   CYS B   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      6  CA  CYS B   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      7  C   CYS B   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      8  O   CYS B   1       1.251   2.389   0.000  1.00  0.00           O
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "7bbj_H_L.pdb"
            pdb_file.write_text(pdb_text)

            result = pdb_to_esmfold2_entry(pdb_file)

            assert result is not None
            assert result["name"] == "7bbj_H_L"
            assert len(result["sequences"]) == 2
            assert result["sequences"][0]["type"] == "protein"
            assert result["sequences"][0]["id"] == "H"
            assert len(result["sequences"][0]["sequence"]) > 0

    def test_with_modifications(self):
        """Test structure with non-standard residues."""
        pdb_text = """ATOM      1  N   MSE A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  MSE A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   MSE A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   MSE A   1       1.251   2.389   0.000  1.00  0.00           O
ATOM      5  SE  MSE A   1       4.000   0.000   0.000  1.00  0.00          SE
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "test_H_L.pdb"  # Need multi-chain format
            pdb_file.write_text(pdb_text)

            result = pdb_to_esmfold2_entry(pdb_file)

            assert result is not None
            assert len(result["sequences"]) == 1
            seq_entry = result["sequences"][0]
            assert "M" in seq_entry["sequence"]  # MSE maps to M
            assert "modifications" in seq_entry
            assert len(seq_entry["modifications"]) == 1
            assert seq_entry["modifications"][0]["ccd"] == "MSE"
            assert seq_entry["modifications"][0]["position"] == 1

    def test_no_modifications_field_for_standard(self):
        """Test that standard residues don't add modifications field."""
        pdb_text = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.389   0.000  1.00  0.00           O
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "test_H_L.pdb"  # Need multi-chain format
            pdb_file.write_text(pdb_text)

            result = pdb_to_esmfold2_entry(pdb_file)

            assert result is not None
            # Standard residues should not have modifications field
            seq_entry = result["sequences"][0]
            assert "modifications" not in seq_entry or len(seq_entry.get("modifications", [])) == 0

    def test_invalid_filename(self):
        """Test handling of invalid filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "invalid.pdb"
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N")

            result = pdb_to_esmfold2_entry(pdb_file)

            assert result is None

    def test_parse_error(self):
        """Test handling of parse errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "test_H_L.pdb"
            # Write truncated PDB that will cause parse error but be caught
            pdb_file.write_text("ATOM   TRUNCATED")

            result = pdb_to_esmfold2_entry(pdb_file)

            # Parser error should be caught and return None
            assert result is None

    def test_concatenated_chain_format(self):
        """Test parsing concatenated chain IDs."""
        pdb_text = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.389   0.000  1.00  0.00           O
ATOM      5  N   CYS B   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      6  CA  CYS B   1       1.458   0.000   0.000  1.00  0.00           C
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "7bbj_HL.pdb"
            pdb_file.write_text(pdb_text)

            result = pdb_to_esmfold2_entry(pdb_file)

            assert result is not None
            assert len(result["sequences"]) == 2

    def test_chain_count_mismatch_warning(self, caplog):
        """Test warning when filename chains don't match PDB chains."""
        import logging

        pdb_text = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Filename says 3 chains but PDB only has 1
            pdb_file = Path(tmpdir) / "7bbj_H_L_A.pdb"
            pdb_file.write_text(pdb_text)

            with caplog.at_level(logging.WARNING):
                result = pdb_to_esmfold2_entry(pdb_file)

            assert "filename encodes" in caplog.text
            assert result is not None

    def test_no_valid_chains_extracted(self, caplog):
        """Test when no valid chains can be extracted."""
        import logging

        # PDB with only HETATM records (no standard residues)
        pdb_text = """HETATM    1  O   HOH A 101       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  O   HOH A 102       1.000   1.000   1.000  1.00  0.00           O
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "test_H_L.pdb"
            pdb_file.write_text(pdb_text)

            with caplog.at_level(logging.ERROR):
                result = pdb_to_esmfold2_entry(pdb_file)

            assert result is None
            assert "no valid chains extracted" in caplog.text

    def test_unknown_residue_skipped(self, caplog):
        """Test that unknown residues are skipped with warning."""
        import logging

        pdb_text = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  N   ZZZ A   2       2.000   2.000   2.000  1.00  0.00           N
ATOM      4  CA  ZZZ A   2       3.000   3.000   3.000  1.00  0.00           C
ATOM      5  N   GLY A   3       4.000   4.000   4.000  1.00  0.00           N
ATOM      6  CA  GLY A   3       5.000   5.000   5.000  1.00  0.00           C
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "test_H_L.pdb"
            pdb_file.write_text(pdb_text)

            with caplog.at_level(logging.WARNING):
                result = pdb_to_esmfold2_entry(pdb_file)

            assert result is not None
            # Unknown residue ZZZ should be skipped, so sequence is AG not AZG
            assert result["sequences"][0]["sequence"] == "AG"
            assert "unrecognised residue 'ZZZ'" in caplog.text


class TestPdbsToEsmfold2Json:
    """Tests for pdbs_to_esmfold2_json."""

    def test_convert_multiple_pdbs(self):
        """Test converting multiple PDB files."""
        pdb_text = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            pdb1 = tmpdir_path / "7anq_H_A.pdb"  # Need multi-chain format
            pdb2 = tmpdir_path / "8qf5_L_B.pdb"
            pdb1.write_text(pdb_text)
            pdb2.write_text(pdb_text)

            output_json = tmpdir_path / "output.json"

            entries = pdbs_to_esmfold2_json([pdb1, pdb2], output_json)

            assert len(entries) == 2
            assert output_json.exists()

            # Check JSON content
            data = json.loads(output_json.read_text())
            assert len(data) == 2
            assert data[0]["name"] == "7anq_H_A"
            assert data[1]["name"] == "8qf5_L_B"

    def test_skip_failed_conversions(self):
        """Test that failed conversions are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            valid_pdb = tmpdir_path / "7anq_H_A.pdb"  # Valid format
            valid_pdb.write_text(
                """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
"""
            )

            invalid_pdb = tmpdir_path / "invalid.pdb"
            invalid_pdb.write_text("invalid")

            output_json = tmpdir_path / "output.json"

            entries = pdbs_to_esmfold2_json([valid_pdb, invalid_pdb], output_json)

            # Only valid entry should be included
            assert len(entries) == 1
            assert entries[0]["name"] == "7anq_H_A"

    def test_creates_output_directory(self):
        """Test that output directory is created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            pdb_file = tmpdir_path / "7anq_H_A.pdb"  # Valid format
            pdb_file.write_text(
                """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
"""
            )

            output_json = tmpdir_path / "subdir" / "output.json"

            pdbs_to_esmfold2_json([pdb_file], output_json)

            assert output_json.exists()
            assert output_json.parent.exists()
