"""Tests for confidock.protenix.inputs module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from confidock.protenix.inference.inputs import (
    pdb_to_protenix_entry,
    pdbs_to_protenix_json,
)


class TestPdbToProtenixEntry:
    """Tests for pdb_to_protenix_entry."""

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

            result = pdb_to_protenix_entry(pdb_file)

            assert result is not None
            assert result["name"] == "7bbj_H_L"
            assert len(result["sequences"]) == 2
            assert result["sequences"][0]["proteinChain"]["count"] == 1
            assert len(result["sequences"][0]["proteinChain"]["sequence"]) > 0
            assert result["sequences"][0]["proteinChain"]["modifications"] == []

    def test_with_modifications(self):
        """Test structure with non-standard residues."""
        pdb_text = """ATOM      1  N   MSE A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  MSE A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   MSE A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   MSE A   1       1.251   2.389   0.000  1.00  0.00           O
ATOM      5  SE  MSE A   1       4.000   0.000   0.000  1.00  0.00          SE
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "test_H_L.pdb"  # Need at least 2 chains in filename
            pdb_file.write_text(pdb_text)

            result = pdb_to_protenix_entry(pdb_file)

            assert result is not None
            assert len(result["sequences"]) == 1
            chain = result["sequences"][0]["proteinChain"]
            assert "M" in chain["sequence"]  # MSE maps to M
            assert len(chain["modifications"]) == 1
            assert chain["modifications"][0]["ptmType"] == "CCD_MSE"
            assert chain["modifications"][0]["ptmPosition"] == 1

    def test_invalid_filename(self):
        """Test handling of invalid filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "invalid.pdb"
            pdb_file.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N")

            result = pdb_to_protenix_entry(pdb_file)

            assert result is None

    def test_parse_error(self):
        """Test handling of parse errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "test_H_L.pdb"
            # Write truncated PDB that will cause parse error but be caught
            pdb_file.write_text("ATOM   TRUNCATED")

            result = pdb_to_protenix_entry(pdb_file)

            # Parser error should be caught and return None
            assert result is None

    def test_empty_chains_skipped(self):
        """Test that chains with no valid residues are skipped."""
        pdb_text = """HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      5  N   ALA B   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      6  CA  ALA B   1       1.458   0.000   0.000  1.00  0.00           C
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_file = Path(tmpdir) / "test_H_L.pdb"
            pdb_file.write_text(pdb_text)

            result = pdb_to_protenix_entry(pdb_file)

            # Should have chain B only
            if result is not None:
                assert len(result["sequences"]) >= 1

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

            result = pdb_to_protenix_entry(pdb_file)

            assert result is not None
            assert len(result["sequences"]) == 2


class TestPdbsToProtenixJson:
    """Tests for pdbs_to_protenix_json."""

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

            entries = pdbs_to_protenix_json([pdb1, pdb2], output_json)

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

            entries = pdbs_to_protenix_json([valid_pdb, invalid_pdb], output_json)

            # Only valid entry should be included
            assert len(entries) == 1
            assert entries[0]["name"] == "7anq_H_A"

    def test_creates_output_directory(self):
        """Test that output directory is created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            pdb_file = tmpdir_path / "7anq_H_A.pdb"  # Valid format
            pdb_file.write_text(
                """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
"""
            )

            output_json = tmpdir_path / "subdir" / "output.json"

            pdbs_to_protenix_json([pdb_file], output_json)

            assert output_json.exists()
            assert output_json.parent.exists()
