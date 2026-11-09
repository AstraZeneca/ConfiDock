"""Test the boltz2 multi-char chain-ID fixup in ``graph_builder.fix_boltz2_pdb``.

Boltz2 writes chain IDs like ``Hc``/``Lc``/``Kx``, which overflow the PDB format's
single chain column (22) and shift the rest of the line one column right. The fixup
maps them to single chars, and must consume the RIGHT number of columns per line:
two for a genuinely 2-char ID (de-shifting it), one for an already-aligned 1-char ID.

Boltz2 mixes both widths inside one file (13.6% of the manifest's .pdb files), so
consuming two columns unconditionally silently shifted 1-char lines LEFT by one. That
only breaks when a coordinate fills its whole 8-column field -- e.g. ``y=-101.187``
fuses with x into the unparseable ``14.275-`` -- which is why a single file out of
49,369 ever failed, as a "Invalid or missing coordinate(s)" that the dataset's retry
logic swallowed.
"""

import pytest
from Bio.PDB import PDBParser

from confidock.dockq_predictor.data.graph_builder import fix_boltz2_pdb

# Columns 31-38 / 39-46 / 47-54 (1-indexed, inclusive) hold x / y / z.
_X, _Y, _Z = slice(30, 38), slice(38, 46), slice(46, 54)

# A 2-char chain ('Hc', shifted right one column) followed by a 1-char chain ('A',
# correctly aligned) whose y needs all 8 columns. Widths here are byte-exact.
MIXED_PDB = (
    "ATOM      1  N   LEU Hc 110     -21.848  51.328   3.593  1.00 86.93           N\n"
    "ATOM      2  CA  LEU Hc 110     -21.972  50.259   5.858  1.00 86.93           C\n"
    "TER       3      LEU Hc 110                                                    \n"
    "ATOM      4  N   GLU A   1      14.275-101.187   1.545  1.00 73.97           N\n"
    "ATOM      5  CA  GLU A   1      14.895-100.699   0.331  1.00 73.97           C\n"
    "END\n"
)


def _write(tmp_path, text):
    p = tmp_path / "mixed.pdb"
    p.write_text(text)
    return str(p)


def test_coordinates_land_on_spec_columns(tmp_path):
    """Both chain widths must end up with x/y/z on PDB columns 31-54."""
    fixed, _ = fix_boltz2_pdb(_write(tmp_path, MIXED_PDB))
    atom_lines = [line for line in open(fixed) if line.startswith("ATOM")]

    # 2-char chain: de-shifted by consuming both columns.
    assert float(atom_lines[0][_X]) == pytest.approx(-21.848)
    assert float(atom_lines[0][_Y]) == pytest.approx(51.328)
    assert float(atom_lines[0][_Z]) == pytest.approx(3.593)

    # 1-char chain: alignment preserved. This is the regression -- consuming two
    # columns here produced x='14.275-', which is not a float. Index 2, not 3:
    # atom_lines skips the TER record.
    assert float(atom_lines[2][_X]) == pytest.approx(14.275)
    assert float(atom_lines[2][_Y]) == pytest.approx(-101.187)
    assert float(atom_lines[2][_Z]) == pytest.approx(1.545)


def test_biopython_parses_the_fixed_file(tmp_path):
    """The whole point of the fixup: PDBParser must accept the result."""
    fixed, chain_map = fix_boltz2_pdb(_write(tmp_path, MIXED_PDB))
    structure = PDBParser(QUIET=True).get_structure("complex", fixed)

    # Every distinct original ID gets its own single-char replacement.
    assert chain_map == {"Hc": "A", "A": "B"}
    assert sorted(c.id for c in structure[0]) == ["A", "B"]

    # No atom is dropped, and the wide coordinate survives the round trip.
    # Biopython stores coords as float32, so compare with a tolerance.
    atoms = list(structure.get_atoms())
    assert len(atoms) == 4
    assert atoms[2].coord[1] == pytest.approx(-101.187, abs=1e-3)


def test_single_char_only_file_is_untouched(tmp_path):
    """A file with no multi-char IDs must pass through byte-identically."""
    plain = (
        "ATOM      1  N   GLU A   1      14.275-101.187   1.545  1.00 73.97           N\n"
        "ATOM      2  CA  GLU B   1      14.895-100.699   0.331  1.00 73.97           C\n"
        "END\n"
    )
    fixed, chain_map = fix_boltz2_pdb(_write(tmp_path, plain))
    # A->A, B->B: distinct inputs, distinct outputs, so content is unchanged.
    assert chain_map == {"A": "A", "B": "B"}
    assert open(fixed).read() == plain


def test_non_atom_records_pass_through(tmp_path):
    """TER/END and other records are copied verbatim, not re-columned."""
    fixed, _ = fix_boltz2_pdb(_write(tmp_path, MIXED_PDB))
    lines = open(fixed).readlines()
    assert lines[2].startswith("TER")
    assert lines[2] == MIXED_PDB.splitlines(keepends=True)[2]
    assert lines[-1].strip() == "END"
