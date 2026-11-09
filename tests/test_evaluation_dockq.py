"""Tests for confidock.evaluation.dockq module."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from confidock.evaluation import dockq


def _install_fake_dockq_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake DockQ module so local imports succeed in tests."""
    dockq_pkg = types.ModuleType("DockQ")
    dockq_mod = types.ModuleType("DockQ.DockQ")
    dockq_mod.load_PDB = lambda *_args, **_kwargs: object()
    dockq_mod.run_on_all_native_interfaces = lambda *_args, **_kwargs: ({}, 0.0)
    dockq_pkg.DockQ = dockq_mod
    monkeypatch.setitem(sys.modules, "DockQ", dockq_pkg)
    monkeypatch.setitem(sys.modules, "DockQ.DockQ", dockq_mod)


def test_is_hydrogen_atom_variants() -> None:
    """Hydrogen atom detection should handle element and naming conventions."""
    assert dockq._is_hydrogen_atom("CA", "H") is True
    assert dockq._is_hydrogen_atom("H1", "") is True
    assert dockq._is_hydrogen_atom("1HG2", "") is True
    assert dockq._is_hydrogen_atom("CA", "C") is False


def test_strip_hydrogens_removes_atom_and_hetatm_records() -> None:
    """Hydrogen ATOM/HETATM records should be removed from PDB text."""
    pdb_text = "\n".join(
        [
            "ATOM      1  H   GLY A   1      11.000  11.000  11.000  1.00 20.00           H",
            "ATOM      2  CA  GLY A   1      12.000  11.000  11.000  1.00 20.00           C",
            "HETATM    3 1HG2 VAL A   1      13.000  11.000  11.000  1.00 20.00           ",
        ]
    )

    stripped = dockq._strip_hydrogens(pdb_text)
    assert " CA " in stripped
    assert " H   GLY" not in stripped
    assert "1HG2" not in stripped


def test_identify_chain_ids_and_remap_and_rewrite() -> None:
    """Multi-character chain IDs should be detected and rewritten using remap."""
    lines = [
        "ATOM      1  CA  GLY A   1      12.0   11.0   11.0  1.00 20.00           C",
        "ATOM      2  CA  GLY Hc  1      12.0   11.0   11.0  1.00 20.00           C",
        "ATOM      3  CA  GLY Lc  1      12.0   11.0   11.0  1.00 20.00           C",
    ]

    single_ids, multi_ids = dockq._identify_chain_ids(lines)
    assert single_ids == {"A"}
    assert multi_ids == {"Hc", "Lc"}

    remap = dockq._create_chain_id_remapping(single_ids, multi_ids)
    assert remap["A"] == "A"
    assert remap["Hc"] != remap["Lc"]

    rewritten = dockq._rewrite_pdb_lines_with_remap(lines, remap)
    # Rewritten lines should have one-character chain IDs in column 22.
    for line in rewritten:
        if line.startswith("ATOM"):
            assert line[22] == " "


def test_normalise_pdb_chains_no_multi_returns_identity() -> None:
    """When no multi-char chains exist, text should be unchanged with identity map."""
    text = "ATOM      1  CA  GLY A   1      12.0   11.0   11.0  1.00 20.00           C"
    out_text, remap = dockq._normalise_pdb_chains(text)
    assert out_text == text
    assert remap == {"A": "A"}


def test_add_occupancy_helpers() -> None:
    """Occupancy helpers should classify atom lines and add occupancy when needed."""
    atom_line = "ATOM 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    not_atom = "HELLO"

    assert dockq._is_atom_line("ATOM foo") is True
    assert dockq._is_atom_line("HETATM foo") is True
    assert dockq._is_atom_line(not_atom) is False

    with_occ = dockq._add_occupancy_to_atom_line(atom_line)
    assert "1.0" in with_occ.split()

    assert dockq._process_cif_line(atom_line, header_done=False) == atom_line
    assert dockq._process_cif_line(atom_line, header_done=True) != atom_line


def test_add_occupancy_field_to_cif_inserts_header_and_values() -> None:
    """CIF occupancy field should be added when missing."""
    cif_text = "\n".join(
        [
            "data_x",
            "loop_",
            "_atom_site.B_iso_or_equiv",
            "ATOM 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
        ]
    )

    out = dockq._add_occupancy_field_to_cif(cif_text)
    assert "_atom_site.occupancy" in out
    assert "1.0" in out


def test_find_available_chain_id() -> None:
    """Available chain ID selection should skip used and assigned IDs."""
    chosen = dockq._find_available_chain_id({"A", "B"}, {"C"})
    assert chosen not in {"A", "B", "C"}


@pytest.mark.parametrize(
    ("used_intended", "expected"),
    [
        (True, "true"),
        (False, "false"),
        (None, ""),
    ],
)
def test_extract_complex_metrics_used_intended_formats(used_intended: bool | None, expected: str) -> None:
    """used_intended_antigen should be serialized to expected string values."""
    metrics = dockq._extract_complex_metrics(
        {"DockQ": 0.5, "iRMS": 1.0, "LRMSD": 2.0, "fnat": 0.8, "fnonnat": 0.1, "F1": 0.7, "clashes": 0},
        selected_antigen_chain="A",
        used_intended_antigen=used_intended,
    )
    assert metrics["used_intended_antigen"] == expected
    assert metrics["selected_antigen_chain"] == "A"


def test_resolve_model_binder_ordering_success_and_missing(caplog: pytest.LogCaptureFixture) -> None:
    """Binder ordering resolution should return mapped list or None when missing chains."""
    mapped = dockq._resolve_model_binder_ordering(("H", "L"), {"H": "X", "L": "Y"})
    assert mapped == ["X", "Y"]

    with caplog.at_level("DEBUG"):
        missing = dockq._resolve_model_binder_ordering(("H", "L"), {"H": "X"})
    assert missing is None
    assert "Skipping - chains" in caplog.text


def test_build_chain_map_and_antigen_order() -> None:
    """Merged chain map and antigen evaluation ordering should prioritize intended antigen."""
    merged = dockq._build_merged_chain_map("HL", "XY", ["A", "B"], {"A": "M", "B": "N"})
    assert merged == {"HL": "XY", "A": "M", "B": "N"}

    order = dockq._build_antigen_eval_order(("H", "L"), ["A", "B"], merged, "B")
    assert order[0] == "B"
    assert set(order) == {"A", "B"}


def test_collect_antigen_candidates_and_select() -> None:
    """Candidate collection and final selection should honor intended antigen policy."""
    interfaces = {
        "HLA": {"DockQ": 0.4},
        "HLB": {"DockQ": 0.8},
    }
    merged_chain_map = {"HL": "XY", "A": "M", "B": "N"}
    candidates = dockq._collect_antigen_candidates(
        interfaces=interfaces,
        antigen_eval_order=["A", "B"],
        merged_chain_map=merged_chain_map,
        merged_binder_id="HL",
        native_chain_ids={"HL", "A", "B"},
        model_chain_ids={"XY", "M", "N"},
    )
    assert len(candidates) == 2

    intended_result, intended_score = dockq._select_complex_candidate(candidates, "A")
    assert intended_result["selected_antigen_chain"] == "A"
    assert intended_score == pytest.approx(0.4)

    best_result, best_score = dockq._select_complex_candidate(candidates, None)
    assert best_result["selected_antigen_chain"] == "B"
    assert best_score == pytest.approx(0.8)


def test_run_complex_level_dockq_returns_base_when_no_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complex scoring should return base result when binder/antigen split is missing."""
    _install_fake_dockq_module(monkeypatch)
    result = dockq.run_complex_level_dockq(
        model_path=Path("model.pdb"),
        native_path=Path("native.pdb"),
        chain_map={},
        native_chain_roles={"A": "binder"},
    )
    assert result == {
        "binder_chains": "A",
        "antigen_chains": "",
        "selected_antigen_chain": "",
        "used_intended_antigen": "",
    }


def test_run_complex_level_dockq_returns_base_for_single_binder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complex scoring should populate interface scores for single binder chain."""
    _install_fake_dockq_module(monkeypatch)

    # Mock run_dockq to return interface scores for the A-B interface
    def mock_run_dockq(*args, **kwargs):
        return (
            {
                "AB": {
                    "DockQ": 0.75,
                    "iRMSD": 1.5,
                    "LRMSD": 3.0,
                    "fnat": 0.8,
                    "fnonnat": 0.1,
                    "F1": 0.85,
                    "clashes": 2,
                }
            },
            0.75,
        )

    monkeypatch.setattr(dockq, "run_dockq", mock_run_dockq)

    result = dockq.run_complex_level_dockq(
        model_path=Path("model.pdb"),
        native_path=Path("native.pdb"),
        chain_map={"A": "X", "B": "Y"},
        native_chain_roles={"A": "binder", "B": "antigen"},
    )
    assert result == {
        "binder_chains": "A",
        "antigen_chains": "B",
        "complex_dockq": 0.75,
        "complex_irmsd": 1.5,
        "complex_lrmsd": 3.0,
        "complex_fnat": 0.8,
        "complex_fnonnat": 0.1,
        "complex_F1": 0.85,
        "complex_clashes": 2,
        "selected_antigen_chain": "B",
        "used_intended_antigen": "",
    }


def test_run_complex_level_dockq_uses_best_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complex scoring should merge base and best result from helper."""
    _install_fake_dockq_module(monkeypatch)

    monkeypatch.setattr(
        dockq,
        "_compute_best_complex_result",
        lambda *args, **kwargs: (
            {
                "complex_dockq": 0.9,
                "selected_antigen_chain": "C",
                "used_intended_antigen": "true",
            },
            0.9,
            ("H", "L"),
            2,
        ),
    )

    result = dockq.run_complex_level_dockq(
        model_path=Path("model.pdb"),
        native_path=Path("native.pdb"),
        chain_map={"H": "X", "L": "Y", "C": "Z"},
        native_chain_roles={"H": "binder", "L": "binder", "C": "antigen"},
        intended_antigen_chain="C",
    )
    assert result is not None
    assert result["binder_chains"] == "H+L"
    assert result["antigen_chains"] == "C"
    assert result["complex_dockq"] == pytest.approx(0.9)
    assert result["selected_antigen_chain"] == "C"


def test_run_complex_level_dockq_no_best_result_returns_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """No valid permutation result should return base labels only."""
    _install_fake_dockq_module(monkeypatch)
    monkeypatch.setattr(dockq, "_compute_best_complex_result", lambda *args, **kwargs: (None, -1.0, None, 4))

    result = dockq.run_complex_level_dockq(
        model_path=Path("model.pdb"),
        native_path=Path("native.pdb"),
        chain_map={"H": "X", "L": "Y", "C": "Z"},
        native_chain_roles={"H": "binder", "L": "binder", "C": "antigen"},
    )
    assert result == {
        "binder_chains": "H+L",
        "antigen_chains": "C",
        "selected_antigen_chain": "",
        "used_intended_antigen": "",
    }


def test_run_complex_level_dockq_handles_internal_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Internal errors should be caught and base result returned."""
    _install_fake_dockq_module(monkeypatch)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(dockq, "_compute_best_complex_result", _boom)

    result = dockq.run_complex_level_dockq(
        model_path=Path("model.pdb"),
        native_path=Path("native.pdb"),
        chain_map={"H": "X", "L": "Y", "C": "Z"},
        native_chain_roles={"H": "binder", "L": "binder", "C": "antigen"},
    )
    assert result == {
        "binder_chains": "H+L",
        "antigen_chains": "C",
        "selected_antigen_chain": "",
        "used_intended_antigen": "",
    }
