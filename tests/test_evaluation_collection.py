"""Tests for confidock.evaluation.collection helper functions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from confidock.evaluation import collection


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("1a2y_BAC", "C"),
        ("1jp5_HcLcC", "C"),
        ("8qrg_H_L_A", "A"),
        ("nounderscore", None),
        ("1abc_", None),
    ],
)
def test_infer_intended_antigen_chain(entry: str, expected: str | None) -> None:
    """Intended antigen chain should be inferred from supported entry naming patterns."""
    assert collection._infer_intended_antigen_chain(entry) == expected


def test_resolve_intended_native_antigen_chain_variants() -> None:
    """Intended chain should resolve from native or model/original chain namespaces."""
    native_roles = {"A": "binder", "B": "antigen", "C": "antigen"}
    chain_map = {"A": "H", "B": "T", "C": "T"}

    # Already native antigen ID.
    assert collection._resolve_intended_native_antigen_chain("B", chain_map, native_roles) == "B"
    # One matching native antigen candidate.
    assert collection._resolve_intended_native_antigen_chain("T", {"B": "T"}, {"B": "antigen"}) == "B"
    # Multiple candidates should return sorted deterministic choice.
    assert collection._resolve_intended_native_antigen_chain("T", chain_map, native_roles) == "B"
    # Missing/unresolvable.
    assert collection._resolve_intended_native_antigen_chain(None, chain_map, native_roles) is None
    assert collection._resolve_intended_native_antigen_chain("Z", chain_map, native_roles) is None


def test_add_epitope_info_to_row_with_and_without_data() -> None:
    """Row epitope fields should be populated from info or with default blanks."""
    row = {}
    collection._add_epitope_info_to_row(row, {"matched_native": "1abc"})
    assert row["matched_native"] == "1abc"

    row2 = {}
    collection._add_epitope_info_to_row(row2, None)
    assert row2["matched_native"] == ""
    assert row2["scored_against_intended"] == ""


def test_apply_epitope_matching_result_extracts_and_mutates_info(tmp_path: Path) -> None:
    """Epitope matching result should extract path/map/roles and remove internal keys."""
    info = {
        "matched_native_path": tmp_path / "x.pdb",
        "matched_chain_map": {"A": "H"},
        "matched_roles": {"A": "binder"},
        "matched_native": "x",
    }

    native, cmap, roles = collection._apply_epitope_matching_result(info)
    assert native == tmp_path / "x.pdb"
    assert cmap == {"A": "H"}
    assert roles == {"A": "binder"}
    assert "matched_native_path" not in info
    assert "matched_chain_map" not in info
    assert "matched_roles" not in info


def test_apply_epitope_matching_result_none_inputs() -> None:
    """Missing epitope info should produce all None outputs."""
    assert collection._apply_epitope_matching_result(None) == (None, None, None)
    assert collection._apply_epitope_matching_result({"foo": "bar"}) == (None, None, None)


def test_select_native_for_scoring_no_match_uses_intended(tmp_path: Path) -> None:
    """No matched native should keep intended target unchanged."""
    intended_native = tmp_path / "intended.pdb"
    out = collection._select_native_for_scoring(
        entry="1abc_HLA",
        matched_native=None,
        matched_chain_map=None,
        matched_roles=None,
        epitope_match_info=None,
        intended_native_path=intended_native,
        intended_chain_map={"A": "H"},
        intended_chain_roles={"A": "binder"},
    )
    assert out == (intended_native, {"A": "H"}, {"A": "binder"}, None)


def test_select_native_for_scoring_matched_without_info(tmp_path: Path) -> None:
    """Matched native should be used directly when no info dict exists."""
    matched_native = tmp_path / "matched.pdb"
    out = collection._select_native_for_scoring(
        entry="1abc_HLA",
        matched_native=matched_native,
        matched_chain_map={"B": "T"},
        matched_roles={"B": "antigen"},
        epitope_match_info=None,
        intended_native_path=tmp_path / "intended.pdb",
        intended_chain_map={"A": "H"},
        intended_chain_roles={"A": "binder"},
    )
    assert out == (matched_native, {"B": "T"}, {"B": "antigen"}, None)


def test_select_native_for_scoring_poor_match_reverts_to_intended(tmp_path: Path) -> None:
    """Poor epitope match should force scoring against intended target."""
    info = {
        "match_quality": "poor_match",
        "matched_native": "2xyz",
        "epitope_distance": 21.3,
        "matched_to_intended_target": False,
    }
    intended_native = tmp_path / "intended.pdb"
    out_native, out_map, out_roles, out_info = collection._select_native_for_scoring(
        entry="2xyz_HLA",
        matched_native=tmp_path / "matched.pdb",
        matched_chain_map={"B": "T"},
        matched_roles={"B": "antigen"},
        epitope_match_info=info,
        intended_native_path=intended_native,
        intended_chain_map={"A": "H"},
        intended_chain_roles={"A": "binder"},
    )
    assert out_native == intended_native
    assert out_map == {"A": "H"}
    assert out_roles == {"A": "binder"}
    assert out_info["scored_against_intended"] is True


def test_select_native_for_scoring_good_match_uses_matched(tmp_path: Path) -> None:
    """Good/ambiguous match should use matched native and preserve intended flag."""
    info = {
        "match_quality": "good_match",
        "matched_native": "2xyz",
        "epitope_distance": 4.1,
        "matched_to_intended_target": False,
    }
    matched_native = tmp_path / "matched.pdb"
    out_native, out_map, out_roles, out_info = collection._select_native_for_scoring(
        entry="2xyz_HLA",
        matched_native=matched_native,
        matched_chain_map={"B": "T"},
        matched_roles={"B": "antigen"},
        epitope_match_info=info,
        intended_native_path=tmp_path / "intended.pdb",
        intended_chain_map={"A": "H"},
        intended_chain_roles={"A": "binder"},
    )
    assert out_native == matched_native
    assert out_map == {"B": "T"}
    assert out_roles == {"B": "antigen"}
    assert out_info["scored_against_intended"] is False


def test_log_epitope_match_details_parses_seed_and_sample(tmp_path: Path) -> None:
    """Epitope log entries should include extracted seed/sample and candidate names."""
    log = []
    collection._log_epitope_match_details(
        epitope_match_log=log,
        pdb_id="1abc",
        entry="1abc_HLA",
        model_path=tmp_path / "pred_seed7_sample3_model.pdb",
        matched_native=tmp_path / "1abc_HLA.pdb",
        distance=2.34,
        quality="good_match",
        native_candidates=[
            (tmp_path / "1abc_HLA.pdb", {}, {}),
            (tmp_path / "1abc_HLB.pdb", {}, {}),
        ],
    )
    assert len(log) == 1
    rec = log[0]
    assert rec["seed"] == "7"
    assert rec["sample"] == "3_model"
    assert rec["matched_to_intended_target"] is True
    assert rec["num_candidates"] == 2
    assert "1abc_HLA" in rec["all_candidates"]


def test_perform_epitope_matching_no_data_short_circuits(tmp_path: Path) -> None:
    """Epitope matching should return None when multiplicity metadata is absent."""
    assert (
        collection._perform_epitope_matching(
            entry="1abc_HLA",
            pdb_id="1abc",
            model_path=tmp_path / "m.pdb",
            native_dir=tmp_path,
            native_chain_roles={"A": "binder", "B": "antigen"},
            json_dir=None,
            multiplicity_data=None,
            epitope_match_log=None,
        )
        is None
    )


def test_is_same_ab_same_ag_case() -> None:
    """Same-antibody same-antigen detection should reflect sequence multiplicity."""
    assert collection._is_same_ab_same_ag_case(
        [
            {"h_seq": "H", "l_seq": "L", "t_seq": "T"},
            {"h_seq": "H", "l_seq": "L", "t_seq": "T"},
        ]
    )
    assert not collection._is_same_ab_same_ag_case(
        [
            {"h_seq": "H1", "l_seq": "L", "t_seq": "T"},
            {"h_seq": "H2", "l_seq": "L", "t_seq": "T"},
        ]
    )


def test_build_native_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Native candidates should be built only for existing candidate PDBs."""
    keep = tmp_path / "1abc_HLA.pdb"
    keep.touch()
    monkeypatch.setattr(collection, "get_chain_map_with_sequence_matching", lambda *args, **kwargs: {"A": "H"})

    candidates = collection._build_native_candidates(
        entries_for_pdb=[
            {"full_entry": "1abc_HLA"},
            {"full_entry": "1abc_missing"},
        ],
        model_path=tmp_path / "model.pdb",
        native_dir=tmp_path,
        native_chain_roles={"A": "binder"},
        json_dir=None,
    )

    assert len(candidates) == 1
    assert candidates[0][0] == keep
    assert candidates[0][1] == {"A": "H"}


def test_perform_epitope_matching_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful epitope matching should return enriched match info and log entry."""
    candidate_a = tmp_path / "1abc_HLA.pdb"
    candidate_b = tmp_path / "1abc_HLB.pdb"
    candidate_a.touch()
    candidate_b.touch()

    monkeypatch.setattr(collection, "get_chain_map_with_sequence_matching", lambda *args, **kwargs: {"A": "H"})
    monkeypatch.setattr(
        collection,
        "match_epitope_by_interface_position",
        lambda *args, **kwargs: (candidate_b, {"B": "T"}, {"B": "antigen"}, 3.2, "good_match"),
    )

    log = []
    info = collection._perform_epitope_matching(
        entry="1abc_HLA",
        pdb_id="1abc",
        model_path=tmp_path / "pred_seed2_sample1.pdb",
        native_dir=tmp_path,
        native_chain_roles={"A": "binder", "B": "antigen"},
        json_dir=None,
        multiplicity_data={
            "1abc": [
                {"full_entry": "1abc_HLA", "h_seq": "H", "l_seq": "L", "t_seq": "T"},
                {"full_entry": "1abc_HLB", "h_seq": "H", "l_seq": "L", "t_seq": "T"},
            ]
        },
        epitope_match_log=log,
    )

    assert info is not None
    assert info["matched_native"] == "1abc_HLB"
    assert info["matched_to_intended_target"] is False
    assert len(log) == 1


def test_perform_epitope_matching_cif_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CIF epitope matching should include numeric seed/sample in the log."""
    candidate_a = tmp_path / "1abc_HLA.pdb"
    candidate_b = tmp_path / "1abc_HLB.pdb"
    candidate_a.touch()
    candidate_b.touch()

    monkeypatch.setattr(collection, "get_chain_map_with_sequence_matching", lambda *args, **kwargs: {"A": "H"})
    monkeypatch.setattr(
        collection,
        "match_epitope_by_interface_position",
        lambda *args, **kwargs: (candidate_a, {"A": "H"}, {"A": "binder"}, 1.5, "good_match"),
    )

    log = []
    info = collection._perform_epitope_matching_cif(
        entry="1abc_HLA",
        pdb_id="1abc",
        cif_path=tmp_path / "pred.cif",
        native_dir=tmp_path,
        native_chain_roles={"A": "binder", "B": "antigen"},
        json_dir=None,
        multiplicity_data={
            "1abc": [
                {"full_entry": "1abc_HLA", "h_seq": "H", "l_seq": "L", "t_seq": "T"},
                {"full_entry": "1abc_HLB", "h_seq": "H", "l_seq": "L", "t_seq": "T"},
            ]
        },
        epitope_match_log=log,
        seed=7,
        sample_idx=3,
    )

    assert info is not None
    assert info["matched_native"] == "1abc_HLA"
    assert len(log) == 1
    assert log[0]["seed"] == 7
    assert log[0]["sample"] == 3


def test_process_sample_boltz_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Boltz sample processing should build one row per interface with default epitope fields."""
    model_path = tmp_path / "pred.pdb"
    native_path = tmp_path / "native.pdb"
    model_path.touch()
    native_path.touch()

    monkeypatch.setattr(
        collection,
        "load_structure_pair",
        lambda *args, **kwargs: (object(), object(), {"H": "H", "T": "T"}),
    )
    monkeypatch.setattr(collection, "chain_ids_in_order", lambda *_args, **_kwargs: ["A", "B"])
    monkeypatch.setattr(collection, "infer_chain_roles", lambda *_args, **_kwargs: {"A": "binder", "B": "antigen"})
    monkeypatch.setattr(collection, "_get_chain_mapping_boltz", lambda *args, **kwargs: {"A": "H", "B": "T"})
    monkeypatch.setattr(collection, "_perform_epitope_matching", lambda *args, **kwargs: None)
    monkeypatch.setattr(collection, "run_dockq", lambda *args, **kwargs: ({"AB": {"DockQ": 0.5}}, 0.5))
    monkeypatch.setattr(collection, "_infer_intended_antigen_chain", lambda *_args, **_kwargs: "B")
    monkeypatch.setattr(collection, "_resolve_intended_native_antigen_chain", lambda *args, **kwargs: "B")
    monkeypatch.setattr(collection, "run_complex_level_dockq", lambda *args, **kwargs: {"complex_dockq": 0.5})
    monkeypatch.setattr(collection, "compute_structure_metrics", lambda *args, **kwargs: ({"rmsd": 1.2}, None))
    monkeypatch.setattr(collection.boltz_eval, "find_confidence_file", lambda *args, **kwargs: tmp_path / "conf.json")
    monkeypatch.setattr(collection.boltz_eval, "load_confidence", lambda *_args, **_kwargs: {"ptm": 0.7})
    monkeypatch.setattr(collection, "validate_interface_coverage", lambda *args, **kwargs: None)
    monkeypatch.setattr(collection, "interface_role_pair", lambda *args, **kwargs: "binder-antigen")
    monkeypatch.setattr(collection, "make_dockq_row", lambda *args, **kwargs: {"entry": args[0], "iface": args[5]})

    rows = collection._process_sample_boltz(
        entry="1abc_HLB",
        seed=1,
        sample_idx=2,
        model_path=model_path,
        native_path=native_path,
        native_dir=tmp_path,
        input_dir=None,
        role_counts_lookup=None,
        json_dir=None,
        use_sequence_matching=True,
        multiplicity_data=None,
        epitope_match_log=None,
    )

    assert len(rows) == 1
    assert rows[0]["iface"] == "AB"
    assert rows[0]["matched_native"] == ""


def test_process_sample_cif_success_with_epitope_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CIF sample processing should merge epitope info into output rows."""
    cif_path = tmp_path / "pred.cif"
    native_path = tmp_path / "native.pdb"
    cif_path.touch()
    native_path.touch()
    model_eval_module = SimpleNamespace(
        chain_map_from_stem=lambda *args, **kwargs: {"A": "H", "B": "T", "C": "U"},
        find_confidence_file=lambda *args, **kwargs: tmp_path / "conf.json",
        load_confidence=lambda *_args, **_kwargs: {"ptm": 0.8},
    )

    match_info = {
        "matched_native": "1abc_HLC",
        "epitope_distance": 2.5,
        "match_quality": "good_match",
        "num_candidates": 2,
        "matched_to_intended_target": False,
    }

    monkeypatch.setattr(
        collection,
        "load_structure_pair",
        lambda *args, **kwargs: (object(), object(), {"H": "H", "T": "T", "U": "U"}),
    )
    monkeypatch.setattr(collection, "chain_ids_in_order", lambda *_args, **_kwargs: ["A", "B", "C"])
    monkeypatch.setattr(
        collection,
        "infer_chain_roles",
        lambda *_args, **_kwargs: {"A": "binder", "B": "antigen", "C": "antigen"},
    )
    monkeypatch.setattr(collection, "_perform_epitope_matching_cif", lambda *args, **kwargs: dict(match_info))
    monkeypatch.setattr(
        collection,
        "_apply_epitope_matching_result",
        lambda info: (
            tmp_path / "matched.pdb",
            {"A": "H", "B": "T", "C": "U"},
            {"A": "binder", "B": "antigen", "C": "antigen"},
        ),
    )
    monkeypatch.setattr(
        collection,
        "_select_native_for_scoring",
        lambda *args, **kwargs: (
            tmp_path / "matched.pdb",
            {"A": "H", "B": "T", "C": "U"},
            {"A": "binder", "B": "antigen", "C": "antigen"},
            dict(match_info),
        ),
    )
    monkeypatch.setattr(collection, "run_dockq", lambda *args, **kwargs: ({"AB": {"DockQ": 0.5}}, 0.5))
    monkeypatch.setattr(collection, "_infer_intended_antigen_chain", lambda *_args, **_kwargs: "C")
    monkeypatch.setattr(collection, "_resolve_intended_native_antigen_chain", lambda *args, **kwargs: "C")
    monkeypatch.setattr(collection, "run_complex_level_dockq", lambda *args, **kwargs: {"complex_dockq": 0.9})
    monkeypatch.setattr(collection, "compute_structure_metrics", lambda *args, **kwargs: ({"rmsd": 1.0}, None))
    monkeypatch.setattr(collection, "validate_interface_coverage", lambda *args, **kwargs: None)
    monkeypatch.setattr(collection, "interface_role_pair", lambda *args, **kwargs: "binder-antigen")
    monkeypatch.setattr(collection, "make_dockq_row", lambda *args, **kwargs: {"entry": args[0], "iface": args[5]})

    rows = collection._process_sample_cif(
        entry="1abc_HLC",
        seed=0,
        sample_idx=3,
        cif_path=cif_path,
        native_path=native_path,
        native_dir=tmp_path,
        role_counts_lookup=None,
        json_dir=None,
        use_sequence_matching=False,
        multiplicity_data=None,
        epitope_match_log=None,
        model_eval_module=model_eval_module,
        confidence_cols=["ptm"],
    )

    assert len(rows) == 1
    assert rows[0]["iface"] == "AB"
    assert rows[0]["matched_native"] == "1abc_HLC"
