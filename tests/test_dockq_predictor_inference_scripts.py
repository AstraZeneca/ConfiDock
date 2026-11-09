"""Tests for the label-free inference entrypoint and the CDR sidecar builder.

These two scripts are what "use the trained model" means in practice, and both fail in
ways that produce a plausible number rather than an error: a pose grouped into the wrong
complex still gets a rank, and a missing CDR sidecar still gets a score. So the tests
here concentrate on the identity of things -- which complex a file belongs to, which JSON
name its metadata is written under, which feature blocks end up live -- rather than on
the numbers themselves.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from confidock.dockq_predictor.data import chain_mapping_cif as cmc

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "dockq_predictor" / "dockq_gnn.yaml"
sys.path.insert(0, str(REPO_ROOT / "workflows" / "scripts" / "inference"))

import build_cdr_sidecar_json as sidecar_script  # noqa: E402
import predict_dockq as predict_script  # noqa: E402
from test_dockq_predictor_pipeline import _smoke_pdb  # noqa: E402

# ---------------------------------------------------------------------------
# The lookup key shared by the loader and the sidecar writer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("1abc_HLA.cif", "1abc_HLA"),
        ("1abc_HLA_sample_0.cif", "1abc_HLA"),
        ("1abc_HLA_sample_12.pdb", "1abc_HLA"),
        ("7jtf_ABE_seed42_sample0.pdb", "7jtf_ABE"),
        ("mypose.pdb", "mypose"),
    ],
)
def test_sabdab_json_key_strips_the_pose_suffixes(filename: str, expected: str) -> None:
    """Every pose of a complex must reduce to the complex's single JSON name."""
    assert cmc.sabdab_json_key(f"/some/dir/{filename}") == expected


def test_sidecar_writer_names_files_the_way_the_loader_looks_them_up() -> None:
    """The builder must not carry its own copy of the naming rule.

    A private copy would let writer and reader drift, and the symptom would be silent:
    no JSON found, CDR labels all zero, no error anywhere.
    """
    assert sidecar_script.sabdab_json_key is cmc.sabdab_json_key


def test_find_sabdab_json_prefers_the_override_directory(tmp_path: Path, monkeypatch) -> None:
    """A sidecar directory adds coverage; it is searched before the shared snapshot."""
    override, fallback = tmp_path / "side", tmp_path / "shared"
    override.mkdir()
    fallback.mkdir()
    (override / "1abc_HLA.json").write_text(json.dumps({"h_seq": "AA"}))
    (fallback / "1abc_HLA.json").write_text(json.dumps({"h_seq": "BB"}))
    monkeypatch.setattr(cmc, "_DEFAULT_SABDAB_JSON_DIR", str(fallback))
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(override))

    assert cmc.find_sabdab_json("/x/1abc_HLA_sample_0.cif") == override / "1abc_HLA.json"
    assert cmc.load_sabdab_json("/x/1abc_HLA_sample_0.cif") == {"h_seq": "AA"}


def test_find_sabdab_json_falls_through_to_the_shared_snapshot(tmp_path: Path, monkeypatch) -> None:
    """An override that lacks a complex must not shadow the snapshot's copy of it."""
    override, fallback = tmp_path / "side", tmp_path / "shared"
    override.mkdir()
    fallback.mkdir()
    (fallback / "1abc_HLA.json").write_text(json.dumps({"h_seq": "BB"}))
    monkeypatch.setattr(cmc, "_DEFAULT_SABDAB_JSON_DIR", str(fallback))
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(override))

    assert cmc.find_sabdab_json("/x/1abc_HLA.cif") == fallback / "1abc_HLA.json"


def test_find_sabdab_json_returns_none_when_no_directory_has_it(tmp_path: Path, monkeypatch) -> None:
    """A missing sidecar is a normal outcome that callers must be able to detect."""
    monkeypatch.setattr(cmc, "_DEFAULT_SABDAB_JSON_DIR", str(tmp_path))
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    assert cmc.find_sabdab_json("/x/9zzz_HLA.cif") is None


# ---------------------------------------------------------------------------
# Structure discovery, shared in spirit by both scripts
# ---------------------------------------------------------------------------


@pytest.fixture
def poses(tmp_path: Path) -> Path:
    """Write a directory of three poses of one complex, plus a non-structure file."""
    directory = tmp_path / "poses"
    directory.mkdir()
    for i in range(3):
        _smoke_pdb(directory / f"cplx1_HA_sample_{i}.pdb")
    (directory / "notes.txt").write_text("not a structure")
    return directory


@pytest.mark.parametrize("iter_files", [predict_script.iter_structure_files, sidecar_script.iter_structure_files])
def test_iter_structure_files_accepts_globs_dirs_and_paths(poses: Path, iter_files) -> None:
    """All three input shapes resolve to the same three structures, never the .txt."""
    expected = sorted(poses.glob("*.pdb"))
    assert iter_files([str(poses)]) == expected
    assert iter_files([f"{poses}/*.pdb"]) == expected
    assert iter_files([str(p) for p in expected]) == expected


@pytest.mark.parametrize("iter_files", [predict_script.iter_structure_files, sidecar_script.iter_structure_files])
def test_iter_structure_files_deduplicates_overlapping_patterns(poses: Path, iter_files) -> None:
    """Overlapping globs are easy to pass and must not score a pose twice."""
    assert iter_files([str(poses), f"{poses}/*.pdb"]) == sorted(poses.glob("*.pdb"))


@pytest.mark.parametrize("iter_files", [predict_script.iter_structure_files, sidecar_script.iter_structure_files])
def test_iter_structure_files_walks_subdirectories(tmp_path: Path, iter_files) -> None:
    """Prediction trees put one subdirectory per complex; a bare dir must find them."""
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        _smoke_pdb(tmp_path / name / f"{name}_HA_sample_0.pdb")
    assert len(iter_files([str(tmp_path)])) == 2


# ---------------------------------------------------------------------------
# predict_dockq: building the input table
# ---------------------------------------------------------------------------


def test_rows_from_structures_applies_one_source_and_chain_string(poses: Path) -> None:
    """With --structures the two annotations are global, since a glob carries neither."""
    df = predict_script.rows_from_structures([f"{poses}/*.pdb"], "boltz2", "HA")
    assert len(df) == 3
    assert set(df.model_source) == {"boltz2"} and set(df.chains) == {"HA"}


def test_rows_from_structures_rejects_a_pattern_that_matches_nothing(tmp_path: Path) -> None:
    """Silently scoring zero structures reads as success; fail instead."""
    with pytest.raises(SystemExit, match="No .cif/.pdb files matched"):
        predict_script.rows_from_structures([f"{tmp_path}/*.pdb"], "boltz2", "HA")


def test_rows_from_manifest_needs_no_dockq_column(tmp_path: Path) -> None:
    """The whole point of this entrypoint: no ground truth is required."""
    manifest = tmp_path / "m.csv"
    pd.DataFrame({"pdb_path": ["/a/x_HA_sample_0.pdb"], "model_source": ["boltz2"], "chains": ["HA"]}).to_csv(
        manifest, index=False
    )
    df = predict_script.rows_from_manifest(manifest, None, None)
    assert "dockq" not in df.columns and len(df) == 1


def test_rows_from_manifest_carries_extra_columns_through(tmp_path: Path) -> None:
    """Whatever the caller wants to group or filter by later must survive the round trip."""
    manifest = tmp_path / "m.csv"
    pd.DataFrame(
        {"pdb_path": ["/a/x_HA_sample_0.pdb"], "model_source": ["boltz2"], "note": ["run7"], "dockq": [0.4]}
    ).to_csv(manifest, index=False)
    df = predict_script.rows_from_manifest(manifest, None, None)
    assert df.loc[0, "note"] == "run7"
    # A labelled manifest is accepted; the label is carried, never read.
    assert df.loc[0, "dockq"] == pytest.approx(0.4)


def test_rows_from_manifest_lets_the_cli_override_the_columns(tmp_path: Path) -> None:
    """--model-source / --chains win over the file, so one manifest can be re-scored."""
    manifest = tmp_path / "m.csv"
    pd.DataFrame({"pdb_path": ["/a/x.pdb"], "model_source": ["boltz2"], "chains": ["HA"]}).to_csv(manifest, index=False)
    df = predict_script.rows_from_manifest(manifest, "protenix", "HLA")
    assert df.loc[0, "model_source"] == "protenix" and df.loc[0, "chains"] == "HLA"


def test_rows_from_manifest_requires_pdb_path(tmp_path: Path) -> None:
    """Without paths there is nothing to score, and the column list helps diagnose it."""
    manifest = tmp_path / "m.csv"
    pd.DataFrame({"structure": ["/a/x.pdb"]}).to_csv(manifest, index=False)
    with pytest.raises(SystemExit, match="no 'pdb_path' column"):
        predict_script.rows_from_manifest(manifest, None, None)


def test_rows_from_manifest_requires_a_model_source_from_somewhere(tmp_path: Path) -> None:
    """model_source selects a one-hot edge feature, so it has no safe default."""
    manifest = tmp_path / "m.csv"
    pd.DataFrame({"pdb_path": ["/a/x.pdb"]}).to_csv(manifest, index=False)
    with pytest.raises(SystemExit, match="no 'model_source' column"):
        predict_script.rows_from_manifest(manifest, None, None)


def test_parse_args_requires_model_source_with_structures(tmp_path: Path) -> None:
    """Same rule at the CLI boundary, where there is no column to fall back on."""
    with pytest.raises(SystemExit):
        predict_script._parse_args(
            ["--checkpoint", "c.ckpt", "--config", "c.yaml", "--structures", f"{tmp_path}/*.pdb"]
        )


def test_parse_args_rejects_structures_and_manifest_together(tmp_path: Path) -> None:
    """Two sources of truth for the input set would silently favour one."""
    with pytest.raises(SystemExit):
        predict_script._parse_args(
            [
                "--checkpoint",
                "c.ckpt",
                "--config",
                "c.yaml",
                "--structures",
                f"{tmp_path}/*.pdb",
                "--manifest",
                "m.csv",
                "--model-source",
                "boltz2",
            ]
        )


# ---------------------------------------------------------------------------
# predict_dockq: warnings for the failure modes that produce a plausible number
# ---------------------------------------------------------------------------


def _no_sidecars(monkeypatch, tmp_path: Path) -> None:
    """Point both JSON search directories at an empty directory."""
    monkeypatch.setattr(cmc, "_DEFAULT_SABDAB_JSON_DIR", str(tmp_path))
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))


def test_warns_when_the_model_source_is_outside_the_trained_one_hot(tmp_path, monkeypatch, capsys) -> None:
    """An unknown source lands on all-zeros, a vector the model never saw."""
    _no_sidecars(monkeypatch, tmp_path)
    df = pd.DataFrame({"pdb_path": ["/a/x_HA_sample_0.pdb"], "model_source": ["alphafold3"], "chains": ["HA"]})
    predict_script.warn_about_features(df)
    assert "alphafold3" in capsys.readouterr().err


def test_does_not_warn_about_a_known_model_source(tmp_path, monkeypatch, capsys) -> None:
    """The three trained sources must not produce noise on every run."""
    _no_sidecars(monkeypatch, tmp_path)
    df = pd.DataFrame({"pdb_path": ["/a/x_HA_sample_0.pdb"], "model_source": ["boltz2"], "chains": ["HA"]})
    predict_script.warn_about_features(df)
    assert "is not one of" not in capsys.readouterr().err


def test_warns_when_no_chain_string_is_given(tmp_path, monkeypatch, capsys) -> None:
    """A mistyped antibody chain wrecks the features without raising."""
    _no_sidecars(monkeypatch, tmp_path)
    df = pd.DataFrame({"pdb_path": ["/a/x_HA_sample_0.pdb"], "model_source": ["boltz2"], "chains": [pd.NA]})
    predict_script.warn_about_features(df)
    assert "no 'chains' given" in capsys.readouterr().err


def test_warns_once_per_complex_about_missing_cdr_metadata(tmp_path, monkeypatch, capsys) -> None:
    """Report complexes, not poses: 40 poses of one complex are one problem."""
    _no_sidecars(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "pdb_path": [f"/a/cplx1_HA_sample_{i}.pdb" for i in range(4)] + ["/a/cplx2_HA_sample_0.pdb"],
            "model_source": "boltz2",
            "chains": "HA",
        }
    )
    predict_script.warn_about_features(df)
    err = capsys.readouterr().err
    assert "2 complex(es)" in err and "cplx1_HA" in err and "cplx2_HA" in err


def test_no_cdr_warning_when_every_complex_has_metadata(tmp_path, monkeypatch, capsys) -> None:
    """The warning has to be silent in the good case or it will be ignored in the bad one."""
    _no_sidecars(monkeypatch, tmp_path)
    (tmp_path / "cplx1_HA.json").write_text(json.dumps({"h_seq": "GG"}))
    df = pd.DataFrame({"pdb_path": ["/a/cplx1_HA_sample_0.pdb"], "model_source": ["boltz2"], "chains": ["HA"]})
    predict_script.warn_about_features(df)
    assert "no CDR metadata" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# predict_dockq: --cdr-json-dir, the visible half of an invisible dependency
# ---------------------------------------------------------------------------


def test_cdr_json_dir_reaches_the_featurisers_lookup(tmp_path: Path, monkeypatch) -> None:
    """The flag has to land where find_sabdab_json reads, which is the environment."""
    _no_sidecars(monkeypatch, tmp_path / "empty")
    sidecars = tmp_path / "cdr_json"
    sidecars.mkdir()
    (sidecars / "cplx1_HA.json").write_text(json.dumps({"h_seq": "GG"}))

    assert cmc.find_sabdab_json("/a/cplx1_HA_sample_0.pdb") is None
    predict_script.apply_cdr_json_dirs([sidecars])
    assert cmc.find_sabdab_json("/a/cplx1_HA_sample_0.pdb") == sidecars / "cplx1_HA.json"


def test_cdr_json_dir_prepends_rather_than_replacing_the_existing_export(tmp_path: Path, monkeypatch) -> None:
    """A sidecar dir for a few complexes must not hide an export covering the rest."""
    exported, flagged = tmp_path / "exported", tmp_path / "flagged"
    exported.mkdir()
    flagged.mkdir()
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(exported))
    predict_script.apply_cdr_json_dirs([flagged])

    dirs = cmc.sabdab_json_search_dirs()
    assert dirs.index(str(flagged.resolve())) < dirs.index(str(exported))


def test_cdr_json_dir_accepts_several_directories_in_order(tmp_path: Path, monkeypatch) -> None:
    """Multiple sidecar sources are the norm once more than one batch has been built."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.delenv("CONFIDOCK_SABDAB_JSON_DIR", raising=False)
    predict_script.apply_cdr_json_dirs([first, second])
    assert cmc.sabdab_json_search_dirs()[:2] == [str(first.resolve()), str(second.resolve())]


def test_cdr_json_dir_rejects_a_directory_that_does_not_exist(tmp_path: Path) -> None:
    """A typo'd path would look exactly like having built no sidecars at all."""
    with pytest.raises(SystemExit, match="not a directory"):
        predict_script.apply_cdr_json_dirs([tmp_path / "nope"])


def test_apply_cdr_json_dirs_is_a_no_op_when_the_flag_is_absent(tmp_path: Path, monkeypatch) -> None:
    """Omitting the flag must leave an existing export exactly as it was."""
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    predict_script.apply_cdr_json_dirs(None)
    assert os.environ["CONFIDOCK_SABDAB_JSON_DIR"] == str(tmp_path)


def test_search_path_is_printed_with_what_was_found_in_each_directory(tmp_path, monkeypatch, capsys) -> None:
    """The run states where it looked, since the command line does not say."""
    sidecars = tmp_path / "cdr_json"
    sidecars.mkdir()
    (sidecars / "cplx1_HA.json").write_text("{}")
    monkeypatch.setattr(cmc, "_DEFAULT_SABDAB_JSON_DIR", str(tmp_path / "gone"))
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(sidecars))

    predict_script.print_cdr_search_path()
    out = capsys.readouterr().out
    assert f"{sidecars}  (1 json)" in out
    assert "(does not exist)" in out  # the absent snapshot is named, not hidden


def test_main_takes_cdr_metadata_from_the_flag_with_no_export(tmp_path: Path, poses: Path, monkeypatch, capsys) -> None:
    """End to end: sidecars built, flag passed, no environment variable set by hand."""
    _no_sidecars(monkeypatch, tmp_path / "empty")
    monkeypatch.delenv("CONFIDOCK_SABDAB_JSON_DIR")
    _stub_anarci(monkeypatch)
    sidecars = tmp_path / "cdr_json"
    sidecar_script.main(["--structures", f"{poses}/*.pdb", "--out-dir", str(sidecars)])

    ckpt, config_path = _tiny_checkpoint(tmp_path)
    predict_script.main(
        [
            "--checkpoint",
            str(ckpt),
            "--config",
            str(config_path),
            "--structures",
            f"{poses}/*.pdb",
            "--model-source",
            "boltz2",
            "--chains",
            "HA",
            "--cdr-json-dir",
            str(sidecars),
            "--output",
            str(tmp_path / "scores.csv"),
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    # The metadata was found, so the run must not have warned about missing CDRs...
    assert "no CDR metadata" not in captured.err
    # ...and it must say where it looked, since the dependency is otherwise invisible.
    assert str(sidecars) in captured.out
    assert cmc.find_sabdab_json(str(poses / "cplx1_HA_sample_0.pdb")) == sidecars / "cplx1_HA.json"
    assert len(pd.read_csv(tmp_path / "scores.csv")) == 3


# ---------------------------------------------------------------------------
# predict_dockq: ranking, which is what the output is actually for
# ---------------------------------------------------------------------------


def test_rank_within_complex_groups_by_the_models_own_key() -> None:
    """Poses are ranked against their own complex, never across complexes."""
    scored = pd.DataFrame(
        {
            "pdb_path": [
                "/a/cplx1_HA_sample_0.pdb",
                "/a/cplx1_HA_sample_1.pdb",
                "/a/cplx2_HA_sample_0.pdb",
            ],
            "dockq_pred": [0.2, 0.8, 0.1],
        }
    )
    ranked = predict_script.rank_within_complex(scored)
    assert list(ranked["complex"]) == ["cplx1_HA", "cplx1_HA", "cplx2_HA"]
    # cplx2's single pose is rank 1 despite the lowest score in the table.
    best = dict(zip(ranked["complex"], ranked["rank"] == 1))
    assert best["cplx2_HA"]
    top = ranked[ranked["rank"] == 1].set_index("complex")["dockq_pred"].to_dict()
    assert top == {"cplx1_HA": 0.8, "cplx2_HA": 0.1}


def test_rank_within_complex_breaks_ties_without_producing_two_rank_ones() -> None:
    """Tied predictions are common on near-identical poses; a rank must still be a rank."""
    scored = pd.DataFrame({"pdb_path": [f"/a/c_HA_sample_{i}.pdb" for i in range(3)], "dockq_pred": [0.5, 0.5, 0.5]})
    ranked = predict_script.rank_within_complex(scored)
    assert sorted(ranked["rank"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# predict_dockq: the whole entrypoint on synthetic structures
# ---------------------------------------------------------------------------


def _tiny_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """Write a real (untrained) SimpleGNN checkpoint plus the config that builds it."""
    from confidock.dockq_predictor.models import SimpleGNN

    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["data"].update(knn_k=4, batch_size=2, num_workers=0, graph_cache_dir=None, cache_graphs=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    ckpt = tmp_path / "model.ckpt"
    torch.save({"state_dict": SimpleGNN(config).state_dict(), "epoch": 0, "global_step": 0}, ckpt)
    return ckpt, config_path


def test_main_scores_and_ranks_structures_without_any_labels(tmp_path: Path, poses: Path, monkeypatch) -> None:
    """The end-to-end contract: structures in, a ranked CSV out, no dockq anywhere.

    Catches what the unit tests above cannot -- a config key the dataset no longer takes,
    a feature width the checkpoint disagrees with, or a graph that never reaches the model.
    """
    _no_sidecars(monkeypatch, tmp_path)
    ckpt, config_path = _tiny_checkpoint(tmp_path)
    out = tmp_path / "scores.csv"

    predict_script.main(
        [
            "--checkpoint",
            str(ckpt),
            "--config",
            str(config_path),
            "--structures",
            f"{poses}/*.pdb",
            "--model-source",
            "boltz2",
            "--chains",
            "HA",
            "--output",
            str(out),
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]
    )

    df = pd.read_csv(out)
    assert len(df) == 3
    assert "dockq" not in df.columns
    assert set(df["complex"]) == {"cplx1_HA"}
    assert sorted(df["rank"]) == [1, 2, 3]
    # Ranks must actually follow the scores, best first.
    ordered = df.sort_values("rank")["dockq_pred"].tolist()
    assert ordered == sorted(ordered, reverse=True)


def test_main_accepts_a_manifest_with_no_dockq_column(tmp_path: Path, poses: Path, monkeypatch) -> None:
    """The manifest route must not inherit the evaluation script's label requirement."""
    _no_sidecars(monkeypatch, tmp_path)
    ckpt, config_path = _tiny_checkpoint(tmp_path)
    manifest = tmp_path / "m.csv"
    pd.DataFrame(
        {"pdb_path": [str(p) for p in sorted(poses.glob("*.pdb"))], "model_source": "boltz2", "chains": "HA"}
    ).to_csv(manifest, index=False)
    out = tmp_path / "scores.csv"

    predict_script.main(
        [
            "--checkpoint",
            str(ckpt),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest),
            "--output",
            str(out),
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert len(pd.read_csv(out)) == 3


def test_main_reports_rather_than_dies_on_unfeaturisable_structures(
    tmp_path: Path, poses: Path, monkeypatch, capsys
) -> None:
    """One corrupt pose in a batch of 40 must not cost the other 39."""
    _no_sidecars(monkeypatch, tmp_path)
    ckpt, config_path = _tiny_checkpoint(tmp_path)
    (poses / "cplx1_HA_sample_9.pdb").write_text("this is not a PDB file\n")
    out = tmp_path / "scores.csv"

    predict_script.main(
        [
            "--checkpoint",
            str(ckpt),
            "--config",
            str(config_path),
            "--structures",
            f"{poses}/*.pdb",
            "--model-source",
            "boltz2",
            "--chains",
            "HA",
            "--output",
            str(out),
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert "failed to featurise" in capsys.readouterr().out
    assert len(pd.read_csv(out)) == 3


# ---------------------------------------------------------------------------
# build_cdr_sidecar_json: the CDR maths, with ANARCI stubbed out
# ---------------------------------------------------------------------------


def test_cdr_positions_uses_chothia_ranges_and_1_based_sequence_positions() -> None:
    """The loader indexes h_cdr/l_cdr into h_seq, so positions are 1-based, not numbers."""
    # Sequence index 0 carries Chothia number 26 (inside H1 26-32) -> position 1.
    pos_map = {0: 26, 1: 27, 2: 40, 3: 52, 4: 95}
    assert sidecar_script.cdr_positions(pos_map, "H") == {"1": [1, 2], "2": [4], "3": [5]}


def test_cdr_positions_omits_cdrs_with_no_residues() -> None:
    """An absent CDR must be absent, not an empty list the loader would iterate."""
    assert sidecar_script.cdr_positions({0: 26}, "H") == {"1": [1]}


# ANARCI's real output shape, trimmed: the header block, the HMM hit line, then numbered
# rows -- one of them a gap ('-'), which consumes a Chothia number but no sequence
# position. Everything downstream of ANARCI is decided by parsing this correctly.
_ANARCI_STDOUT = """\
# q
# ANARCI numbered
# Domain 1 of 1
# Most significant HMM hit
#|species|chain_type|e-value|score|seqstart_index|seqend_index|
#|human|H|6.5e-56|179.1|0|118|
# Scheme = chothia
H 1       -
H 2       V
H 3       Q
H 26      G
H 52      A
//
"""


def test_hit_line_gives_the_chain_class_and_the_sequence_offset() -> None:
    """The '#|species|...' header must not be mistaken for the hit line below it."""
    assert sidecar_script._hit_chain_class(_ANARCI_STDOUT) == ("H", 0)


@pytest.mark.parametrize(
    ("chain_type", "expected"),
    [("H", "H"), ("L", "L"), ("K", "L"), ("A", None)],  # kappa is a light chain; A is neither
)
def test_hit_line_collapses_anarci_domain_types(chain_type: str, expected) -> None:
    """CHOTHIA_CDR is keyed H/L only, so K must fold into L and anything else drop out."""
    stdout = _ANARCI_STDOUT.replace("|human|H|", f"|human|{chain_type}|")
    assert sidecar_script._hit_chain_class(stdout)[0] == expected


def test_hit_line_offset_shifts_the_numbering_onto_the_full_sequence() -> None:
    """ANARCI numbers the domain it found, which need not start at residue 1."""
    stdout = _ANARCI_STDOUT.replace("|6.5e-56|179.1|0|118|", "|6.5e-56|179.1|4|118|")
    chain_class, seqstart = sidecar_script._hit_chain_class(stdout)
    assert (chain_class, seqstart) == ("H", 4)
    assert min(sidecar_script._numbered_positions(stdout, seqstart)) == 4


def test_numbered_positions_skips_gap_rows() -> None:
    """A gap row consumes a Chothia number but no sequence position -- the crux of the map."""
    # H1 is a gap, so sequence index 0 is H2, not H1.
    assert sidecar_script._numbered_positions(_ANARCI_STDOUT, 0) == {0: 2, 1: 3, 2: 26, 3: 52}


def test_no_hit_line_means_not_an_antibody_chain() -> None:
    """An antigen chain gets numbered output with no domain hit; that is not an error."""
    stdout = "\n".join(ln for ln in _ANARCI_STDOUT.splitlines() if not ln.startswith("#|human"))
    assert sidecar_script._hit_chain_class(stdout) == (None, 0)


def test_anarci_number_returns_nothing_when_anarci_produced_no_output(monkeypatch) -> None:
    """A non-zero exit, empty output or a timeout all mean "no numbering for this chain"."""
    monkeypatch.setattr(sidecar_script, "_run_anarci", lambda seq: None)
    assert sidecar_script.anarci_number("GGG") == (None, {})


def test_anarci_number_parses_a_real_anarci_run(monkeypatch) -> None:
    """The end-to-end parse: stdout in, (class, index -> Chothia number) out."""
    monkeypatch.setattr(sidecar_script, "_run_anarci", lambda seq: _ANARCI_STDOUT)
    assert sidecar_script.anarci_number("VQGA") == ("H", {0: 2, 1: 3, 2: 26, 3: 52})


def test_anarci_cache_numbers_each_distinct_sequence_once(monkeypatch) -> None:
    """ANARCI dominates the runtime, and complexes often share an antibody."""
    calls = []

    def fake(seq):
        calls.append(seq)
        return "H", {0: 26}

    monkeypatch.setattr(sidecar_script, "anarci_number", fake)
    cache = sidecar_script._AnarciCache()
    for seq in ("AAA", "AAA", "BBB", "AAA"):
        cache.number(seq)
    assert calls == ["AAA", "BBB"] and cache.calls == 2


def test_group_by_key_collapses_every_pose_of_a_complex(tmp_path: Path) -> None:
    """One JSON per complex, not per pose -- poses share a sequence."""
    files = [Path(f"/a/cplx1_HA_sample_{i}.pdb") for i in range(3)] + [Path("/a/cplx2_HA_seed1_sample0.pdb")]
    groups = sidecar_script.group_by_key(files)
    assert list(groups) == ["cplx1_HA", "cplx2_HA"]
    assert len(groups["cplx1_HA"]) == 3


def _stub_anarci(monkeypatch, heavy: str = "GGGGGG") -> None:
    """Type sequence ``heavy`` as an antibody heavy chain and everything else as antigen."""

    def fake(seq):
        return ("H", {i: 26 + i for i in range(len(seq))}) if seq == heavy else (None, {})

    monkeypatch.setattr(sidecar_script, "anarci_number", fake)


def _antibody_antigen_pdb(path: Path) -> Path:
    """Write the smoke structure with its antigen chain mutated to poly-alanine.

    Numbering is memoised by sequence, so the two chains have to differ for one to come
    back heavy and the other antigen -- which is the case the payload schema is about.
    """
    _smoke_pdb(path)
    path.write_text(path.read_text().replace("GLY A", "ALA A"))
    return path


def test_first_model_sequences_takes_one_model_and_skips_residue_free_chains(tmp_path: Path) -> None:
    """Every sequence here becomes an ANARCI call, so both filters are load-bearing.

    A second model is the same chains again -- numbering them would double the run's cost
    and add a duplicate heavy chain that ``build_payload`` would report as a second Fv. A
    solvent/ligand chain has no standard residue, so its sequence is empty and ANARCI has
    nothing to number.
    """
    path = tmp_path / "c_HA_sample_0.pdb"
    atoms = [ln for ln in _smoke_pdb(path).read_text().splitlines() if ln.startswith("ATOM")]
    water = "HETATM 9001  O   HOH W   1      99.000  99.000  99.000  1.00  0.00           O"
    path.write_text("\n".join(["MODEL        1", *atoms, water, "ENDMDL", "MODEL        2", *atoms, "ENDMDL", "END"]))

    structure = sidecar_script.parse_structure(path)
    assert len(structure) == 2  # the file really does hold two models
    assert sidecar_script.first_model_sequences(structure) == ["GGGGGG", "GGGGGG"]


def test_build_payload_returns_none_without_an_antibody_heavy_chain(tmp_path: Path, monkeypatch) -> None:
    """An antigen-only structure has no CDRs; writing a JSON would assert otherwise."""
    monkeypatch.setattr(sidecar_script, "anarci_number", lambda seq: (None, {}))
    structure = sidecar_script.parse_structure(_smoke_pdb(tmp_path / "c_HA_sample_0.pdb"))
    assert sidecar_script.build_payload(structure, sidecar_script._AnarciCache()) is None


def test_build_payload_writes_the_schema_the_loader_reads(tmp_path: Path, monkeypatch) -> None:
    """h_seq + 1-based h_cdr positions, and the antigen kept as t_seq."""
    _stub_anarci(monkeypatch)
    structure = sidecar_script.parse_structure(_antibody_antigen_pdb(tmp_path / "c_HA_sample_0.pdb"))
    payload = sidecar_script.build_payload(structure, sidecar_script._AnarciCache())
    assert payload["h_seq"] == "GGGGGG"
    assert payload["t_seq"] == "AAAAAA"  # the antigen chain, kept but not CDR-labelled
    assert payload["h_cdr"]["1"] == [1, 2, 3, 4, 5, 6]  # Chothia 26-31, all inside H1
    assert max(max(v) for v in payload["h_cdr"].values()) <= len(payload["h_seq"])
    assert "extra_antibody_chains" not in payload


def test_build_payload_flags_a_second_fv_it_cannot_express(tmp_path: Path, monkeypatch) -> None:
    """The schema holds one heavy chain, so a bispecific's second Fv goes unlabelled.

    Recorded in the payload rather than dropped silently, so the run can say so.
    """
    _stub_anarci(monkeypatch)  # both chains of the smoke structure are "GGGGGG" -> both heavy
    structure = sidecar_script.parse_structure(_smoke_pdb(tmp_path / "c_HH_sample_0.pdb"))
    payload = sidecar_script.build_payload(structure, sidecar_script._AnarciCache())
    assert payload["extra_antibody_chains"] == 1
    assert payload["h_seq"] == "GGGGGG" and "t_seq" not in payload


def test_main_writes_one_json_per_complex_named_for_the_loader(tmp_path: Path, poses: Path, monkeypatch) -> None:
    """The output must land exactly where find_sabdab_json will look for it."""
    _stub_anarci(monkeypatch)
    out_dir = tmp_path / "cdr_json"
    sidecar_script.main(["--structures", f"{poses}/*.pdb", "--out-dir", str(out_dir)])

    assert [p.name for p in out_dir.glob("*.json")] == ["cplx1_HA.json"]
    monkeypatch.setattr(cmc, "_DEFAULT_SABDAB_JSON_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(out_dir))
    assert cmc.load_sabdab_json(str(poses / "cplx1_HA_sample_0.pdb"))["h_seq"] == "GGGGGG"


def test_main_leaves_existing_jsons_alone_unless_overwrite(tmp_path: Path, poses: Path, monkeypatch) -> None:
    """Re-running over a partly-built directory must not redo the expensive part."""
    _stub_anarci(monkeypatch)
    out_dir = tmp_path / "cdr_json"
    out_dir.mkdir()
    (out_dir / "cplx1_HA.json").write_text(json.dumps({"h_seq": "PRESERVED"}))

    sidecar_script.main(["--structures", f"{poses}/*.pdb", "--out-dir", str(out_dir)])
    assert json.loads((out_dir / "cplx1_HA.json").read_text())["h_seq"] == "PRESERVED"

    sidecar_script.main(["--structures", f"{poses}/*.pdb", "--out-dir", str(out_dir), "--overwrite"])
    assert json.loads((out_dir / "cplx1_HA.json").read_text())["h_seq"] == "GGGGGG"


def test_main_tallies_each_complex_under_the_right_outcome(tmp_path: Path, monkeypatch, capsys) -> None:
    """The summary is the only report of what a long run did or did not label.

    Four complexes, one per outcome: written, written-but-multi-Fv, already-there, and
    skipped for having no heavy chain. An under-count here is the failure that matters --
    it reads as "all labelled" while some complex silently goes into a scoring run with
    zero CDR features. The multi-Fv one is also a write, so it counts twice over.
    """
    _stub_anarci(monkeypatch)
    structures, out_dir = tmp_path / "poses", tmp_path / "cdr_json"
    structures.mkdir()
    out_dir.mkdir()

    _antibody_antigen_pdb(structures / "written_HA_sample_0.pdb")
    _smoke_pdb(structures / "bispecific_HH_sample_0.pdb")  # both chains GGGGGG -> two heavy chains
    # No GLY at all -> both chains come back antigen -> nothing to label.
    antigen_only = _smoke_pdb(structures / "nofv_AB_sample_0.pdb")
    antigen_only.write_text(antigen_only.read_text().replace("GLY", "ALA"))
    _antibody_antigen_pdb(structures / "existing_HA_sample_0.pdb")
    (out_dir / "existing_HA.json").write_text(json.dumps({"h_seq": "PRESERVED"}))

    sidecar_script.main(["--structures", f"{structures}/*.pdb", "--out-dir", str(out_dir)])

    out = capsys.readouterr().out
    assert "Wrote 2 JSON(s)" in out  # the plain one and the multi-Fv one
    assert "1 complex(es) had >1 Fv" in out
    assert "1 already existed" in out
    assert "1 skipped: no immunoglobulin heavy chain" in out
    assert json.loads((out_dir / "existing_HA.json").read_text())["h_seq"] == "PRESERVED"


def test_main_rejects_a_pattern_that_matches_nothing(tmp_path: Path) -> None:
    """Reporting "wrote 0 JSONs" as success would send someone into a silent-CDR run."""
    with pytest.raises(SystemExit, match="No .cif/.pdb files matched"):
        sidecar_script.main(["--structures", f"{tmp_path}/*.cif", "--out-dir", str(tmp_path / "out")])


# ---------------------------------------------------------------------------
# The reason the sidecar exists: the feature blocks it brings back to life
# ---------------------------------------------------------------------------


def test_a_sidecar_revives_the_cdr_feature_blocks(tmp_path: Path, monkeypatch) -> None:
    """Without metadata, CDR-dependent node and edge features are silently all-zero.

    This is the whole justification for the sidecar step, so it is asserted on the real
    featuriser rather than described in a docstring.
    """
    from confidock.dockq_predictor.data.dataset import DockQDataset

    pdb = _smoke_pdb(tmp_path / "cplx1_HA_sample_0.pdb")
    entry = {"pdb_path": str(pdb), "model_source": "boltz2", "dockq": 0.0, "chain_str": "HA"}

    def live_columns() -> tuple[int, int]:
        data = DockQDataset(
            entries=[entry], interface_cutoff=6.0, knn_k=4, cache_graphs=False, graph_cache_dir=None
        ).get(0)
        return int((data.x.abs().sum(0) > 0).sum()), int((data.cdr_label > 0).sum())

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(cmc, "_DEFAULT_SABDAB_JSON_DIR", str(empty))
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(empty))
    cols_without, cdrs_without = live_columns()
    assert cdrs_without == 0

    _stub_anarci(monkeypatch)
    out_dir = tmp_path / "cdr_json"
    sidecar_script.main(["--structures", str(pdb), "--out-dir", str(out_dir)])
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(out_dir))
    cols_with, cdrs_with = live_columns()

    assert cdrs_with > 0, "the sidecar did not reach the featuriser"
    assert cols_with > cols_without, "CDR labels landed but lit up no feature columns"
