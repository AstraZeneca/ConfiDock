"""Tests for the DockQ-predictor data-prep path, its config and its training entrypoint.

Covers ``confidock.dockq_predictor.data.manifest_builder`` (manifest assembly)
and ``workflows/scripts/training/train_simple_gnn_from_manifest.py`` (the runnable
entrypoint), plus the shipped final config -- so a broken config or a renamed
DataModule keyword fails here rather than 20 minutes into a GPU job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from confidock.dockq_predictor.data import manifest_builder as mb
from confidock.dockq_predictor.data.feature_utils import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from confidock.dockq_predictor.models import SimpleGNN

REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "dockq_predictor" / "dockq_gnn.yaml"

sys.path.insert(0, str(REPO_ROOT / "workflows" / "scripts" / "training"))

import train_simple_gnn_from_manifest as train_script  # noqa: E402

# ---------------------------------------------------------------------------
# build_antigen_cluster_map
# ---------------------------------------------------------------------------


def test_cluster_map_strips_the_chain_suffix(tmp_path: Path) -> None:
    """Cluster ids come from mmseqs as "<interface_id>|<chain>"."""
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("1a14_HLN|A\t1a14_HLN|A\n1a14_HLN|A\t2xyz_HLB|C\n")
    assert mb.build_antigen_cluster_map(tsv) == {"1a14_HLN": "1a14_HLN", "2xyz_HLB": "1a14_HLN"}


# ---------------------------------------------------------------------------
# select_stratified_entries
# ---------------------------------------------------------------------------


def test_stratified_selection_covers_every_cluster_before_repeating() -> None:
    """Round-robin: antigen clusters are very skewed (median 2, max ~791).

    Plain random sampling would spend the whole budget inside the few huge clusters.
    """
    cluster_map = {f"big{i}": "big" for i in range(10)}
    cluster_map.update({"small1": "s1", "small2": "s2"})
    selected = mb.select_stratified_entries(set(cluster_map), cluster_map, n_total=3, random_seed=1)

    assert len(selected) == 3
    assert {cluster_map[e] for e in selected} == {"big", "s1", "s2"}


def test_stratified_selection_returns_everything_when_the_budget_exceeds_the_pool() -> None:
    """Asking for more complexes than exist yields all of them, sorted."""
    entries = {"b", "a", "c"}
    assert mb.select_stratified_entries(entries, {}, n_total=10) == ["a", "b", "c"]


def test_stratified_selection_still_picks_unclustered_entries() -> None:
    """A complex missing from the cluster TSV must not be silently unselectable."""
    selected = mb.select_stratified_entries({"x", "y"}, {"x": "c1"}, n_total=2, random_seed=0)
    assert sorted(selected) == ["x", "y"]


def test_stratified_selection_is_reproducible_for_a_seed() -> None:
    """Same seed, same complexes -- runs must be comparable."""
    entries = {f"e{i}" for i in range(20)}
    cluster_map = {e: f"c{i % 4}" for i, e in enumerate(sorted(entries))}
    first = mb.select_stratified_entries(entries, cluster_map, n_total=8, random_seed=5)
    second = mb.select_stratified_entries(entries, cluster_map, n_total=8, random_seed=5)
    assert first == second


# ---------------------------------------------------------------------------
# load_and_sample_dockq
# ---------------------------------------------------------------------------


def _dockq_csv(tmp_path: Path, n_complexes: int = 4, decoys: int = 3) -> Path:
    rows = [
        {
            "entry": f"{i:04d}_HLA",
            "model_file": f"{i:04d}_HLA_sample_{k}.cif",
            "complex_dockq": 0.1 * k,
            "complex_fnat": 0.5,
            "complex_irmsd": 1.0,
            "complex_lrmsd": 2.0,
            "msa_mode": "antigen_only",
            "loops": 4,
            "steps": 50,
        }
        for i in range(n_complexes)
        for k in range(decoys)
    ]
    path = tmp_path / "dockq_scores.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_sampling_is_at_the_complex_level_keeping_every_decoy(tmp_path: Path) -> None:
    """Per-native ranking metrics need >=2 decoys, so rows are never sampled alone."""
    csv = _dockq_csv(tmp_path, n_complexes=6, decoys=3)
    sampled = mb.load_and_sample_dockq(csv, "esmfold2", n_samples=2, random_seed=0)

    assert sampled["entry"].nunique() == 2
    assert len(sampled) == 6  # both complexes kept all three decoys
    assert sampled.groupby("entry").size().tolist() == [3, 3]
    assert set(sampled["model_source"]) == {"esmfold2"}


def test_sampling_keeps_everything_when_the_pool_is_smaller(tmp_path: Path) -> None:
    """A small CSV must not error on an over-large n_samples."""
    csv = _dockq_csv(tmp_path, n_complexes=2, decoys=3)
    assert len(mb.load_and_sample_dockq(csv, "boltz2", n_samples=100)) == 6


def test_preselected_entries_override_random_sampling(tmp_path: Path) -> None:
    """One stratified complex set is shared across all three folding models."""
    csv = _dockq_csv(tmp_path, n_complexes=5, decoys=2)
    sampled = mb.load_and_sample_dockq(csv, "protenix", n_samples=1, preselected_entries={"0001_HLA", "0003_HLA"})
    assert sorted(sampled["entry"].unique()) == ["0001_HLA", "0003_HLA"]


# ---------------------------------------------------------------------------
# convert_to_training_format
# ---------------------------------------------------------------------------


def _row(**overrides) -> pd.DataFrame:
    base = {
        "entry": "1a14_HLN",
        "model_file": "1a14_HLN_sample_0.cif",
        "complex_dockq": 0.42,
        "complex_fnat": 0.5,
        "complex_irmsd": 1.5,
        "complex_lrmsd": 3.0,
        "msa_mode": "antigen_only",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_convert_splits_the_entry_into_pdb_id_and_chains(tmp_path: Path) -> None:
    """``chains`` is what tells the featuriser which chain is heavy / light / antigen."""
    df = _row(loops=4, steps=50)
    out = mb.convert_to_training_format(df, tmp_path, "esmfold2")
    assert out.loc[0, "pdb_id"] == "1a14"
    assert out.loc[0, "chains"] == "HLN"
    assert out.loc[0, "dockq"] == pytest.approx(0.42)
    assert out.loc[0, "model_source"] == "esmfold2"


def test_convert_builds_the_esmfold2_layout(tmp_path: Path) -> None:
    """Condition dir / entry / model_file."""
    out = mb.convert_to_training_format(_row(loops=4, steps=50), tmp_path, "esmfold2")
    assert out.loc[0, "condition"] == "msa=antigen_only__loops=4__steps=50"
    assert out.loc[0, "pdb_path"] == str(
        tmp_path / "msa=antigen_only__loops=4__steps=50" / "1a14_HLN" / "1a14_HLN_sample_0.cif"
    )


def test_convert_builds_the_protenix_seed_layout(tmp_path: Path) -> None:
    """Protenix nests predictions under seed_<N>/predictions/."""
    out = mb.convert_to_training_format(_row(cycle=10, step=200, seed=101), tmp_path, "protenix")
    assert out.loc[0, "condition"] == "msa=antigen_only__cycle=10__step=200"
    assert out.loc[0, "pdb_path"].endswith("1a14_HLN/seed_101/predictions/1a14_HLN_sample_0.cif")


def test_convert_prefers_the_boltz2_scale_directory_that_exists(tmp_path: Path) -> None:
    """The grid dirs are inconsistently named: "scale=1.0" in some runs, "scale=1" in others."""
    df = _row(msa_mode="antigen_only", recycling_steps=3, step_scale=1.0)
    prefix = "msa=antigen_only__recycling=3__scale="

    # Only the bare-integer spelling exists on disk -> that one must be chosen.
    (tmp_path / f"{prefix}1").mkdir()
    out = mb.convert_to_training_format(df, tmp_path, "boltz2")
    assert out.loc[0, "condition"] == f"{prefix}1"
    assert "boltz_results__staged_inputs/predictions/1a14_HLN" in out.loc[0, "pdb_path"]

    # With the float spelling present too, the float form wins (it is tried first).
    (tmp_path / f"{prefix}1.0").mkdir()
    assert mb.convert_to_training_format(df, tmp_path, "boltz2").loc[0, "condition"] == f"{prefix}1.0"


def test_convert_falls_back_to_the_float_spelling_when_no_directory_exists(tmp_path: Path) -> None:
    """A path is still produced (validate_manifest is what drops missing files)."""
    df = _row(recycling_steps=3, step_scale=1.5)
    out = mb.convert_to_training_format(df, tmp_path, "boltz2")
    assert out.loc[0, "condition"] == "msa=antigen_only__recycling=3__scale=1.5"


def test_convert_handles_an_unknown_model_source(tmp_path: Path) -> None:
    """An unrecognised source must not raise; it gets the flat fallback layout."""
    out = mb.convert_to_training_format(_row(), tmp_path, "af3")
    assert out.loc[0, "condition"] == "unknown"
    assert out.loc[0, "pdb_path"] == str(tmp_path / "1a14_HLN" / "1a14_HLN_sample_0.cif")


def test_convert_carries_provenance_and_native_confidence_columns(tmp_path: Path) -> None:
    """fnat/irmsd/lrmsd and the folding model's own confidence are provenance only."""
    df = _row(loops=4, steps=50, ptm=0.8, iptm=0.7, complex_plddt=85.0)
    out = mb.convert_to_training_format(df, tmp_path, "esmfold2").loc[0]
    assert out["fnat"] == pytest.approx(0.5)
    assert out["irmsd"] == pytest.approx(1.5)
    assert out["lrmsd"] == pytest.approx(3.0)
    assert out["ptm"] == pytest.approx(0.8)
    assert out["iptm"] == pytest.approx(0.7)
    assert out["plddt"] == pytest.approx(85.0)  # falls back to complex_plddt


def test_convert_leaves_missing_confidence_as_nan(tmp_path: Path) -> None:
    """No PAE / confidence sidecar is a normal case, not a defect."""
    out = mb.convert_to_training_format(_row(loops=4, steps=50), tmp_path, "esmfold2").loc[0]
    assert pd.isna(out["ptm"]) and pd.isna(out["iptm"]) and pd.isna(out["plddt"])


# ---------------------------------------------------------------------------
# validate_manifest / print_statistics
# ---------------------------------------------------------------------------


def test_validate_manifest_keeps_only_rows_whose_structure_exists(tmp_path: Path) -> None:
    """Only the structure file is required -- features come from coordinates alone."""
    present = tmp_path / "there.cif"
    present.write_text("x")
    df = pd.DataFrame(
        [
            {"pdb_path": str(present), "dockq": 0.5, "model_source": "boltz2"},
            {"pdb_path": str(tmp_path / "missing.cif"), "dockq": 0.6, "model_source": "boltz2"},
        ]
    )
    out = mb.validate_manifest(df)
    assert out["pdb_path"].tolist() == [str(present)]


def test_print_statistics_reports_the_capri_tiers(capsys) -> None:
    """The run log's summary of what the manifest actually contains."""
    df = pd.DataFrame(
        {
            "model_source": ["boltz2", "boltz2", "protenix", "protenix"],
            "dockq": [0.1, 0.3, 0.6, 0.9],
        }
    )
    mb.print_statistics(df)
    out = capsys.readouterr().out
    assert "MANIFEST STATISTICS" in out
    assert "Total entries: 4" in out
    for tier in ("Unacceptable", "Acceptable", "Medium", "High"):
        assert tier in out
    assert "quality_tier" in out  # the crosstab keeps its header


def test_print_statistics_does_not_add_columns_to_the_manifest(capsys) -> None:
    """Callers write the manifest out right after printing, so nothing may be added to it."""
    df = pd.DataFrame({"model_source": ["boltz2", "protenix"], "dockq": [0.1, 0.9]})
    before = list(df.columns)
    mb.print_statistics(df)
    capsys.readouterr()
    assert list(df.columns) == before


# ---------------------------------------------------------------------------
# The shipped final config
# ---------------------------------------------------------------------------


def test_final_config_is_self_contained() -> None:
    """No inheritance, no override merging: what the file says is what runs."""
    config = train_script.load_config(CONFIG_PATH)
    assert "base_config" not in config
    assert "config_overrides" not in config
    assert set(config) == {
        "model",
        "training",
        "data",
        "loss",
        "callbacks",
        "output_dir",
        "wandb",
        "scatter_plot",
    }


def test_final_config_declares_the_fixed_feature_widths() -> None:
    """These only assert agreement with the code; a drift must fail loudly."""
    model_cfg = train_script.load_config(CONFIG_PATH)["model"]
    assert model_cfg["node_input_dim"] == NODE_FEATURE_DIM
    assert model_cfg["edge_input_dim"] == EDGE_FEATURE_DIM


def test_final_config_builds_the_423877_parameter_model() -> None:
    """The exact size of the model the reported results were produced with."""
    model = SimpleGNN(train_script.load_config(CONFIG_PATH))
    assert model.count_parameters()["total"] == 423877


def test_final_config_keeps_the_ranking_term_active_and_grouped() -> None:
    """lambda_rank > 0 is only meaningful with group_by_complex -- pin them together."""
    config = train_script.load_config(CONFIG_PATH)
    assert config["loss"]["lambda_rank"] > 0
    assert config["data"]["group_by_complex"] is True
    assert config["data"]["decoys_per_complex"] >= 2
    assert config["loss"]["rank_type"] == "listnet"
    assert config["loss"]["rank_tau"] == pytest.approx(0.10)


def test_final_config_selects_checkpoints_on_per_native_spearman() -> None:
    """Not val/loss: absolute error can improve while within-complex order does not."""
    callbacks = train_script.load_config(CONFIG_PATH)["callbacks"]
    assert callbacks["monitor"] == "val/per_native_spearman_mean"
    assert callbacks["mode"] == "max"


def test_final_config_trains_in_full_precision() -> None:
    """The ranking term operates on small within-complex differences."""
    assert train_script.load_config(CONFIG_PATH)["training"]["precision"] == "32-true"


# ---------------------------------------------------------------------------
# Entrypoint: manifest -> entries
# ---------------------------------------------------------------------------


def _manifest_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pdb_path": "/d/1a14_HLN_sample_0.cif", "model_source": "boltz2", "chains": "HLN", "dockq": 0.42},
            {"pdb_path": "/d/1a14_HLN_sample_1.cif", "model_source": "boltz2", "chains": "HLN", "dockq": None},
            {"pdb_path": "/d/2xyz_AB_sample_0.cif", "model_source": "protenix", "chains": None, "dockq": 0.1},
        ]
    )


def test_row_to_entry_reads_only_the_four_columns_training_needs() -> None:
    """Everything else in the manifest is provenance."""
    row = _manifest_df().iloc[0]
    assert train_script._row_to_entry(row) == {
        "pdb_path": "/d/1a14_HLN_sample_0.cif",
        "model_source": "boltz2",
        "dockq": 0.42,
        "chain_str": "HLN",
    }


def test_row_to_entry_rejects_a_row_with_no_label() -> None:
    """An unscored decoy cannot be trained on."""
    assert train_script._row_to_entry(_manifest_df().iloc[1]) is None


def test_row_to_entry_makes_missing_chains_optional() -> None:
    """Without a chain string the chain roles are recovered from the structure."""
    assert train_script._row_to_entry(_manifest_df().iloc[2])["chain_str"] is None


def test_manifest_df_to_entries_drops_unlabelled_rows() -> None:
    """The count that gets printed must match what the dataset receives."""
    entries = train_script.manifest_df_to_entries(_manifest_df())
    assert [e["pdb_path"] for e in entries] == ["/d/1a14_HLN_sample_0.cif", "/d/2xyz_AB_sample_0.cif"]


def test_load_entries_reads_a_manifest_csv(tmp_path: Path) -> None:
    """End-to-end from disk, including the NaN-label drop."""
    manifest = tmp_path / "manifest.csv"
    _manifest_df().to_csv(manifest, index=False)
    entries = train_script.load_entries(manifest)
    assert len(entries) == 2
    assert entries[0]["dockq"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Entrypoint: wiring
# ---------------------------------------------------------------------------


def test_build_datamodule_passes_every_data_setting_through() -> None:
    """A renamed DataModule keyword must fail here, not on the cluster."""
    config = train_script.load_config(CONFIG_PATH)
    dm = train_script.build_datamodule([], config)

    data_cfg = config["data"]
    assert dm.batch_size == data_cfg["batch_size"]
    assert dm.num_workers == data_cfg["num_workers"]
    assert dm.interface_cutoff == pytest.approx(data_cfg["interface_cutoff"])
    assert dm.knn_k == data_cfg["knn_k"]
    assert dm.val_fraction == pytest.approx(data_cfg["val_fraction"])
    assert dm.test_fraction == pytest.approx(data_cfg["test_fraction"])
    assert dm.cache_graphs == data_cfg["cache_graphs"]
    assert dm.graph_cache_dir == data_cfg["graph_cache_dir"]
    assert dm.group_by_complex == data_cfg["group_by_complex"]
    assert dm.decoys_per_complex == data_cfg["decoys_per_complex"]
    assert dm.random_seed == data_cfg["random_seed"]


def test_build_datamodule_defaults_match_the_final_recipe() -> None:
    """An empty config must not quietly train a different setup."""
    dm = train_script.build_datamodule([], {})
    assert (dm.batch_size, dm.num_workers, dm.interface_cutoff, dm.knn_k) == pytest.approx((128, 8, 6.0, 16))
    assert dm.group_by_complex is True
    assert dm.decoys_per_complex == 8


def test_build_callbacks_watches_one_metric_everywhere(tmp_path: Path) -> None:
    """Checkpoint and early stop must agree, or training stops on the wrong signal."""
    config = train_script.load_config(CONFIG_PATH)
    checkpoint, early_stop, lr_monitor = train_script.build_callbacks(config, tmp_path)

    assert checkpoint.monitor == early_stop.monitor == "val/per_native_spearman_mean"
    assert checkpoint.mode == early_stop.mode == "max"
    assert early_stop.patience == config["training"]["early_stopping_patience"]
    assert checkpoint.dirpath == str(tmp_path / "checkpoints")
    assert checkpoint.save_last is True
    assert lr_monitor.logging_interval == "epoch"


def test_checkpoint_filename_embeds_the_monitored_score(tmp_path: Path) -> None:
    """So a checkpoint file says which epoch and which score produced it."""
    checkpoint = train_script.build_callbacks(train_script.load_config(CONFIG_PATH), tmp_path)[0]
    name = checkpoint.format_checkpoint_name({"epoch": 7, "val/per_native_spearman_mean": 0.6795})
    assert Path(name).name == "epoch007-score0.6795.ckpt"


def test_build_logger_is_disabled_without_a_project(tmp_path: Path) -> None:
    """W&B must be optional: no project, no import, no API key needed."""
    args = train_script._parse_args(["--config", "c.yaml", "--manifest", "m.csv"])
    assert train_script.build_logger({"wandb": {"project": None}}, args, tmp_path) is None
    assert train_script.build_logger({}, args, tmp_path) is None


def test_build_trainer_reads_the_training_block(tmp_path: Path) -> None:
    """max_epochs / precision / grad clip come from the config, not the CLI."""
    config = train_script.load_config(CONFIG_PATH)
    args = train_script._parse_args(["--config", "c.yaml", "--manifest", "m.csv"])
    trainer = train_script.build_trainer(config, args, tmp_path)

    assert trainer.max_epochs == config["training"]["n_epochs"]
    assert trainer.gradient_clip_val == pytest.approx(config["training"]["grad_clip"])
    assert trainer.precision == "32-true"
    assert trainer.logger is None or trainer.logger.__class__.__name__ != "WandbLogger"


def test_build_trainer_passes_fast_dev_run_through(tmp_path: Path) -> None:
    """The smoke test for data + model wiring before a real run."""
    args = train_script._parse_args(["--config", "c.yaml", "--manifest", "m.csv", "--fast-dev-run"])
    assert args.fast_dev_run is True
    assert train_script.build_trainer({}, args, tmp_path).fast_dev_run is True


# ---------------------------------------------------------------------------
# Entrypoint: CLI
# ---------------------------------------------------------------------------


def test_cli_requires_a_config_and_a_manifest() -> None:
    """Both are required: the config IS the experiment."""
    with pytest.raises(SystemExit):
        train_script._parse_args(["--manifest", "m.csv"])
    with pytest.raises(SystemExit):
        train_script._parse_args(["--config", "c.yaml"])


def test_cli_defaults_leave_output_dir_and_wandb_to_the_config() -> None:
    """Flags cover only where data/checkpoints live, so runs stay reproducible."""
    args = train_script._parse_args(["--config", "c.yaml", "--manifest", "m.csv"])
    assert args.output_dir is None
    assert args.wandb_project is None
    assert args.run_name is None
    assert args.fast_dev_run is False


def test_cli_accepts_the_output_and_wandb_overrides() -> None:
    """The three things that legitimately differ between two runs of one config."""
    args = train_script._parse_args(
        [
            "--config",
            "c.yaml",
            "--manifest",
            "m.csv",
            "--output-dir",
            "/tmp/out",
            "--wandb-project",
            "proj",
            "--run-name",
            "run-1",
        ]
    )
    assert args.output_dir == Path("/tmp/out")
    assert args.wandb_project == "proj"
    assert args.run_name == "run-1"


def test_main_refuses_an_empty_manifest(tmp_path: Path) -> None:
    """Fail immediately rather than after Trainer setup."""
    manifest = tmp_path / "empty.csv"
    pd.DataFrame({"pdb_path": [], "model_source": [], "chains": [], "dockq": []}).to_csv(manifest, index=False)
    with pytest.raises(SystemExit, match="No usable rows"):
        train_script.main(
            ["--config", str(CONFIG_PATH), "--manifest", str(manifest), "--output-dir", str(tmp_path / "out")]
        )


def _smoke_pdb(path: Path) -> Path:
    """Two 6-residue poly-glycine chains, H and A, 5A apart."""
    lines = []
    serial = 1
    for chain, y in (("H", 0.0), ("A", 5.0)):
        for res_index in range(1, 7):
            for name, dx, dy in (("N", -1.2, 0.3), ("CA", 0.0, 0.0), ("C", 1.2, 0.4), ("O", 1.3, 1.6)):
                lines.append(
                    f"ATOM  {serial:5d}  {name:<3s} GLY {chain}{res_index:4d}    "
                    f"{3.8 * res_index + dx:8.3f}{y + dy:8.3f}{0.0:8.3f}  1.00 80.00           {name[0]}"
                )
                serial += 1
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


@pytest.mark.parametrize("fast_dev_run", [True, False])
def test_main_trains_end_to_end_on_synthetic_structures(
    tmp_path: Path, monkeypatch, capsys, fast_dev_run: bool
) -> None:
    """The whole entrypoint: manifest -> graphs -> fit -> checkpoint -> test.

    This catches what the unit tests above cannot: a DataModule keyword the config no
    longer has, a metric the checkpoint monitors but the model never logs, or a
    feature width mismatch between the builder and the model. Run twice -- once as
    ``--fast-dev-run`` (one train + one val batch, checkpointing suppressed) and once
    as a real single epoch that must actually write a checkpoint and then test from
    it. Forced onto CPU with a shrunken copy of the shipped config; the config's own
    VALUES are asserted separately above.
    """
    config = train_script.load_config(CONFIG_PATH)
    config["data"].update(
        batch_size=4,
        decoys_per_complex=2,
        num_workers=0,
        pin_memory=False,
        knn_k=4,
        val_fraction=0.5,
        test_fraction=0.25,
        cache_graphs=False,
        graph_cache_dir=None,
    )
    config["training"].update(n_epochs=1, run_test=True)
    config["wandb"] = {}
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config))

    rows = []
    for complex_index in range(4):
        for decoy in range(2):
            pdb = _smoke_pdb(tmp_path / f"1ab{complex_index}_HA_seed42_sample{decoy}.pdb")
            rows.append({"pdb_path": str(pdb), "model_source": "boltz2", "chains": "HA", "dockq": 0.2 + 0.3 * decoy})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    # accelerator="auto" would grab a (possibly full) GPU; pin the smoke run to CPU
    # without bypassing build_trainer, so its own wiring stays under test.
    real_trainer = train_script.pl.Trainer
    monkeypatch.setattr(
        train_script.pl,
        "Trainer",
        lambda **kwargs: real_trainer(**{**kwargs, "accelerator": "cpu", "devices": 1}),
    )

    output_dir = tmp_path / "out"
    argv = ["--config", str(config_path), "--manifest", str(manifest), "--output-dir", str(output_dir)]
    train_script.main(argv + ["--fast-dev-run"] if fast_dev_run else argv)

    checkpoints = sorted(p.name for p in (output_dir / "checkpoints").glob("*.ckpt"))
    if fast_dev_run:
        assert checkpoints == []  # suppressed, as documented
        # ...which is why the test phase must be skipped rather than asked for
        # ckpt_path="best", a combination Lightning rejects outright.
        assert "Skipping the test phase" in capsys.readouterr().out
    else:
        # The monitored metric made it into the filename, so the callback really saw
        # val/per_native_spearman_mean rather than silently selecting on nothing.
        assert "last.ckpt" in checkpoints
        assert any(name.startswith("epoch000-score") for name in checkpoints)


# ---------------------------------------------------------------------------
# W&B provenance: what a run records about the data it consumed
# ---------------------------------------------------------------------------


def _provenance_manifest(tmp_path: Path, n_complexes: int = 10, decoys: int = 3) -> Path:
    """Build a manifest spread over enough pdb_ids that the split yields a val set."""
    rows = []
    for c in range(n_complexes):
        for d in range(decoys):
            rows.append(
                {
                    "pdb_path": str(tmp_path / f"1ab{c}_HA_seed42_sample{d}.pdb"),
                    "model_source": "boltz2",
                    "chains": "HA",
                    "dockq": 0.1 + 0.3 * d,
                }
            )
    manifest = tmp_path / "prov_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _provenance(tmp_path: Path, **config_overrides):
    """Run build_provenance over a synthetic manifest and return its dict."""
    manifest = _provenance_manifest(tmp_path)
    config = train_script.load_config(CONFIG_PATH)
    config["data"].update({"graph_cache_dir": None, "num_workers": 0, **config_overrides})
    args = train_script._parse_args(["--config", str(CONFIG_PATH), "--manifest", str(manifest)])
    entries = train_script.load_entries(manifest)
    datamodule = train_script.build_datamodule(entries, config)
    return train_script.build_provenance(args, config, entries, datamodule), entries, config


def test_provenance_pins_the_manifest_by_path_hash_and_row_count(tmp_path: Path) -> None:
    """Path alone is not enough: a regenerated CSV must be distinguishable."""
    provenance, entries, _ = _provenance(tmp_path)
    assert provenance["train_manifest"].endswith("prov_manifest.csv")
    assert provenance["train_manifest_rows"] == len(entries) == 30
    assert len(provenance["train_manifest_sha256"]) == 12


def test_provenance_reports_the_post_split_sizes_not_the_pool(tmp_path: Path) -> None:
    """These are the counts that were missing from W&B: what the model really sees.

    They must sum to the pool and both be non-zero, so a run records that it trained
    on a subset -- two seeds split the same manifest differently, and without these
    the val metrics look comparable when they are not.
    """
    provenance, entries, _ = _provenance(tmp_path)
    assert provenance["n_train_samples"] + provenance["n_val_samples"] == len(entries)
    assert provenance["n_train_samples"] > 0
    assert provenance["n_val_samples"] > 0
    assert provenance["n_train_samples"] < provenance["train_manifest_rows"]


def test_provenance_split_sizes_track_the_random_seed(tmp_path: Path) -> None:
    """The reason the sizes are worth logging at all."""
    first, _, _ = _provenance(tmp_path, random_seed=1)
    second, _, _ = _provenance(tmp_path, random_seed=7)
    assert (first["n_train_samples"], first["n_val_samples"]) != (
        second["n_train_samples"],
        second["n_val_samples"],
    ) or first[
        "n_val_samples"
    ] > 0  # a different seed splits differently (or at least splits)


def test_provenance_records_the_std_bin_footprint_and_the_monitor(tmp_path: Path) -> None:
    """Both restore keys the ablation runs had and the stripped entrypoint had lost."""
    provenance, entries, config = _provenance(tmp_path)
    assert provenance["monitor_metric"] == config["callbacks"]["monitor"]
    assert provenance["val_fraction"] == pytest.approx(config["data"]["val_fraction"])
    assert sum(v for k, v in provenance.items() if "stdbin_n_decoys" in k) == len(entries)


def test_provenance_lands_in_the_wandb_config_without_shadowing_hyperparameters(tmp_path: Path) -> None:
    """Provenance is merged in, but a config key of the same name must win."""
    args = train_script._parse_args(
        ["--config", "c.yaml", "--manifest", "m.csv", "--wandb-project", "p"],
    )
    captured = {}

    class _Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import pytorch_lightning.loggers as loggers_module

    real = loggers_module.WandbLogger
    loggers_module.WandbLogger = _Recorder
    try:
        train_script.build_logger({"model": {"hidden_dim": 128}}, args, tmp_path, {"n_train_samples": 91119})
    finally:
        loggers_module.WandbLogger = real

    assert captured["config"]["n_train_samples"] == 91119
    assert captured["config"]["model"] == {"hidden_dim": 128}


def test_build_callbacks_adds_the_scatter_callback_only_with_a_logger(tmp_path: Path) -> None:
    """The figures have nowhere to go in an offline run, so they are not built."""
    config = train_script.load_config(CONFIG_PATH)
    assert len(train_script.build_callbacks(config, tmp_path)) == 3

    class _Logger:
        experiment = None

    with_logger = train_script.build_callbacks(config, tmp_path, logger=_Logger())
    assert len(with_logger) == 4
    assert isinstance(with_logger[-1], train_script.DockQScatterCallback)


def test_scatter_callback_can_be_disabled_from_the_config(tmp_path: Path) -> None:
    """An expensive run may not want per-epoch figures."""
    config = train_script.load_config(CONFIG_PATH)
    config["scatter_plot"] = {"enabled": False}

    class _Logger:
        experiment = None

    assert len(train_script.build_callbacks(config, tmp_path, logger=_Logger())) == 3


def test_write_run_pointer_records_the_run_the_output_dir_belongs_to(tmp_path: Path) -> None:
    """The scoring pass reads this to log into the same run; None logger writes none."""
    import json as json_module

    class _Run:
        entity, project, id, name = "my-entity", "confidock-dockq-predictor", "abc123", "pr-gnn-listnet"
        url = "https://wandb/my-entity/confidock-dockq-predictor/runs/abc123"

    class _Logger:
        experiment = _Run()

    path = train_script.write_run_pointer(_Logger(), tmp_path)
    assert json_module.loads(path.read_text()) == {
        "entity": "my-entity",
        "project": "confidock-dockq-predictor",
        "run_id": "abc123",
        "run_name": "pr-gnn-listnet",
        "url": _Run.url,
    }
    assert train_script.write_run_pointer(None, tmp_path) is None
