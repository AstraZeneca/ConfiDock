"""Tests for the inference entrypoint: manifest + checkpoint -> predictions and metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data

from confidock.dockq_predictor.models import SimpleGNN
from confidock.dockq_predictor.training.metrics import DOCKQ_STD_BINS

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "dockq_predictor" / "dockq_gnn.yaml"
sys.path.insert(0, str(REPO_ROOT / "workflows" / "scripts" / "training"))

import predict_dockq_from_manifest as predict_script  # noqa: E402
from test_dockq_predictor_pipeline import _smoke_pdb  # noqa: E402


def _manifest(tmp_path: Path, n_complexes: int = 3, decoys: int = 4, bad_rows: int = 0) -> Path:
    """Write a manifest of synthetic structures, optionally with unreadable paths."""
    pdb = _smoke_pdb(tmp_path / "decoy.pdb")
    rows = []
    for c in range(n_complexes):
        for d in range(decoys):
            rows.append(
                {
                    "pdb_id": f"cplx{c}",
                    "chains": "HA",
                    "entry": f"cplx{c}_HA",
                    "pdb_path": str(pdb),
                    "model_source": "boltz2",
                    "condition": "cond",
                    "dockq": 0.1 + 0.2 * d,
                }
            )
    for b in range(bad_rows):
        rows.append(
            {
                "pdb_id": "gone",
                "chains": "HA",
                "entry": "gone_HA",
                "pdb_path": str(tmp_path / f"missing{b}.pdb"),
                "model_source": "boltz2",
                "condition": "cond",
                "dockq": 0.5,
            }
        )
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _checkpoint(tmp_path: Path) -> Path:
    """Save an untrained model in the shape the script loads (a Lightning ckpt dict).

    Untrained is fine: these tests assert plumbing -- shapes, joins, output files --
    not predictive quality.
    """
    import yaml

    config = yaml.safe_load(CONFIG_PATH.read_text())
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": SimpleGNN(config).state_dict(), "epoch": 7, "global_step": 700}, path)
    return path


def _run(tmp_path: Path, manifest: Path, *extra: str) -> Path:
    """Invoke the entrypoint on CPU and return its output directory."""
    out = tmp_path / "out"
    predict_script.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--checkpoint",
            str(_checkpoint(tmp_path)),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(out),
            "--num-workers",
            "0",
            "--batch-size",
            "4",
            "--no-graph-cache",
            *extra,
        ]
    )
    return out


# ---------------------------------------------------------------------------
# Row tagging: the join back onto the manifest
# ---------------------------------------------------------------------------


def test_indexed_dataset_tags_each_graph_with_its_manifest_row(tmp_path: Path) -> None:
    """Predictions are joined by this tag, not by dataloader position."""
    pdb = _smoke_pdb(tmp_path / "d.pdb")
    entries = [{"pdb_path": str(pdb), "model_source": "boltz2", "dockq": 0.4, "chain_str": "HA"}] * 3
    dataset = predict_script.IndexedDockQDataset(entries, interface_cutoff=6.0, knn_k=4, cache_graphs=False)
    assert [int(dataset[i].entry_row) for i in range(3)] == [0, 1, 2]


def test_row_tags_survive_batching_unchanged() -> None:
    """Regression: PyG increments any attribute whose NAME contains "index".

    ``Data.__inc__`` adds the cumulative node count to such attributes when batching,
    so naming this ``entry_index`` silently turned the row numbers into node offsets
    and the join raised an out-of-bounds IndexError. Named ``entry_row``, it is
    concatenated verbatim.
    """
    graphs = []
    for row, n_nodes in enumerate((5, 9, 4)):
        data = Data(x=torch.zeros(n_nodes, 18), edge_index=torch.zeros(2, 0, dtype=torch.long))
        data.entry_row = torch.tensor([row], dtype=torch.long)
        graphs.append(data)
    assert Batch.from_data_list(graphs).entry_row.tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# Collation: a failed structure is dropped, never resampled
# ---------------------------------------------------------------------------


def test_collate_drops_the_graphs_that_failed_to_build() -> None:
    """A structure that cannot be featurised is absent, not substituted."""
    graphs = [Data(x=torch.zeros(3, 18), edge_index=torch.zeros(2, 0, dtype=torch.long)) for _ in range(2)]
    batch = predict_script._collate_skipping_failures([graphs[0], None, graphs[1]])
    assert batch.num_graphs == 2


def test_collate_returns_none_when_nothing_in_the_batch_built() -> None:
    """The scoring loop skips such a batch rather than crashing on an empty list."""
    assert predict_script._collate_skipping_failures([None, None]) is None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_evaluate_prefixes_every_metric_and_reports_per_native() -> None:
    """The prefix keeps two scored manifests distinguishable in one metrics dict."""
    scored = pd.DataFrame(
        {
            "entry": ["a_HA", "a_HA", "b_HA", "b_HA"],
            "model_source": ["boltz2"] * 4,
            "dockq": [0.1, 0.8, 0.2, 0.9],
            "dockq_pred": [0.2, 0.7, 0.3, 0.85],
        }
    )
    metrics = predict_script.evaluate(scored, prefix="val_hard/")
    assert all(k.startswith("val_hard/") for k in metrics)
    # Ordering is perfect within both complexes, so per-native Spearman is 1.
    assert metrics["val_hard/per_native_spearman_mean"] == pytest.approx(1.0)
    assert all(isinstance(v, float) for v in metrics.values())


def test_evaluate_groups_per_native_by_entry_not_by_row() -> None:
    """Grouping by the complex is what makes the headline number a RANKING number."""
    scored = pd.DataFrame(
        {
            "entry": ["a_HA", "a_HA", "b_HA", "b_HA"],
            "model_source": ["boltz2"] * 4,
            "dockq": [0.1, 0.8, 0.2, 0.9],
            "dockq_pred": [0.8, 0.1, 0.3, 0.85],  # a inverted, b correct -> mean 0
        }
    )
    metrics = predict_script.evaluate(scored, prefix="p/")
    assert metrics["p/per_native_spearman_mean"] == pytest.approx(0.0)


def test_native_ids_prefer_the_entry_column() -> None:
    """A builder-made manifest carries ``entry``; that is the authoritative key."""
    df = pd.DataFrame({"entry": ["a_HA", "b_HL"], "pdb_path": ["/d/zzz_XX_sample_0.pdb"] * 2})
    assert predict_script.native_ids(df) == ["a_HA", "b_HL"]


def test_native_ids_fall_back_to_the_path_derived_key() -> None:
    """Derive the key the way training does when ``entry`` is absent.

    Only pdb_path/model_source/dockq are needed to SCORE a manifest, so evaluating
    one must not require ``entry`` either.
    """
    df = pd.DataFrame({"pdb_path": ["/d/1a3r_HLP_sample_0.cif", "/d/1a3r_HLP_sample_1.cif", "/d/7y8j_HLA.cif"]})
    assert predict_script.native_ids(df) == ["1a3r_HLP", "1a3r_HLP", "7y8j_HLA"]


def test_evaluate_without_an_entry_column_still_groups_per_native() -> None:
    """The regression this guards: a KeyError on 'entry' after the model has run."""
    scored = pd.DataFrame(
        {
            "pdb_path": ["/d/a_HA_sample_0.pdb", "/d/a_HA_sample_1.pdb", "/d/b_HA_sample_0.pdb"],
            "model_source": ["boltz2"] * 3,
            "dockq": [0.1, 0.8, 0.2],
            "dockq_pred": [0.2, 0.7, 0.3],
        }
    )
    metrics = predict_script.evaluate(scored, prefix="p/")
    # Two complexes, only "a_HA" rankable (b_HA has a single decoy) -> Spearman 1.
    assert metrics["p/per_native_spearman_mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Entrypoint, end to end
# ---------------------------------------------------------------------------


def test_main_writes_predictions_and_metrics_for_every_row(tmp_path: Path) -> None:
    """The whole path: manifest -> graphs -> checkpoint -> predictions.csv + metrics.json."""
    out = _run(tmp_path, _manifest(tmp_path, n_complexes=3, decoys=4))

    predictions = pd.read_csv(out / "predictions.csv")
    assert len(predictions) == 12  # nothing sampled away: every row is scored
    assert "dockq_pred" in predictions.columns
    assert bool(((predictions["dockq_pred"] >= 0) & (predictions["dockq_pred"] <= 1)).all())
    # The manifest columns are carried through, so predictions can be re-analysed
    # per model source / condition without re-running the model.
    for column in ("pdb_path", "entry", "model_source", "condition", "dockq"):
        assert column in predictions.columns

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics, "no metrics were written"
    assert all(k.startswith("manifest/") for k in metrics)  # default prefix is the manifest stem


def test_main_keeps_labels_aligned_when_a_structure_fails(tmp_path: Path) -> None:
    """The failed row is dropped and the survivors keep THEIR labels.

    A positional join would shift every label after the failure by one.
    """
    manifest = _manifest(tmp_path, n_complexes=2, decoys=3, bad_rows=2)
    out = _run(tmp_path, manifest)

    predictions = pd.read_csv(out / "predictions.csv")
    assert len(predictions) == 6  # the two missing files are gone
    assert "gone_HA" not in set(predictions["entry"])
    original = pd.read_csv(manifest)
    # validate="one_to_one": the triple is unique per row on BOTH sides, so a
    # surviving prediction must match exactly one manifest row. Without it a
    # duplicated key would fan out into a cartesian product, and the row count
    # below could then be satisfied by the wrong pairing.
    merged = predictions.merge(original, on=["entry", "pdb_path", "dockq"], how="inner", validate="one_to_one")
    assert len(merged) == len(predictions)


def test_main_accepts_an_explicit_metric_prefix(tmp_path: Path) -> None:
    """Two manifests can be scored into one metrics namespace without collisions."""
    out = _run(tmp_path, _manifest(tmp_path, n_complexes=2, decoys=3), "--prefix", "external/")
    assert all(k.startswith("external/") for k in json.loads((out / "metrics.json").read_text()))


def test_main_refuses_a_manifest_with_no_labels(tmp_path: Path) -> None:
    """Fail before loading the checkpoint, not after scoring nothing."""
    manifest = tmp_path / "empty.csv"
    pd.DataFrame({"pdb_path": [], "model_source": [], "chains": [], "entry": [], "dockq": []}).to_csv(
        manifest, index=False
    )
    with pytest.raises(SystemExit, match="No rows with a DockQ label"):
        predict_script.main(
            [
                "--config",
                str(CONFIG_PATH),
                "--checkpoint",
                str(_checkpoint(tmp_path)),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_main_raises_when_no_structure_could_be_featurised(tmp_path: Path) -> None:
    """An all-missing manifest must not write an empty predictions file."""
    manifest = _manifest(tmp_path, n_complexes=0, decoys=0, bad_rows=3)
    with pytest.raises(SystemExit, match="nothing to score"):
        _run(tmp_path, manifest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_requires_config_checkpoint_manifest_and_output() -> None:
    """All four identify the experiment; none has a safe default."""
    complete = ["--config", "c.yaml", "--checkpoint", "m.ckpt", "--manifest", "m.csv", "--output-dir", "o"]
    assert predict_script._parse_args(complete).prefix is None
    for drop in ("--config", "--checkpoint", "--manifest", "--output-dir"):
        partial = [a for i, a in enumerate(complete) if a != drop and complete[i - 1] != drop]
        with pytest.raises(SystemExit):
            predict_script._parse_args(partial)


def test_no_graph_cache_flag_disables_the_configured_cache_dir(tmp_path: Path) -> None:
    """A single scoring pass gains nothing from writing a ~3 MB/graph cache."""
    manifest = _manifest(tmp_path, n_complexes=1, decoys=2)
    out = _run(tmp_path, manifest)  # _run always passes --no-graph-cache
    assert (out / "predictions.csv").exists()
    assert (
        predict_script._parse_args(
            ["--config", "c.yaml", "--checkpoint", "m.ckpt", "--manifest", "m.csv", "--output-dir", "o"]
        ).no_graph_cache
        is False
    )


# ---------------------------------------------------------------------------
# W&B: held-out metrics land in the run that produced the checkpoint
# ---------------------------------------------------------------------------


def _pointer(path: Path, run_id: str = "abc123") -> Path:
    """Write a wandb_run.json like the training entrypoint does."""
    path.write_text(
        json.dumps(
            {
                "entity": "my-entity",
                "project": "confidock-dockq-predictor",
                "run_id": run_id,
                "run_name": "pr-gnn-listnet",
                "url": "https://wandb/x",
            }
        )
    )
    return path


def test_run_pointer_is_found_one_level_up(tmp_path: Path) -> None:
    """The real layout: <run>/wandb_run.json with the scores in <run>/val_hard/."""
    _pointer(tmp_path / "wandb_run.json")
    scored_dir = tmp_path / "val_hard"
    scored_dir.mkdir()
    assert predict_script.find_run_pointer(scored_dir) == tmp_path / "wandb_run.json"


def test_run_pointer_in_the_output_dir_itself_wins(tmp_path: Path) -> None:
    """A pointer beside the metrics is more specific than the parent's."""
    out = tmp_path / "out"
    out.mkdir()
    _pointer(tmp_path / "wandb_run.json", "parent")
    _pointer(out / "wandb_run.json", "own")
    assert predict_script.find_run_pointer(out) == out / "wandb_run.json"


def test_run_pointer_is_absent_rather_than_an_error(tmp_path: Path) -> None:
    """Scoring without W&B is a supported mode, not a failure."""
    assert predict_script.find_run_pointer(tmp_path) is None
    assert predict_script.find_run_pointer(tmp_path, tmp_path / "nope.json") is None


def test_explicit_run_json_overrides_the_search(tmp_path: Path) -> None:
    """--wandb-run-json points at a run in an unrelated directory."""
    explicit = _pointer(tmp_path / "elsewhere.json")
    assert predict_script.find_run_pointer(tmp_path / "out", explicit) == explicit


def test_metrics_and_provenance_are_pushed_to_the_named_run(tmp_path: Path, monkeypatch) -> None:
    """Summary gets the metrics, config gets the provenance, and update() commits."""
    recorded = {"summary": {}, "config": {}, "updated": 0, "path": None}

    class _Run:
        summary = recorded["summary"]
        config = recorded["config"]

        def update(self):
            recorded["updated"] += 1

    class _Api:
        def run(self, path):
            recorded["path"] = path
            return _Run()

    fake_wandb = type("W", (), {"Api": _Api})
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    assert (
        predict_script.log_to_wandb(
            {"val_hard/per_native_spearman_mean": 0.42},
            {"val_hard/manifest_rows": 13655},
            _pointer(tmp_path / "wandb_run.json"),
        )
        is True
    )
    assert recorded["path"] == "my-entity/confidock-dockq-predictor/abc123"
    assert recorded["summary"]["val_hard/per_native_spearman_mean"] == pytest.approx(0.42)
    assert recorded["config"]["val_hard/manifest_rows"] == 13655
    assert recorded["updated"] == 1


def test_wandb_failure_never_breaks_a_scoring_pass(tmp_path: Path, monkeypatch, capsys) -> None:
    """metrics.json is the authoritative artefact; the expensive work is already done."""

    class _Api:
        def run(self, path):
            raise RuntimeError("wandb unreachable")

    monkeypatch.setitem(sys.modules, "wandb", type("W", (), {"Api": _Api}))
    assert predict_script.log_to_wandb({"m": 1.0}, {}, _pointer(tmp_path / "wandb_run.json")) is False
    assert "W&B logging skipped" in capsys.readouterr().out


def test_a_corrupt_pointer_is_reported_not_raised(tmp_path: Path, capsys) -> None:
    """A truncated JSON file must not lose the predictions."""
    bad = tmp_path / "wandb_run.json"
    bad.write_text("{not json")
    assert predict_script.log_to_wandb({"m": 1.0}, {}, bad) is False
    assert "W&B logging skipped" in capsys.readouterr().out


def test_main_logs_the_scored_metrics_when_a_pointer_is_present(tmp_path: Path, monkeypatch) -> None:
    """End to end: scoring a manifest updates the training run's summary."""
    calls = []
    monkeypatch.setattr(
        predict_script, "log_to_wandb", lambda metrics, provenance, pointer: calls.append((metrics, provenance)) or True
    )
    out = tmp_path / "out"
    _pointer(tmp_path / "wandb_run.json")
    manifest = _manifest(tmp_path, n_complexes=2, decoys=3)
    predict_script.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--checkpoint",
            str(_checkpoint(tmp_path)),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(out),
            "--prefix",
            "val_hard/",
            "--num-workers",
            "0",
            "--batch-size",
            "4",
            "--no-graph-cache",
            "--wandb-run-json",
            str(tmp_path / "wandb_run.json"),
        ]
    )
    # Unpack rather than index: this asserts "exactly one call" on its own.
    [(metrics, provenance)] = calls
    assert all(k.startswith("val_hard/") for k in metrics)
    # The scored manifest is pinned the same way the training manifest is.
    assert provenance["val_hard/manifest_rows"] == 6
    assert provenance["val_hard/n_scored"] == 6
    assert len(provenance["val_hard/manifest_sha256"]) == 12
    assert provenance["val_hard/checkpoint"].endswith("model.ckpt")


def test_no_wandb_flag_skips_logging_entirely(tmp_path: Path, monkeypatch) -> None:
    """An explicit opt-out, even with a pointer sitting right there."""
    calls = []
    monkeypatch.setattr(predict_script, "log_to_wandb", lambda *a: calls.append(a) or True)
    _pointer(tmp_path / "wandb_run.json")
    _run(tmp_path, _manifest(tmp_path, n_complexes=1, decoys=2), "--no-wandb")
    assert calls == []


def test_scoring_without_any_pointer_still_writes_its_outputs(tmp_path: Path, capsys) -> None:
    """The default path when the training run had W&B disabled."""
    out = _run(tmp_path, _manifest(tmp_path, n_complexes=1, decoys=2))
    assert (out / "metrics.json").exists()
    assert "No wandb_run.json found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# print_summary: the DockQ-std breakdown
# ---------------------------------------------------------------------------


def _stdbin_metrics(prefix: str = "val_easy/") -> dict:
    """Build a metric dict carrying one value per std bin, as the eval set does."""
    metrics = {f"{prefix}pred_std": 0.163, f"{prefix}true_std": 0.282, f"{prefix}std_ratio": 0.578}
    for i, (lo, hi) in enumerate(zip(DOCKQ_STD_BINS[:-1], DOCKQ_STD_BINS[1:])):
        tag = f"{lo:.2f}_{hi:.2f}"
        metrics[f"{prefix}stdbin_n_complexes/{tag}"] = float(100 + i)
        metrics[f"{prefix}stdbin_n_decoys/{tag}"] = float(1000 + i)
        metrics[f"{prefix}stdbin_pernative_spearman/{tag}"] = 0.1 * i
        metrics[f"{prefix}stdbin_global_spearman/{tag}"] = 0.2 * i
    return metrics


def test_summary_prints_the_high_variance_bin(capsys) -> None:
    """The bin the loss arms are compared on must be readable without opening the JSON."""
    predict_script.print_summary(_stdbin_metrics(), "val_easy/", "val_easy")
    out = capsys.readouterr().out
    assert "0.20_1.01" in out
    # Bin index 4 -> per-native 0.4, global 0.8.
    hi_var_line = next(line for line in out.splitlines() if "0.20_1.01" in line)
    assert "0.4000" in hi_var_line and "0.8000" in hi_var_line


def test_summary_prints_every_bin_with_its_counts(capsys) -> None:
    """All five bins, so a bin that is nearly empty cannot be mistaken for a result."""
    predict_script.print_summary(_stdbin_metrics(), "val_easy/", "val_easy")
    out = capsys.readouterr().out
    for lo, hi in zip(DOCKQ_STD_BINS[:-1], DOCKQ_STD_BINS[1:]):
        assert f"{lo:.2f}_{hi:.2f}" in out
    assert "104" in out and "1004" in out  # n_complexes / n_decoys of the top bin


def test_summary_prints_the_prediction_scale(capsys) -> None:
    """std_ratio is how a collapsed prediction scale becomes visible."""
    predict_script.print_summary(_stdbin_metrics(), "val_easy/", "val_easy")
    out = capsys.readouterr().out
    assert "std_ratio" in out and "0.5780" in out


def test_summary_without_stdbin_metrics_omits_the_table(capsys) -> None:
    """Guards a metric dict from an older run: no bins means no empty table rows."""
    predict_script.print_summary({"val_easy/spearman_r": 0.67}, "val_easy/", "val_easy")
    out = capsys.readouterr().out
    assert "0.20_1.01" not in out
    assert "0.6700" in out
