#!/usr/bin/env python
"""Train the DockQ-predictor GNN from a manifest CSV.

    python workflows/scripts/training/train_simple_gnn_from_manifest.py \
        --config configs/dockq_predictor/dockq_gnn.yaml \
        --manifest data/dockq_predictor/dockq_training_manifest.csv \
        --output-dir checkpoints/dockq_gnn

Build the manifest first with ``workflows/scripts/analysis/sample_dockq_for_training.py``;
one row is one decoy, and only ``pdb_path``, ``model_source``, ``chains`` and ``dockq``
are read here.

Every hyperparameter comes from the YAML config -- the flags below only cover
where the data is, where checkpoints go, and whether to log to W&B, so that two
runs of the same config are the same experiment. See the config for why the
non-obvious values (grouped batching, per-native-Spearman checkpoint selection,
32-bit precision) are what they are.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from confidock.dockq_predictor.data import DockQDataModule
from confidock.dockq_predictor.models import SimpleGNN
from confidock.dockq_predictor.training.metrics import DOCKQ_STD_BINS
from confidock.dockq_predictor.training.wandb_logging import (
    DockQScatterCallback,
    file_sha256,
    train_std_bin_histogram,
)


def load_config(config_path: Path) -> dict:
    """Load the training config YAML."""
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _row_to_entry(row) -> Optional[Dict]:
    """Convert one manifest row into a ``DockQDataset`` entry, or None if unusable.

    Only ``dockq`` is a label; ``chains`` (e.g. ``HLN``) tells the featuriser which
    chains are heavy / light / antigen, and is optional -- without it the chain roles
    are inferred from the structure.
    """
    if pd.isna(row["dockq"]):
        return None
    return {
        "pdb_path": row["pdb_path"],
        "model_source": row["model_source"],
        "dockq": float(row["dockq"]),
        "chain_str": row["chains"] if "chains" in row and pd.notna(row["chains"]) else None,
    }


def manifest_df_to_entries(df: pd.DataFrame) -> List[Dict]:
    """Convert a manifest DataFrame into ``DockQDataset`` entry dicts."""
    return [e for e in (_row_to_entry(row) for _, row in df.iterrows()) if e is not None]


def load_entries(manifest_path: Path) -> List[Dict]:
    """Load decoy entries from a manifest CSV, dropping rows with no DockQ label."""
    df = pd.read_csv(manifest_path).dropna(subset=["dockq"])
    entries = manifest_df_to_entries(df)
    print(f"Manifest {manifest_path}: {len(entries)} decoys with a DockQ label")
    return entries


def build_datamodule(entries: List[Dict], config: dict) -> DockQDataModule:
    """Build the DataModule from the config's ``data`` block."""
    data_cfg = config.get("data", {})
    return DockQDataModule(
        entries=entries,
        val_fraction=float(data_cfg.get("val_fraction", 0.15)),
        test_fraction=float(data_cfg.get("test_fraction", 0.0)),
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 8)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        interface_cutoff=float(data_cfg.get("interface_cutoff", 6.0)),
        knn_k=int(data_cfg.get("knn_k", 16)),
        random_seed=int(data_cfg.get("random_seed", 42)),
        cache_graphs=bool(data_cfg.get("cache_graphs", False)),
        graph_cache_dir=data_cfg.get("graph_cache_dir"),
        group_by_complex=bool(data_cfg.get("group_by_complex", True)),
        decoys_per_complex=int(data_cfg.get("decoys_per_complex", 8)),
    )


def build_callbacks(config: dict, output_dir: Path, logger=None) -> list:
    """Build checkpoint / early-stopping / LR-logging callbacks.

    Both the checkpoint and the early stop watch the SAME metric, so training stops
    when the metric that selects the checkpoint stops improving.

    With a W&B logger, the predicted-vs-true visualisations are appended too; they
    have nowhere to go without one, so they are not built for an offline run.
    """
    cb_cfg = config.get("callbacks", {})
    monitor = cb_cfg.get("monitor", "val/per_native_spearman_mean")
    mode = cb_cfg.get("mode", "max")
    patience = int(config.get("training", {}).get("early_stopping_patience", 15))

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="epoch{epoch:03d}-score{" + monitor + ":.4f}",
            auto_insert_metric_name=False,
            monitor=monitor,
            mode=mode,
            save_top_k=int(cb_cfg.get("save_top_k", 1)),
            save_last=True,
        ),
        EarlyStopping(monitor=monitor, mode=mode, patience=patience, verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    scatter_cfg = config.get("scatter_plot", {}) or {}
    if logger is not None and scatter_cfg.get("enabled", True):
        callbacks.append(
            DockQScatterCallback(
                logger,
                every_n_epochs=int(scatter_cfg.get("every_n_epochs", 5)),
                max_points=int(scatter_cfg.get("max_points", 5000)),
                n_native_scatters=int(scatter_cfg.get("n_native_scatters", 10)),
                # Every epoch would upload one table per epoch for the whole run;
                # 20 keeps a few checkpoints of the raw points without the bloat.
                table_every_n_epochs=int(scatter_cfg.get("table_every_n_epochs", 20)),
                random_seed=int(config.get("data", {}).get("random_seed", 42)),
            )
        )
    return callbacks


def build_provenance(args: argparse.Namespace, config: dict, entries: List[Dict], datamodule) -> Dict:
    """Assemble the run-provenance fields that go into the W&B config.

    The scalar metrics say how well a run did; these say WHAT it did it on, which is
    what makes two runs comparable months later:

    * ``train_manifest`` + ``_rows`` + ``_sha256`` -- pins the exact CSV content, so a
      regenerated or edited manifest cannot silently masquerade as the same data.
    * ``n_train_samples`` / ``n_val_samples`` -- sizes AFTER the internal split, i.e.
      what the model really trains and validates on. ``train_manifest_rows`` is the
      pre-split pool, so these differ by ``val_fraction``; two runs at different
      ``random_seed`` split the same manifest differently and their val numbers are
      NOT directly comparable, which is only visible if these are recorded.
    * ``train/stdbin_n_*`` -- the training pool's DockQ-std footprint, same bins as
      the eval-side ``stdbin_*`` metrics.
    """
    data_cfg = config.get("data", {})
    # setup('fit') caches its split, so this is safe to call before Lightning does.
    if datamodule.train_dataset is None:
        datamodule.setup("fit")
    n_train = len(datamodule.train_dataset)
    n_val = len(datamodule.val_dataset) if datamodule.val_dataset is not None else 0
    print(f"  dataset sizes: train={n_train} decoys, val={n_val} decoys (from {len(entries)} pooled)")

    histogram = train_std_bin_histogram(entries)
    print("  train DockQ-std bins (n_complexes | n_decoys):")
    for lo, hi in zip(DOCKQ_STD_BINS[:-1], DOCKQ_STD_BINS[1:]):
        tag = f"{lo:.2f}_{hi:.2f}"
        n_cplx = int(histogram[f"train/stdbin_n_complexes/{tag}"])
        n_decoys = int(histogram[f"train/stdbin_n_decoys/{tag}"])
        print(f"    [{tag}] {n_cplx:5d} | {n_decoys:7d}")

    return {
        "train_manifest": str(args.manifest),
        "train_manifest_sha256": file_sha256(args.manifest),
        "train_manifest_rows": len(entries),
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "val_fraction": data_cfg.get("val_fraction"),
        "monitor_metric": config.get("callbacks", {}).get("monitor", "val/per_native_spearman_mean"),
        **histogram,
    }


def build_logger(config: dict, args: argparse.Namespace, output_dir: Path, provenance: Optional[Dict] = None):
    """Return a W&B logger if a project is configured, else None (offline run).

    W&B is imported lazily so training does not depend on it being installed or on
    ``WANDB_API_KEY`` being set.
    """
    wandb_cfg = config.get("wandb", {}) or {}
    project = args.wandb_project or wandb_cfg.get("project")
    if not project:
        print("W&B disabled (no wandb.project in config and no --wandb-project); metrics go to stdout.")
        return None

    from pytorch_lightning.loggers import WandbLogger

    return WandbLogger(
        project=project,
        entity=wandb_cfg.get("entity"),
        name=args.run_name or wandb_cfg.get("run_name"),
        save_dir=str(output_dir),
        # Provenance first so a colliding key in the config wins -- the config is the
        # authoritative record of the hyperparameters.
        config={**(provenance or {}), **config},
    )


def write_run_pointer(logger, output_dir: Path) -> Optional[Path]:
    """Record which W&B run this output dir belongs to, and return the file written.

    ``predict_dockq_from_manifest.py`` reads this to log its held-out metrics into
    the SAME run as the training that produced the checkpoint, instead of leaving
    them in a JSON file that has to be correlated by hand.
    """
    if logger is None:
        return None
    run = logger.experiment  # forces wandb.init, so id/url below are populated
    pointer = {
        "entity": run.entity,
        "project": run.project,
        "run_id": run.id,
        "run_name": run.name,
        "url": run.url,
    }
    path = output_dir / "wandb_run.json"
    with open(path, "w") as fh:
        json.dump(pointer, fh, indent=2)
    print(f"W&B run {run.name} ({run.id}) -> {path}")
    return path


def build_trainer(config: dict, args: argparse.Namespace, output_dir: Path, logger=None) -> pl.Trainer:
    """Build the Lightning Trainer from the config's ``training`` block.

    ``logger=None`` builds one from the config; ``main`` passes an already-built
    logger so the provenance and the gradient watch are attached to it first.
    """
    train_cfg = config.get("training", {})
    if logger is None:
        logger = build_logger(config, args, output_dir)
    return pl.Trainer(
        max_epochs=int(train_cfg.get("n_epochs", 100)),
        accelerator="auto",
        devices=1,
        precision=train_cfg.get("precision", "32-true"),
        gradient_clip_val=float(train_cfg.get("grad_clip", 1.0)),
        callbacks=build_callbacks(config, output_dir, logger=logger),
        logger=logger,
        default_root_dir=str(output_dir),
        log_every_n_steps=25,
        fast_dev_run=args.fast_dev_run,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML")
    parser.add_argument("--manifest", type=Path, required=True, help="Training manifest CSV (one row per decoy)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where checkpoints and logs go (default: output_dir from the config)",
    )
    parser.add_argument("--wandb-project", default=None, help="W&B project; overrides wandb.project in the config")
    parser.add_argument("--run-name", default=None, help="W&B run name; overrides wandb.run_name in the config")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run one train + one val batch and exit -- smoke-tests the data and model wiring",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Train the model end to end."""
    args = _parse_args(argv)

    config = load_config(args.config)
    output_dir = args.output_dir or Path(config.get("output_dir", "checkpoints/dockq_gnn"))
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(config.get("data", {}).get("random_seed", 42)), workers=True)

    entries = load_entries(args.manifest)
    if not entries:
        raise SystemExit(f"No usable rows in {args.manifest}")

    datamodule = build_datamodule(entries, config)
    model = SimpleGNN(config)
    print(f"Model parameters: {json.dumps(model.count_parameters(), indent=2)}")

    provenance = build_provenance(args, config, entries, datamodule)
    logger = build_logger(config, args, output_dir, provenance)
    if logger is not None:
        # Per-parameter gradient histograms: the diagnostic for a layer that has
        # stopped learning, which no scalar metric shows.
        logger.watch(model, log="gradients", log_freq=100)
        write_run_pointer(logger, output_dir)

    trainer = build_trainer(config, args, output_dir, logger=logger)
    trainer.fit(model, datamodule=datamodule)

    if config.get("training", {}).get("run_test", False):
        if args.fast_dev_run:
            # fast_dev_run suppresses checkpointing, so ckpt_path="best" would raise.
            print("Skipping the test phase: --fast-dev-run writes no checkpoint to test from.")
        else:
            trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"Done. Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
