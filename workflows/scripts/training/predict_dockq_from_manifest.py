#!/usr/bin/env python
"""Score a manifest with a trained DockQ-predictor checkpoint.

    python workflows/scripts/training/predict_dockq_from_manifest.py \
        --config configs/dockq_predictor/dockq_gnn.yaml \
        --checkpoint checkpoints/dockq_gnn/checkpoints/epoch012-score0.6431.ckpt \
        --manifest data/dockq_predictor/val_hard_manifest.csv \
        --output-dir results/dockq_gnn/val_hard

This is the evaluation counterpart to ``train_simple_gnn_from_manifest.py``: it runs
the model over EVERY row of a manifest (no splitting, no sampling) and writes

* ``predictions.csv`` -- one row per decoy: its manifest columns plus ``dockq_pred``
* ``metrics.json``    -- the same comprehensive metric set the training run logs

Use it for a held-out or external validation set. The manifest is scored whole, so
the metrics are over the full set rather than a fraction of it, and the per-decoy CSV
lets the same predictions be re-analysed (per model source, per condition, per
antigen cluster) without re-running the model.

The config must be the one the checkpoint was trained with: the checkpoint holds only
weights, so the architecture and the featurisation settings (``interface_cutoff``,
``knn_k``) are read from the config here. Loading a checkpoint under a config that
disagrees raises on the state-dict shapes rather than silently mis-scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from confidock.dockq_predictor.data.dataset import DockQDataset, _native_id_from_path
from confidock.dockq_predictor.models import SimpleGNN
from confidock.dockq_predictor.training.metrics import DOCKQ_STD_BINS, compute_comprehensive_metrics
from confidock.dockq_predictor.training.wandb_logging import file_sha256

# The training entrypoint owns manifest -> entry conversion and config loading; reuse
# them so the two scripts cannot drift into reading a manifest differently.
sys.path.insert(0, str(Path(__file__).parent))

from train_simple_gnn_from_manifest import load_config, manifest_df_to_entries  # noqa: E402


class IndexedDockQDataset(DockQDataset):
    """``DockQDataset`` that tags each graph with its row index in the manifest.

    Predictions have to be joined back onto the manifest rows they came from, and
    dataloader order is not a reliable key: a structure that fails to featurise
    yields None and is dropped from its batch, so positions shift. Carrying the
    index on the graph makes the join exact.
    """

    def get(self, idx: int):
        """Build the graph for row ``idx``, tagged with that index."""
        data = super().get(idx)
        if data is not None:
            # NOT "entry_index": PyG's Data.__inc__ adds the cumulative node count to
            # any attribute whose name contains "index" when batching, which would
            # turn these row numbers into garbage.
            data.entry_row = torch.tensor([idx], dtype=torch.long)
        return data


def _collate_skipping_failures(samples: List) -> Optional[Batch]:
    """Batch the graphs that built, dropping the Nones.

    Unlike training, a failed structure must NOT be replaced by a random resample
    (``FilteredDockQDataset``): that would score some decoys twice and others never.
    Here the row is simply absent from the predictions, and reported as a miss.
    """
    kept = [s for s in samples if s is not None]
    return Batch.from_data_list(kept) if kept else None


def score_manifest(
    df: pd.DataFrame,
    model: SimpleGNN,
    config: dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    """Run the model over every usable manifest row and return the predictions.

    Returns a copy of the manifest rows that were scored, with a ``dockq_pred``
    column appended.
    """
    entries = manifest_df_to_entries(df)
    data_cfg = config.get("data", {})
    dataset = IndexedDockQDataset(
        entries=entries,
        interface_cutoff=float(data_cfg.get("interface_cutoff", 6.0)),
        knn_k=int(data_cfg.get("knn_k", 16)),
        cache_graphs=False,  # one pass over the data: an in-memory cache is dead weight
        graph_cache_dir=data_cfg.get("graph_cache_dir"),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_skipping_failures,
        pin_memory=False,
    )

    model = model.to(device).eval()
    rows: List[int] = []
    preds: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue
            batch = batch.to(device)
            preds.append(model(batch)["dockq"].detach().cpu())
            rows.extend(batch.entry_row.cpu().tolist())
            if (batch_idx + 1) % 25 == 0:
                print(f"  scored {len(rows)}/{len(dataset)} decoys", flush=True)

    if not preds:
        raise SystemExit("No decoy could be featurised; nothing to score.")

    scored = df.iloc[rows].copy()
    scored["dockq_pred"] = torch.cat(preds).numpy()
    n_missing = len(dataset) - len(scored)
    print(f"Scored {len(scored)}/{len(dataset)} decoys ({n_missing} could not be featurised)")
    return scored


def native_ids(df: pd.DataFrame) -> List[str]:
    """Per-row complex key: the manifest's ``entry`` column, else derived from the path.

    Training never reads ``entry`` -- it groups by ``_native_id_from_path(pdb_path)``,
    and the builder writes ``entry`` to match that ("{pdb_id}_{chains}"). So either
    source gives the same key for a builder-made manifest, while a hand-written or
    externally produced one (only ``pdb_path`` / ``model_source`` / ``dockq`` are
    required to SCORE it) stays evaluable instead of dying on a missing column.
    """
    if "entry" in df.columns:
        return df["entry"].tolist()
    return [_native_id_from_path(p) for p in df["pdb_path"]]


def evaluate(scored: pd.DataFrame, prefix: str) -> Dict[str, float]:
    """Compute the comprehensive metric set over the scored rows.

    ``native_id`` is the complex key the per-native (per-complex) ranking metrics
    group by -- the deployment-relevant question is ordering the decoys WITHIN a
    complex, so this grouping is what the headline number depends on.
    """
    metrics = compute_comprehensive_metrics(
        {"dockq": torch.tensor(scored["dockq_pred"].to_numpy(), dtype=torch.float32)},
        {"dockq": torch.tensor(scored["dockq"].to_numpy(), dtype=torch.float32)},
        {
            "model_source": scored["model_source"].tolist(),
            "native_id": native_ids(scored),
        },
        prefix=prefix,
    )
    return {k: float(v) for k, v in metrics.items()}


def print_summary(metrics: Dict[str, float], prefix: str, label: str) -> None:
    """Print the headline metrics; the full set goes to ``metrics.json``."""
    print("\n" + "=" * 80)
    print(f"{label}: {len(metrics)} metrics")
    print("=" * 80)
    for name in (
        "spearman_r",
        "pearson_r",
        "kendall_tau",
        "rmse",
        "mae",
        "per_native_spearman_mean",
        "per_native_spearman_mean_rankable",
        "per_native_n_rankable",
        "top1_success_0_49",
        "auc_0.49",
    ):
        key = prefix + name
        if key in metrics:
            print(f"  {name:36s} {metrics[key]:.4f}")

    for key in sorted(metrics):
        if "model_" in key and key.endswith("_per_native_spearman_mean"):
            source = key.split("model_")[1].split("_per_native_spearman_mean")[0]
            print(f"  per-native Spearman [{source:9s}]  {metrics[key]:.4f}")

    # The DockQ-std breakdown. The pooled per-native mean above is dominated by
    # near-flat complexes, where a Spearman over decoys that all score alike is noise
    # around zero -- which is why it reads ~0.25 while the top bin reads ~0.58. The
    # HIGH-variance bin is the one that answers the deployment question (given decoys
    # that genuinely differ, can we order them) and is the number the loss ablations
    # were compared on. Printed because it is otherwise 5 keys out of 113 in the JSON.
    print("-" * 80)
    print(f"  {'DockQ-std bin':14s} {'n_cplx':>7s} {'n_decoys':>9s} {'per-native rho':>15s} {'global rho':>12s}")
    for lo, hi in zip(DOCKQ_STD_BINS[:-1], DOCKQ_STD_BINS[1:]):
        tag = f"{lo:.2f}_{hi:.2f}"
        n_complexes = metrics.get(f"{prefix}stdbin_n_complexes/{tag}")
        if n_complexes is None:
            continue
        per_native = metrics.get(f"{prefix}stdbin_pernative_spearman/{tag}", float("nan"))
        global_rho = metrics.get(f"{prefix}stdbin_global_spearman/{tag}", float("nan"))
        n_decoys = metrics.get(f"{prefix}stdbin_n_decoys/{tag}", 0.0)
        print(f"  {tag:14s} {n_complexes:7.0f} {n_decoys:9.0f} {per_native:15.4f} {global_rho:12.4f}")

    # std_ratio = pred_std / true_std. Well below 1 means the model has compressed the
    # prediction scale, the known cost of a ranking term that overwhelms the Huber
    # term. Ranking metrics alone cannot show it: a compressed prediction can still
    # order decoys correctly, and then calibrated absolute DockQ is what is lost.
    for name in ("pred_std", "true_std", "std_ratio"):
        if prefix + name in metrics:
            print(f"  {name:36s} {metrics[prefix + name]:.4f}")
    print("=" * 80)


def find_run_pointer(output_dir: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    """Locate the ``wandb_run.json`` written by the training run, or return None.

    Searched rather than required, because the usual layout puts the scored sets in
    subdirectories of the training output dir (``<run>/val_easy``), so the pointer is
    one level up. An absent pointer is not an error: the metrics still go to
    ``metrics.json`` and scoring is unaffected.
    """
    if explicit is not None:
        return explicit if explicit.exists() else None
    for candidate in (output_dir / "wandb_run.json", output_dir.parent / "wandb_run.json"):
        if candidate.exists():
            return candidate
    return None


def log_to_wandb(metrics: Dict[str, float], provenance: Dict, pointer_path: Path) -> bool:
    """Add the scored metrics to the training run named by ``pointer_path``.

    Written through the public API (summary + config) rather than by re-initialising
    the run: this appends no history step, works whether or not the training run has
    finished, and cannot disturb a run that is still training.

    Never raises -- ``metrics.json`` on disk is the authoritative artefact, so a
    missing key, an unreachable server or an absent wandb install must not fail a
    scoring pass that has already done the expensive work.
    """
    try:
        import wandb

        pointer = json.loads(pointer_path.read_text())
        api = wandb.Api()
        run = api.run(f"{pointer['entity']}/{pointer['project']}/{pointer['run_id']}")
        run.summary.update(metrics)
        run.config.update(provenance)
        run.update()
        print(f"Logged {len(metrics)} metrics to W&B run {pointer.get('run_name')} ({pointer['run_id']})")
        return True
    except Exception as exc:  # noqa: BLE001 -- see docstring: logging is best-effort
        print(f"W&B logging skipped ({type(exc).__name__}: {exc}); metrics.json is unaffected.")
        return False


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="The config the checkpoint was trained with")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Lightning .ckpt to score with")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest CSV to score (every row is scored)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where predictions.csv and metrics.json go")
    parser.add_argument("--prefix", default=None, help="Metric-name prefix (default: the manifest stem)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the config's data.batch_size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override the config's data.num_workers")
    parser.add_argument(
        "--no-graph-cache",
        action="store_true",
        help="Ignore data.graph_cache_dir -- a single scoring pass gains nothing from writing a cache",
    )
    parser.add_argument(
        "--wandb-run-json",
        type=Path,
        default=None,
        help="wandb_run.json of the run to log into (default: search --output-dir and its parent)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Write metrics.json only; do not touch W&B")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Score a manifest end to end."""
    args = _parse_args(argv)

    config = load_config(args.config)
    data_cfg = config.setdefault("data", {})
    if args.no_graph_cache:
        data_cfg["graph_cache_dir"] = None
    batch_size = args.batch_size or int(data_cfg.get("batch_size", 128))
    num_workers = args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 8))

    df = pd.read_csv(args.manifest).dropna(subset=["dockq"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No rows with a DockQ label in {args.manifest}")
    print(f"Manifest {args.manifest}: {len(df)} labelled decoys, {len(set(native_ids(df)))} complexes")

    # Construct from the config and load the weights, rather than
    # ``load_from_checkpoint``: the model takes a single ``config`` dict, which is not
    # a signature Lightning can reconstruct from saved hparams.
    # ``weights_only=True``: only tensors and plain containers are unpickled, so a
    # checkpoint from elsewhere cannot execute code on load. Nothing here needs more --
    # the config is read from the yaml, not from the checkpoint's hparams.
    model = SimpleGNN(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    print(f"Loaded {args.checkpoint} (epoch {state.get('epoch', '?')}, step {state.get('global_step', '?')})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scored = score_manifest(df, model, config, batch_size, num_workers, device)

    prefix = args.prefix if args.prefix is not None else f"{args.manifest.stem}/"
    metrics = evaluate(scored, prefix)
    print_summary(metrics, prefix, label=str(args.manifest.stem))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "predictions.csv", index=False)
    with open(args.output_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print(f"\nWrote {args.output_dir / 'predictions.csv'} and {args.output_dir / 'metrics.json'}")

    # Push the metrics into the training run, so a held-out score is never separated
    # from the run that produced it. Same provenance fields as the training manifest:
    # path + content hash + row count, plus how many rows actually scored.
    if not args.no_wandb:
        pointer = find_run_pointer(args.output_dir, args.wandb_run_json)
        if pointer is None:
            print("No wandb_run.json found; skipping W&B (pass --wandb-run-json to point at one).")
        else:
            log_to_wandb(
                metrics,
                {
                    f"{prefix}manifest": str(args.manifest),
                    f"{prefix}manifest_sha256": file_sha256(args.manifest),
                    f"{prefix}manifest_rows": len(df),
                    f"{prefix}n_scored": len(scored),
                    f"{prefix}checkpoint": str(args.checkpoint),
                },
                pointer,
            )


if __name__ == "__main__":
    main()
