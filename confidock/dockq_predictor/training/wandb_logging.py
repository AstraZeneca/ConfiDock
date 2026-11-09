"""W&B provenance and visualisation helpers for the DockQ-predictor training run.

Three things a run needs in W&B beyond the scalar metrics, all of which were logged
by the ablation harness and are re-exported here so the single training entrypoint
records the same information:

* ``file_sha256``            -- pin the exact manifest content a run consumed
* ``train_std_bin_histogram`` -- how much training data sits in each DockQ-std regime
* ``DockQScatterCallback``   -- predicted-vs-true figures and a queryable table

Kept out of the entrypoint so each piece is importable and testable on its own.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch

from ..data.dataset import _native_id_from_path
from .metrics import DOCKQ_STD_BINS

__all__ = ["DockQScatterCallback", "file_sha256", "train_std_bin_histogram"]


def file_sha256(path: Optional[str | Path], length: int = 12) -> Optional[str]:
    """Return a short SHA-256 hex digest of a file, or None if it doesn't exist.

    Logged alongside the manifest path so a run pins the exact data it used -- the
    path alone can drift if the CSV is later regenerated or edited, which silently
    makes two runs incomparable.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def train_std_bin_histogram(entries: Sequence[Dict], bins: Sequence[float] = DOCKQ_STD_BINS) -> Dict[str, float]:
    """Bin the training pool by each complex's TRUE DockQ spread.

    Same bins and same native key as the eval-time ``stdbin_*`` metrics, so the
    train-side footprint lines up with the val-side breakdown. A run therefore
    records how many decoys it trains on in each std regime -- the direct footprint
    of added rigid-body-perturbed decoys, which land in the low-std/flat bins the
    augmentation targets.

    Returns flat ``train/stdbin_n_decoys/<lo>_<hi>`` and
    ``train/stdbin_n_complexes/<lo>_<hi>`` scalars, ready for the W&B config.
    """
    import numpy as np

    by_native: Dict[str, List[float]] = {}
    for entry in entries:
        by_native.setdefault(_native_id_from_path(entry["pdb_path"]), []).append(float(entry["dockq"]))

    complex_std = {native: float(np.std(values)) for native, values in by_native.items()}

    histogram: Dict[str, float] = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        tag = f"{lo:.2f}_{hi:.2f}"
        in_bin = [native for native, std in complex_std.items() if lo <= std < hi]
        histogram[f"train/stdbin_n_complexes/{tag}"] = float(len(in_bin))
        histogram[f"train/stdbin_n_decoys/{tag}"] = float(sum(len(by_native[n]) for n in in_bin))
    return histogram


class DockQScatterCallback(pl.Callback):
    """Log validation predicted-vs-true DockQ visualisations to W&B.

    Accumulates predictions/targets during validation independently of the model's
    own step buffer (the model clears its buffer in its own epoch-end hook), then
    logs, on a configurable cadence:

    - ``val/pred_vs_true`` -- global scatter, coloured by folding model, annotated
      with the epoch's metrics, every ``every_n_epochs`` epochs.
    - ``val/per_native_scatters`` -- small-multiples grid of ``n_native_scatters``
      sampled complexes, each subplot one complex's decoys with its per-native
      Spearman. The complexes are chosen once and reused, so the grid tracks the
      SAME complexes across training rather than resampling each epoch.
    - ``val/pred_vs_true_table_epoch_<n>`` -- a ``wandb.Table`` of the raw rows, so
      the underlying points stay queryable in the UI.
    """

    def __init__(
        self,
        wandb_logger,
        every_n_epochs: int = 5,
        max_points: int = 5000,
        n_native_scatters: int = 10,
        table_every_n_epochs: int = 20,
        random_seed: int = 42,
    ):
        super().__init__()
        self._logger = wandb_logger
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.max_points = int(max_points)
        self.n_native_scatters = int(n_native_scatters)
        self.table_every_n_epochs = max(1, int(table_every_n_epochs))
        self.random_seed = int(random_seed)
        self._sampled_natives: Optional[List] = None
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._models: List = []
        self._natives: List = []

    def _scatter_epoch(self, trainer: pl.Trainer) -> bool:
        return (trainer.current_epoch + 1) % self.every_n_epochs == 0

    def _table_epoch(self, trainer: pl.Trainer) -> bool:
        return (trainer.current_epoch + 1) % self.table_every_n_epochs == 0

    def _active(self, trainer: pl.Trainer) -> bool:
        """Accumulate only when some output is due -- and never during sanity check."""
        if trainer.sanity_checking:
            return False
        return self._scatter_epoch(trainer) or self._table_epoch(trainer)

    def on_validation_epoch_start(self, trainer, pl_module):
        """Reset the per-epoch accumulators."""
        self._preds, self._targets, self._models, self._natives = [], [], [], []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect predictions/targets from the model's ``validation_step`` output."""
        if not self._active(trainer) or not isinstance(outputs, dict):
            return
        if "predictions_dockq" not in outputs or "targets_dockq" not in outputs:
            return
        self._preds.append(outputs["predictions_dockq"].detach().float().cpu())
        self._targets.append(outputs["targets_dockq"].detach().float().cpu())
        self._models.extend(outputs.get("model_source", []))
        self._natives.extend(outputs.get("native_id", []))

    def on_validation_epoch_end(self, trainer, pl_module):
        """Render and log the visualisations due this epoch."""
        if not self._active(trainer) or not self._preds:
            return

        import numpy as np

        preds = torch.cat(self._preds).numpy()
        targets = torch.cat(self._targets).numpy()
        # Metadata is only usable if it lines up 1:1 with the predictions.
        models = list(self._models) if len(self._models) == len(preds) else None
        natives = list(self._natives) if len(self._natives) == len(preds) else None
        epoch = trainer.current_epoch

        # Read the metrics Lightning already computed this epoch, so the figure's
        # annotation matches the logged scalars exactly rather than recomputing them.
        metrics = trainer.callback_metrics

        def _metric(key):
            value = metrics.get(key)
            return float(value) if value is not None else None

        if self._scatter_epoch(trainer):
            self._log_scatter(preds, targets, models, epoch, _metric, np)
            if natives is not None and self.n_native_scatters > 0:
                self._log_per_native_scatters(preds, targets, models, natives, epoch, np)

        if self._table_epoch(trainer):
            self._log_table(preds, targets, models, natives, epoch)

    # -- renderers ----------------------------------------------------------

    _PALETTE = {"boltz2": "#4C72B0", "esmfold2": "#DD8452", "protenix": "#55A868", "perturbed": "#C44E52"}

    def _log_per_native_scatters(self, preds, targets, models, natives, epoch, np) -> None:
        """Small-multiples grid: the decoys of N sampled complexes, one subplot each."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr

        natives_arr = np.asarray(natives)

        # Choose the complexes once (only those with >=2 decoys, so a rank exists).
        if self._sampled_natives is None:
            uniq, counts = np.unique(natives_arr, return_counts=True)
            eligible = sorted(uniq[counts >= 2].tolist())
            rng = np.random.default_rng(self.random_seed)
            k = min(self.n_native_scatters, len(eligible))
            self._sampled_natives = rng.choice(eligible, size=k, replace=False).tolist() if k > 0 else []
        chosen = self._sampled_natives
        if not chosen:
            return

        models_arr = np.asarray(models) if models is not None else None
        ncols = min(5, len(chosen))
        nrows = int(np.ceil(len(chosen) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

        for idx, native in enumerate(chosen):
            ax = axes[idx // ncols][idx % ncols]
            mask = natives_arr == native
            true, pred = targets[mask], preds[mask]
            if models_arr is not None:
                for source in np.unique(models_arr[mask]):
                    sub = models_arr[mask] == source
                    ax.scatter(true[sub], pred[sub], s=18, alpha=0.7, color=self._PALETTE.get(str(source), "#888888"))
            else:
                ax.scatter(true, pred, s=18, alpha=0.7, color="#4C72B0")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            rho = spearmanr(pred, true).correlation if len(pred) >= 2 else float("nan")
            ax.set_title(f"{native}  rho={rho:.2f} (n={int(mask.sum())})", fontsize=8)
            ax.tick_params(labelsize=6)

        for j in range(len(chosen), nrows * ncols):  # blank the unused cells
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(f"Per-complex validation DockQ (epoch {epoch})", fontsize=11)
        fig.supxlabel("True DockQ", fontsize=9)
        fig.supylabel("Predicted DockQ", fontsize=9)
        fig.tight_layout()

        import wandb

        self._logger.experiment.log({"val/per_native_scatters": wandb.Image(fig), "epoch": epoch})
        plt.close(fig)

    def _log_table(self, preds, targets, models, natives, epoch) -> None:
        """Log the raw per-sample rows as a ``wandb.Table`` for this epoch."""
        import wandb

        table = wandb.Table(columns=["epoch", "native_id", "model_source", "pred_dockq", "true_dockq"])
        for i in range(len(preds)):
            table.add_data(
                epoch,
                natives[i] if natives is not None else "",
                models[i] if models is not None else "",
                float(preds[i]),
                float(targets[i]),
            )
        # Unique key per epoch, so successive tables do not overwrite each other.
        self._logger.experiment.log({f"val/pred_vs_true_table_epoch_{epoch}": table, "epoch": epoch})

    def _log_scatter(self, preds, targets, models, epoch, metric_fn, np) -> None:
        """Global predicted-vs-true scatter, coloured by folding model."""
        import matplotlib

        matplotlib.use("Agg")  # headless: no display in a k8s job
        import matplotlib.pyplot as plt

        # Stride-subsample for a readable, lightweight figure. Deterministic: no RNG.
        n = len(preds)
        if n > self.max_points:
            stride = int(np.ceil(n / self.max_points))
            preds, targets = preds[::stride], targets[::stride]
            if models is not None:
                models = list(models)[::stride]

        fig, ax = plt.subplots(figsize=(6, 6))
        if models is not None:
            models_arr = np.asarray(models)
            for source in np.unique(models_arr):
                mask = models_arr == source
                ax.scatter(
                    targets[mask],
                    preds[mask],
                    s=10,
                    alpha=0.4,
                    color=self._PALETTE.get(str(source), "#888888"),
                    label=str(source),
                )
            ax.legend(title="model", loc="upper left", fontsize=8, framealpha=0.9)
        else:
            ax.scatter(targets, preds, s=10, alpha=0.4, color="#4C72B0")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("True DockQ")
        ax.set_ylabel("Predicted DockQ")
        ax.set_title(f"Validation DockQ (epoch {epoch})")

        lines = []
        for label, key in [
            ("per-complex Spearman", "val/per_native_spearman_mean"),
            ("Spearman", "val/spearman_r"),
            ("Pearson", "val/pearson_r"),
            ("RMSE", "val/rmse"),
        ]:
            value = metric_fn(key)
            if value is not None:
                lines.append(f"{label}: {value:.3f}")
        lines.append(f"n = {len(preds)}")
        ax.text(
            0.97,
            0.03,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        fig.tight_layout()

        import wandb

        self._logger.experiment.log({"val/pred_vs_true": wandb.Image(fig), "epoch": epoch})
        plt.close(fig)
