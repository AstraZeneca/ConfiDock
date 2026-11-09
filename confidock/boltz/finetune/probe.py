"""Train the DockQ head on pre-extracted pooled latents.

This is the frozen-confidence-module case: because the latents were computed once
with the module in ``eval()``, an epoch here is a pass over a 640-wide table
rather than 97k confidence-module forwards.  Seconds instead of ~13.5 h, which is
what makes this the right tool for choosing the loss weighting, hidden width and
learning rate before spending GPU-days on the full fine-tune.

It is also the honest baseline for that fine-tune.  If unfreezing 25M parameters
on 2,945 entries does not beat what a 0.2M-parameter MLP extracts from the same
frozen latents, the extra capacity bought overfitting rather than signal.

Batching mirrors the end-to-end path: one optimiser step sees whole entries, so
the within-entry ranking term always has same-native pairs to compare.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from confidock.boltz.finetune.dockq_head import (
    DockQHead,
    DockQLoss,
    component_targets,
)
from confidock.boltz.finetune.latent_cache import LatentTable
from confidock.boltz.finetune.trainer import correlation_report

LOGGER = logging.getLogger(__name__)

__all__ = [
    "evaluate_probe",
    "standardise",
    "table_targets",
    "train_probe",
]


def table_targets(table: LatentTable, rows: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    """Build the composite loss's target dict for ``rows`` of ``table``.

    Mirrors :func:`~.trainer.batch_targets` on the end-to-end path, so the probe and
    the fine-tune supervise on identical quantities and their losses are comparable
    numbers rather than merely similar ones.
    """

    def as_tensor(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(values[rows]).to(device)

    return {
        "dockq": as_tensor(table.dockq),
        "components": component_targets(as_tensor(table.fnat), as_tensor(table.irmsd), as_tensor(table.lrmsd)),
    }


def standardise(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Z-score features using **training** statistics only.

    Fitting the scaler on the validation rows too would leak their distribution
    into the model selection, which on a 315-entry holdout is not a rounding
    error.  Zero-variance columns are left alone rather than divided by ~0.
    """
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return tuple((a - mean) / std for a in (train, *others))


def _entry_batches(table: LatentTable, *, shuffle: bool, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Row indices grouped by entry, so each step is one native's decoys."""
    groups: dict[int, list[int]] = defaultdict(list)
    for row, entry in enumerate(table.entry_index):
        groups[int(entry)].append(row)
    batches = [np.asarray(v, dtype=np.int64) for v in groups.values()]
    if shuffle and rng is not None:
        rng.shuffle(batches)
    return batches


def train_probe(
    head: DockQHead,
    loss_fn: DockQLoss,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    table: LatentTable,
    *,
    device: str = "cuda",
    epoch: int = 0,
    seed: int = 0,
    accumulate: int = 4,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Run one epoch over the latent table; returns mean loss components."""
    head.train()
    rng = np.random.default_rng((seed, epoch))
    batches = _entry_batches(table, shuffle=True, rng=rng)

    totals: dict[str, float] = defaultdict(float)
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for index, rows in enumerate(batches):
        x = torch.from_numpy(features[rows]).to(device)

        predictions = head.from_pooled(x)
        loss, parts = loss_fn(predictions, table_targets(table, rows, device))
        (loss / accumulate).backward()

        if (index + 1) % accumulate == 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for key, value in parts.items():
            totals[key] += value
        n_steps += 1

    out = {k: v / max(n_steps, 1) for k, v in totals.items()}
    out["n_steps"] = float(n_steps)
    return out


@torch.no_grad()
def evaluate_probe(
    head: DockQHead,
    features: np.ndarray,
    table: LatentTable,
    *,
    device: str = "cuda",
    batch_size: int = 4096,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every row and build the same report the end-to-end path produces."""
    head.eval()
    preds = np.empty(len(table), dtype=np.float32)
    # The three sub-scores are kept alongside the mean: which one ranks a given
    # native best varies by complex, so a best-of-three analysis needs all three.
    comps = np.empty((len(table), 3), dtype=np.float32)
    for start in range(0, len(table), batch_size):
        stop = min(start + batch_size, len(table))
        x = torch.from_numpy(features[start:stop]).to(device)
        out = head.from_pooled(x)
        preds[start:stop] = out["dockq"].float().cpu().numpy()
        comps[start:stop] = out["components"].float().cpu().numpy()

    rows: list[dict[str, Any]] = []
    for i in range(len(table)):
        row = {
            "entry": table.entries[int(table.entry_index[i])],
            "decoy_id": table.decoy_ids[i],
            "source": table.sources[i],
            "dockq": float(table.dockq[i]),
            "dockq_pred": float(preds[i]),
            "fnat_pred": float(comps[i, 0]),
            "s_irmsd_pred": float(comps[i, 1]),
            "s_lrmsd_pred": float(comps[i, 2]),
        }
        for key, values in table.baseline.items():
            value = float(values[i])
            if not np.isnan(value):
                row[key] = value
        rows.append(row)

    return correlation_report(rows), rows
