"""Report the SimpleGNN ablation's metric suite for a Boltz confidence run.

Why this exists
---------------
``train_ablation.py`` logs a large, fixed set of metrics for every SimpleGNN variant
(global Pearson/Spearman/Kendall/NDCG/RMSE/MAE, top-k success at the CAPRI
thresholds, AUCs, distribution shape, per-native and per-folding-model breakdowns,
and a per-std-bin slice).  The confidence fine-tune reported a much smaller set --
global and per-entry Spearman per baseline metric, plus top-1 selection quality --
which answers "did the head beat Boltz's own confidence" but *not* "is this better
or worse than the GNN that was trained on the same complexes".  Answering the second
question from two runs needs the same numbers under the same key names, computed by
the same code, which is what this module arranges: the metric functions themselves
are the vendored copy in :mod:`.gnn_metrics`, and everything here is plumbing from
this trainer's per-decoy rows into the shapes that copy expects.

Both suites are logged, and they do NOT agree by construction
------------------------------------------------------------
:func:`~.trainer.correlation_report` and the GNN suite compute per-native Spearman
under different conventions, so the run now carries two numbers for what sounds like
one quantity:

* ``val/per_entry/dockq_pred`` (this trainer) -- entries with fewer than 3 decoys or
  an undefined correlation are *excluded* from the mean.
* ``val/per_native_spearman_mean`` (GNN convention) -- groups of >= 2 decoys count,
  and a degenerate group scores ``0.0`` and stays in the mean.

So the GNN-convention number is the lower of the two whenever the set contains flat
or tiny groups, and equal when it does not.  The internal split contains
``perturbed`` decoys, some of whose natives are near-flat, so expect a gap there;
on ``val_easy`` the two agreed to 4 decimals.  Keeping both is deliberate -- the
first is what this run's checkpoint selection uses and what the earlier BCE run
logged, the second is what the ablation table holds.

Group semantics match the GNN's, which is the part that makes the comparison fair: a
"native" there pools every folding model's decoys for one complex, and an "entry"
here does the same (``boltz2``/``esmfold2``/``protenix``, plus ``perturbed`` on the
internal split).  Hence these keys line up with the ablation's plain
``per_native_spearman_mean``, and the per-model split below is what lines up with its
``per_native_spearman_mean_permodel``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .gnn_metrics import DOCKQ_STD_BINS, compute_comprehensive_metrics

__all__ = [
    "external_suite_prefix",
    "suite_from_rows",
    "train_std_bin_histogram",
]


def suite_from_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    """Compute the ablation metric suite for ``rows`` under ``prefix``.

    ``rows`` are this trainer's per-decoy records (``entry``, ``source``, ``dockq``,
    ``dockq_pred``, ...), i.e. exactly what is written to ``predictions_*.csv``.  The
    row set is the *gathered* one, so on multiple ranks this must be called with the
    output of :func:`~.distributed.all_gather_rows` -- a per-shard suite would report
    a per-native mean over a fraction of the natives, which is a different and
    noisier number than the one the ablation table holds.

    ``prefix`` is a bare namespace (``"val"``, ``"val_easy"``); the trailing slash the
    metric functions expect is added here so callers cannot get it subtly wrong.
    Returns ``{}`` for an empty row set rather than a dict of NaNs, so a skipped
    evaluation logs nothing instead of a wall of empty charts.

    Only ``dockq_pred`` gets the full suite.  Running it for all nine Boltz baselines
    as well would multiply ~150 keys by nine for no new information: the baselines
    already appear in :func:`~.trainer.correlation_report` on the axes that matter
    for them (global, per-entry, per-entry-rankable, top-1 DockQ).
    """
    if not rows:
        return {}

    predictions = {"dockq": torch.tensor([float(r["dockq_pred"]) for r in rows], dtype=torch.float32)}
    targets = {"dockq": torch.tensor([float(r["dockq"]) for r in rows], dtype=torch.float32)}
    metadata = {
        # The ablation calls this native_id; an entry here IS the native id.
        "native_id": [str(r["entry"]) for r in rows],
        # Folding model, so the per-model breakdown keys are named the same way the
        # ablation names them (model_boltz2_*, model_esmfold2_*, model_protenix_*),
        # with `perturbed` appearing as a fourth "model" on the internal split.
        "model_source": [str(r.get("source", "")) for r in rows],
    }
    suite = compute_comprehensive_metrics(predictions, targets, metadata, prefix=f"{prefix.rstrip('/')}/")
    return {k: float(v) for k, v in suite.items()}


def train_std_bin_histogram(
    records: Mapping[str, Iterable[Mapping[str, Any]]], bins: list[float] = DOCKQ_STD_BINS
) -> dict[str, float]:
    """Count training decoys/complexes per per-native DockQ-std bin.

    The ablation's ``train_std_bin_histogram`` does this over manifest rows; here the
    input is :class:`~.cache_dataset.EntryGroupedCacheDataset`'s ``_records`` mapping
    (entry -> its score rows), which is already grouped by native, so no id parsing is
    needed.  Same bins and same key names, so the two runs' training footprints are
    directly comparable, and the bins line up with the eval-side ``stdbin_*`` slices.

    This is the one place a run records HOW MUCH of its training data sits in each std
    regime -- the direct footprint of the ``perturbed`` decoys, which land in the low-std
    bins.  Without it, a change in ranking metrics cannot be attributed to the data
    versus the loss.
    """
    complex_std = {
        entry: float(np.std([float(r["dockq"]) for r in entry_rows])) for entry, entry_rows in records.items()
    }
    counts = {entry: len(list(entry_rows)) for entry, entry_rows in records.items()}

    hist: dict[str, float] = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        tag = f"{lo:.2f}_{hi:.2f}"
        in_bin = [entry for entry, std in complex_std.items() if lo <= std < hi]
        hist[f"train/stdbin_n_complexes/{tag}"] = float(len(in_bin))
        hist[f"train/stdbin_n_decoys/{tag}"] = float(sum(counts[entry] for entry in in_bin))
    return hist


def external_suite_prefix(cache_dir: Path) -> str:
    """Namespace for the OOD pass's suite keys, derived from the cache path.

    The ablation logs its external set as ``val_easy/``, and this run's OOD cache is
    built from the same ``sabdab_val_easy`` complexes, so using that name puts the two
    on one axis with no renaming at query time.  A cache from anywhere else must not
    borrow the name -- it would silently mislabel a different set as the one the
    ablation table reports -- so anything unrecognised falls back to ``external``,
    matching the prefix :func:`~.tracking.flatten_report` already uses there.
    """
    name = cache_dir.parent.name if cache_dir.name == "confidence_cache" else cache_dir.name
    return "val_easy" if "val_easy" in name else "external"
