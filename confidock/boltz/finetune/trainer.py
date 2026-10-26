"""Training and evaluation loop for the DockQ head.

The frozen part
---------------
Only :class:`~confidock.boltz.finetune.dockq_head.DockQHead` and (optionally)
the confidence module are trained.  Three things have to be right or the "frozen
trunk" claim is false:

1. ``pred_distogram_logits`` is computed from ``model.distogram_module``, which is
   a **trunk** module, so it runs under ``no_grad``.
2. ``run_confidence_head`` in ``..inference.cache_runner`` is decorated
   ``@torch.no_grad()``; this module calls ``model.confidence_module`` directly.
3. Every module in the confidence stack needs
   ``activation_checkpointing = True``.  Boltz chunks triangular attention only
   when ``not self.training`` (``pairformer.py:180-186``) and otherwise relies on
   checkpointing, which defaults to off -- so the training path gets neither and
   OOMs at 39 GiB on a 352-token entry.  With it on, the same entry peaks at
   8.6 GiB.  This only bites under ``--unfreeze_confidence``; with the module
   frozen its forward runs under ``no_grad`` in ``eval()`` mode and gets the
   chunked path.

Memory
------
Three knobs bound peak memory, and they are not independent:

* ``forward_chunk`` -- a real bound, because activation checkpointing keeps only
  one chunk's activations live at a time.  At 600 tokens, 4 decoys fits in 33.6 GiB
  at chunk=1 and OOMs at 36.6 with chunk=4.
* ``amp_dtype`` -- bf16 autocast, which is *not* a uniform halving: at chunk=4 it
  measured slightly worse than fp32 (34.9 vs 33.6 GiB), because autocast holds bf16
  casts alongside the fp32 masters and the stack's ``autocast(enabled=False)``
  regions stay fp32.  It pays only together with chunk=1, where 600 tokens/8 decoys
  goes from OOM to 29.4 GiB.
* the decoy count per batch (:func:`decoy_budget`), quadratic in tokens.

An earlier version of this docstring claimed ``forward_chunk`` was not a memory
bound.  That measurement varied the chunk while holding the decoy count at a size
that could not fit either way; see the ``_MEM_*`` comments in ``cache_dataset`` for
the fitted model and the two runs the mistake cost.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from confidock.boltz.finetune.cache_dataset import EntryBatch, decoy_budget
from confidock.boltz.finetune.dockq_head import (
    DockQHead,
    DockQLoss,
    component_targets,
    pool_confidence_latents,
)
from confidock.boltz.inference.cache_runner import CONFIDENCE_SCALARS
from confidock.paths import default_boltz_cache

LOGGER = logging.getLogger(__name__)

#: Minimum true-DockQ standard deviation for a native to count as rankable.  Kept
#: numerically identical to ``dockq_predictor.training.metrics.RANKABLE_MIN_STD``
#: so the SimpleGNN ablation and this report measure the same subset; if one moves,
#: both must.
RANKABLE_MIN_STD = 0.05

#: Baseline scalars computed from the cached ones rather than read from the cache.
#: They are reported alongside :data:`CONFIDENCE_SCALARS` wherever a row carries them.
DERIVED_SCALARS: tuple[str, ...] = ("confidence_score",)

__all__ = [
    "DERIVED_SCALARS",
    "RANKABLE_MIN_STD",
    "FinetuneConfig",
    "batch_targets",
    "boltz_confidence_score",
    "correlation_report",
    "decoy_budget",
    "decoy_slice",
    "eval_entry_predictions",
    "evaluate",
    "prepare_confidence_module",
    "run_head",
    "train_one_epoch",
]


def boltz_confidence_score(row: Mapping[str, Any]) -> float:
    """Reconstruct Boltz-2's headline ``confidence_score`` from the cached scalars.

    ``(4 * complex_plddt + iptm) / 5``, falling back to ``ptm`` when ``iptm`` is
    identically zero, which is Boltz's own definition verbatim
    (``boltz/model/models/boltz2.py:1085-1094``).  The caching runner records the
    ingredients but not this combination, and it is the number every pre-fine-tuning
    comparison is quoted under -- the ``sabdab_val_easy`` correlation analysis reports
    ``boltz2`` at 0.130 per-native (0.328 rankable) for ``confidence_score``
    specifically, and Boltz uses it to rank its own samples.  Deriving it here means
    the run's CSVs and tables carry that exact axis instead of leaving the comparison
    to be reassembled by hand later.

    Returns NaN when an ingredient is missing, so a row from an older cache simply
    has no ``confidence_score`` rather than a silently wrong one.
    """
    plddt = row.get("complex_plddt")
    iptm = row.get("iptm")
    ptm = row.get("ptm")
    if plddt is None or (iptm is None and ptm is None):
        return float("nan")

    # Zero iptm is a SENTINEL, not a small measurement: Boltz emits exactly 0.0 when
    # there is no scored interface (monomer, or a ligand-free complex) and falls back
    # to ptm.  So this is deliberately an exact-zero test and must not become a
    # tolerance comparison, which would also swallow a genuinely tiny iptm.  Written
    # as a truthiness check because it is equivalent for every float -- including
    # -0.0 (falsy) and NaN (truthy) -- without a float ``==``.
    iptm_value = float(iptm) if iptm is not None else 0.0
    if iptm_value:
        interface = iptm_value
    else:
        interface = float(ptm) if ptm is not None else float("nan")

    return (4.0 * float(plddt) + interface) / 5.0


def batch_targets(batch: EntryBatch) -> dict[str, torch.Tensor]:
    """Build the composite loss's target dict from one :class:`EntryBatch`.

    The three sub-scores are cached as they are reported -- ``fnat`` as a fraction,
    ``irmsd``/``lrmsd`` in Angstrom -- so the DockQ distance scaling is applied here
    rather than at cache time.  Doing it in one place keeps the training and DDP
    paths from drifting apart on which of the two conventions reaches the loss.
    """
    return {
        "dockq": batch.dockq,
        "components": component_targets(batch.fnat, batch.irmsd, batch.lrmsd),
    }


@dataclass
class FinetuneConfig:
    """Hyperparameters for a DockQ-head fine-tuning run."""

    cache_dir: Path
    output_dir: Path
    #: Where the Boltz-2 weights and CCD files live. Site-specific, so pass it
    #: explicitly (``--boltz_cache``) or set ``CONFIDOCK_BOLTZ_CACHE``; the fallback is
    #: a per-user cache directory, not a shared one.
    boltz_cache: Path = field(default_factory=default_boltz_cache)
    checkpoint: Path | None = None

    epochs: int = 3
    lr: float = 1e-3
    #: Separate, much smaller LR for the pretrained confidence module when
    #: ``unfreeze_confidence`` is set -- the head is random, the module is not.
    confidence_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    #: Optimiser steps to accumulate; one step is one entry, so this is the
    #: effective "entries per update".
    accumulate: int = 4

    hidden: int = 256
    dropout: float = 0.1
    #: Composite-loss weights; see :class:`~..dockq_head.DockQLoss`.  Defaults match
    #: the SimpleGNN ``comp_sup`` ablation arm that won the loss sweep.
    lambda_component: float = 1.0
    lambda_rank: float = 0.5
    huber_delta: float = 1.0
    rank_tau: float = 0.05

    max_decoys_per_batch: int = 34
    val_fraction: float = 0.1
    split_seed: int = 0
    #: Cap the token count.  Cached ``sabdab_train`` entries top out at 600, so
    #: this is a guard for other caches rather than the binding memory constraint
    #: -- see ``forward_chunk``.
    max_tokens: int | None = 768
    #: Decoys per confidence-module forward.  ``z_conf`` is
    #: ``(chunk, L, L, token_z)`` = 5.3 GiB at chunk=34/L=600, so the decoy axis,
    #: not the token axis, is what has to be split to fit a 40 GiB card.
    forward_chunk: int = 4

    #: Train the confidence module too.  Off by default: 25M pretrained params on
    #: ~2,900 entries overfits far more readily than the ~0.2M-param head.
    unfreeze_confidence: bool = False

    device: str = "cuda"
    num_workers: int = 6
    seed: int = 42
    log_every: int = 50
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None

    def to_json(self) -> str:
        """Serialise for the run record."""
        payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        return json.dumps(payload, indent=2, sort_keys=True)


def prepare_confidence_module(model: Any, *, unfreeze: bool = False) -> None:
    """Freeze the trunk, enable checkpointing, and expose the latents.

    Sets ``return_latent_feats`` so ``s_conf``/``z_conf`` come back, and turns on
    ``activation_checkpointing`` throughout the confidence stack -- without which
    a training-mode forward OOMs (see the module docstring).
    """
    for param in model.parameters():
        param.requires_grad_(False)

    head = model.confidence_module
    head.return_latent_feats = True
    for module in head.modules():
        if hasattr(module, "activation_checkpointing"):
            module.activation_checkpointing = True

    if unfreeze:
        for param in head.parameters():
            param.requires_grad_(True)
        head.train()
    else:
        # eval() also disables dropout in the confidence pairformer, so the
        # latents the head learns from match what inference will produce.
        head.eval()


def run_head(
    model: Any,
    dockq_head: DockQHead,
    batch: EntryBatch,
    *,
    use_kernels: bool = False,
    chunk: int = 4,
    confidence_requires_grad: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Score every decoy of ``batch``: ``components`` ``(n, 3)`` and ``dockq`` ``(n,)``.

    The confidence module is invoked with ``multiplicity = n_decoys_in_chunk``,
    which is how Boltz itself scores multiple diffusion samples against one trunk
    state: ``z`` is expanded internally by ``repeat_interleave`` rather than by us
    materialising N copies of an O(L^2) tensor.

    Memory
    ------
    ``z_conf`` alone is ``(multiplicity, L, L, token_z)`` -- 5.3 GiB in fp32 for a
    600-token entry with 34 decoys, before the pairformer's own triangular
    intermediates.  ``max_tokens`` cannot bound this because the cached entries top
    out at 600 tokens *and still OOM* at full multiplicity; the decoy axis is what
    has to be split.  So decoys are processed ``chunk`` at a time and only the
    pooled ``(chunk, in_dim)`` vectors are carried forward.

    When the confidence module is frozen (the default) its forward runs under
    ``no_grad`` and nothing but the ~0.2M-parameter head needs a graph, which is
    what makes a small ``chunk`` cheap rather than merely survivable.  With
    ``--unfreeze_confidence`` the graph is required, so the chunk forward is kept
    grad-enabled and the caller must shrink ``chunk`` accordingly.

    ``amp_dtype``
    -------------
    Passing ``torch.bfloat16`` runs the confidence forward under autocast, which
    roughly halves the stored activations -- the difference between a 600-token
    entry fitting a decoy pair and not fitting one.  This is safe here rather than
    merely expedient: Boltz's own inference path runs ``bf16-mixed``
    (``boltz/main.py:1262``) and every numerically delicate op in the stack --
    triangular attention, the softmax reductions, the confidence losses -- already
    forces ``autocast(enabled=False)`` around itself.  The pooled latents are cast
    back to the head's dtype so the head and its loss stay in fp32.
    """
    n = len(batch)
    amp = torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None)

    with torch.no_grad(), amp:
        pred_distogram_logits = model.distogram_module(batch.z)[:, :, :, 0]

    pooled_parts: list[torch.Tensor] = []
    for start in range(0, n, max(chunk, 1)):
        stop = min(start + max(chunk, 1), n)
        x_chunk = batch.x_pred[start:stop]

        # Pass the cached batch-1 ``feats`` through untouched.  The head applies
        # ``repeat_interleave(multiplicity, 0)`` to ``s``/``z``/``s_inputs`` and to
        # ``token_to_rep_atom`` itself (confidencev2.py:173, 190), so pre-repeating
        # here would repeat twice and fail with a batch mismatch of n*n vs n.
        with torch.set_grad_enabled(confidence_requires_grad), amp:
            out = model.confidence_module(
                s_inputs=batch.s_inputs,
                s=batch.s,
                z=batch.z,
                x_pred=x_chunk,
                feats=batch.feats,
                pred_distogram_logits=pred_distogram_logits,
                multiplicity=stop - start,
                run_sequentially=False,
                use_kernels=use_kernels,
            )
            # ``s_conf``/``z_conf`` come back with batch (stop - start) while
            # ``feats`` stays at 1; pool_confidence_latents expands the masks.
            pooled_parts.append(pool_confidence_latents(out["s_conf"], out["z_conf"], batch.feats))
        del out

    # Leave autocast before the head: its input dtype now matches its fp32 weights,
    # so the loss and the sub-scores are computed at full precision either way.
    pooled = torch.cat(pooled_parts, dim=0).to(dockq_head.norm.weight.dtype)
    return dockq_head.from_pooled(pooled)


def _step_optimizer(optimizer: torch.optim.Optimizer, config: FinetuneConfig) -> None:
    """Clip gradients over every optimised parameter, step, and zero."""
    params = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _log_train_progress(
    totals: dict[str, float], n_steps: int, config: FinetuneConfig, *, epoch: int, elapsed: float
) -> None:
    """Emit the running per-step loss means, if this step is a logging step."""
    if not config.log_every or n_steps % config.log_every != 0:
        return
    LOGGER.info(
        "epoch %d step %d/%s loss %.4f (dockq %.4f comp %.4f rank %.4f) %.2f entries/s",
        epoch,
        n_steps,
        config.limit_train_batches or "?",
        totals["loss"] / n_steps,
        totals["huber_dockq"] / n_steps,
        totals["huber_comp"] / n_steps,
        totals["rank"] / n_steps,
        n_steps / max(elapsed, 1e-9),
    )


def train_one_epoch(
    model: Any,
    dockq_head: DockQHead,
    loss_fn: DockQLoss,
    optimizer: torch.optim.Optimizer,
    loader: Iterable[EntryBatch],
    config: FinetuneConfig,
    *,
    epoch: int,
) -> dict[str, float]:
    """Run one training epoch; returns mean loss components."""
    dockq_head.train()
    totals: dict[str, float] = defaultdict(float)
    n_steps = 0
    n_skipped = 0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    for index, batch in enumerate(loader):
        if config.limit_train_batches and index >= config.limit_train_batches:
            break

        n_tokens = batch.s.shape[1]
        if config.max_tokens and n_tokens > config.max_tokens:
            n_skipped += 1
            continue

        batch = batch.to(config.device)
        try:
            predictions = run_head(
                model,
                dockq_head,
                batch,
                chunk=config.forward_chunk,
                confidence_requires_grad=config.unfreeze_confidence,
            )
            loss, parts = loss_fn(predictions, batch_targets(batch))
            (loss / config.accumulate).backward()
        except torch.OutOfMemoryError:
            # One oversized entry must not end the epoch.
            LOGGER.warning(
                "OOM on %s (%d tokens, %d decoys); skipping",
                batch.entry,
                n_tokens,
                len(batch),
            )
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            n_skipped += 1
            continue

        if (index + 1) % config.accumulate == 0:
            _step_optimizer(optimizer, config)

        for key, value in parts.items():
            totals[key] += value
        n_steps += 1

        _log_train_progress(totals, n_steps, config, epoch=epoch, elapsed=time.time() - t0)

    out = {k: v / max(n_steps, 1) for k, v in totals.items()}
    out["n_steps"] = float(n_steps)
    out["n_skipped"] = float(n_skipped)
    out["seconds"] = time.time() - t0
    return out


def decoy_slice(batch: EntryBatch, start: int, stop: int) -> EntryBatch:
    """Return ``batch`` restricted to decoys ``[start:stop]``, sharing tensors.

    ``s``/``z``/``feats`` are per-entry and untouched, so this costs no extra GPU
    memory -- ``z`` alone is the O(L^2) tensor, and copying it per slice would
    defeat the point.

    The range is arbitrary rather than a prefix because the two callers want
    different things from it.  Training accepts a truncated batch: a smaller decoy
    sample is a valid, if weaker, step.  Evaluation cannot, since every decoy needs
    a score, so it splits the batch into consecutive windows instead.
    """
    clone = copy.copy(batch)
    clone.x_pred = batch.x_pred[start:stop]
    clone.dockq = batch.dockq[start:stop]
    clone.fnat = batch.fnat[start:stop]
    clone.irmsd = batch.irmsd[start:stop]
    clone.lrmsd = batch.lrmsd[start:stop]
    clone.decoy_ids = batch.decoy_ids[start:stop]
    clone.sources = batch.sources[start:stop]
    clone.baseline = {k: v[start:stop] for k, v in batch.baseline.items()}
    return clone


def eval_entry_predictions(
    model: Any,
    dockq_head: DockQHead,
    batch: EntryBatch,
    *,
    forward_chunk: int,
    amp_dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor] | None, int]:
    """Score every decoy of ``batch``, halving the group size on OOM.

    Returns ``(predictions, group_size)``; ``predictions`` is ``None`` only if a
    *single* decoy did not fit, and ``group_size`` is what finally worked, so the
    caller can report how often the full batch had to be split.  Each group's
    ``components``/``dockq`` are concatenated back along the decoy axis, so the
    result is indistinguishable from a single un-split forward.

    The retry matters more here than it looks.  Peak memory grows with token count
    (see the module docstring), so an OOM is not a random event -- it is nearly
    certain at 600 tokens and nearly impossible at 350.  Dropping the entry on OOM,
    as both evaluation paths used to, therefore does not just shrink the report: it
    removes the *largest* complexes from it, and the remaining natives are a
    systematically easier subset.  Every correlation computed from such a run is
    biased upward by an unknown amount, which is indistinguishable from a real
    improvement.  Scoring in smaller groups keeps the entry in the report instead,
    at no cost to the numbers: the confidence module sees the same coordinates
    either way, so group size affects only memory, never the predictions.
    """
    n = len(batch)
    group = n
    while group >= 1:
        try:
            parts = [
                run_head(
                    model,
                    dockq_head,
                    batch if group >= n else decoy_slice(batch, start, min(start + group, n)),
                    chunk=min(forward_chunk, group),
                    amp_dtype=amp_dtype,
                )
                for start in range(0, n, group)
            ]
            return {key: torch.cat([p[key] for p in parts], dim=0) for key in parts[0]}, group
        except torch.OutOfMemoryError:
            # Free the failed attempt's blocks, or the retry inherits its
            # fragmentation and fails just as fast.
            torch.cuda.empty_cache()
            if group == 1:
                return None, group
            group = max(group // 2, 1)
    return None, 0


@contextmanager
def _confidence_in_eval(model: Any) -> Iterator[None]:
    """Put the confidence module in ``eval()`` for the block, restoring the caller's mode.

    The module has to be switched explicitly.  Under ``--unfreeze_confidence``
    :func:`prepare_confidence_module` leaves it in ``train()``, and nothing else in
    :func:`evaluate` would undo that -- ``dockq_head.eval()`` only touches the head.
    The cost is memory, not numerics: Boltz-2 ships the confidence stack with
    ``dropout: 0.0`` (``boltz/main.py:73,85``), but triangular attention is chunked
    only when ``not self.training`` (``pairformer.py:174-180``).  Measured at 600
    tokens under ``no_grad``, bf16: 4 decoys peak at 16.79 GiB in train() versus
    10.43 in eval(), and 8 decoys OOM in train() where eval() fits in 20.0.

    Restoring on the way out matters because a subsequent training epoch still needs
    checkpointing's train-mode path.
    """
    confidence = getattr(model, "confidence_module", None)
    if confidence is None:
        yield
        return

    was_training = confidence.training
    confidence.eval()
    try:
        yield
    finally:
        if was_training:
            confidence.train()


def _entry_rows(batch: EntryBatch, predictions: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    """One record per decoy of ``batch``: true DockQ, the head's outputs, the baselines."""
    scored = len(predictions["dockq"])
    if scored != len(batch):  # pragma: no cover - defensive
        raise RuntimeError(f"scored {scored} of {len(batch)} decoys for {batch.entry}")

    pred = predictions["dockq"].float().cpu().numpy()
    # Keep the three predicted sub-scores per decoy.  On the SimpleGNN side
    # WHICH component ranks a given native best varies by complex (best-of-
    # three rho 0.958 against 0.915 for the best fixed one), so an analysis
    # that only has the mean cannot reproduce that result without re-running
    # the model.
    comp = predictions["components"].float().cpu().numpy()
    dockq = batch.dockq.float().cpu().numpy()

    rows: list[dict[str, Any]] = []
    for i, decoy_id in enumerate(batch.decoy_ids):
        row = {
            "entry": batch.entry,
            "decoy_id": decoy_id,
            "source": batch.sources[i],
            "dockq": float(dockq[i]),
            "dockq_pred": float(pred[i]),
            "fnat_pred": float(comp[i, 0]),
            "s_irmsd_pred": float(comp[i, 1]),
            "s_lrmsd_pred": float(comp[i, 2]),
        }
        for key, values in batch.baseline.items():
            row[key] = float(values[i].item())
        row["confidence_score"] = boltz_confidence_score(row)
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate(
    model: Any,
    dockq_head: DockQHead,
    loader: Iterable[EntryBatch],
    config: FinetuneConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score the validation set and report correlations.

    Returns ``(metrics, rows)``; ``rows`` is one record per decoy so the caller can
    write a CSV and re-analyse without re-running the model.
    """
    dockq_head.eval()
    rows: list[dict[str, Any]] = []
    n_regrouped = 0

    with _confidence_in_eval(model):
        for index, batch in enumerate(loader):
            if config.limit_val_batches and index >= config.limit_val_batches:
                break
            if config.max_tokens and batch.s.shape[1] > config.max_tokens:
                continue

            batch = batch.to(config.device)
            predictions, group = eval_entry_predictions(model, dockq_head, batch, forward_chunk=config.forward_chunk)
            if predictions is None:
                LOGGER.warning(
                    "OOM on val entry %s (%d tokens) even at 1 decoy; skipping",
                    batch.entry,
                    batch.s.shape[1],
                )
                continue
            if group < len(batch):
                n_regrouped += 1

            rows.extend(_entry_rows(batch, predictions))

    if n_regrouped:
        LOGGER.info("regrouped %d val entries to fit", n_regrouped)
    return correlation_report(rows), rows


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation, NaN if either side is constant."""
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    value = spearmanr(x, y).statistic
    return float(value) if value is not None and not math.isnan(value) else float("nan")


def _per_entry_correlations(
    by_entry: dict[str, list[dict[str, Any]]], rankable: set[str], metric: str
) -> tuple[list[float], list[float]]:
    """Spearman of ``metric`` against DockQ within each native.

    Returns ``(all_entries, rankable_only)``.  Entries with fewer than 3 decoys and
    entries whose correlation is undefined are omitted from both rather than
    counted as zero -- see :func:`correlation_report` for why that differs from the
    ablation-side ``metrics.spearman_r``.
    """
    per: list[float] = []
    per_rankable: list[float] = []
    for entry, entry_rows in by_entry.items():
        if len(entry_rows) < 3:
            continue
        value = _spearman(
            np.array([r[metric] for r in entry_rows], dtype=float),
            np.array([r["dockq"] for r in entry_rows], dtype=float),
        )
        if not math.isnan(value):
            per.append(value)
            if entry in rankable:
                per_rankable.append(value)
    return per, per_rankable


def correlation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare ``dockq_pred`` against every baseline confidence metric.

    Reports two families, because they answer different questions:

    * **global** -- one correlation over all decoys pooled.  Inflated by
      between-native variance: a metric can score well here purely by separating
      easy targets from hard ones.
    * **per_entry** -- correlation computed within each native, then averaged.
      This is the model-selection number: given several decoys of *one* target,
      can the metric pick the best?  Boltz's ``iptm`` measures 0.599 global but
      only 0.038 per-entry on ``val_easy``, so this is the axis with headroom.
    * **per_entry_rankable** -- the same average, restricted to natives whose true
      DockQ has standard deviation >= :data:`RANKABLE_MIN_STD`.  On ``val_easy``
      248 of 763 natives have a near-flat DockQ spread (median per-entry std
      0.113, against 0.292 on the internal split, which includes ``perturbed``
      decoys); ranking those is measuring noise, and averaging that noise in
      halves every metric roughly uniformly.  This mirrors
      ``dockq_predictor.training.metrics.RANKABLE_MIN_STD`` so the two metric
      suites report the same quantity.

    Note on the ``per_entry`` definition: entries with fewer than 3 decoys and
    entries whose correlation is undefined (constant on either side) are excluded
    from the mean rather than counted as zero.  ``metrics.spearman_r`` on the
    ablation side returns 0.0 in those cases instead.  On ``val_easy`` no entry
    triggers either branch, so the two agree exactly -- verified to 4 decimals
    across every metric -- but on a set that does contain such entries they will
    not, and this function's number will be the higher of the two.
    """
    if not rows:
        return {"n_decoys": 0, "n_entries": 0, "global": {}, "per_entry": {}}

    metrics = ["dockq_pred", *[m for m in (*CONFIDENCE_SCALARS, *DERIVED_SCALARS) if m in rows[0]]]
    dockq = np.array([r["dockq"] for r in rows], dtype=float)

    report: dict[str, Any] = {
        "n_decoys": len(rows),
        "n_entries": len({r["entry"] for r in rows}),
        "global": {},
        "per_entry": {},
        "per_entry_n": {},
        "per_entry_rankable": {},
        "per_entry_rankable_n": {},
    }

    for metric in metrics:
        values = np.array([r[metric] for r in rows], dtype=float)
        report["global"][metric] = _spearman(values, dockq)

    by_entry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_entry[row["entry"]].append(row)

    rankable = {
        entry
        for entry, entry_rows in by_entry.items()
        if float(np.std([r["dockq"] for r in entry_rows])) >= RANKABLE_MIN_STD
    }
    report["n_entries_rankable"] = len(rankable)

    for metric in metrics:
        per, per_rankable = _per_entry_correlations(by_entry, rankable, metric)
        report["per_entry"][metric] = float(np.mean(per)) if per else float("nan")
        report["per_entry_n"][metric] = len(per)
        report["per_entry_rankable"][metric] = float(np.mean(per_rankable)) if per_rankable else float("nan")
        report["per_entry_rankable_n"][metric] = len(per_rankable)

    # Top-1 selection quality: how good is the decoy each metric would pick?
    report["top1_mean_dockq"] = {}
    for metric in metrics:
        picked = [max(entry_rows, key=lambda r: r[metric])["dockq"] for entry_rows in by_entry.values()]
        report["top1_mean_dockq"][metric] = float(np.mean(picked)) if picked else float("nan")
    report["oracle_mean_dockq"] = float(np.mean([max(r["dockq"] for r in v) for v in by_entry.values()]))

    return report
