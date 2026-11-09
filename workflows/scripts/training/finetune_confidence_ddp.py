r"""Fine-tune the **full Boltz-2 confidence module** plus a DockQ head, on 6 GPUs.

What trains
-----------
All 25.0M parameters of ``model.confidence_module`` (452 tensors) *and* the new
0.20M-parameter :class:`DockQHead`.  The 481.5M-parameter trunk stays frozen -- its
outputs are what the cache holds, so training it would invalidate every cached
``s``/``z``.  The confidence module shares zero parameter tensors with the trunk,
which is what makes this separation exact rather than approximate.

Two learning rates, because the two parts start in very different places: the head
is randomly initialised (``--lr``, default 1e-3) while the confidence module is
pretrained and only needs nudging (``--confidence_lr``, default 1e-5).  A single LR
large enough to train the head would destroy the pretrained module in a few steps.

Why not DDP
-----------
The backward graph is not static -- decoy chunking varies per entry, activation
checkpointing re-runs the forward, and ~140 of 452 tensors get no gradient from a
DockQ-only loss.  Each of those independently breaks a DDP assumption.  Gradients
are therefore averaged by hand after backward; see
``confidock.boltz.finetune.distributed``.

Baseline to beat
----------------
On ``val_easy``, Boltz's ``iptm`` scores 0.599 global Spearman against DockQ but
only **0.038 per-entry**.  Per-entry is the model-selection number and the axis
with headroom.  Compare the ``dockq_pred`` row against every other row of the
report table; ``train_dockq_probe.py`` gives the frozen-module baseline that says
whether unfreezing 25M parameters actually bought anything.

Usage (6 GPUs)::

    workflows/scripts/training/finetune_confidence_ddp.sh runs/confidence_ft_v1

Equivalently, by hand::

    poetry run torchrun --nproc_per_node=6 \
        workflows/scripts/training/finetune_confidence_ddp.py \
        --output_dir runs/confidence_ft_v1 --epochs 2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch

from confidock.boltz.finetune.cache_dataset import (
    EntryGroupedCacheDataset,
    collate_entry_batches,
    split_entries,
)
from confidock.boltz.finetune.distributed import (
    agreed_step_count,
    all_gather_rows,
    average_gradients,
    barrier,
    cleanup_distributed,
    is_main,
    setup_distributed,
    shard_entries,
    sum_scalar,
)
from confidock.boltz.finetune.dockq_head import DockQHead, DockQLoss
from confidock.boltz.finetune.gnn_metrics import DOCKQ_STD_BINS
from confidock.boltz.finetune.metric_suite import (
    external_suite_prefix,
    suite_from_rows,
    train_std_bin_histogram,
)
from confidock.boltz.finetune.tracking import (
    finish_run,
    flatten_report,
    init_run,
    log_metrics,
)
from confidock.boltz.finetune.trainer import (
    batch_targets,
    boltz_confidence_score,
    correlation_report,
    decoy_slice,
    eval_entry_predictions,
    prepare_confidence_module,
    run_head,
)
from confidock.boltz.inference.cache_runner import CacheRunConfig, load_cache_model
from confidock.paths import default_boltz_cache

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE = Path("/data/confidock/sabdab_train/confidence_cache")
DEFAULT_BOLTZ_CACHE = default_boltz_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--boltz_cache", type=Path, default=DEFAULT_BOLTZ_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval_cache_dir", type=Path, default=None)
    parser.add_argument(
        "--external_cache_dir",
        type=Path,
        default=None,
        help="held-out OOD cache (e.g. sabdab_val_easy) scored ONCE after training, "
        "with the best epoch's weights.  Unlike --eval_cache_dir this never replaces "
        "the validation set, so model selection stays on the internal holdout; "
        "selecting on the OOD set would make it a second validation set and the "
        "number would stop being out-of-distribution",
    )
    parser.add_argument(
        "--ft_checkpoint",
        type=Path,
        default=None,
        help="a confidence_ft_best.pt to load before training (or, with --epochs 0, "
        "to score an existing run against --external_cache_dir without retraining)",
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3, help="new DockQ head LR")
    parser.add_argument(
        "--confidence_lr",
        type=float,
        default=1e-5,
        help="pretrained confidence-module LR; much smaller than --lr on purpose",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Composite-loss weights.  Defaults are the SimpleGNN ``comp_sup`` ablation arm
    # (configs/dockq_predictor/ablations/loss_objective.yaml), which won that sweep;
    # keeping the numbers identical is what makes the two models comparable.
    parser.add_argument(
        "--lambda_component",
        type=float,
        default=1.0,
        help="weight on supervising the three DockQ sub-scores individually",
    )
    parser.add_argument(
        "--lambda_rank",
        type=float,
        default=0.5,
        help="weight on the within-entry ranking term; the effective weight is lambda_rank/rank_tau",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help="delta for every Huber term; >=1 is plain MSE here since DockQ is in [0, 1]",
    )
    parser.add_argument(
        "--rank_type",
        choices=DockQLoss.RANK_TYPES,
        default="spearman",
        help="within-entry ranking objective: soft-Spearman on ranks, or listwise ListNet",
    )
    parser.add_argument(
        "--rank_tau",
        type=float,
        default=0.05,
        help="ranking temperature; load-bearing as a weight, since the gradient scales as 1/rank_tau",
    )

    parser.add_argument(
        "--max_decoys_per_batch",
        type=int,
        default=16,
        help="decoys per optimiser step; lower than the frozen path because every chunk now holds a backward graph",
    )
    parser.add_argument(
        "--forward_chunk",
        type=int,
        default=1,
        help="decoys per confidence forward.  1 is not merely conservative: under "
        "activation checkpointing only one chunk's activations are live, so at 600 "
        "tokens chunk=1 fits 8 decoys in 29.4 GiB where chunk=4 OOMs at 4.  Raising "
        "it invalidates the --memory_limit_gib calibration",
    )
    parser.add_argument(
        "--token_aware_batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="size each entry's decoy batch from its token count.  On by default: "
        "peak memory is quadratic in tokens, and a flat cap OOM'd 49%% of batches "
        "(all 324-600 tokens), training the model on small complexes only",
    )
    parser.add_argument(
        "--min_decoys_on_oom",
        type=int,
        default=1,
        help="floor for the OOM retry.  1, not 2: a lone decoy loses the ranking "
        "term but DockQLoss still regresses its DockQ and sub-scores, and at 2 the "
        "largest entries were dropped entirely (11%% of batches, nearly all >=500 tokens)",
    )
    parser.add_argument(
        "--amp_dtype",
        choices=("bf16", "none"),
        default="bf16",
        help="autocast dtype for the confidence forward.  Only pays in combination "
        "with --forward_chunk 1, where 600 tokens/8 decoys goes from OOM to 29.4 "
        "GiB; at chunk=4 bf16 measured slightly worse than fp32.  Safe because "
        "Boltz's own inference runs bf16-mixed and its numerically sensitive ops "
        "force autocast(enabled=False) themselves",
    )
    parser.add_argument(
        "--memory_limit_gib",
        type=float,
        default=34.0,
        help="per-GPU budget for the token-aware decoy cap (A100-40GB: 34)",
    )
    parser.add_argument(
        "--eval_decoys_per_batch",
        type=int,
        default=64,
        help="decoys per eval item; can far exceed --max_decoys_per_batch because "
        "eval holds no backward graph.  Each item re-reads the entry's z (up to "
        "92 MiB at 600 tokens), so a small value multiplies eval's disk reads",
    )
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument(
        "--freeze_confidence",
        action="store_true",
        help="train only the new head (the linear-probe baseline)",
    )

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--max_entries", type=int, default=None)
    parser.add_argument("--log_level", default="INFO")

    # Its own project rather than the GNN's `dockq-predictor`: the two share the
    # DockQ target and the composite loss but nothing else, and mixing them would
    # put runs with incomparable step axes on the same charts.
    parser.add_argument("--wandb_project", default="confidock-boltz_ft")
    parser.add_argument("--wandb_run_name", default=None, help="defaults to the output dir's name")
    parser.add_argument("--wandb_group", default=None, help="group related runs, e.g. a loss sweep")
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--no_wandb", action="store_true", help="disable W&B for this run")
    return parser.parse_args(argv)


def make_loader(dataset, *, num_workers: int, shuffle: bool):
    """Wrap a dataset; ``batch_size=1`` since each item is already a decoy batch."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_entry_batches,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _amp_dtype(args: argparse.Namespace) -> torch.dtype | None:
    """Translate ``--amp_dtype`` into the dtype :func:`run_head` expects."""
    return torch.bfloat16 if args.amp_dtype == "bf16" else None


def _backward_with_retry(
    model,
    head: DockQHead,
    loss_fn: DockQLoss,
    batch,
    args: argparse.Namespace,
    *,
    min_decoys: int = 1,
) -> tuple[dict[str, float] | None, int]:
    """Forward+backward one batch, halving the decoy count on OOM.

    Predicting peak memory from token count alone proved unreliable: an isolated
    forward at 600 tokens / 2 decoys peaks at 29.4 GiB, yet inside the training
    loop the same shape OOMs at *one* decoy, because the allocator is fragmented by
    the preceding entries' wildly different shapes.  Any static budget is therefore
    either too optimistic (and drops data) or too pessimistic (and wastes the
    card).  Retrying smaller measures the real limit at the moment it matters.

    Halving preserves the ranking signal as far as it can -- a pair still compares.
    Below a pair the batch is still worth running: :class:`DockQLoss` skips its
    ranking term when only one decoy is present, and the Huber terms on DockQ and on
    the three sub-scores still teach absolute quality.  That matters because at 600
    tokens a pair frequently does not
    fit under real memory pressure, and 29% of the training entries are >=500
    tokens -- a floor of 2 dropped 11% of batches, concentrated entirely on the
    largest complexes, which is exactly the bias this retry exists to prevent.

    Returns ``(loss_parts, n_halvings)``; ``loss_parts`` is ``None`` only if even
    ``min_decoys`` did not fit.
    """
    n = len(batch)
    halvings = 0
    while n >= min_decoys:
        attempt = batch if n == len(batch) else decoy_slice(batch, 0, n)
        try:
            predictions = run_head(
                model,
                head,
                attempt,
                chunk=args.forward_chunk,
                confidence_requires_grad=not args.freeze_confidence,
                amp_dtype=_amp_dtype(args),
            )
            loss, parts = loss_fn(predictions, batch_targets(attempt))
            (loss / args.accumulate).backward()
            # How many decoys the loss actually saw -- ``len(batch)`` after however
            # many halvings this card needed.  Reported through ``parts`` rather than
            # the return tuple so the counter cannot be silently dropped by a caller
            # that only wants the loss terms; ``train_epoch`` sums only the keys it
            # declares, so an extra key here is inert everywhere else.
            parts["n_decoys_used"] = float(n)
            return parts, halvings
        except torch.OutOfMemoryError:
            # The partial graph from the failed attempt has to go before retrying,
            # or the retry inherits its fragmentation and fails just as fast.
            torch.cuda.empty_cache()
            # Clamp to the floor rather than letting integer halving jump past it:
            # from 3 with a floor of 2, ``n //= 2`` lands on 1 and the loop exits
            # without ever having tried 2.
            n = max(n // 2, min_decoys) if n > min_decoys else n - 1
            halvings += 1

    LOGGER.warning(
        "OOM on %s (%d tokens) even at %d decoys; skipping",
        batch.entry,
        batch.s.shape[1],
        min_decoys,
    )
    torch.cuda.empty_cache()
    return None, halvings


def _attempt_batch(
    model,
    head: DockQHead,
    loss_fn: DockQLoss,
    batch,
    args: argparse.Namespace,
    *,
    device: str,
) -> tuple[dict[str, float] | None, int, int]:
    """Run one batch's forward/backward, or decline it as oversized.

    Returns ``(loss_parts, n_halvings, n_skipped)``.  ``loss_parts`` is ``None``
    when the batch contributed nothing -- either it exceeded ``max_tokens`` or it
    OOM'd even at the decoy floor.  The caller must still take its turn in the
    gradient-averaging collective in that case; see :func:`train_epoch`.
    """
    if args.max_tokens and batch.s.shape[1] > args.max_tokens:
        return None, 0, 1
    batch = batch.to(device)
    parts, halvings = _backward_with_retry(
        model,
        head,
        loss_fn,
        batch,
        args,
        min_decoys=args.min_decoys_on_oom,
    )
    return parts, halvings, 0 if parts is not None else 1


def _log_step(
    totals: dict[str, float],
    n_steps: int,
    args: argparse.Namespace,
    *,
    epoch: int,
    index: int,
    steps: int,
    elapsed: float,
    decoys: dict[str, float],
    last: dict[str, float],
) -> None:
    """Emit running loss means and decoy counts on rank 0, on logging steps.

    ``decoys`` are this rank's running totals (see :func:`train_epoch`) and ``last``
    is the step just finished, so the line answers both "how many decoys is a step
    getting" (the instantaneous pair, which is what moves when the OOM retry bites)
    and "how many on average" (the running mean, which is the number to quote).
    """
    if not args.log_every or n_steps % args.log_every != 0 or not is_main():
        return
    means = {k: v / n_steps for k, v in totals.items()}
    rate = n_steps / max(elapsed, 1e-9)
    offered = decoys["offered"]
    used = decoys["used"]
    LOGGER.info(
        "epoch %d step %d/%d loss %.4f (dockq %.4f comp %.4f rank %.4f) "
        "decoys %d/%d this step, %.1f/%.1f mean (%.0f%% used, %.0f%% ranked) %.3f entries/s/rank",
        epoch,
        index + 1,
        steps,
        means["loss"],
        means["huber_dockq"],
        means["huber_comp"],
        means["rank"],
        int(last["used"]),
        int(last["offered"]),
        used / n_steps,
        offered / max(decoys["steps_offered"], 1.0),
        100.0 * used / max(offered, 1.0),
        100.0 * decoys["ranked"] / max(used, 1.0),
        rate,
    )
    # Running means, not the instantaneous batch: with one entry per step the
    # per-batch loss swings by more than the trend it is being read for.  The decoy
    # counters follow the same convention, and `decoys_used_frac`/`decoys_ranked_frac`
    # are the two that need watching: the first falls when the OOM retry halves
    # batches, the second when it halves them down to a single decoy, at which point
    # those steps train absolute quality with no ranking signal at all.
    log_metrics(
        {f"train/{k}": v for k, v in means.items()}
        | {
            "train/entries_per_s_per_rank": rate,
            "train/decoys_offered_per_step": offered / max(decoys["steps_offered"], 1.0),
            "train/decoys_used_per_step": used / n_steps,
            "train/decoys_used_frac": used / max(offered, 1.0),
            "train/decoys_ranked_frac": decoys["ranked"] / max(used, 1.0),
            "epoch": epoch,
        },
        step=(epoch - 1) * steps + index + 1,
    )


def _cycled_batches(loader, steps: int) -> Iterator:
    """Yield exactly ``steps`` batches, restarting ``loader`` if it runs out.

    The epoch length is the *longest* shard across ranks (see
    :func:`~confidock.boltz.finetune.distributed.agreed_step_count`), so a rank
    with fewer items than that has to keep going past the end of its own shard rather
    than leave its peers blocked in an all-reduce.  Restarting the loader reshuffles
    it, so the padding steps are a random sample of this rank's items instead of its
    first few in order.  The first pass is what guarantees coverage: every item on
    every rank is served before any is served twice.
    """
    yielded = 0
    while yielded < steps:
        empty = True
        for batch in loader:
            empty = False
            yield batch
            yielded += 1
            if yielded == steps:
                return
        if empty:
            # Nothing to pad with; without this the loop would spin forever.
            raise RuntimeError(f"train loader is empty but {steps} steps were agreed")


def train_epoch(
    model,
    head: DockQHead,
    loss_fn: DockQLoss,
    optimizer: torch.optim.Optimizer,
    loader,
    args: argparse.Namespace,
    *,
    epoch: int,
    device: str,
    trainable: list[torch.nn.Parameter],
) -> dict[str, float]:
    """One distributed epoch with manual gradient averaging.

    Every rank walks a step count agreed up front and reaches
    :func:`average_gradients` on exactly the same iterations.  This matters more
    than it looks: a rank that skips past an accumulate boundary -- because its
    entry was too big or OOM'd -- performs one fewer all-reduce than its peers,
    and every subsequent collective is then matched against the wrong one.  The
    symptom is not a crash but a 600 s NCCL watchdog timeout with hundreds of
    enqueued and a handful of completed operations.  So a skipped batch still takes
    its turn in the collective, contributing the zero gradient it in fact has.
    """
    head.train()
    if not args.freeze_confidence:
        model.confidence_module.train()

    # Shards differ in item count even when entry counts match (decoys per entry
    # vary), so the longest shard sets the epoch length for everyone and the shorter
    # ones wrap round to fill it.  That way no shard is truncated: every item of every
    # rank is trained on in every epoch, at the cost of the short shards re-visiting
    # `n_padded` of their items.
    steps = agreed_step_count(
        len(loader),
        accumulate=args.accumulate,
        limit=args.limit_train_batches,
        device=device,
    )
    n_padded = max(0, steps - len(loader))
    if is_main():
        LOGGER.info(
            "epoch %d: %d steps/rank (longest shard; %d items of this rank's %d re-visited to fill it)",
            epoch,
            steps,
            n_padded,
            len(loader),
        )

    totals: dict[str, float] = {"loss": 0.0, "huber_dockq": 0.0, "huber_comp": 0.0, "rank": 0.0}
    # Decoy accounting, per rank.  `offered` is what the dataloader handed over
    # (already sized by decoy_budget from the entry's token count), `used` is what
    # reached the loss after any OOM halving, and `ranked` is the subset that reached
    # the ranking term, which needs at least a pair.  `steps_offered` counts every
    # step including declined ones, so the offered mean is not inflated by dropping
    # the batches that were hardest to fit.
    decoys: dict[str, float] = {"offered": 0.0, "used": 0.0, "ranked": 0.0, "steps_offered": 0.0, "singles": 0.0}
    n_steps = 0
    n_skipped = 0
    n_halved = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for index, batch in enumerate(_cycled_batches(loader, steps)):
        # Read before the attempt: `_attempt_batch` may decline the batch entirely,
        # and a declined batch's decoys are exactly what needs counting.
        offered = float(len(batch))
        parts, halvings, skipped = _attempt_batch(model, head, loss_fn, batch, args, device=device)
        n_halved += halvings
        n_skipped += skipped
        # A decoy count is a whole number; it is a float only because it rides back
        # through `parts` alongside the loss terms.  Keeping the int form is what lets
        # the single-decoy steps be counted without comparing floats for equality.
        n_used = int(parts["n_decoys_used"]) if parts is not None else 0
        used = float(n_used)
        decoys["offered"] += offered
        decoys["steps_offered"] += 1.0
        decoys["used"] += used
        decoys["ranked"] += used if n_used >= 2 else 0.0
        decoys["singles"] += 1.0 if n_used == 1 else 0.0
        last = {"offered": offered, "used": used}

        if (index + 1) % args.accumulate == 0:
            # Average across ranks *before* clipping, so every rank clips the same
            # gradient and the optimiser states stay identical.
            average_gradients(trainable)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if parts is None:
            continue

        for key in totals:
            totals[key] += parts[key]
        n_steps += 1

        _log_step(
            totals,
            n_steps,
            args,
            epoch=epoch,
            index=index,
            steps=steps,
            elapsed=time.time() - t0,
            decoys=decoys,
            last=last,
        )

    out = {k: v / max(n_steps, 1) for k, v in totals.items()}
    out["n_steps"] = sum_scalar(float(n_steps), device)
    out["n_skipped"] = sum_scalar(float(n_skipped), device)
    out["n_halved"] = sum_scalar(float(n_halved), device)
    # Whole-epoch decoy totals across every rank, so the epoch line reports the pool
    # the model was actually trained on rather than one shard's view of it.
    out["n_decoys_offered"] = sum_scalar(decoys["offered"], device)
    out["n_decoys_used"] = sum_scalar(decoys["used"], device)
    out["n_decoys_ranked"] = sum_scalar(decoys["ranked"], device)
    out["n_single_decoy_steps"] = sum_scalar(decoys["singles"], device)
    # How much of the epoch was padding: items re-visited so a short shard could reach
    # the agreed step count.  Zero on the longest shard, and the whole fleet's total is
    # what to watch -- if it grows past a few percent the split is badly unbalanced.
    out["n_padded_steps"] = sum_scalar(float(n_padded), device)
    out["seconds"] = time.time() - t0
    # The planned per-rank count, so the caller can put epoch-level metrics on the
    # same monotonic step axis the per-step logging above uses.
    out["steps_planned"] = float(steps)
    return out


def _prediction_rows(batch, predictions: dict) -> list[dict]:
    """One row per decoy: the true DockQ, the head's outputs, and the cached baselines.

    Everything is pulled to the CPU once per entry rather than once per decoy: the
    tensors are per-batch, so indexing them on the device would mean one sync per row.
    """
    pred = predictions["dockq"].float().cpu().numpy()
    # The three sub-scores are kept per decoy: which one ranks a given native
    # best varies by complex, so an offline best-of-three analysis needs them.
    comp = predictions["components"].float().cpu().numpy()
    dockq = batch.dockq.float().cpu().numpy()
    rows: list[dict] = []
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
        # Boltz's own headline scalar, rebuilt from the cached ingredients: it is
        # the axis every pre-fine-tuning val_easy number is quoted under, so the
        # CSV should not make a reader recombine it.
        row["confidence_score"] = boltz_confidence_score(row)
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_distributed(
    model, head: DockQHead, loader, args: argparse.Namespace, *, device: str
) -> tuple[dict, list[dict]]:
    """Score this rank's val shard, gather all rows, and report on the whole set.

    ``model.confidence_module.eval()`` is load-bearing, not tidiness: triangular
    attention is chunked only when ``not self.training`` (``pairformer.py:174-180``),
    so a module left in ``train()`` by ``--unfreeze_confidence`` evaluates on the
    unchunked path.  Measured at 600 tokens under ``no_grad``/bf16, 4 decoys peak at
    16.79 GiB in ``train()`` versus 10.43 in ``eval()``, and 8 decoys OOM in
    ``train()`` where ``eval()`` fits in 20.0.
    """
    head.eval()
    model.confidence_module.eval()
    rows: list[dict] = []
    n_regrouped = 0

    for index, batch in enumerate(loader):
        if args.limit_val_batches and index >= args.limit_val_batches:
            break
        if args.max_tokens and batch.s.shape[1] > args.max_tokens:
            continue

        batch = batch.to(device)
        predictions, group = eval_entry_predictions(
            model,
            head,
            batch,
            forward_chunk=args.forward_chunk,
            amp_dtype=_amp_dtype(args),
        )
        if predictions is None:
            LOGGER.warning(
                "OOM on val entry %s (%d tokens) even at 1 decoy; skipping",
                batch.entry,
                batch.s.shape[1],
            )
            continue
        if group < len(batch):
            n_regrouped += 1
        rows.extend(_prediction_rows(batch, predictions))

    if n_regrouped:
        LOGGER.info("regrouped %d val entries to fit", n_regrouped)
    all_rows = all_gather_rows(rows)
    return correlation_report(all_rows), all_rows


def log_report(report: dict, label: str) -> None:
    """Log the metric comparison table."""
    per = report.get("per_entry", {})
    glob = report.get("global", {})
    top1 = report.get("top1_mean_dockq", {})
    LOGGER.info(
        "%s: %d decoys / %d entries",
        label,
        report.get("n_decoys"),
        report.get("n_entries"),
    )
    LOGGER.info("  %-16s %8s %10s %10s", "metric", "global", "per_entry", "top1_dockq")
    # Best correlation first, NaN last: a constant metric (e.g. a baseline that does
    # not vary within an entry) correlates to NaN and must not sort above real scores.
    for metric in sorted(per, key=lambda m: 9.0 if math.isnan(per[m]) else -per[m]):
        LOGGER.info(
            "  %-16s %8.4f %10.4f %10.4f",
            metric,
            glob.get(metric, float("nan")),
            per.get(metric, float("nan")),
            top1.get(metric, float("nan")),
        )
    LOGGER.info(
        "  %-16s %8s %10s %10.4f",
        "ORACLE",
        "-",
        "-",
        report.get("oracle_mean_dockq", float("nan")),
    )


def write_predictions(path: Path, rows: list[dict]) -> None:
    """Write per-decoy predictions for offline analysis."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _attach_suite(report: dict, rows: list[dict], prefix: str) -> dict[str, float]:
    """Compute the ablation metric suite, store it in ``report``, return it flat.

    Two consumers, one computation: the flat dict goes to W&B under GNN key names
    (``val/per_native_spearman_mean`` and friends), and the same numbers are stashed
    under ``report["suite"]`` so ``report_*.json`` / ``history.json`` carry them
    offline.  Without the second, comparing a finished run against an ablation arm
    would mean re-deriving the suite from the predictions CSV and hoping the code
    matched.

    ``rows`` must be the gathered set -- see :func:`suite_from_rows`.
    """
    suite = suite_from_rows(rows, prefix)
    if suite:
        report["suite"] = suite
    return suite


def _log_training_footprint(
    train_ds: EntryGroupedCacheDataset, val_ds: EntryGroupedCacheDataset, *, device: str
) -> None:
    """Log the DockQ-std-bin footprint of the training pool and the split sizes.

    Every rank must call this: the datasets are per-rank shards, so the full-pool
    counts come from summing each rank's histogram.  Summing is exact here (shards
    partition the entries, and the ablation logs counts, not averages) and costs ten
    small all-reduces, versus rebuilding the unsharded dataset on rank 0 just to count
    -- which on the current cache is a 45 s scan of 2945 score files.

    A side benefit on multi-rank runs: these are the first collectives of the job, so
    NCCL builds its communicator here, before Boltz is loaded and while device memory
    is still empty, rather than mid-training.
    """
    hist = {key: sum_scalar(value, device) for key, value in sorted(train_std_bin_histogram(train_ds.records).items())}
    item_sizes = train_ds.item_sizes
    sizes = {
        # Named as the ablation names them, so a run comparison lines the two up.
        "n_train_samples": sum_scalar(float(train_ds.n_decoys), device),
        "n_val_samples": sum_scalar(float(val_ds.n_decoys), device),
        "n_train_complexes": sum_scalar(float(len(train_ds.records)), device),
        "n_val_complexes": sum_scalar(float(len(val_ds.records)), device),
        # The planned decoys-per-step distribution, i.e. what token-aware batching
        # decided before any OOM retry.  Compare against `train/decoys_used_per_step`
        # during the epoch: the gap between the two IS the memory pressure.
        "n_train_items": sum_scalar(float(len(item_sizes)), device),
        "train/decoys_per_item_planned": sum_scalar(float(sum(item_sizes)), device)
        / max(sum_scalar(float(len(item_sizes)), device), 1.0),
        "train/single_decoy_items_planned": sum_scalar(float(sum(1 for n in item_sizes if n < 2)), device),
    }
    if not is_main():
        return
    LOGGER.info(
        "train items: %d, planned %.1f decoys/item (%d items hold a single decoy, no ranking term)",
        int(sizes["n_train_items"]),
        sizes["train/decoys_per_item_planned"],
        int(sizes["train/single_decoy_items_planned"]),
    )
    LOGGER.info("train DockQ-std bins (n_complexes | n_decoys):")
    for lo, hi in zip(DOCKQ_STD_BINS[:-1], DOCKQ_STD_BINS[1:]):
        tag = f"{lo:.2f}_{hi:.2f}"
        LOGGER.info(
            "  [%s] %5d | %7d",
            tag,
            int(hist[f"train/stdbin_n_complexes/{tag}"]),
            int(hist[f"train/stdbin_n_decoys/{tag}"]),
        )
    log_metrics(hist | sizes | {"epoch": 0}, step=0)


def _build_optimizer(
    model, head: DockQHead, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Build the two-LR optimiser; returns ``(optimizer, trainable, conf_params)``.

    The head is randomly initialised and the confidence module is pretrained, so they
    get separate learning rates -- a single LR big enough for the head would wreck the
    module in a few steps.  Under ``--freeze_confidence`` there are no confidence
    parameters and the second group is omitted entirely.
    """
    conf_params = [p for p in model.confidence_module.parameters() if p.requires_grad]
    groups = [{"params": list(head.parameters()), "lr": args.lr}]
    if conf_params:
        groups.append({"params": conf_params, "lr": args.confidence_lr})
    # Every group sets its own `lr`, so AdamW's default is never used; passing
    # args.lr as well keeps the effective head LR visible at the call site.
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    trainable = [p for group in groups for p in group["params"]]
    return optimizer, trainable, conf_params


def _build_datasets(
    args: argparse.Namespace, *, rank: int, world_size: int, train_entries: list[str], val_entries: list[str]
) -> tuple[EntryGroupedCacheDataset, EntryGroupedCacheDataset]:
    """Shard the split across ranks and build this rank's train/val datasets."""
    train_ds = EntryGroupedCacheDataset(
        args.cache_dir,
        shard_entries(train_entries, rank, world_size),
        max_decoys_per_batch=args.max_decoys_per_batch,
        token_aware_batching=args.token_aware_batching,
        memory_limit_gib=args.memory_limit_gib,
        shuffle_decoys=True,
        seed=args.seed,
    )

    val_cache = args.eval_cache_dir or args.cache_dir
    if args.eval_cache_dir is None:
        my_val = shard_entries(val_entries, rank, world_size)
    else:
        # A separate eval cache is a standalone set (e.g. val_easy), so take all of
        # its entries rather than the held-out slice of the training split.
        my_val = shard_entries(
            sorted(p.stem for p in (val_cache / "entries").glob("*.pt")),
            rank,
            world_size,
        )
    val_ds = EntryGroupedCacheDataset(
        val_cache,
        my_val,
        max_decoys_per_batch=args.eval_decoys_per_batch,
        shuffle_decoys=False,
        require_label_spread=False,
        seed=args.seed,
    )
    return train_ds, val_ds


def _external_dataset(cache_dir: Path, args: argparse.Namespace, *, rank: int, world_size: int):
    """Build the OOD dataset: every entry in ``cache_dir``, sharded across ranks.

    Returns ``(dataset, entry_ids)``; the ids are the *unsharded* list so rank 0 can
    check them against the training split.
    """
    entries = sorted(p.stem for p in (cache_dir / "entries").glob("*.pt"))
    dataset = EntryGroupedCacheDataset(
        cache_dir,
        shard_entries(entries, rank, world_size),
        max_decoys_per_batch=args.eval_decoys_per_batch,
        shuffle_decoys=False,
        require_label_spread=False,
        seed=args.seed,
    )
    return dataset, entries


def _load_ft_checkpoint(path: Path, model, head: DockQHead, *, device: str) -> int:
    """Load a ``confidence_ft_best.pt`` into ``model.confidence_module`` and ``head``.

    Every rank loads the same file from shared storage rather than broadcasting, which
    keeps this usable from a single-rank scoring run too.  ``confidence_module`` is
    absent from checkpoints written with ``--freeze_confidence``, where the pretrained
    module is already the right weights.  Returns the checkpoint's epoch.

    Both modules are left in ``eval()``, which is the mode a just-loaded checkpoint is
    read in: the two callers either score it straight away or are about to be resumed
    from, and :func:`train_epoch` puts back exactly the parts it trains.  Leaving the
    mode to whatever it happened to be is what makes the difference invisible -- in
    ``eval()`` the confidence pairformer runs dropout-free and chunks its triangular
    attention, so a module left in ``train()`` scores differently *and* needs roughly
    60% more memory (see :func:`evaluate_distributed`).
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    head.load_state_dict({k: v.to(device) for k, v in payload["dockq_head"].items()})
    head.eval()
    if "confidence_module" in payload:
        model.confidence_module.load_state_dict({k: v.to(device) for k, v in payload["confidence_module"].items()})
        model.confidence_module.eval()
    elif is_main():
        LOGGER.info("%s holds no confidence_module (head-only run); keeping pretrained weights", path.name)
    return int(payload.get("epoch", 0))


def _score_external(
    args: argparse.Namespace, model, head: DockQHead, *, rank: int, world_size: int, device: str
) -> None:
    """Score the OOD cache once and write/log the report.  Rank 0 does the writing."""
    external_ds, entries = _external_dataset(args.external_cache_dir, args, rank=rank, world_size=world_size)
    if not entries:
        if is_main():
            LOGGER.error("no entry caches under %s; skipping the OOD pass", args.external_cache_dir / "entries")
        return

    loader = make_loader(external_ds, num_workers=args.num_workers, shuffle=False)
    report, rows = evaluate_distributed(model, head, loader, args, device=device)
    if not is_main():
        return
    log_report(report, f"external OOD ({args.external_cache_dir.parent.name})")
    # The suite goes under `val_easy/` when the cache is the val_easy one, which is the
    # namespace train_ablation.py logs its external set to -- so the ablation arm and
    # this run answer the same query.  The pre-existing `external/*` keys are kept
    # untouched: the BCE run this replaces logged those, and renaming them would break
    # that comparison to fix a different one.
    suite = _attach_suite(report, rows, external_suite_prefix(args.external_cache_dir))
    (args.output_dir / "report_external.json").write_text(json.dumps(report, indent=2))
    write_predictions(args.output_dir / "predictions_external.csv", rows)
    # No step: scored once with the selected weights, so it is a summary of the run
    # rather than a point on the epoch axis.
    log_metrics(flatten_report(report, prefix="external") | suite | {"n_val_easy_samples": float(len(rows))})


def _load_model_and_head(args: argparse.Namespace, *, rank: int, world_size: int, device: str):
    """Load Boltz-2 on this rank and build a head whose weights match every peer.

    Returns ``(model, head, token_s, token_z)``.
    """
    if is_main():
        LOGGER.info("loading Boltz-2 on each rank")
    t0 = time.time()
    model = load_cache_model(
        CacheRunConfig(
            processed_dir=args.cache_dir,
            boltz_cache=args.boltz_cache,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            device=device,
        )
    )
    prepare_confidence_module(model, unfreeze=not args.freeze_confidence)
    if is_main():
        LOGGER.info("model ready in %.1fs", time.time() - t0)

    token_s = model.confidence_module.s_to_z.in_features
    token_z = model.confidence_module.s_to_z.out_features
    head = DockQHead(token_s, token_z, hidden=args.hidden, dropout=args.dropout).to(device)
    # Every rank must start from identical head weights or the averaged gradient
    # is applied to divergent parameters.
    if world_size > 1:
        state = head.state_dict()
        objects = [state if rank == 0 else None]
        torch.distributed.broadcast_object_list(objects, src=0)
        head.load_state_dict({k: v.to(device) for k, v in objects[0].items()})
    return model, head, token_s, token_z


def _save_best(
    args: argparse.Namespace,
    head: DockQHead,
    model,
    report: dict,
    rows: list[dict],
    *,
    epoch: int,
    score: float,
    token_s: int,
    token_z: int,
) -> None:
    """Persist the checkpoint, predictions, and report for a new best epoch."""
    payload = {
        "dockq_head": head.state_dict(),
        "epoch": epoch,
        "per_entry_spearman": score,
        "token_s": token_s,
        "token_z": token_z,
    }
    if not args.freeze_confidence:
        payload["confidence_module"] = model.confidence_module.state_dict()
    torch.save(payload, args.output_dir / "confidence_ft_best.pt")
    # Only the best epoch's predictions are kept: an intermediate epoch is not a
    # result anyone reports, and each CSV is ~34k rows.  The per-epoch reports in
    # history.json still show the whole trajectory.
    write_predictions(args.output_dir / "predictions_best.csv", rows)
    (args.output_dir / "report_best.json").write_text(json.dumps(report, indent=2))
    LOGGER.info("new best per-entry Spearman %.4f (epoch %d)", score, epoch)


def _configure_logging(args: argparse.Namespace, *, rank: int) -> None:
    """Log at the requested level on rank 0 and warnings only on its peers.

    Every rank computes the same averaged scalars, so six ranks at INFO would be five
    copies of every line.  The rank tag stays in the format either way, so the warnings
    that do get through are attributable to the rank that raised them.
    """
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()) if is_main() else logging.WARNING,
        format=f"%(asctime)s r{rank} %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _start_run(args: argparse.Namespace, *, world_size: int) -> None:
    """Rank 0: make the output dir, save the resolved config, open the W&B run.

    Rank 0 only: every rank logs the same averaged scalars, so six ranks would mean
    five redundant runs cluttering the project.
    """
    if not is_main():
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items()} | {"world_size": world_size}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    init_run(
        None if args.no_wandb else args.wandb_project,
        name=args.wandb_run_name or args.output_dir.name,
        group=args.wandb_group,
        config=config,
        tags=args.wandb_tags,
    )


def _resolve_split(args: argparse.Namespace, *, world_size: int) -> tuple[list[str], list[str]] | None:
    """Draw the train/val entry split, identically on every rank.

    ``None`` means the cache held no entries, which is the caller's cue to exit non-zero.
    Rank 0 writes ``split.json`` so a later run can be checked against what this one
    held out -- the split is seeded, but the seed alone does not say which entries
    existed when it was drawn.
    """
    all_entries = sorted(p.stem for p in (args.cache_dir / "entries").glob("*.pt"))
    if args.max_entries:
        all_entries = all_entries[: args.max_entries]
    if not all_entries:
        LOGGER.error("no entry caches under %s", args.cache_dir / "entries")
        return None

    train_entries, val_entries = split_entries(all_entries, val_fraction=args.val_fraction, seed=args.split_seed)
    if is_main():
        LOGGER.info(
            "split: %d train / %d val entries over %d ranks",
            len(train_entries),
            len(val_entries),
            world_size,
        )
        (args.output_dir / "split.json").write_text(json.dumps({"train": train_entries, "val": val_entries}, indent=2))
    return train_entries, val_entries


def _external_overlaps_train(args: argparse.Namespace, train_entries: list[str]) -> bool:
    """Whether ``--external_cache_dir`` shares any entry with the training split.

    Checked before the model loads: an overlapping OOD set is not a degraded result, it
    is a wrong one, and finding out after 40 h of training is too late.
    """
    if args.external_cache_dir is None:
        return False
    shared = {p.stem for p in (args.external_cache_dir / "entries").glob("*.pt")} & set(train_entries)
    if not shared:
        return False
    if is_main():
        LOGGER.error(
            "%d entries are in both --cache_dir's train split and --external_cache_dir (e.g. %s); "
            "the OOD report would be scored on trained-on entries",
            len(shared),
            sorted(shared)[:3],
        )
    return True


def _resume_if_requested(args: argparse.Namespace, model, head: DockQHead, *, device: str) -> None:
    """Load ``--ft_checkpoint`` over the freshly built module and head, if one was given."""
    if args.ft_checkpoint is None:
        return
    loaded_epoch = _load_ft_checkpoint(args.ft_checkpoint, model, head, device=device)
    if is_main():
        LOGGER.info("loaded %s (epoch %d)", args.ft_checkpoint, loaded_epoch)


def _log_trainable(head: DockQHead, trainable: list[torch.nn.Parameter], conf_params: list[torch.nn.Parameter]) -> None:
    """Rank 0: what the optimiser is actually updating, head and confidence module apart."""
    if not is_main():
        return
    LOGGER.info(
        "trainable: head %.2fM + confidence %.2fM = %.2fM params (%d tensors)",
        sum(p.numel() for p in head.parameters()) / 1e6,
        sum(p.numel() for p in conf_params) / 1e6,
        sum(p.numel() for p in trainable) / 1e6,
        len(trainable),
    )


def _baseline_report(
    args: argparse.Namespace, model, head: DockQHead, val_loader, *, device: str
) -> tuple[dict, list[dict]]:
    """Score the eval set before training and write rank 0's epoch-0 artifacts.

    Returns the report and rows, which are what ``--epochs 0`` ends the run with.
    """
    report, rows = evaluate_distributed(model, head, val_loader, args, device=device)
    if is_main():
        log_report(report, "epoch 0 (untrained head)")
        suite = _attach_suite(report, rows, "val")
        (args.output_dir / "report_epoch0.json").write_text(json.dumps(report, indent=2))
        # Step 0 carries the Boltz baselines as well as the untrained head, so every
        # W&B chart starts with the number this run has to beat.
        log_metrics(flatten_report(report) | suite | {"epoch": 0}, step=0)
        # Kept despite `dockq_pred` being a random head here: every other column is
        # Boltz's own confidence output on the eval set, which is the baseline the
        # fine-tuned model has to beat.  Saving it now means the comparison never
        # depends on re-running inference.
        write_predictions(args.output_dir / "predictions_baseline.csv", rows)
    return report, rows


def _log_epoch_stats(stats: dict[str, float], *, epoch: int) -> None:
    """Emit the epoch's two training lines: the loss terms, then the decoy accounting."""
    LOGGER.info(
        "epoch %d: loss %.4f (dockq %.4f comp %.4f rank %.4f) %d steps, %d halved, %d skipped, %.1f min",
        epoch,
        stats["loss"],
        stats["huber_dockq"],
        stats["huber_comp"],
        stats["rank"],
        int(stats["n_steps"]),
        int(stats["n_halved"]),
        int(stats["n_skipped"]),
        stats["seconds"] / 60,
    )
    # Decoy accounting on its own line: with token-aware batching and the OOM
    # retry, "how many decoys did this epoch actually train on" is not derivable
    # from the step count, and the answer decides how much of the ranking loss
    # the run got.  `offered` is what the dataloader served across all ranks.
    LOGGER.info(
        "epoch %d decoys: %d used of %d offered (%.1f%%), %d in the ranking term "
        "(%.1f%%), %.1f used/step, %d single-decoy steps, %d padding steps",
        epoch,
        int(stats["n_decoys_used"]),
        int(stats["n_decoys_offered"]),
        100.0 * stats["n_decoys_used"] / max(stats["n_decoys_offered"], 1.0),
        int(stats["n_decoys_ranked"]),
        100.0 * stats["n_decoys_ranked"] / max(stats["n_decoys_used"], 1.0),
        stats["n_decoys_used"] / max(stats["n_steps"], 1.0),
        int(stats["n_single_decoy_steps"]),
        int(stats["n_padded_steps"]),
    )


def _record_epoch(
    args: argparse.Namespace,
    model,
    head: DockQHead,
    stats: dict[str, float],
    report: dict,
    rows: list[dict],
    *,
    epoch: int,
    best: float,
    history: list[dict],
    token_s: int,
    token_z: int,
) -> float:
    """Rank 0: log the epoch, save its report, append to ``history``, keep it if best.

    Returns the running best per-entry Spearman, unchanged on the other ranks -- only
    rank 0 has the artifacts to compare, and only rank 0 writes the checkpoint.
    """
    score = report["per_entry"].get("dockq_pred", float("nan"))
    if not is_main():
        return best

    _log_epoch_stats(stats, epoch=epoch)
    log_report(report, f"epoch {epoch}")
    suite = _attach_suite(report, rows, "val")
    (args.output_dir / f"report_epoch{epoch}.json").write_text(json.dumps(report, indent=2))
    history.append({"epoch": epoch, "train": stats, "val": report})
    log_metrics(
        flatten_report(report) | suite | {f"train/epoch_{k}": v for k, v in stats.items()} | {"epoch": epoch},
        step=epoch * int(stats["steps_planned"]),
    )

    # A NaN score must never win: it means the report had no rankable
    # entries, not a good epoch.
    if math.isnan(score) or score <= best:
        return best
    _save_best(
        args,
        head,
        model,
        report,
        rows,
        epoch=epoch,
        score=score,
        token_s=token_s,
        token_z=token_z,
    )
    return score


def _write_final_artifacts(
    args: argparse.Namespace, report: dict, rows: list[dict], history: list[dict], *, best: float
) -> None:
    """Rank 0: the last epoch's predictions and report, the history, the run's summary."""
    if not is_main():
        return
    if args.epochs > 0:
        # The final weights' predictions, written unconditionally so the run always
        # ends with one CSV per complex-decoy at a known name: true DockQ, the head's
        # DockQ and its three sub-scores, and every Boltz confidence scalar including
        # the derived `confidence_score`.  `predictions_best.csv` holds the selected
        # epoch instead, which is the same file whenever the last epoch won -- the
        # duplication is worth not making every downstream script branch on which
        # epoch that was.
        write_predictions(args.output_dir / "predictions_final.csv", rows)
        (args.output_dir / "report_final.json").write_text(json.dumps(report, indent=2))
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    LOGGER.info("done; best per-entry Spearman %.4f", best)
    # Summary, not a step metric: this is what a sweep table sorts on.
    log_metrics({"best/per_entry_spearman": best})


def _run_external_pass(
    args: argparse.Namespace, model, head: DockQHead, *, rank: int, world_size: int, device: str
) -> None:
    """Score the OOD cache with the *selected* epoch's weights, if one was configured.

    The training loop leaves the model at the last epoch, which is generally overfit
    relative to the checkpoint, so scoring as-is would understate the OOD number.
    """
    if args.external_cache_dir is None:
        return
    # The checkpoint is written by rank 0, so every rank waits for it before reading.
    barrier()
    checkpoint = args.output_dir / "confidence_ft_best.pt"
    if checkpoint.exists():
        epoch = _load_ft_checkpoint(checkpoint, model, head, device=device)
        if is_main():
            LOGGER.info("OOD pass using the epoch-%d weights", epoch)
    elif is_main():
        # --epochs 0 with --ft_checkpoint lands here by design: the weights loaded
        # at startup are already the ones to score.
        LOGGER.info("no %s; scoring whatever weights are loaded", checkpoint.name)
    _score_external(args, model, head, rank=rank, world_size=world_size, device=device)


def main(argv: list[str] | None = None) -> int:
    """Fine-tune the confidence module and the DockQ head across all GPUs."""
    args = parse_args(argv)
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    _configure_logging(args, rank=rank)
    torch.manual_seed(args.seed + rank)
    _start_run(args, world_size=world_size)

    # ---- split (identical on every rank, then sharded) -------------------
    # Both preflight failures are fatal rather than degrading: nothing to train on, or
    # an OOD set that overlaps the train split.
    split = _resolve_split(args, world_size=world_size)
    if split is None or _external_overlaps_train(args, split[0]):
        cleanup_distributed()
        return 1
    train_entries, val_entries = split

    train_ds, val_ds = _build_datasets(
        args,
        rank=rank,
        world_size=world_size,
        train_entries=train_entries,
        val_entries=val_entries,
    )
    _log_training_footprint(train_ds, val_ds, device=device)

    # ---- model -----------------------------------------------------------
    model, head, token_s, token_z = _load_model_and_head(args, rank=rank, world_size=world_size, device=device)
    _resume_if_requested(args, model, head, device=device)
    optimizer, trainable, conf_params = _build_optimizer(model, head, args)
    _log_trainable(head, trainable, conf_params)

    loss_fn = DockQLoss(
        lambda_component=args.lambda_component,
        lambda_rank=args.lambda_rank,
        huber_delta=args.huber_delta,
        rank_type=args.rank_type,
        rank_tau=args.rank_tau,
    )
    val_loader = make_loader(val_ds, num_workers=args.num_workers, shuffle=False)
    report, rows = _baseline_report(args, model, head, val_loader, device=device)

    history: list[dict] = []
    best = -float("inf")
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train_loader = make_loader(train_ds, num_workers=args.num_workers, shuffle=True)
        stats = train_epoch(
            model,
            head,
            loss_fn,
            optimizer,
            train_loader,
            args,
            epoch=epoch,
            device=device,
            trainable=trainable,
        )
        report, rows = evaluate_distributed(model, head, val_loader, args, device=device)
        best = _record_epoch(
            args,
            model,
            head,
            stats,
            report,
            rows,
            epoch=epoch,
            best=best,
            history=history,
            token_s=token_s,
            token_z=token_z,
        )
        barrier()

    _write_final_artifacts(args, report, rows, history, best=best)
    _run_external_pass(args, model, head, rank=rank, world_size=world_size, device=device)

    if is_main():
        finish_run()
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
