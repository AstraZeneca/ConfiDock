"""Multi-GPU support for fine-tuning the confidence module.

Why manual gradient all-reduce instead of ``DistributedDataParallel``
---------------------------------------------------------------------
DDP installs autograd hooks that assume a fixed backward graph per step.  Three
properties of this model break that assumption at once:

1. ``run_head`` loops over decoy chunks, so the number of confidence-module
   forwards per step varies with the entry's decoy count (28 vs 63).  The graph is
   not static, which is exactly what ``static_graph=True`` promises it is.
2. Activation checkpointing (required, or a 600-token entry OOMs) re-runs the
   forward during backward, which DDP's hooks double-count unless static_graph is
   set -- and (1) forbids setting it.
3. Roughly 140 of the confidence module's 452 parameter tensors never receive a
   gradient from a DockQ-only loss, so DDP needs ``find_unused_parameters=True``,
   which is itself incompatible with reentrant checkpointing.

Averaging gradients by hand after backward sidesteps all three: the optimiser sees
the same averaged gradient DDP would have produced, but nothing depends on the
graph being shaped the same way twice.  The cost is one all-reduce per optimiser
step (~100 MB for 25M params), which is negligible against a step that spends
seconds in the confidence module.

Sharding is by **entry**, matching the one-entry-per-step design: every rank holds
whole natives, so the within-entry ranking loss is unaffected by the split.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)

__all__ = [
    "agreed_step_count",
    "all_gather_rows",
    "average_gradients",
    "barrier",
    "cleanup_distributed",
    "is_main",
    "max_scalar",
    "min_scalar",
    "setup_distributed",
    "shard_entries",
    "sum_scalar",
]


def setup_distributed() -> tuple[int, int, int]:
    """Initialise the process group from torchrun's environment.

    Returns ``(rank, world_size, local_rank)``.  Falls back to single-process
    ``(0, 1, 0)`` when not launched under torchrun, so the same code path serves
    both cases.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if not dist.is_initialized():
        # NCCL for the gradient all-reduce; the row gather goes through gloo-style
        # object collectives, which NCCL supports via device tensors.
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    LOGGER.info("rank %d/%d on local GPU %d", rank, world_size, local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Tear the process group down, if there is one."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    """Whether this is rank 0 (the process that writes artifacts)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    """Synchronise all ranks, if there is more than one.

    The ``world_size == 1`` guard matches every other collective here, and it is not
    just tidiness: a one-rank job never all-reduces, so the *barrier* is where NCCL
    first has to build its communicator, and building one allocates device buffers.
    At the end of an epoch the model, its optimizer state and the eval's cached
    activations are still resident, so that allocation fails outright --
    ``ncclUnhandledCudaError: Failed to CUDA calloc async 4 bytes`` -- and takes down
    a run that had already finished training.  Multi-rank runs built the communicator
    during the first gradient average and are unaffected.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def shard_entries(entries: list[str], rank: int, world_size: int) -> list[str]:
    """Round-robin entries across ranks.

    Round-robin rather than contiguous: entry ids are sorted and partly correlate
    with size, so contiguous blocks would give one rank the large tail and leave
    the others waiting at every all-reduce.
    """
    if world_size <= 1:
        return entries
    return [e for i, e in enumerate(entries) if i % world_size == rank]


def average_gradients(params: list[torch.nn.Parameter]) -> None:
    """All-reduce and average gradients across ranks.

    Parameters with ``grad is None`` on this rank are skipped, but a parameter that
    is ``None`` on *some* ranks and not others would desynchronise the collective
    and hang.  So a zero tensor is materialised for any parameter that requires
    grad and lacks one -- the DockQ loss genuinely does not reach ~140 of the
    confidence module's tensors, and which ones depends on the entry.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    world_size = dist.get_world_size()
    for param in params:
        if not param.requires_grad:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size


def sum_scalar(value: float, device: str | torch.device = "cuda") -> float:
    """Sum a Python scalar across ranks (for step/skip counters)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def min_scalar(value: float, device: str | torch.device = "cuda") -> float:
    """Take the minimum of a scalar across ranks."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return float(tensor.item())


def max_scalar(value: float, device: str | torch.device = "cuda") -> float:
    """Take the maximum of a scalar across ranks.

    Used to agree on a step count before an epoch starts.  Every rank **must**
    execute the same number of collectives: shards differ in length (entries have
    different decoy counts, so item counts differ even when entry counts match), and
    a rank that runs out of data early leaves its peers waiting in an all-reduce until
    the 600 s NCCL watchdog kills the job.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def agreed_step_count(
    local_batches: int,
    *,
    accumulate: int,
    limit: int | None = None,
    device: str | torch.device = "cuda",
) -> int:
    """Return the number of batches every rank must consume this epoch.

    Takes the maximum across ranks and rounds *up* to a whole multiple of
    ``accumulate``.  The maximum means no shard is truncated, so every item of every
    rank is trained on in every epoch; the caller owes the padding, i.e. a rank whose
    shard is shorter than the agreed count must keep drawing (see ``_cycled_batches``
    in ``finetune_confidence_ddp``), and those extra steps re-visit items it has
    already seen this epoch.  This deliberately trades a little duplication on the
    short shards for full coverage -- taking the minimum instead dropped the longest
    shard's tail, 2.0% of items per epoch at 6 ranks on the SAbDab split.  Rounding up
    to a multiple of ``accumulate`` guarantees the epoch ends on an optimiser step, so
    no rank is left holding accumulated gradients that the others already applied.

    ``limit`` (the ``--limit_train_batches`` smoke knob) stays a hard ceiling and is
    itself trimmed *down* to a whole number of optimiser steps.
    """
    accumulate = max(accumulate, 1)
    steps = int(max_scalar(float(local_batches), device))
    steps += -steps % accumulate
    if limit:
        steps = min(steps, limit - limit % accumulate)
    return steps


def all_gather_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gather per-decoy prediction dicts from every rank onto every rank.

    The correlation report is computed over the whole validation set, so a rank
    that only saw its own shard would report a per-entry mean over a fraction of
    the natives -- a different and noisier number than the one being compared
    against ``iptm``.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return rows

    gathered: list[list[dict[str, Any]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, rows)
    out: list[dict[str, Any]] = []
    for part in gathered:
        if part:
            out.extend(part)
    return out
