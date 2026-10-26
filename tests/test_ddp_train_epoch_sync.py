"""Two-process regression test for the collective-desync hang in ``train_epoch``.

The bug this pins down: ranks got shards of different length, and a rank could also
skip a batch (oversized entry, or OOM) by jumping past the gradient-averaging block.
Either way it performed fewer all-reduces than its peers, every later collective was
matched against the wrong one, and the job died 600 s later in the NCCL watchdog --
no traceback, no failed assertion, just a hang.  A unit test on the arithmetic
cannot catch that; it needs two processes actually running the loop.

Runs on gloo/CPU with stubs in place of the 506M-parameter model, so it costs a
couple of seconds and no GPU.  Both failure modes are provoked at once: rank 1 gets
a shorter shard *and* an entry that trips the ``max_tokens`` skip.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1] / "workflows" / "scripts"
sys.path.insert(0, str(WORKFLOWS_ROOT / "training"))

from confidock.boltz.finetune.dockq_head import DockQHead, DockQLoss  # noqa: E402


class _StubBatch:
    """Minimal stand-in for ``EntryBatch``: only what ``train_epoch`` touches."""

    def __init__(self, entry: str, n_decoys: int, n_tokens: int):
        self.entry = entry
        self.s = torch.zeros(1, n_tokens, 4)
        self.dockq = torch.linspace(0.1, 0.9, n_decoys)
        # The composite loss reads the three sub-scores too, via ``batch_targets``.
        # In the cached convention: fnat a fraction, both RMSDs in Angstrom.
        self.fnat = torch.linspace(0.2, 0.95, n_decoys)
        self.irmsd = torch.linspace(4.0, 0.5, n_decoys)
        self.lrmsd = torch.linspace(20.0, 2.0, n_decoys)
        self._n = n_decoys

    def __len__(self) -> int:
        return self._n

    def to(self, device):  # noqa: ANN001 - mirrors the real signature
        return self


def _make_loader(n_batches: int, *, oversized_at: int | None = None):
    """A list of stub batches; ``oversized_at`` trips the max_tokens skip."""
    return [
        _StubBatch(
            f"entry{i}",
            n_decoys=4,
            n_tokens=9999 if i == oversized_at else 32,
        )
        for i in range(n_batches)
    ]


def _worker(rank: int, world_size: int, port: int, result_queue) -> None:  # noqa: ANN001
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
    try:
        import finetune_confidence_ddp as ddp

        # The real script broadcasts head state from rank 0; a shared seed is the
        # equivalent here, and is required for the identical-parameters assertion
        # to mean "gradients were averaged" rather than "inits happened to differ".
        torch.manual_seed(0)
        # A real DockQHead rather than a stub, so this exercises train_epoch's
        # declared parameter type.  token_s=4/token_z=2 gives in_dim = 2*2+4 = 8,
        # the width `_run_head` feeds it below; dropout=0 keeps the two ranks
        # deterministic, which the identical-parameters assertion requires.
        head = DockQHead(4, 2, dropout=0.0)
        # Both auxiliary terms on: this test is about collective *counts*, and a
        # term that contributed no gradient would be a term whose all-reduces were
        # never exercised.
        loss_fn = DockQLoss(lambda_component=1.0, lambda_rank=0.5)
        # weight_decay=0 explicitly: this test asserts the two ranks end with
        # identical parameters, so decay must not be a second thing moving them.
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)
        trainable = list(head.parameters())

        # The asymmetry that used to hang: rank 1 has fewer batches AND skips one.
        if rank == 0:
            loader = _make_loader(8)
        else:
            loader = _make_loader(6, oversized_at=2)

        def _run_head(model, dockq_head, batch, **kwargs):  # noqa: ANN001, ARG001
            # Stand in for the real pooling of confidence latents, which needs the
            # 506M-parameter trunk.  Rank-dependent inputs make the per-rank
            # gradients differ, so averaging them is observable.
            pooled = torch.ones(len(batch), dockq_head.in_dim) * (0.1 * (rank + 1))
            return dockq_head.from_pooled(pooled)

        ddp.run_head = _run_head

        class _StubModel:
            confidence_module = torch.nn.Linear(2, 2)

        args = argparse.Namespace(
            freeze_confidence=False,
            max_tokens=768,
            forward_chunk=2,
            accumulate=2,
            grad_clip=1.0,
            limit_train_batches=None,
            log_every=0,
            min_decoys_on_oom=1,
            amp_dtype="none",
        )

        stats = ddp.train_epoch(
            _StubModel(),
            head,
            loss_fn,
            optimizer,
            loader,
            args,
            epoch=1,
            device="cpu",
            trainable=trainable,
        )
        # Parameters must be bit-identical across ranks after averaged steps.  Send a
        # hash of the raw bytes rather than the ~68k parameters themselves: the parent
        # joins before draining the queue, so a payload above the pipe buffer would
        # block the child in queue.put and look exactly like a collective hang.
        flat = torch.cat([p.detach().flatten() for p in trainable])
        digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
        result_queue.put(
            (
                rank,
                stats["n_steps"],
                stats["n_skipped"],
                digest,
                # Decoy accounting: also all-reduced sums, so both ranks must report
                # the same totals for the whole epoch.
                (stats["n_decoys_offered"], stats["n_decoys_used"], stats["n_decoys_ranked"]),
                stats["n_padded_steps"],
            )
        )
    finally:
        dist.destroy_process_group()


def test_unequal_shards_and_skips_do_not_desync_collectives():
    """Would hang (then NCCL-timeout) before the agreed-step-count fix."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(rank, 2, 29613, queue)) for rank in range(2)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=110)

    # A desync manifests as a live process still blocked in a collective.  Kill it
    # before asserting, or a regression leaves orphans holding the port.
    hung = [i for i, proc in enumerate(procs) if proc.is_alive()]
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
    assert not hung, f"ranks {hung} hung in a collective -- shards desynchronised"

    for proc in procs:
        assert proc.exitcode == 0, f"rank died with exitcode {proc.exitcode}"

    results = {}
    while not queue.empty():
        rank, n_steps, n_skipped, digest, decoys, n_padded = queue.get()
        results[rank] = (n_steps, n_skipped, digest, decoys, n_padded)
    assert set(results) == {0, 1}

    # Longest shard is 8, already a multiple of accumulate=2 -> 8 steps each, with rank
    # 1 wrapping its 6-item loader to fill the last two (items 0 and 1 again; its
    # oversized entry sits at index 2 and so is still skipped exactly once).  16
    # attempted steps across the two ranks, 15 contributing.
    # n_steps/n_skipped are all-reduced sums over both ranks -- integer counts that
    # are only floats because they travelled through a float all-reduce, so compare
    # them as the exact integers they are rather than with a tolerance.
    assert int(results[0][0]) == int(results[1][0]) == 15
    assert int(results[0][1]) == int(results[1][1]) == 1

    # 16 attempted steps of 4 decoys are offered across the two ranks; the skipped
    # oversized entry costs 4 of them, and every surviving step has enough decoys for
    # the ranking term.  This is the accounting the per-step log reports, so it is
    # worth pinning here where the skip path is actually exercised.
    assert tuple(int(v) for v in results[0][3]) == tuple(int(v) for v in results[1][3]) == (64, 60, 60)

    # Two of those 16 steps are rank 1's padding.  Zero here would mean the short shard
    # had been truncated instead -- the thing this arrangement is meant to prevent.
    assert int(results[0][4]) == int(results[1][4]) == 2

    # The real payoff: identical parameters, i.e. the gradients really were averaged
    # over matched collectives rather than silently drifting apart.  Comparing digests
    # of the raw bytes is the same bit-identity check as comparing the tensors.
    assert results[0][2] == results[1][2]
