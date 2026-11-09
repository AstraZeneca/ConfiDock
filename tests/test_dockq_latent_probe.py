"""Tests for latent extraction, the probe trainer, and the distributed helpers.

The pieces that need 6 GPUs and a 506M-parameter checkpoint cannot be tested here;
what is covered is everything that decides *correctness* rather than throughput:
the round-trip of the latent artifact, the version guard that stops a stale cache
being trained on silently, train-only standardisation, entry-grouped batching, and
the sharding/averaging arithmetic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "scripts" / "training"))

from confidock.boltz.finetune import distributed as dist_helpers
from confidock.boltz.finetune.cache_dataset import (
    EntryGroupedCacheDataset,
    decoy_budget,
)
from confidock.boltz.finetune.distributed import (
    agreed_step_count,
    average_gradients,
    barrier,
    is_main,
    min_scalar,
    shard_entries,
    sum_scalar,
)
from confidock.boltz.finetune.dockq_head import DockQHead, DockQLoss
from confidock.boltz.finetune.latent_cache import (
    COMPONENT_KEYS,
    POOLING_VERSION,
    load_latents,
    save_latent_shard,
)
from confidock.boltz.finetune.probe import (
    _entry_batches,
    evaluate_probe,
    standardise,
    table_targets,
    train_probe,
)


def _components(dockqs):
    """DockQ sub-scores as they are cached: fnat a fraction, both RMSDs in Angstrom.

    Consistent with the DockQ definition -- fnat rises with quality, the distances
    fall -- but deliberately not constructed so their scaled mean equals ``dockq``,
    so nothing can pass by relying on that identity.
    """
    return {
        "fnat": [min(1.0, d * 1.5) for d in dockqs],
        "irmsd": [max(0.0, 3.0 - 3.0 * d) for d in dockqs],
        "lrmsd": [max(0.0, 15.0 - 15.0 * d) for d in dockqs],
    }


def _write_latents(root, entry, dockqs, *, in_dim=8, iptm=None, version=None, drop_components=False):
    """Write one entry's pooled-latent shard."""
    n = len(dockqs)
    rng = np.random.default_rng(abs(hash(entry)) % 2**32)
    save_latent_shard(
        root / "latents" / f"{entry}.npz",
        entry=entry,
        pooled=torch.from_numpy(rng.normal(size=(n, in_dim)).astype(np.float32)),
        dockq=list(dockqs),
        components=_components(dockqs),
        decoy_ids=[f"src__d{i}" for i in range(n)],
        sources=["boltz2"] * n,
        baseline={"iptm": list(iptm) if iptm is not None else [0.5 + 0.01 * i for i in range(n)]},
    )
    if drop_components:
        # Reproduce a shard written before the composite loss existed.
        import json

        path = root / "latents" / f"{entry}.npz"
        blob = {k: v for k, v in np.load(path, allow_pickle=True).items() if k not in COMPONENT_KEYS}
        blob["metadata"] = json.dumps({"entry": entry, "pooling_version": POOLING_VERSION})
        np.savez_compressed(path, **blob)
    if version is not None:
        # Rewrite the metadata with a bogus version to exercise the guard.
        import json

        path = root / "latents" / f"{entry}.npz"
        blob = dict(np.load(path, allow_pickle=True))
        blob["metadata"] = json.dumps({"entry": entry, "pooling_version": version})
        np.savez_compressed(path, **blob)


class TestLatentRoundTrip:
    def test_round_trips_pooled_and_labels(self, tmp_path):
        _write_latents(tmp_path, "1abc_HL", [0.1, 0.5, 0.9])
        table = load_latents(tmp_path)
        assert len(table) == 3
        assert table.in_dim == 8
        assert np.allclose(table.dockq, [0.1, 0.5, 0.9])
        assert table.entries == ["1abc_HL"]

    def test_concatenates_multiple_entries_with_correct_grouping(self, tmp_path):
        _write_latents(tmp_path, "1abc_HL", [0.1, 0.9])
        _write_latents(tmp_path, "2xyz_HL", [0.2, 0.4, 0.6])
        table = load_latents(tmp_path)
        assert len(table) == 5
        assert sorted(table.entries) == ["1abc_HL", "2xyz_HL"]
        # entry_index must partition the rows, not interleave them.
        counts = np.bincount(table.entry_index)
        assert sorted(counts.tolist()) == [2, 3]

    def test_carries_baseline_metrics(self, tmp_path):
        _write_latents(tmp_path, "1abc_HL", [0.1, 0.9], iptm=[0.3, 0.7])
        table = load_latents(tmp_path)
        assert np.allclose(table.baseline["iptm"], [0.3, 0.7])

    def test_round_trips_the_dockq_sub_scores(self, tmp_path):
        """The composite head is supervised on these, so they must survive the npz.

        They are stored as cached -- ``fnat`` a fraction, the RMSDs in Angstrom --
        because the scaling belongs to ``component_targets``.
        """
        _write_latents(tmp_path, "1abc_HL", [0.0, 0.5, 1.0])
        table = load_latents(tmp_path)
        assert np.allclose(table.fnat, [0.0, 0.75, 1.0])
        assert np.allclose(table.irmsd, [3.0, 1.5, 0.0])
        assert np.allclose(table.lrmsd, [15.0, 7.5, 0.0])

    def test_rejects_a_shard_without_sub_scores(self, tmp_path):
        """A pre-composite shard must fail loudly, not train on missing labels."""
        _write_latents(tmp_path, "1abc_HL", [0.1, 0.9], drop_components=True)
        with pytest.raises(ValueError, match="backfill_dockq_components"):
            load_latents(tmp_path)

    def test_save_rejects_incomplete_components(self, tmp_path):
        with pytest.raises(ValueError, match="components missing"):
            save_latent_shard(
                tmp_path / "latents" / "1abc_HL.npz",
                entry="1abc_HL",
                pooled=torch.zeros(2, 8),
                dockq=[0.1, 0.9],
                components={"fnat": [0.1, 0.9]},
                decoy_ids=["a", "b"],
                sources=["boltz2", "boltz2"],
                baseline={},
            )

    def test_rejects_stale_pooling_version(self, tmp_path):
        _write_latents(tmp_path, "1abc_HL", [0.1, 0.9], version=POOLING_VERSION + 1)
        with pytest.raises(ValueError, match="pooling_version"):
            load_latents(tmp_path)

    def test_missing_dir_is_an_error_not_an_empty_table(self, tmp_path):
        with pytest.raises(ValueError, match="no latent shards"):
            load_latents(tmp_path / "nothing")

    def test_entry_filter_selects_a_subset(self, tmp_path):
        _write_latents(tmp_path, "1abc_HL", [0.1, 0.9])
        _write_latents(tmp_path, "2xyz_HL", [0.2, 0.4])
        table = load_latents(tmp_path, entries=["2xyz_HL"])
        assert table.entries == ["2xyz_HL"]
        assert len(table) == 2

    def test_absent_metric_is_dropped_not_all_nan(self, tmp_path):
        _write_latents(tmp_path, "1abc_HL", [0.1, 0.9])
        table = load_latents(tmp_path)
        assert "iptm" in table.baseline
        assert "ligand_iptm" not in table.baseline


class TestSubset:
    def test_keeps_only_requested_entries_and_remaps_index(self, tmp_path):
        _write_latents(tmp_path, "1abc_HL", [0.1, 0.9])
        _write_latents(tmp_path, "2xyz_HL", [0.2, 0.4, 0.6])
        table = load_latents(tmp_path)
        subset = table.subset({"2xyz_HL"})
        assert subset.entries == ["2xyz_HL"]
        assert len(subset) == 3
        # entry_index must be re-based onto the new (shorter) entries list.
        assert subset.entry_index.max() < len(subset.entries)
        assert np.allclose(sorted(subset.dockq), [0.2, 0.4, 0.6])
        # A mask applied to `dockq` but not to the sub-scores would pair each row
        # with another decoy's labels -- wrong supervision, no error.
        assert np.allclose(sorted(subset.fnat), [0.3, 0.6, 0.9])
        assert len(subset.irmsd) == len(subset)
        assert len(subset.lrmsd) == len(subset)

    def test_subsets_are_disjoint_and_complete(self, tmp_path):
        for i, entry in enumerate(["1abc_HL", "2xyz_HL", "3def_HL"]):
            _write_latents(tmp_path, entry, [0.1 * i, 0.5, 0.9])
        table = load_latents(tmp_path)
        a = table.subset({"1abc_HL"})
        b = table.subset({"2xyz_HL", "3def_HL"})
        assert len(a) + len(b) == len(table)


class TestStandardise:
    def test_train_becomes_zero_mean_unit_variance(self):
        train = np.random.default_rng(0).normal(5, 3, size=(100, 4)).astype(np.float32)
        (out,) = standardise(train)
        assert np.allclose(out.mean(axis=0), 0, atol=1e-5)
        assert np.allclose(out.std(axis=0), 1, atol=1e-4)

    def test_val_uses_train_statistics_only(self):
        """Val must be transformed by train stats, else the split leaks."""
        train = np.zeros((10, 3), dtype=np.float32)
        train[:, 0] = np.arange(10)
        val = np.full((5, 3), 100.0, dtype=np.float32)
        _, val_out = standardise(train, val)
        # If val had been standardised by its own stats it would be all-zero.
        assert not np.allclose(val_out[:, 0], 0.0)

    def test_constant_column_does_not_divide_by_zero(self):
        train = np.ones((10, 2), dtype=np.float32)
        (out,) = standardise(train)
        assert np.isfinite(out).all()


class TestEntryBatches:
    def test_each_batch_is_one_entry(self, tmp_path):
        _write_latents(tmp_path, "1abc_HL", [0.1, 0.9])
        _write_latents(tmp_path, "2xyz_HL", [0.2, 0.4, 0.6])
        table = load_latents(tmp_path)
        batches = _entry_batches(table, shuffle=False)
        assert len(batches) == 2
        for rows in batches:
            assert len(set(table.entry_index[rows].tolist())) == 1

    def test_batches_cover_every_row_exactly_once(self, tmp_path):
        _write_latents(tmp_path, "1abc_HL", [0.1, 0.9])
        _write_latents(tmp_path, "2xyz_HL", [0.2, 0.4, 0.6])
        table = load_latents(tmp_path)
        rows = np.concatenate(_entry_batches(table, shuffle=False))
        assert sorted(rows.tolist()) == list(range(len(table)))


class TestProbeTraining:
    def _table(self, tmp_path, n_entries=6):
        for i in range(n_entries):
            _write_latents(tmp_path, f"{i}abc_HL", [0.1, 0.4, 0.7, 0.95])
        return load_latents(tmp_path)

    def test_training_reduces_loss_on_a_learnable_signal(self, tmp_path):
        """Make the target a function of the features, then check the loss falls."""
        table = self._table(tmp_path)
        # Overwrite feature column 0 with the label so a fit is possible at all.
        table.pooled[:, 0] = table.dockq * 10.0
        head = DockQHead(4, 2, dropout=0.0).to("cpu")
        loss_fn = DockQLoss(lambda_rank=1.0)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0.0)

        first = train_probe(head, loss_fn, opt, table.pooled, table, device="cpu", epoch=1)
        for epoch in range(2, 25):
            last = train_probe(head, loss_fn, opt, table.pooled, table, device="cpu", epoch=epoch)
        assert last["loss"] < first["loss"]

    def test_reports_every_composite_loss_term(self, tmp_path):
        """The epoch log prints these by name, so a renamed key is a KeyError."""
        table = self._table(tmp_path, n_entries=2)
        head = DockQHead(4, 2, dropout=0.0)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)
        stats = train_probe(head, DockQLoss(), opt, table.pooled, table, device="cpu", epoch=1)
        assert {"loss", "huber_dockq", "huber_comp", "rank"} <= set(stats)

    def test_evaluate_returns_one_row_per_decoy(self, tmp_path):
        table = self._table(tmp_path, n_entries=3)
        head = DockQHead(4, 2, dropout=0.0)
        report, rows = evaluate_probe(head, table.pooled, table, device="cpu")
        assert len(rows) == len(table)
        assert report["n_entries"] == 3
        assert "dockq_pred" in report["per_entry"]

    def test_evaluate_keeps_the_three_predicted_sub_scores(self, tmp_path):
        """Which sub-score ranks a native best varies, so all three must be logged."""
        table = self._table(tmp_path, n_entries=2)
        head = DockQHead(4, 2, dropout=0.0)
        _, rows = evaluate_probe(head, table.pooled, table, device="cpu")
        assert {"fnat_pred", "s_irmsd_pred", "s_lrmsd_pred"} <= set(rows[0])
        mean = (rows[0]["fnat_pred"] + rows[0]["s_irmsd_pred"] + rows[0]["s_lrmsd_pred"]) / 3.0
        assert rows[0]["dockq_pred"] == pytest.approx(mean, abs=1e-6)

    def test_evaluate_includes_baseline_metrics_for_comparison(self, tmp_path):
        table = self._table(tmp_path, n_entries=3)
        head = DockQHead(4, 2, dropout=0.0)
        report, rows = evaluate_probe(head, table.pooled, table, device="cpu")
        assert "iptm" in rows[0]
        assert "iptm" in report["global"]


class TestTableTargets:
    """The probe must supervise on exactly what the end-to-end path supervises on."""

    def _table(self, tmp_path):
        _write_latents(tmp_path, "1abc_HL", [0.0, 0.5, 1.0])
        return load_latents(tmp_path)

    def test_scales_the_rmsds_into_the_unit_interval(self, tmp_path):
        table = self._table(tmp_path)
        targets = table_targets(table, np.arange(3), "cpu")
        assert targets["components"].shape == (3, 3)
        assert (targets["components"] >= 0.0).all()
        assert (targets["components"] <= 1.0).all()

    def test_selects_only_the_requested_rows(self, tmp_path):
        table = self._table(tmp_path)
        targets = table_targets(table, np.array([2, 0]), "cpu")
        assert torch.allclose(targets["dockq"], torch.tensor([1.0, 0.0]))
        # Row order must follow `rows`, not the table, or a shuffled entry batch
        # would pair features with the wrong labels.
        assert targets["components"][0, 0] > targets["components"][1, 0]


class TestShardEntries:
    def test_single_rank_gets_everything(self):
        entries = [f"{i}abc_H" for i in range(10)]
        assert shard_entries(entries, 0, 1) == entries

    def test_shards_are_disjoint_and_cover_all_entries(self):
        entries = [f"{i}abc_H" for i in range(23)]
        shards = [shard_entries(entries, r, 6) for r in range(6)]
        flat = [e for s in shards for e in s]
        assert sorted(flat) == sorted(entries)
        assert len(flat) == len(set(flat))

    def test_shards_are_balanced_within_one(self):
        entries = [f"{i}abc_H" for i in range(100)]
        sizes = [len(shard_entries(entries, r, 6)) for r in range(6)]
        assert max(sizes) - min(sizes) <= 1


class TestDistributedFallbacks:
    """Without a process group the helpers must be exact no-ops."""

    def test_is_main_is_true_single_process(self):
        assert is_main()

    def test_sum_scalar_passes_value_through(self):
        # abs=0 asserts bit-exact equality, not a tolerance: the fallback returns
        # `value` untouched, whereas the distributed path round-trips through a
        # float64 tensor.  Anything short of exact would let a real transform pass.
        value = 7.0
        assert sum_scalar(value, device="cpu") == pytest.approx(value, abs=0)

    def test_average_gradients_leaves_grads_untouched(self):
        head = DockQHead(4, 2)
        head.from_pooled(torch.randn(3, 8))["dockq"].sum().backward()
        before = [p.grad.clone() for p in head.parameters() if p.grad is not None]
        average_gradients(list(head.parameters()))
        after = [p.grad for p in head.parameters() if p.grad is not None]
        assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))

    def test_average_gradients_tolerates_missing_grads(self):
        head = DockQHead(4, 2)
        params = list(head.parameters())
        average_gradients(params)  # must not raise: no backward has run
        # Single-process fallback returns before touching anything, so it must not
        # have materialised the zero grads the distributed path creates.
        assert all(p.grad is None for p in params)

    def test_min_scalar_passes_value_through(self):
        value = 3.0
        assert min_scalar(value, device="cpu") == pytest.approx(value, abs=0)

    def test_barrier_without_a_process_group_does_not_raise(self):
        barrier()


class TestBarrierSkipsOneRankGroups:
    """A one-rank group must not reach NCCL, even though a group *is* initialised.

    Found on a 1-GPU job: nothing all-reduces in a one-rank run, so the barrier is
    where NCCL first builds its communicator, and that build allocates device memory.
    At the end of an epoch -- model, optimizer state and the eval's activations still
    resident -- it failed with ``Failed to CUDA calloc async 4 bytes`` and killed a run
    whose training had already finished.  Multi-rank runs must still synchronise, so
    both directions are pinned.
    """

    def _spy(self, monkeypatch, world_size: int) -> list[int]:
        calls: list[int] = []
        monkeypatch.setattr(dist_helpers.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist_helpers.dist, "get_world_size", lambda: world_size)
        monkeypatch.setattr(dist_helpers.dist, "barrier", lambda *a, **k: calls.append(world_size))
        return calls

    def test_one_rank_does_not_call_the_collective(self, monkeypatch):
        calls = self._spy(monkeypatch, 1)
        barrier()
        assert calls == []

    def test_multi_rank_still_synchronises(self, monkeypatch):
        calls = self._spy(monkeypatch, 6)
        barrier()
        assert calls == [6]


class TestAgreedStepCount:
    """Guards the NCCL-hang fix: ranks must agree on the collective count.

    The original bug was a rank skipping past an accumulate boundary, so it ran one
    fewer all-reduce than its peers and every later collective was mismatched --
    surfacing as a 600 s watchdog timeout, not an error.

    The count rounds *up* to a whole accumulation window and takes the maximum over
    ranks, so no shard is truncated and every item is trained on every epoch; the
    caller pads a short shard by wrapping its loader.  Rounding down instead -- what
    this did originally -- silently dropped the longest shard's tail, so these tests
    are also what stops that from creeping back.
    """

    def test_rounds_up_to_a_whole_accumulation_window(self):
        # 10 batches at accumulate=4 leaves 2 dangling; two more steps of padding are
        # cheaper than dropping 2 items, and the epoch still ends on an optimiser step.
        assert agreed_step_count(10, accumulate=4, device="cpu") == 12

    def test_exact_multiple_is_unchanged(self):
        assert agreed_step_count(12, accumulate=4, device="cpu") == 12

    def test_accumulate_one_keeps_every_batch(self):
        assert agreed_step_count(7, accumulate=1, device="cpu") == 7

    def test_limit_caps_the_count(self):
        # The smoke knob stays a hard ceiling, trimmed *down* to whole windows: an
        # epoch that overshot --limit_train_batches would defeat the point of it.
        assert agreed_step_count(100, accumulate=2, limit=5, device="cpu") == 4

    def test_shorter_than_one_window_still_takes_a_step(self):
        # 3 items at accumulate=4: one padded step beats an epoch that trains nothing.
        assert agreed_step_count(3, accumulate=4, device="cpu") == 4

    def test_result_is_identical_for_every_shard_size_in_a_single_process(self):
        # Single-process fallback: no collective, so each call sees only its own
        # length.  The distributed guarantee is max_scalar's; this pins the
        # arithmetic that runs on top of it.
        assert agreed_step_count(9, accumulate=2, device="cpu") == 10
        assert agreed_step_count(8, accumulate=2, device="cpu") == 8

    def test_zero_accumulate_does_not_divide_by_zero(self):
        assert agreed_step_count(5, accumulate=0, device="cpu") == 5

    def test_never_returns_fewer_steps_than_the_shard_has_items(self):
        # The coverage guarantee, as arithmetic: whatever the accumulate window, the
        # agreed count is at least the shard length, so the wrap-around padding can
        # only ever repeat items the rank has already been served this epoch.
        for local in range(1, 20):
            for accumulate in (1, 2, 4, 8):
                assert agreed_step_count(local, accumulate=accumulate, device="cpu") >= local


class TestCycledBatches:
    """The padding side of the coverage guarantee.

    A rank whose shard is shorter than the agreed step count keeps drawing from its own
    loader rather than dropping out of the collectives.  What matters is the order:
    every item once before any item twice, or "no shard is truncated" would not hold.
    """

    def _cycled(self, items, steps):
        import finetune_confidence_ddp as ddp

        return list(ddp._cycled_batches(items, steps))

    def test_short_shard_is_padded_by_wrapping(self):
        assert self._cycled(["a", "b", "c"], 5) == ["a", "b", "c", "a", "b"]

    def test_exact_length_does_not_wrap(self):
        assert self._cycled(["a", "b", "c"], 3) == ["a", "b", "c"]

    def test_stops_early_when_fewer_steps_than_items(self):
        # The --limit_train_batches case: the cap wins and the tail is simply not read.
        assert self._cycled(["a", "b", "c"], 2) == ["a", "b"]

    def test_every_item_is_served_before_any_is_repeated(self):
        served = self._cycled(list(range(7)), 20)
        assert served[:7] == list(range(7))
        assert set(served) == set(range(7))
        assert len(served) == 20

    def test_empty_loader_raises_instead_of_spinning(self):
        # An empty shard with a nonzero agreed count would otherwise loop forever,
        # which in a 6-rank job looks exactly like the NCCL hang this all guards.
        with pytest.raises(RuntimeError, match="empty"):
            self._cycled([], 3)


class TestDecoyBudget:
    """Guards the token-aware batch cap that replaced a flat max_decoys_per_batch.

    The first fine-tune run OOM'd on 49% of its training batches -- 3,324 of 6,792,
    every one between 324 and 600 tokens -- because peak memory is quadratic in
    token count and the cap was flat.  The trainer catches OutOfMemoryError and
    skips, so this was silent: the model trained almost only on small complexes.
    """

    def test_large_entries_get_fewer_decoys_than_small_ones(self):
        assert decoy_budget(600) < decoy_budget(300)

    def test_monotonically_non_increasing_in_tokens(self):
        budgets = [decoy_budget(t) for t in range(100, 700, 25)]
        assert budgets == sorted(budgets, reverse=True)

    def test_stays_under_the_measured_600_token_oom(self):
        """At 600 tokens, 8 decoys peaked at 29.4 GiB and 12 OOM'd at 34.8."""
        assert decoy_budget(600, limit_gib=34.0) < 12

    def test_matches_the_measured_438_token_headroom(self):
        """438 tokens / 16 decoys peaked at 22.9 GiB, so the cap must not bite."""
        assert decoy_budget(438, limit_gib=34.0, max_decoys=16) == 16

    def test_matches_the_measured_544_token_headroom(self):
        """544 / 16 peaked at 35.5 GiB -- just over budget, so this must trim."""
        assert 10 <= decoy_budget(544, limit_gib=34.0, max_decoys=16) < 16

    def test_never_returns_fewer_than_a_comparable_pair(self):
        # A within-entry ranking loss needs two decoys to compare, so the estimate
        # always asks for a pair.  When even that does not fit, it is the trainer's
        # OOM retry -- not this function -- that falls back to a single decoy.
        assert decoy_budget(100_000) >= 2

    def test_the_largest_entries_get_a_usable_batch(self):
        """600 tokens must fit well above a pair, or the >=500-token 29% starves.

        Measured at chunk=1 under bf16: 8 decoys peak at 29.4 GiB and 12 OOM.  This
        is the regression that matters -- the fp32/chunk=4 calibration returned 3
        here, which in a live loop degraded to give-ups on 68% of batches.
        """
        assert 6 <= decoy_budget(600, limit_gib=34.0) <= 11

    def test_respects_the_max_decoys_ceiling(self):
        assert decoy_budget(50, max_decoys=8) == 8

    def test_a_bigger_budget_allows_more_decoys(self):
        assert decoy_budget(600, limit_gib=80.0) > decoy_budget(600, limit_gib=34.0)

    def test_unreadable_entry_falls_back_to_the_smallest_batch(self, tmp_path):
        """A metadata read failure must degrade to slow, never to OOM."""
        bad = tmp_path / "corrupt.pt"
        bad.write_bytes(b"not a torch file")
        tokens = EntryGroupedCacheDataset._entry_tokens(bad)
        assert decoy_budget(tokens) == 2


class _RetryBatch:
    """Stub batch whose decoy count is all ``_backward_with_retry`` inspects."""

    def __init__(self, n_decoys: int, n_tokens: int = 600):
        self.entry = "stub_HL"
        self.s = torch.zeros(1, n_tokens, 4)
        self.dockq = torch.linspace(0.1, 0.9, n_decoys)
        # Raw sub-scores, as the cache stores them: ``batch_targets`` scales them.
        self.fnat = self.dockq * 0.9
        self.irmsd = 3.0 - 3.0 * self.dockq
        self.lrmsd = 15.0 - 15.0 * self.dockq
        self.x_pred = torch.zeros(n_decoys, 1, 4, 3)
        self.decoy_ids = [f"d{i}" for i in range(n_decoys)]
        self.sources = ["stub"] * n_decoys
        self.baseline = {}

    def __len__(self) -> int:
        return len(self.decoy_ids)


class TestBackwardRetryLadder:
    """The OOM retry decides which entries reach the optimiser at all.

    With a floor of 2 it dropped 11% of training batches, nearly all >=500 tokens,
    so the ladder's exact rungs are a correctness concern rather than a tuning
    detail: whichever sizes it declines to try are sizes the largest complexes never
    get trained at.
    """

    @staticmethod
    def _run(n_decoys, *, fits_at, min_decoys):
        """Return (parts, halvings, sizes tried) for a card that fits ``fits_at``."""
        import finetune_confidence_ddp as ddp

        tried: list[int] = []

        def fake_run_head(model, head, batch, **kwargs):
            tried.append(len(batch))
            if len(batch) > fits_at:
                raise torch.OutOfMemoryError("stub")
            components = torch.zeros(len(batch), 3, requires_grad=True)
            return {"components": components, "dockq": components.mean(dim=-1)}

        original = ddp.run_head
        ddp.run_head = fake_run_head
        try:
            parts, halvings = ddp._backward_with_retry(
                None,
                # A real head to match the declared type, though `fake_run_head`
                # above never consults it -- this test is about the retry ladder.
                DockQHead(4, 2, dropout=0.0),
                DockQLoss(lambda_rank=1.0),
                _RetryBatch(n_decoys),
                argparse.Namespace(
                    forward_chunk=2,
                    freeze_confidence=False,
                    accumulate=1,
                    amp_dtype="none",
                ),
                min_decoys=min_decoys,
            )
        finally:
            ddp.run_head = original
        return parts, halvings, tried

    def test_halves_until_it_fits(self):
        parts, halvings, tried = self._run(16, fits_at=4, min_decoys=1)
        assert tried == [16, 8, 4]
        assert halvings == 2
        assert parts is not None

    def test_does_not_halve_past_the_floor(self):
        """From 3 with a floor of 2, plain ``n //= 2`` lands on 1 and never tries 2.

        That silently turned a floor of 2 into "give up", which is how the largest
        entries went missing.
        """
        _, _, tried = self._run(3, fits_at=2, min_decoys=2)
        assert 2 in tried

    def test_a_single_decoy_still_trains(self):
        """One decoy loses the ranking term, but both Huber terms still apply."""
        parts, _, tried = self._run(8, fits_at=1, min_decoys=1)
        assert tried[-1] == 1
        assert parts is not None
        assert parts["rank"] == pytest.approx(0.0)
        assert parts["huber_dockq"] > 0.0
        assert parts["huber_comp"] > 0.0

    def test_reports_how_many_decoys_the_loss_saw(self):
        """``n_decoys_used`` is the size that fit, not the size that was offered.

        This is the counter the per-step log reads, and it is the only record of how
        much of the batch the loss actually got: a run whose entries silently halve
        to a single decoy trains no ranking signal, and nothing else in the loop
        would say so.
        """
        # A count that is only a float because it shares the loss-parts dict, so
        # compare it as the integer it is rather than with a tolerance.
        parts, _, _ = self._run(16, fits_at=4, min_decoys=1)
        assert int(parts["n_decoys_used"]) == 4

        parts, _, _ = self._run(8, fits_at=8, min_decoys=1)
        assert int(parts["n_decoys_used"]) == 8

    def test_gives_up_only_below_the_floor(self):
        parts, _, tried = self._run(4, fits_at=0, min_decoys=1)
        assert parts is None
        assert min(tried) == 1
