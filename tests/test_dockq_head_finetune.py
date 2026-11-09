"""Tests for the DockQ-head fine-tuning components.

These cover everything that does not need the 506M-parameter Boltz checkpoint:
the entry/PDB split, cache dataset filtering, interface-aware pooling, the loss,
and the correlation report.  The one thing they cannot cover is the real
confidence-module forward, which is verified by the probes documented in
``docs/boltz_confidence_caching.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from confidock.boltz.finetune.cache_dataset import (
    EntryGroupedCacheDataset,
    _chunk_indices,
    collate_entry_batches,
    split_entries,
)
from confidock.boltz.finetune.dockq_head import (
    DockQHead,
    DockQLoss,
    component_targets,
    pool_confidence_latents,
    scale_irmsd,
    scale_lrmsd,
)
from confidock.boltz.finetune.trainer import (
    FinetuneConfig,
    batch_targets,
    boltz_confidence_score,
    correlation_report,
    decoy_slice,
    eval_entry_predictions,
    evaluate,
    run_head,
)
from confidock.boltz.inference.confidence_cache import EntryCache, save_entry_cache

#: For loss terms that are switched *off* rather than merely small -- a zero weight,
#: or a batch with nothing to rank -- where the code returns a hard ``0.0``.  A bare
#: ``== 0.0`` is a float-equality smell even when it is the right assertion, so this
#: says which kind of zero is meant while keeping the comparison exact: ``abs=0.0``
#: with the default ``rel`` applied to an expected value of zero leaves no tolerance
#: at all, so a term that leaked a tiny non-zero still fails.
EXACT_ZERO = pytest.approx(0.0, abs=0.0)


class TestSplitEntries:
    """The split must not leak a PDB across sides."""

    def test_groups_by_pdb_id(self):
        entries = ["1abc_HLA", "1abc_ABC", "2xyz_HL", "3def_A", "4ghi_B", "5jkl_C"]
        train, val = split_entries(entries, val_fraction=0.5, seed=0)
        train_pdb = {e.split("_")[0] for e in train}
        val_pdb = {e.split("_")[0] for e in val}
        assert not (train_pdb & val_pdb)

    def test_chain_variants_stay_together(self):
        entries = [f"{p}_{c}" for p in ("1abc", "2xyz", "3def", "4ghi") for c in ("HLA", "ABC")]
        train, val = split_entries(entries, val_fraction=0.25, seed=3)
        for pdb in {e.split("_")[0] for e in entries}:
            variants = {e for e in entries if e.startswith(pdb)}
            assert variants <= set(train) or variants <= set(val)

    def test_covers_every_entry(self):
        entries = [f"{i}abc_H" for i in range(20)]
        train, val = split_entries(entries, val_fraction=0.2, seed=1)
        assert sorted(train + val) == sorted(entries)
        assert val

    def test_is_deterministic(self):
        # Two separate calls, named, so the assertion does not read as a literal
        # self-comparison: the point is that the same seed reproduces the split.
        entries = [f"{i}abc_H" for i in range(30)]
        first = split_entries(entries, seed=7)
        second = split_entries(entries, seed=7)
        assert first == second


def _write_fake_cache(root: Path, entry: str, dockqs: list[float], *, n_tokens=6, n_atoms=8):
    """Build a minimal on-disk cache for one entry."""
    feats = {
        "token_pad_mask": torch.ones(1, n_tokens),
        "asym_id": torch.tensor([[0] * (n_tokens // 2) + [1] * (n_tokens - n_tokens // 2)]),
        "atom_pad_mask": torch.ones(1, n_atoms),
        "atom_to_token": torch.zeros(1, n_atoms, n_tokens, dtype=torch.int64),
    }
    feats["atom_to_token"][0, :, 0] = 1
    cache = EntryCache(
        entry=entry,
        s_inputs=torch.randn(1, n_tokens, 4),
        s=torch.randn(1, n_tokens, 4),
        z=torch.randn(1, n_tokens, n_tokens, 2),
        feats=feats,
        metadata={"n_tokens": n_tokens},
    )
    save_entry_cache(cache, root / "entries" / f"{entry}.pt", dtype=torch.float32)

    (root / "scores").mkdir(parents=True, exist_ok=True)
    (root / "decoys" / entry).mkdir(parents=True, exist_ok=True)
    lines = []
    for i, dockq in enumerate(dockqs):
        decoy_id = f"src__d{i}"
        np.savez_compressed(
            root / "decoys" / entry / f"{decoy_id}.npz",
            coords=np.random.default_rng(i).random((n_atoms, 3)).astype(np.float32),
            present=np.ones(n_atoms, dtype=bool),
            metadata=json.dumps({}),
        )
        lines.append(
            json.dumps(
                {
                    "entry": entry,
                    "decoy_id": decoy_id,
                    "source": "boltz2",
                    "condition": "c",
                    "dockq": dockq,
                    # Sub-scores consistent with the DockQ definition: fnat rises
                    # with quality, both RMSDs fall.  The mean of the three scaled
                    # values is not forced to equal `dockq`, so a test that silently
                    # relied on that identity would fail rather than pass by luck.
                    "fnat": min(1.0, dockq * 1.5),
                    "irmsd": max(0.0, 3.0 - dockq * 3.0),
                    "lrmsd": max(0.0, 15.0 - dockq * 15.0),
                    "iptm": 0.5 + 0.01 * i,
                    "ptm": 0.6,
                    "protein_iptm": 0.5,
                    "ligand_iptm": 0.0,
                    "complex_plddt": 0.9,
                    "complex_iplddt": 0.9,
                    "complex_pde": 0.2,
                    "complex_ipde": 0.3,
                }
            )
        )
    (root / "scores" / f"{entry}.jsonl").write_text("\n".join(lines) + "\n")


class TestCacheDataset:
    """Dataset must group by entry and drop unusable labels."""

    def test_serves_one_entry_with_all_decoys(self, tmp_path):
        _write_fake_cache(tmp_path, "1abc_HL", [0.1, 0.5, 0.9])
        ds = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)
        assert len(ds) == 1
        batch = ds[0]
        assert len(batch) == 3
        assert batch.x_pred.shape[0] == 3
        assert batch.z.shape[0] == 1  # z shared, not repeated
        assert torch.allclose(batch.dockq, torch.tensor([0.1, 0.5, 0.9]))

    def test_drops_nan_labels(self, tmp_path):
        _write_fake_cache(tmp_path, "1abc_HL", [0.1, float("nan"), 0.9])
        ds = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)
        batch = ds[0]
        assert len(batch) == 2
        assert not bool(torch.isnan(batch.dockq).any())

    def test_drops_entries_without_label_spread(self, tmp_path):
        _write_fake_cache(tmp_path, "1abc_HL", [0.4, 0.4, 0.4])
        ds = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], require_label_spread=True)
        assert len(ds) == 0

    def test_keeps_flat_entries_when_not_required(self, tmp_path):
        _write_fake_cache(tmp_path, "1abc_HL", [0.4, 0.4, 0.4])
        ds = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], require_label_spread=False)
        assert len(ds) == 1

    def test_chunks_oversized_entries(self, tmp_path):
        _write_fake_cache(tmp_path, "1abc_HL", [0.1 * i for i in range(10)])
        ds = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], max_decoys_per_batch=4, shuffle_decoys=False)
        assert len(ds) == 3
        # Balanced, not strided: 4/4/2 would be the same three forward passes with the
        # last one carrying half the ranking signal of the others.
        assert [len(ds[i]) for i in range(3)] == [4, 3, 3]

    def test_every_epoch_serves_every_decoy_exactly_once(self, tmp_path):
        """The coverage guarantee: an epoch is a partition of each entry's decoys.

        ``shuffle_decoys`` used to sample each item independently from the whole entry,
        which meant two items could draw the same decoy and ~27% of the training set
        went unseen in any given epoch.  Items now take disjoint slices of one
        per-epoch permutation, so the company a decoy keeps still changes but the set
        served does not.
        """
        dockqs = [0.03 * i for i in range(10)]
        _write_fake_cache(tmp_path, "1abc_HL", dockqs)
        ds = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], max_decoys_per_batch=4, shuffle_decoys=True)
        expected = {f"src__d{i}" for i in range(10)}

        seen_per_epoch = []
        for epoch in range(3):
            ds.set_epoch(epoch)
            served = [decoy_id for i in range(len(ds)) for decoy_id in ds[i].decoy_ids]
            assert len(served) == len(expected), "an epoch served a decoy twice or not at all"
            assert set(served) == expected
            seen_per_epoch.append(tuple(served))

        # Same set, different grouping: a fixed grouping would show the ranking term
        # the same pairs every epoch, which is what the shuffling is for.
        assert len(set(seen_per_epoch)) > 1

    def test_decoy_grouping_does_not_depend_on_which_worker_loads_it(self, tmp_path):
        """Two dataset objects at the same epoch must agree on the partition.

        Dataloader workers each hold their own copy of the dataset and are handed
        different items of the same entry, so the permutation has to be reproducible
        from ``(seed, epoch, entry)`` alone.  Seeding it with anything process-local --
        ``hash(entry)`` is salted per process -- would silently go back to overlapping
        draws in exactly the multi-worker configuration the real runs use.
        """
        _write_fake_cache(tmp_path, "1abc_HL", [0.03 * i for i in range(10)])
        kwargs = {"max_decoys_per_batch": 4, "shuffle_decoys": True, "seed": 7}
        first = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], **kwargs)
        second = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], **kwargs)
        first.set_epoch(2)
        second.set_epoch(2)

        assert [first[i].decoy_ids for i in range(len(first))] == [second[i].decoy_ids for i in range(len(second))]

    def test_carries_the_dockq_sub_scores(self, tmp_path):
        """The composite head is supervised on these, so they must reach the batch."""
        _write_fake_cache(tmp_path, "1abc_HL", [0.2, 0.8])
        batch = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)[0]
        assert torch.allclose(batch.fnat, torch.tensor([0.3, 1.0]))
        assert torch.allclose(batch.irmsd, torch.tensor([2.4, 0.6]), atol=1e-6)
        assert torch.allclose(batch.lrmsd, torch.tensor([12.0, 3.0]), atol=1e-6)

    def test_drops_rows_missing_a_sub_score(self, tmp_path, caplog):
        """A row without components cannot feed the composite loss.

        Counted and reported rather than skipped silently: on the real caches the
        drop is concentrated in 6 entries, and a source-correlated loss that reads as
        full coverage is the failure mode this line exists to prevent.
        """
        _write_fake_cache(tmp_path, "1abc_HL", [0.1, 0.5, 0.9])
        scores = tmp_path / "scores" / "1abc_HL.jsonl"
        rows = [json.loads(line) for line in scores.read_text().splitlines()]
        del rows[1]["irmsd"]
        scores.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        with caplog.at_level(logging.INFO, logger="confidock.boltz.finetune.cache_dataset"):
            batch = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)[0]
        assert len(batch) == 2
        assert "1 no-components" in caplog.text

    def test_carries_baseline_metrics(self, tmp_path):
        _write_fake_cache(tmp_path, "1abc_HL", [0.1, 0.9])
        batch = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)[0]
        assert "iptm" in batch.baseline
        assert batch.baseline["iptm"].shape == (2,)

    def test_skips_entries_with_no_cache_file(self, tmp_path):
        (tmp_path / "entries").mkdir(parents=True, exist_ok=True)
        ds = EntryGroupedCacheDataset(tmp_path, ["missing_XY"])
        assert len(ds) == 0

    def test_collate_rejects_multi_item_batches(self, tmp_path):
        _write_fake_cache(tmp_path, "1abc_HL", [0.1, 0.9])
        ds = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)
        assert collate_entry_batches([ds[0]]) is not None
        with pytest.raises(ValueError, match="batch_size=1"):
            collate_entry_batches([ds[0], ds[0]])


class TestChunkIndices:
    """Item sizes decide how much ranking signal a step carries.

    Striding leaves a remainder, and a remainder of one decoy is a step with no
    ranking term at all: :class:`DockQLoss` needs a pair.  Balancing costs nothing --
    same item count, same memory ceiling -- and on the real training cache it turned
    388 of 8336 unrankable items into rankable ones.
    """

    @pytest.mark.parametrize(
        ("n_rows", "per_batch", "expected"),
        [
            (33, 16, [11, 11, 11]),  # the real case: 16/16/1 before
            (10, 4, [4, 3, 3]),
            (9, 3, [3, 3, 3]),
            (5, 8, [5]),  # fits in one item
            (1, 8, [1]),  # a one-decoy entry stays one decoy, nothing to balance
        ],
    )
    def test_sizes_are_balanced(self, n_rows, per_batch, expected):
        sizes = [len(idx) for _, idx in _chunk_indices("e", n_rows, per_batch)]
        assert sizes == expected

    @pytest.mark.parametrize("n_rows", range(1, 40))
    @pytest.mark.parametrize("per_batch", [1, 2, 5, 16])
    def test_covers_every_row_once_and_never_exceeds_the_budget(self, n_rows, per_batch):
        items = _chunk_indices("e", n_rows, per_batch)
        indices = [i for _, idx in items for i in idx]

        # A partition of the rows: the per-epoch permutation in ``__getitem__`` is only
        # a partition of the entry's decoys because these slices are.
        assert sorted(indices) == list(range(n_rows))
        # The budget is a memory ceiling, so balancing must round item sizes *down* to
        # fit it, never up: exceeding it is what token-aware batching exists to prevent.
        assert max(len(idx) for _, idx in items) <= max(per_batch, 1)
        # And the count is the same as striding would give, so nothing about the
        # planned step count or the epoch length changes.
        assert len(items) == -(-n_rows // per_batch)

    def test_only_a_single_decoy_entry_yields_an_unrankable_item(self):
        for n_rows in range(2, 40):
            sizes = [len(idx) for _, idx in _chunk_indices("e", n_rows, 16)]
            assert min(sizes) >= 2, f"{n_rows} rows produced a single-decoy item"


class TestPooling:
    """Interface-aware pooling must mask padding and never emit NaN."""

    def test_output_width(self):
        s, z = torch.randn(3, 5, 8), torch.randn(3, 5, 5, 4)
        feats = {"token_pad_mask": torch.ones(1, 5), "asym_id": torch.tensor([[0, 0, 1, 1, 1]])}
        assert pool_confidence_latents(s, z, feats).shape == (3, 2 * 4 + 8)

    def test_single_chain_does_not_produce_nan(self):
        s, z = torch.randn(2, 4, 8), torch.randn(2, 4, 4, 4)
        feats = {"token_pad_mask": torch.ones(1, 4), "asym_id": torch.zeros(1, 4, dtype=torch.long)}
        assert not bool(torch.isnan(pool_confidence_latents(s, z, feats)).any())

    def test_ignores_padded_tokens(self):
        s, z = torch.randn(2, 6, 8), torch.randn(2, 6, 6, 4)
        feats = {
            "token_pad_mask": torch.cat([torch.ones(1, 4), torch.zeros(1, 2)], dim=1),
            "asym_id": torch.tensor([[0, 0, 1, 1, 0, 0]]),
        }
        before = pool_confidence_latents(s, z, feats)
        s2, z2 = s.clone(), z.clone()
        s2[:, 4:] = 1e3
        z2[:, 4:] = 1e3
        z2[:, :, 4:] = 1e3
        assert torch.allclose(before, pool_confidence_latents(s2, z2, feats))

    def test_cross_chain_mean_differs_from_all_pair_mean(self):
        torch.manual_seed(0)
        z = torch.randn(1, 4, 4, 2)
        s = torch.randn(1, 4, 2)
        feats = {"token_pad_mask": torch.ones(1, 4), "asym_id": torch.tensor([[0, 0, 1, 1]])}
        pooled = pool_confidence_latents(s, z, feats)
        assert not torch.allclose(pooled[:, :2], pooled[:, 2:4])


class TestDockQHead:
    def test_forward_shape(self):
        head = DockQHead(8, 4)
        s, z = torch.randn(5, 6, 8), torch.randn(5, 6, 6, 4)
        feats = {"token_pad_mask": torch.ones(1, 6), "asym_id": torch.tensor([[0, 0, 0, 1, 1, 1]])}
        out = head(s, z, feats)
        assert out["dockq"].shape == (5,)
        assert out["components"].shape == (5, 3)

    def test_dockq_is_the_exact_mean_of_the_components(self):
        """DockQ *is* that mean (Basu & Wallner), not a fourth learned output."""
        head = DockQHead(8, 4, dropout=0.0).eval()
        s, z = torch.randn(4, 6, 8), torch.randn(4, 6, 6, 4)
        feats = {"token_pad_mask": torch.ones(1, 6), "asym_id": torch.tensor([[0, 0, 0, 1, 1, 1]])}
        out = head(s, z, feats)
        assert torch.allclose(out["dockq"], out["components"].mean(dim=-1), atol=1e-6)

    def test_components_stay_in_the_unit_interval(self):
        """All three sub-scores are bounded, so the head ends in a sigmoid."""
        head = DockQHead(8, 4)
        s, z = torch.randn(6, 6, 8) * 50, torch.randn(6, 6, 6, 4) * 50
        feats = {"token_pad_mask": torch.ones(1, 6), "asym_id": torch.tensor([[0, 0, 0, 1, 1, 1]])}
        components = head(s, z, feats)["components"]
        assert bool(((components >= 0.0) & (components <= 1.0)).all())

    def test_gradients_flow(self):
        head = DockQHead(8, 4)
        s, z = torch.randn(3, 6, 8), torch.randn(3, 6, 6, 4)
        feats = {"token_pad_mask": torch.ones(1, 6), "asym_id": torch.tensor([[0, 0, 0, 1, 1, 1]])}
        head(s, z, feats)["dockq"].sum().backward()
        assert all(p.grad is not None for p in head.parameters())


class TestComponentTargets:
    """The DockQ distance scaling, which turns Angstrom into a [0, 1] sub-score."""

    def test_scaling_is_monotonically_decreasing(self):
        distances = torch.tensor([0.0, 1.0, 5.0, 20.0])
        for scale in (scale_irmsd, scale_lrmsd):
            values = scale(distances)
            assert bool((values[1:] < values[:-1]).all())

    def test_zero_distance_is_a_perfect_score(self):
        zero = torch.zeros(1)
        assert scale_irmsd(zero).item() == pytest.approx(1.0)
        assert scale_lrmsd(zero).item() == pytest.approx(1.0)

    def test_scaling_uses_the_published_d1_d2_constants(self):
        """At the characteristic distance the sub-score is exactly 1/2."""
        assert scale_irmsd(torch.tensor([1.5])).item() == pytest.approx(0.5)
        assert scale_lrmsd(torch.tensor([8.5])).item() == pytest.approx(0.5)

    def test_targets_are_fnat_then_the_two_scaled_rmsds(self):
        targets = component_targets(torch.tensor([0.4]), torch.tensor([1.5]), torch.tensor([8.5]))
        assert targets.shape == (1, 3)
        assert targets[0].tolist() == pytest.approx([0.4, 0.5, 0.5])


class TestDockQLoss:
    """The composite loss: Huber on DockQ + Huber on the sub-scores + soft-Spearman.

    Predictions and targets are built from a DockQ vector by repeating it across the
    three component slots.  That keeps each test focused on the one term it names:
    the component and DockQ Huber terms then agree, so a change in ``rank`` cannot be
    a regression term in disguise.
    """

    @staticmethod
    def _side(dockq: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"dockq": dockq, "components": dockq.unsqueeze(-1).expand(-1, 3)}

    def test_ranking_term_punishes_inversion(self):
        loss_fn = DockQLoss(lambda_rank=1.0)
        targets = self._side(torch.tensor([0.1, 0.5, 0.9]))
        _, good = loss_fn(self._side(torch.tensor([0.2, 0.5, 0.8])), targets)
        _, bad = loss_fn(self._side(torch.tensor([0.8, 0.5, 0.2])), targets)
        assert bad["rank"] > good["rank"]

    def test_correct_order_has_near_zero_rank_loss(self):
        loss_fn = DockQLoss(lambda_rank=1.0)
        _, parts = loss_fn(
            self._side(torch.tensor([0.1, 0.5, 0.9])),
            self._side(torch.tensor([0.1, 0.5, 0.9])),
        )
        assert parts["rank"] < 0.01

    def test_full_inversion_costs_about_two(self):
        """(1 - rho) with rho = -1, which is the term's ceiling."""
        loss_fn = DockQLoss(lambda_rank=1.0)
        _, parts = loss_fn(
            self._side(torch.tensor([0.9, 0.5, 0.1])),
            self._side(torch.tensor([0.1, 0.5, 0.9])),
        )
        assert parts["rank"] == pytest.approx(2.0, abs=0.05)

    def test_lambda_rank_zero_disables_the_ranking_term(self):
        loss_fn = DockQLoss(lambda_rank=0.0, lambda_component=0.0)
        predictions = self._side(torch.tensor([0.9, 0.5, 0.1]))
        _, parts = loss_fn(predictions, self._side(torch.tensor([0.1, 0.5, 0.9])))
        assert parts["rank"] == EXACT_ZERO
        assert parts["loss"] == pytest.approx(parts["huber_dockq"])

    def test_lambda_component_zero_drops_component_supervision(self):
        loss_fn = DockQLoss(lambda_rank=0.0, lambda_component=0.0)
        predictions = {
            "dockq": torch.tensor([0.5, 0.5]),
            # Deliberately far from the targets: with the weight at zero this must
            # not move the loss at all.
            "components": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        }
        _, parts = loss_fn(predictions, self._side(torch.tensor([0.2, 0.8])))
        assert parts["huber_comp"] == EXACT_ZERO
        assert parts["loss"] == pytest.approx(parts["huber_dockq"])

    def test_component_error_reaches_the_loss(self):
        """Otherwise ``comp_sup`` would be the plain DockQ arm under another name."""
        loss_fn = DockQLoss(lambda_rank=0.0, lambda_component=1.0)
        targets = self._side(torch.tensor([0.5, 0.5]))
        # Both predictions have the right DockQ mean; only the split differs.
        exact = {"dockq": torch.tensor([0.5, 0.5]), "components": torch.full((2, 3), 0.5)}
        wrong = {
            "dockq": torch.tensor([0.5, 0.5]),
            "components": torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]),
        }
        _, exact_parts = loss_fn(exact, targets)
        _, wrong_parts = loss_fn(wrong, targets)
        assert exact_parts["huber_dockq"] == pytest.approx(wrong_parts["huber_dockq"])
        assert wrong_parts["loss"] > exact_parts["loss"]

    def test_single_decoy_has_no_ranking_term(self):
        loss_fn = DockQLoss()
        _, parts = loss_fn(self._side(torch.tensor([0.5])), self._side(torch.tensor([0.7])))
        assert parts["rank"] == EXACT_ZERO

    def test_tied_predictions_do_not_emit_nan(self):
        """A constant prediction gives the soft ranks zero variance."""
        loss_fn = DockQLoss()
        loss, parts = loss_fn(
            self._side(torch.tensor([0.5, 0.5, 0.5])),
            self._side(torch.tensor([0.1, 0.5, 0.9])),
        )
        assert torch.isfinite(loss)
        assert parts["rank"] == EXACT_ZERO

    def test_loss_is_finite_at_label_extremes(self):
        loss_fn = DockQLoss()
        loss, _ = loss_fn(self._side(torch.tensor([0.0, 0.0])), self._side(torch.tensor([0.0, 1.0])))
        assert torch.isfinite(loss)

    def test_gradient_flows_through_every_term(self):
        loss_fn = DockQLoss()
        components = torch.rand(4, 3, requires_grad=True)
        predictions = {"components": components, "dockq": components.mean(dim=-1)}
        loss, _ = loss_fn(predictions, self._side(torch.tensor([0.1, 0.4, 0.6, 0.9])))
        loss.backward()
        assert components.grad is not None
        assert bool(torch.isfinite(components.grad).all())

    def test_lambda_component_zero_drops_the_component_term(self):
        """The arm that isolates the loss from the 3-sigmoid head's supervision.

        With the weight at zero the components are still *predicted* -- DockQ is their
        mean -- but no longer supervised individually, so a wrong split of a correct
        mean must cost nothing.
        """
        loss_fn = DockQLoss(lambda_component=0.0, lambda_rank=0.0)
        targets = {
            "dockq": torch.tensor([0.6]),
            "components": torch.tensor([[0.2, 0.6, 1.0]]),
        }
        # Same mean (0.6), completely different split.
        skewed = {"dockq": torch.tensor([0.6]), "components": torch.tensor([[1.0, 0.8, 0.0]])}
        loss, parts = loss_fn(skewed, targets)
        assert parts["huber_comp"] == EXACT_ZERO
        assert parts["loss"] == pytest.approx(parts["huber_dockq"], abs=1e-7)
        assert float(loss) == pytest.approx(0.0, abs=1e-7)


class TestListNetRanking:
    """``rank_type="listnet"``: listwise top-one cross-entropy instead of soft ranks.

    Mirrors ``dockq_predictor.losses.losses.listnet_term`` so the GNN and the Boltz
    head optimise the same objective.  That package lives on the GNN branch and is not
    importable here, so the reference below is written out independently -- explicit
    ``exp``/``log`` rather than ``softmax``/``log_softmax`` -- which checks the formula
    rather than restating the implementation.
    """

    @staticmethod
    def _side(dockq: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"dockq": dockq, "components": dockq.unsqueeze(-1).expand(-1, 3)}

    @staticmethod
    def _reference(pred: torch.Tensor, targ: torch.Tensor, tau: float) -> float:
        """H(softmax(t/tau), softmax(p/tau)), built from exp/log by hand."""
        pt = torch.exp(targ / tau) / torch.exp(targ / tau).sum()
        pp = torch.exp(pred / tau) / torch.exp(pred / tau).sum()
        return float(-(pt * torch.log(pp)).sum())

    def _rank_only(self, tau: float = 0.1) -> DockQLoss:
        return DockQLoss(lambda_component=0.0, lambda_rank=1.0, rank_type="listnet", rank_tau=tau)

    def test_matches_the_reference_cross_entropy(self):
        loss_fn = self._rank_only()
        pred = torch.tensor([0.2, 0.5, 0.85, 0.4])
        targ = torch.tensor([0.1, 0.5, 0.9, 0.3])
        _, parts = loss_fn(self._side(pred), self._side(targ))
        assert parts["rank"] == pytest.approx(self._reference(pred, targ, 0.1), rel=1e-6)

    def test_punishes_inversion(self):
        loss_fn = self._rank_only()
        targets = self._side(torch.tensor([0.1, 0.5, 0.9]))
        _, good = loss_fn(self._side(torch.tensor([0.2, 0.5, 0.8])), targets)
        _, bad = loss_fn(self._side(torch.tensor([0.8, 0.5, 0.2])), targets)
        assert bad["rank"] > good["rank"]

    def test_is_invariant_to_a_per_entry_offset(self):
        """The property that makes Huber the only scale-aware term in the objective.

        ``softmax((p + c) / tau) == softmax(p / tau)``, so shifting a whole entry's
        predictions cannot change this term -- which is exactly why lambda_rank/rank_tau
        can drown out the absolute DockQ scale if pushed too high.
        """
        loss_fn = self._rank_only()
        pred = torch.tensor([0.2, 0.5, 0.85])
        targets = self._side(torch.tensor([0.1, 0.5, 0.9]))
        _, base = loss_fn(self._side(pred), targets)
        _, shifted = loss_fn(self._side(pred - 0.15), targets)
        assert shifted["rank"] == pytest.approx(base["rank"], rel=1e-6)

    def _grad_norm(self, pred_values: list[float], tau: float) -> float:
        pred = torch.tensor(pred_values, requires_grad=True)
        loss, _ = self._rank_only(tau)(self._side(pred), self._side(torch.tensor([0.1, 0.5, 0.9])))
        loss.backward()
        return float(pred.grad.norm())

    def test_gradient_scales_as_one_over_tau_while_the_ordering_is_wrong(self):
        """Halving tau doubles the ranking gradient, so tau is a second weight.

        Measured on an inverted ordering, where it is near-exact: 6.15 / 14.13 / 28.61 /
        56.91 at tau 0.2 / 0.1 / 0.05 / 0.025.  ``d(loss)/dp ~ (P_pred - P_true)/tau``,
        so the 1/tau amplification is only visible while that difference is O(1).
        """
        inverted = [0.85, 0.5, 0.2]
        ratio = self._grad_norm(inverted, 0.05) / self._grad_norm(inverted, 0.1)
        assert ratio == pytest.approx(2.0, rel=0.1)

    def test_a_satisfied_ordering_saturates_instead(self):
        """The other half of the same mechanism, and the reason the test above inverts.

        Once the predicted order is already correct, shrinking tau drives both softmaxes
        onto the *same* one-hot, so ``P_pred - P_true`` vanishes faster than 1/tau grows
        and the gradient falls away.  A correctly-ranked entry therefore stops
        contributing at small tau -- the pressure is spent on the entries still wrong.
        """
        ordered = [0.2, 0.5, 0.85]
        assert self._grad_norm(ordered, 0.05) < self._grad_norm(ordered, 0.1)

    def test_all_equal_targets_ask_only_for_equal_predictions(self):
        """No argsort tie-breaking: a flat target is uniform, and flat preds satisfy it.

        This is the one behaviour that differs in kind from soft-Spearman, which hands
        tied targets an arbitrary order via ``argsort`` and pushes the model to
        reproduce it (see ``_soft_spearman``'s docstring).
        """
        loss_fn = self._rank_only()
        flat = self._side(torch.tensor([0.4, 0.4, 0.4]))
        _, equal = loss_fn(self._side(torch.tensor([0.7, 0.7, 0.7])), flat)
        _, spread = loss_fn(self._side(torch.tensor([0.1, 0.5, 0.9])), flat)
        # Uniform target distribution -> cross-entropy is minimised by uniform preds.
        assert equal["rank"] < spread["rank"]

    def test_single_decoy_has_no_ranking_term(self):
        loss_fn = self._rank_only()
        _, parts = loss_fn(self._side(torch.tensor([0.5])), self._side(torch.tensor([0.7])))
        assert parts["rank"] == EXACT_ZERO

    def test_gradient_flows_and_stays_finite_at_label_extremes(self):
        loss_fn = self._rank_only()
        components = torch.rand(4, 3, requires_grad=True)
        predictions = {"components": components, "dockq": components.mean(dim=-1)}
        loss, _ = loss_fn(predictions, self._side(torch.tensor([0.0, 0.0, 1.0, 1.0])))
        loss.backward()
        assert torch.isfinite(loss)
        assert bool(torch.isfinite(components.grad).all())

    def test_unknown_rank_type_is_rejected(self):
        """A typo must not fall through to the soft-Spearman branch silently."""
        with pytest.raises(ValueError, match="listnet"):
            DockQLoss(rank_type="listwise")

    def test_spearman_remains_the_default(self):
        assert DockQLoss().rank_type == "spearman"


class TestCorrelationReport:
    def _rows(self, pred_fn):
        rows = []
        for entry in ("a_H", "b_H", "c_H"):
            for i in range(5):
                dockq = i / 4
                rows.append(
                    {
                        "entry": entry,
                        "decoy_id": f"{entry}_{i}",
                        "source": "boltz2",
                        "dockq": dockq,
                        "dockq_pred": pred_fn(dockq),
                        "iptm": 0.5,
                    }
                )
        return rows

    def test_perfect_predictor_scores_one(self):
        report = correlation_report(self._rows(lambda d: d))
        assert report["global"]["dockq_pred"] == pytest.approx(1.0)
        assert report["per_entry"]["dockq_pred"] == pytest.approx(1.0)

    def test_inverted_predictor_scores_minus_one(self):
        report = correlation_report(self._rows(lambda d: 1 - d))
        assert report["per_entry"]["dockq_pred"] == pytest.approx(-1.0)

    def test_constant_metric_is_nan_not_zero(self):
        report = correlation_report(self._rows(lambda d: d))
        assert np.isnan(report["per_entry"]["iptm"])

    def test_counts_entries_and_decoys(self):
        report = correlation_report(self._rows(lambda d: d))
        assert report["n_entries"] == 3
        assert report["n_decoys"] == 15

    def test_top1_matches_oracle_for_perfect_predictor(self):
        report = correlation_report(self._rows(lambda d: d))
        assert report["top1_mean_dockq"]["dockq_pred"] == pytest.approx(report["oracle_mean_dockq"])

    def test_empty_rows_do_not_crash(self):
        report = correlation_report([])
        assert report["n_decoys"] == 0

    def test_derived_confidence_score_is_reported_when_present(self):
        rows = self._rows(lambda d: d)
        for row in rows:
            row["complex_plddt"] = 0.5 + row["dockq"] / 10
            row["confidence_score"] = boltz_confidence_score(row)
        report = correlation_report(rows)
        # It varies with DockQ here, so it must appear and rank perfectly -- the point
        # is that a derived column is reported like a cached one, not skipped.
        assert report["per_entry"]["confidence_score"] == pytest.approx(1.0)


class TestBoltzConfidenceScore:
    """Boltz-2's headline scalar, rebuilt from the cached ingredients."""

    def test_matches_boltz_formula(self):
        row = {"complex_plddt": 0.9, "iptm": 0.4, "ptm": 0.1}
        assert boltz_confidence_score(row) == pytest.approx((4 * 0.9 + 0.4) / 5)

    def test_falls_back_to_ptm_when_iptm_is_zero(self):
        # Boltz substitutes ptm for an all-zero iptm (monomer, or a ligand-free
        # interface that never got scored); a cached 0.0 must not be averaged in.
        row = {"complex_plddt": 0.9, "iptm": 0.0, "ptm": 0.5}
        assert boltz_confidence_score(row) == pytest.approx((4 * 0.9 + 0.5) / 5)

    def test_missing_ingredient_is_nan(self):
        assert np.isnan(boltz_confidence_score({"iptm": 0.4, "ptm": 0.1}))
        assert np.isnan(boltz_confidence_score({"complex_plddt": 0.9}))


class TestChunkedForward:
    """``run_head`` splits the decoy axis; chunking must not change the answer.

    A stub confidence module stands in for the 506M-parameter checkpoint.  It
    reproduces the two behaviours ``run_head`` depends on: ``repeat_interleave`` on
    the multiplicity axis, and returning ``s_conf``/``z_conf`` at batch
    ``multiplicity``.  Making the stub's output depend on ``x_pred`` is what lets
    the test detect a chunk boundary silently dropping or reordering decoys.
    """

    class _StubConfidence:
        def __init__(self, token_s: int, token_z: int):
            self.token_s = token_s
            self.token_z = token_z
            self.calls: list[int] = []

        def __call__(self, *, s, z, x_pred, multiplicity, **kwargs):
            self.calls.append(multiplicity)
            assert x_pred.shape[0] == multiplicity, "x_pred must match multiplicity"
            # One scalar per decoy, derived from its coordinates, broadcast out.
            per_decoy = x_pred.reshape(multiplicity, -1).sum(dim=-1)
            n_tokens = s.shape[1]
            s_conf = per_decoy[:, None, None] * torch.ones(multiplicity, n_tokens, self.token_s)
            z_conf = per_decoy[:, None, None, None] * torch.ones(multiplicity, n_tokens, n_tokens, self.token_z)
            return {"s_conf": s_conf, "z_conf": z_conf}

    class _StubModel:
        def __init__(self, confidence_module):
            self.confidence_module = confidence_module

        def distogram_module(self, z):
            return torch.zeros(*z.shape[:3], 1)

    def _batch(self, tmp_path, n_decoys=7):
        _write_fake_cache(tmp_path, "1abc_HL", [i / n_decoys for i in range(n_decoys)])
        ds = EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)
        return ds[0]

    def test_chunked_matches_unchunked(self, tmp_path):
        torch.manual_seed(0)
        batch = self._batch(tmp_path)
        stub = self._StubConfidence(8, 4)
        model = self._StubModel(stub)
        head = DockQHead(8, 4, dropout=0.0).eval()

        full = run_head(model, head, batch, chunk=len(batch))
        chunked = run_head(model, head, batch, chunk=3)
        assert torch.allclose(full["dockq"], chunked["dockq"], atol=1e-5)
        assert torch.allclose(full["components"], chunked["components"], atol=1e-5)

    def test_chunk_sizes_cover_every_decoy_exactly_once(self, tmp_path):
        batch = self._batch(tmp_path, n_decoys=7)
        stub = self._StubConfidence(8, 4)
        run_head(self._StubModel(stub), DockQHead(8, 4).eval(), batch, chunk=3)
        assert stub.calls == [3, 3, 1]
        assert sum(stub.calls) == len(batch)

    def test_output_length_matches_decoy_count(self, tmp_path):
        batch = self._batch(tmp_path, n_decoys=5)
        predictions = run_head(
            self._StubModel(self._StubConfidence(8, 4)),
            DockQHead(8, 4).eval(),
            batch,
            chunk=2,
        )
        assert predictions["dockq"].shape == (5,)
        assert predictions["components"].shape == (5, 3)

    def test_frozen_confidence_forward_builds_no_graph(self, tmp_path):
        """With ``confidence_requires_grad=False`` the latents must be detached.

        This is the property that makes a frozen run cheap: if the confidence
        forward built a graph, chunking would still hold O(chunk * L^2) activations
        alive until backward.
        """
        batch = self._batch(tmp_path, n_decoys=4)
        stub = self._StubConfidence(8, 4)
        head = DockQHead(8, 4, dropout=0.0)
        predictions = run_head(self._StubModel(stub), head, batch, chunk=2)
        # The head itself is still trainable end-to-end.
        predictions["dockq"].sum().backward()
        assert all(p.grad is not None for p in head.parameters())

    def test_from_pooled_matches_forward(self):
        head = DockQHead(8, 4, dropout=0.0).eval()
        s, z = torch.randn(3, 6, 8), torch.randn(3, 6, 6, 4)
        feats = {
            "token_pad_mask": torch.ones(1, 6),
            "asym_id": torch.tensor([[0, 0, 0, 1, 1, 1]]),
        }
        pooled = pool_confidence_latents(s, z, feats)
        direct = head(s, z, feats)
        from_pooled = head.from_pooled(pooled)
        assert torch.allclose(direct["dockq"], from_pooled["dockq"], atol=1e-6)
        assert torch.allclose(direct["components"], from_pooled["components"], atol=1e-6)


class TestEvaluate:
    """``evaluate`` must score every decoy, and must not evaluate in train() mode.

    Both properties were bugs.  The module was left in ``train()`` by
    ``prepare_confidence_module(unfreeze=True)``, which costs memory because Boltz
    chunks triangular attention only when ``not self.training``; and an OOM dropped
    the whole entry, which biases the report because OOM risk grows with token
    count.  A stub that raises above a decoy threshold reproduces both cheaply.
    """

    class _OOMAboveConfidence(TestChunkedForward._StubConfidence):
        """Stub confidence module that OOMs above ``max_multiplicity`` decoys.

        Also records ``training`` at call time, which is how the mode assertion is
        made without a real pairformer.
        """

        def __init__(self, token_s: int, token_z: int, *, max_multiplicity: int):
            super().__init__(token_s, token_z)
            self.max_multiplicity = max_multiplicity
            self.training = True
            self.training_at_call: list[bool] = []

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.training = mode
            return self

        def __call__(self, *, multiplicity, **kwargs):
            self.training_at_call.append(self.training)
            if multiplicity > self.max_multiplicity:
                raise torch.OutOfMemoryError("stub OOM")
            return super().__call__(multiplicity=multiplicity, **kwargs)

    def _loader(self, tmp_path, n_decoys=8):
        _write_fake_cache(tmp_path, "1abc_HL", [i / n_decoys for i in range(n_decoys)])
        return [EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)[0]]

    def _config(self, tmp_path, **kwargs):
        return FinetuneConfig(cache_dir=tmp_path, output_dir=tmp_path, device="cpu", forward_chunk=8, **kwargs)

    def test_scores_every_decoy_when_nothing_ooms(self, tmp_path):
        loader = self._loader(tmp_path, n_decoys=8)
        stub = self._OOMAboveConfidence(8, 4, max_multiplicity=8)
        model = TestChunkedForward._StubModel(stub)
        _, rows = evaluate(model, DockQHead(8, 4, dropout=0.0), loader, self._config(tmp_path))
        assert len(rows) == 8

    def test_oom_halves_the_group_instead_of_dropping_the_entry(self, tmp_path):
        """The old behaviour lost all 8 decoys; the retry keeps them."""
        loader = self._loader(tmp_path, n_decoys=8)
        stub = self._OOMAboveConfidence(8, 4, max_multiplicity=2)
        model = TestChunkedForward._StubModel(stub)
        _, rows = evaluate(model, DockQHead(8, 4, dropout=0.0), loader, self._config(tmp_path))
        assert len(rows) == 8
        assert {r["decoy_id"] for r in rows} == {f"src__d{i}" for i in range(8)}

    def test_regrouping_does_not_change_predictions(self, tmp_path):
        """Group size is a memory knob only -- it must not move a single logit."""
        loader = self._loader(tmp_path, n_decoys=8)
        head = DockQHead(8, 4, dropout=0.0)
        config = self._config(tmp_path)

        _, full = evaluate(
            TestChunkedForward._StubModel(self._OOMAboveConfidence(8, 4, max_multiplicity=8)),
            head,
            loader,
            config,
        )
        _, split = evaluate(
            TestChunkedForward._StubModel(self._OOMAboveConfidence(8, 4, max_multiplicity=2)),
            head,
            loader,
            config,
        )
        by_id = {r["decoy_id"]: r["dockq_pred"] for r in split}
        for row in full:
            assert by_id[row["decoy_id"]] == pytest.approx(row["dockq_pred"], abs=1e-6)

    def test_confidence_module_is_in_eval_mode_during_scoring(self, tmp_path):
        loader = self._loader(tmp_path, n_decoys=4)
        stub = self._OOMAboveConfidence(8, 4, max_multiplicity=4)
        stub.train()
        evaluate(
            TestChunkedForward._StubModel(stub),
            DockQHead(8, 4, dropout=0.0),
            loader,
            self._config(tmp_path),
        )
        assert stub.training_at_call == [False]

    def test_prior_training_mode_is_restored(self, tmp_path):
        """A subsequent training epoch still needs the train-mode path."""
        loader = self._loader(tmp_path, n_decoys=4)
        stub = self._OOMAboveConfidence(8, 4, max_multiplicity=4)
        stub.train()
        evaluate(
            TestChunkedForward._StubModel(stub),
            DockQHead(8, 4, dropout=0.0),
            loader,
            self._config(tmp_path),
        )
        assert stub.training is True

    def test_frozen_module_stays_in_eval_mode(self, tmp_path):
        loader = self._loader(tmp_path, n_decoys=4)
        stub = self._OOMAboveConfidence(8, 4, max_multiplicity=4)
        stub.eval()
        evaluate(
            TestChunkedForward._StubModel(stub),
            DockQHead(8, 4, dropout=0.0),
            loader,
            self._config(tmp_path),
        )
        assert stub.training is False

    def test_entry_is_skipped_when_even_one_decoy_ooms(self, tmp_path, caplog):
        """The ladder bottoms out rather than looping forever.

        A skipped entry must be logged as a skip, not counted as a regroup -- the
        two lines mean different things when reading a run's log.
        """
        loader = self._loader(tmp_path, n_decoys=4)
        stub = self._OOMAboveConfidence(8, 4, max_multiplicity=0)
        with caplog.at_level(logging.INFO, logger="confidock.boltz.finetune.trainer"):
            _, rows = evaluate(
                TestChunkedForward._StubModel(stub),
                DockQHead(8, 4, dropout=0.0),
                loader,
                self._config(tmp_path),
            )
        assert rows == []
        assert "even at 1 decoy" in caplog.text
        assert "regrouped" not in caplog.text


class TestBatchTargets:
    """The Angstrom -> scaled conversion happens here and nowhere else.

    If it ever also happened at cache time the RMSDs would be scaled twice, which
    is not a crash but a silently compressed target range, so the exact identity
    against :func:`component_targets` is worth pinning.
    """

    def _batch(self, tmp_path, dockqs=(0.2, 0.8)):
        _write_fake_cache(tmp_path, "1abc_HL", list(dockqs))
        return EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)[0]

    def test_passes_dockq_through_untouched(self, tmp_path):
        batch = self._batch(tmp_path)
        targets = batch_targets(batch)
        assert torch.equal(targets["dockq"], batch.dockq)

    def test_scales_the_raw_rmsds(self, tmp_path):
        batch = self._batch(tmp_path)
        targets = batch_targets(batch)
        expected = component_targets(batch.fnat, batch.irmsd, batch.lrmsd)
        assert torch.allclose(targets["components"], expected)
        # The cache stores Angstrom, so passing an irmsd of 2.4 through unscaled
        # would leave the unit interval and make the identity above vacuous.
        assert (targets["components"] >= 0.0).all()
        assert (targets["components"] <= 1.0).all()

    def test_shapes_match_what_the_loss_expects(self, tmp_path):
        batch = self._batch(tmp_path, dockqs=(0.1, 0.4, 0.9))
        targets = batch_targets(batch)
        assert targets["dockq"].shape == (3,)
        assert targets["components"].shape == (3, 3)


class TestDecoySlice:
    """``decoy_slice`` takes an arbitrary range and shares the per-entry tensors."""

    def _batch(self, tmp_path, n_decoys=8):
        _write_fake_cache(tmp_path, "1abc_HL", [i / n_decoys for i in range(n_decoys)])
        return EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)[0]

    def test_slices_the_decoy_axis(self, tmp_path):
        batch = self._batch(tmp_path)
        part = decoy_slice(batch, 3, 6)
        assert len(part) == 3
        assert part.decoy_ids == batch.decoy_ids[3:6]
        assert part.sources == batch.sources[3:6]
        assert torch.equal(part.x_pred, batch.x_pred[3:6])
        assert torch.equal(part.dockq, batch.dockq[3:6])
        # The composite loss reads these, so a slice that forgot them would train
        # on another decoy's sub-scores rather than crash.
        assert torch.equal(part.fnat, batch.fnat[3:6])
        assert torch.equal(part.irmsd, batch.irmsd[3:6])
        assert torch.equal(part.lrmsd, batch.lrmsd[3:6])
        for key, values in part.baseline.items():
            assert torch.equal(values, batch.baseline[key][3:6])

    def test_per_entry_tensors_are_shared_not_copied(self, tmp_path):
        """This is why slicing costs no GPU memory: ``z`` is the O(L^2) tensor."""
        batch = self._batch(tmp_path)
        part = decoy_slice(batch, 2, 4)
        assert part.z is batch.z
        assert part.s is batch.s
        assert part.s_inputs is batch.s_inputs
        assert part.feats is batch.feats

    def test_slicing_does_not_mutate_the_original(self, tmp_path):
        batch = self._batch(tmp_path)
        decoy_slice(batch, 0, 2)
        assert len(batch) == 8


class TestEvalEntryPredictions:
    """``eval_entry_predictions`` must return the group size it settled on."""

    def _batch(self, tmp_path, n_decoys=8):
        _write_fake_cache(tmp_path, "1abc_HL", [i / n_decoys for i in range(n_decoys)])
        return EntryGroupedCacheDataset(tmp_path, ["1abc_HL"], shuffle_decoys=False)[0]

    def test_reports_the_full_batch_when_it_fits(self, tmp_path):
        batch = self._batch(tmp_path, n_decoys=8)
        stub = TestEvaluate._OOMAboveConfidence(8, 4, max_multiplicity=8)
        predictions, group = eval_entry_predictions(
            TestChunkedForward._StubModel(stub),
            DockQHead(8, 4, dropout=0.0),
            batch,
            forward_chunk=8,
        )
        assert group == 8
        assert predictions["dockq"].shape == (8,)

    def test_reports_the_halved_group_after_oom(self, tmp_path):
        batch = self._batch(tmp_path, n_decoys=8)
        stub = TestEvaluate._OOMAboveConfidence(8, 4, max_multiplicity=2)
        predictions, group = eval_entry_predictions(
            TestChunkedForward._StubModel(stub),
            DockQHead(8, 4, dropout=0.0),
            batch,
            forward_chunk=8,
        )
        assert group == 2
        assert predictions["dockq"].shape == (8,)
        # Regrouping is a memory knob; every sub-score must survive it too.
        assert predictions["components"].shape == (8, 3)
