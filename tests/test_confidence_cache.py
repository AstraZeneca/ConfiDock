"""Tests for the Boltz-2 confidence-head input cache.

These cover the pure-tensor and path-resolution logic; the GPU end-to-end path
(trunk split, head re-scoring) is validated by running
``workflows/scripts/inference/cache_boltz2_confidence.py`` on a few entries.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from confidock.boltz.inference.cache_runner import build_x_pred, feats_for_confidence
from confidock.boltz.inference.confidence_cache import (
    CONFIDENCE_FEAT_KEYS,
    EntryCache,
    compact_feats,
    expand_feats,
    load_entry_cache,
    save_decoy_coords,
    save_entry_cache,
)
from confidock.boltz.inference.decoy_sources import SOURCES, collect_decoys


def _one_hot(n_rows: int, width: int, *, empty_rows: tuple[int, ...] = ()) -> torch.Tensor:
    """Build a (1, n_rows, width) one-hot, optionally with all-zero rows."""
    out = torch.zeros((1, n_rows, width), dtype=torch.int64)
    for i in range(n_rows):
        if i not in empty_rows:
            out[0, i, i % width] = 1
    return out


class TestCompactFeats:
    """One-hot compaction must be exactly reversible."""

    def test_one_hot_round_trip(self):
        feats = {"atom_to_token": _one_hot(64, 16)}
        back = expand_feats(compact_feats(feats))
        assert torch.equal(back["atom_to_token"], feats["atom_to_token"])

    def test_empty_rows_round_trip(self):
        """Padded atoms give all-zero rows; they must not become a 1 in column 0."""
        feats = {"atom_to_token": _one_hot(64, 16, empty_rows=(3, 17, 63))}
        compact = compact_feats(feats)
        assert int(compact["atom_to_token"][0, 3]) == -1
        back = expand_feats(compact)
        assert torch.equal(back["atom_to_token"], feats["atom_to_token"])
        assert int(back["atom_to_token"][0, 3].sum()) == 0

    def test_compaction_shrinks_one_hots(self):
        feats = {"atom_to_token": _one_hot(512, 128)}
        before = feats["atom_to_token"].numel() * feats["atom_to_token"].element_size()
        compact = compact_feats(feats)
        after = sum(v.numel() * v.element_size() for v in compact.values())
        assert after < before / 10

    def test_ambiguous_row_raises(self):
        """A row summing to 2 cannot be stored as an index; fail rather than corrupt."""
        bad = _one_hot(8, 4)
        bad[0, 2, 3] = 1  # row 2 now sums to 2
        with pytest.raises(ValueError, match="not one-hot"):
            compact_feats({"atom_to_token": bad})

    def test_int64_narrowing_is_reversible(self):
        feats = {"residue_index": torch.arange(64, dtype=torch.int64)[None]}
        compact = compact_feats(feats)
        assert compact["residue_index"].dtype == torch.int32
        back = expand_feats(compact)
        assert back["residue_index"].dtype == torch.int64
        assert torch.equal(back["residue_index"], feats["residue_index"])

    def test_bool_masks_preserved(self):
        feats = {"frame_resolved_mask": torch.tensor([[True, False, True]])}
        back = expand_feats(compact_feats(feats))
        assert back["frame_resolved_mask"].dtype == torch.bool
        assert torch.equal(back["frame_resolved_mask"], feats["frame_resolved_mask"])


class TestEntryCacheIO:
    """Saving and loading an entry cache."""

    def _cache(self) -> EntryCache:
        return EntryCache(
            entry="1abc_HLA",
            s_inputs=torch.randn(1, 8, 16),
            s=torch.randn(1, 8, 16),
            z=torch.randn(1, 8, 8, 4),
            feats={
                "atom_to_token": _one_hot(32, 8),
                "token_pad_mask": torch.ones(1, 8),
                "residue_index": torch.arange(8, dtype=torch.int64)[None],
            },
            metadata={"n_tokens": 8, "seed": 42},
        )

    def test_round_trip_preserves_shapes_and_metadata(self, tmp_path):
        cache = self._cache()
        path = tmp_path / "1abc_HLA.pt"
        save_entry_cache(cache, path)
        loaded = load_entry_cache(path)

        assert loaded.entry == cache.entry
        assert loaded.s.shape == cache.s.shape
        assert loaded.z.shape == cache.z.shape
        assert loaded.metadata["seed"] == 42
        assert torch.equal(loaded.feats["atom_to_token"], cache.feats["atom_to_token"])

    def test_fp16_storage_is_approximate_not_exact(self, tmp_path):
        cache = self._cache()
        path = tmp_path / "e.pt"
        save_entry_cache(cache, path, dtype=torch.float16)
        loaded = load_entry_cache(path)
        assert torch.allclose(loaded.z, cache.z, atol=1e-2)

    def test_fp32_storage_is_exact(self, tmp_path):
        cache = self._cache()
        path = tmp_path / "e.pt"
        save_entry_cache(cache, path, dtype=torch.float32)
        loaded = load_entry_cache(path)
        assert torch.equal(loaded.z, cache.z)

    def test_no_temp_file_left_behind(self, tmp_path):
        save_entry_cache(self._cache(), tmp_path / "e.pt")
        assert list(tmp_path.glob("*.tmp*")) == []


class TestDecoyCoordsIO:
    def test_saves_to_the_requested_path(self, tmp_path):
        """np.savez_compressed appends .npz; the temp name must not defeat it."""
        path = tmp_path / "decoy.npz"
        coords = np.random.default_rng(0).standard_normal((100, 3)).astype(np.float32)
        present = np.ones(100, dtype=bool)
        save_decoy_coords(path, coords, present, {"dockq": 0.5})

        assert path.is_file()
        assert list(tmp_path.glob("*.tmp*")) == []

    def test_coords_stored_as_float32(self, tmp_path):
        """fp16 quantises to ~0.016 A at these magnitudes, which moves the scores."""
        path = tmp_path / "decoy.npz"
        coords = (np.random.default_rng(1).standard_normal((50, 3)) * 100).astype(np.float32)
        save_decoy_coords(path, coords, np.ones(50, dtype=bool), {})

        data = np.load(path)
        assert data["coords"].dtype == np.float32
        assert np.array_equal(data["coords"], coords)


class TestBuildXPred:
    """x_pred must match the featurizer's padded atom width."""

    def test_pads_to_the_featurizer_width(self):
        feats = {"coords": torch.zeros(1, 1, 2752, 3)}
        coords = np.random.default_rng(2).standard_normal((2725, 3)).astype(np.float32)
        x = build_x_pred(coords, feats)

        assert x.shape == (1, 1, 2752, 3)
        assert torch.allclose(x[0, 0, :2725], torch.from_numpy(coords))
        assert torch.all(x[0, 0, 2725:] == 0)  # padded rows stay at zero, as Boltz does

    def test_exact_width_is_unchanged(self):
        feats = {"coords": torch.zeros(1, 1, 100, 3)}
        coords = np.random.default_rng(3).standard_normal((100, 3)).astype(np.float32)
        x = build_x_pred(coords, feats)
        assert torch.allclose(x[0, 0], torch.from_numpy(coords))

    def test_too_many_atoms_raises(self):
        feats = {"coords": torch.zeros(1, 1, 50, 3)}
        with pytest.raises(ValueError, match="allocated only"):
            build_x_pred(np.zeros((60, 3), dtype=np.float32), feats)

    def test_accepts_a_tensor(self):
        feats = {"coords": torch.zeros(1, 1, 10, 3)}
        x = build_x_pred(torch.randn(10, 3), feats)
        assert x.shape == (1, 1, 10, 3)
        assert x.dtype == torch.float32


class TestFeatsForConfidence:
    def test_adds_a_coords_placeholder(self):
        """CONFIDENCE_FEAT_KEYS excludes coords, but build_x_pred reads its width."""
        cached = {"atom_pad_mask": torch.ones(1, 2752)}
        feats = feats_for_confidence(cached)
        assert feats["coords"].shape == (1, 1, 2752, 3)

    def test_does_not_clobber_existing_coords(self):
        coords = torch.randn(1, 1, 10, 3)
        feats = feats_for_confidence({"atom_pad_mask": torch.ones(1, 10), "coords": coords})
        assert torch.equal(feats["coords"], coords)

    def test_does_not_mutate_the_input(self):
        cached = {"atom_pad_mask": torch.ones(1, 10)}
        feats_for_confidence(cached)
        assert "coords" not in cached

    def test_coords_is_not_a_cached_key(self):
        assert "coords" not in CONFIDENCE_FEAT_KEYS


class TestDecoySources:
    def test_all_four_sources_are_registered(self):
        assert {s.name for s in SOURCES} == {
            "boltz2",
            "esmfold2",
            "protenix",
            "perturbed",
        }

    def test_protenix_puts_the_seed_in_the_id(self):
        """Protenix reuses sample_0.cif across seeds, so ids collide without it."""
        protenix = next(s for s in SOURCES if s.name == "protenix")
        assert "seed" in protenix.id_columns

    def test_missing_csv_is_skipped_not_fatal(self, tmp_path):
        assert collect_decoys(tmp_path) == []
