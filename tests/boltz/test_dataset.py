"""Unit tests for Boltz fine-tuning dataset."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
import torch

from confidock.boltz.dataset import BoltzFinetuneDataset

REQUIRED_KEYS = {
    "coords",
    "atom_pad_mask",
    "atom_resolved_mask",
    "token_pad_mask",
    "token_index",
    "residue_index",
    "chain_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "mol_type",
    "token_bonds",
    "msa",
    "msa_mask",
    "msa_profile",
    "deletion_mean",
    "frames_idx",
    "frame_resolved_mask",
    "pdb_id",
    "idx_dataset",
}


def _distinct(tensor: torch.Tensor) -> set[float]:
    """Every distinct value anywhere in ``tensor``, ignoring its layout.

    These are whole-tensor domain checks -- "this mask only ever holds 0 or 1" --
    so the flattening is the point rather than an accident of the default
    ``torch.unique(dim=None)``.  Shapes are pinned separately by
    :func:`test_dataset_tensor_shapes`.
    """
    return set(tensor.flatten().tolist())


@pytest.fixture
def prepared_dataset(tmp_path: Path) -> BoltzFinetuneDataset:
    structure_dir = tmp_path / "structures"
    processed_dir = tmp_path / "processed"
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    fixture = Path("tests/boltz/fixtures/test_structure.pdb")
    target = structure_dir / "test_structure.pdb"
    shutil.copyfile(fixture, target)

    manifest = tmp_path / "train.csv"
    manifest.write_text("pdb_id,split,resolution\ntest_structure,train,1.8\n", encoding="utf-8")

    return BoltzFinetuneDataset(
        structure_dir=str(structure_dir),
        processed_dir=str(processed_dir),
        split_manifest=str(manifest),
        max_tokens=512,
        max_atoms=4096,
    )


def test_dataset_returns_required_keys(prepared_dataset: BoltzFinetuneDataset) -> None:
    item = prepared_dataset[0]
    assert REQUIRED_KEYS.issubset(set(item.keys()))


def test_dataset_tensor_shapes(prepared_dataset: BoltzFinetuneDataset) -> None:
    item = prepared_dataset[0]

    l_atoms = item["coords"].shape[2]
    l_tokens = item["token_pad_mask"].shape[1]

    assert item["coords"].shape == (1, 1, l_atoms, 3)
    assert item["atom_pad_mask"].shape == (1, l_atoms)
    assert item["atom_resolved_mask"].shape == (1, l_atoms)

    assert item["token_pad_mask"].shape == (1, l_tokens)
    assert item["token_index"].shape == (1, l_tokens)
    assert item["residue_index"].shape == (1, l_tokens)
    assert item["chain_index"].shape == (1, l_tokens)
    assert item["asym_id"].shape == (1, l_tokens)
    assert item["entity_id"].shape == (1, l_tokens)
    assert item["sym_id"].shape == (1, l_tokens)
    assert item["mol_type"].shape == (1, l_tokens)

    assert item["token_bonds"].shape == (1, l_tokens, l_tokens, 1)
    assert item["msa"].shape == (1, 1, l_tokens)
    assert item["msa_mask"].shape == (1, 1, l_tokens)
    assert item["msa_profile"].shape == (1, l_tokens, 22)
    assert item["deletion_mean"].shape == (1, l_tokens)

    assert item["frames_idx"].shape == (1, 1, l_tokens, 3)
    assert item["frame_resolved_mask"].shape == (1, 1, l_tokens)
    assert item["idx_dataset"].shape == (1,)


def test_dataset_value_ranges(prepared_dataset: BoltzFinetuneDataset) -> None:
    item = prepared_dataset[0]

    atom_pad_vals = _distinct(item["atom_pad_mask"])
    token_pad_vals = _distinct(item["token_pad_mask"])
    atom_resolved_vals = _distinct(item["atom_resolved_mask"])
    mol_type_vals = _distinct(item["mol_type"])

    assert atom_pad_vals.issubset({0.0, 1.0})
    assert token_pad_vals.issubset({0.0, 1.0})
    assert atom_resolved_vals.issubset({0.0, 1.0})
    assert mol_type_vals.issubset({0, 1, 2, 3})
    assert torch.isfinite(item["coords"]).all()


def test_dataset_caching(prepared_dataset: BoltzFinetuneDataset) -> None:
    _ = prepared_dataset[0]
    cache_file = prepared_dataset.processed_dir / "test_structure.pt"
    assert cache_file.exists()

    # ``st_mtime_ns`` rather than ``st_mtime``: the assertion is "this file was not
    # rewritten", which needs to be exact, and the integer nanosecond field compares
    # exactly where the float seconds field only compares approximately.
    mtime_before = cache_file.stat().st_mtime_ns
    time.sleep(0.05)
    _ = prepared_dataset[0]
    mtime_after = cache_file.stat().st_mtime_ns

    assert mtime_before == mtime_after


def test_missing_pdb_is_handled_gracefully(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text("pdb_id,split,resolution\nmissing_id,train,2.0\n", encoding="utf-8")

    dataset = BoltzFinetuneDataset(
        structure_dir=str(tmp_path / "structures"),
        processed_dir=str(tmp_path / "processed"),
        split_manifest=str(manifest),
    )

    assert len(dataset) == 0
    with pytest.raises(IndexError):
        _ = dataset[0]
