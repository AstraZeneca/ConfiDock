"""Tests for loading a fine-tune checkpoint back, which the OOD pass depends on.

``confidence_ft_best.pt`` is written by rank 0 at the best epoch, and the external
report is only meaningful if those exact weights come back.  The failure mode is
silent: a checkpoint that half-loads still produces a plausible-looking Spearman,
just of the wrong model.  So the round trip is pinned rather than assumed, including
the head-only case where the payload carries no ``confidence_module`` at all.

CPU-only, with a stub in place of the 506M-parameter Boltz model.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1] / "workflows" / "scripts"
sys.path.insert(0, str(WORKFLOWS_ROOT / "training"))

from finetune_confidence_ddp import _load_ft_checkpoint  # noqa: E402

from confidock.boltz.finetune.dockq_head import DockQHead  # noqa: E402


class _StubModel:
    """Only the attribute ``_load_ft_checkpoint`` touches."""

    def __init__(self):
        self.confidence_module = nn.Linear(4, 4)


def _head() -> DockQHead:
    return DockQHead(token_s=8, token_z=4, hidden=8, dropout=0.0)


def _payload(head: DockQHead, model: _StubModel | None, *, epoch: int = 3) -> dict:
    blob = {"dockq_head": head.state_dict(), "epoch": epoch, "token_s": 8, "token_z": 4}
    if model is not None:
        blob["confidence_module"] = model.confidence_module.state_dict()
    return blob


def _same(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


class TestLoadFtCheckpoint:
    def test_restores_both_the_head_and_the_confidence_module(self, tmp_path):
        saved_head, saved_model = _head(), _StubModel()
        path = tmp_path / "confidence_ft_best.pt"
        torch.save(_payload(saved_head, saved_model), path)

        # Fresh objects with different random init, so an unloaded tensor is visible.
        head, model = _head(), _StubModel()
        assert not _same(head, saved_head)
        assert not _same(model.confidence_module, saved_model.confidence_module)

        epoch = _load_ft_checkpoint(path, model, head, device="cpu")
        assert epoch == 3
        assert _same(head, saved_head)
        assert _same(model.confidence_module, saved_model.confidence_module)

    def test_a_head_only_checkpoint_leaves_the_pretrained_module_alone(self, tmp_path, caplog):
        """``--freeze_confidence`` writes no ``confidence_module``.

        The pretrained weights are then already correct, so the load must be a no-op
        on that module rather than a KeyError -- the probe-baseline checkpoints are
        exactly this shape.
        """
        saved_head = _head()
        path = tmp_path / "head_only.pt"
        torch.save(_payload(saved_head, None), path)

        head, model = _head(), _StubModel()
        before = {k: v.clone() for k, v in model.confidence_module.state_dict().items()}
        with caplog.at_level(logging.INFO):
            _load_ft_checkpoint(path, model, head, device="cpu")

        assert _same(head, saved_head)
        after = model.confidence_module.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
        assert "no confidence_module" in caplog.text
        # Its mode is left alone as well: nothing was loaded into it, so nothing about
        # how it is meant to be run has been decided here.
        assert model.confidence_module.training

    def test_both_loaded_modules_come_back_in_eval_mode(self, tmp_path):
        """A checkpoint is loaded to be scored, and the mode decides what scoring means.

        In ``train()`` the confidence pairformer keeps dropout on and skips the chunked
        triangular-attention path, so an implicit mode would leave the OOD number
        depending on whichever mode the module happened to be left in.
        """
        path = tmp_path / "confidence_ft_best.pt"
        torch.save(_payload(_head(), _StubModel()), path)

        head, model = _head(), _StubModel()
        head.train()
        model.confidence_module.train()

        _load_ft_checkpoint(path, model, head, device="cpu")

        assert not head.training
        assert not model.confidence_module.training

    def test_a_missing_epoch_defaults_to_zero_rather_than_raising(self, tmp_path):
        """Only used for a log line, so it must not be the thing that kills the pass."""
        path = tmp_path / "no_epoch.pt"
        torch.save({"dockq_head": _head().state_dict()}, path)
        assert _load_ft_checkpoint(path, _StubModel(), _head(), device="cpu") == 0

    def test_a_shape_mismatch_is_loud(self, tmp_path):
        """A checkpoint from a different --hidden must not silently half-load."""
        path = tmp_path / "wrong_width.pt"
        torch.save(_payload(DockQHead(token_s=8, token_z=4, hidden=16, dropout=0.0), None), path)
        with pytest.raises(RuntimeError):
            _load_ft_checkpoint(path, _StubModel(), _head(), device="cpu")
