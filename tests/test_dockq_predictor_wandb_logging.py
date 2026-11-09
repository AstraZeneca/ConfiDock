"""Tests for the W&B provenance helpers and the pred-vs-true visualisation callback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import pytest
import torch

from confidock.dockq_predictor.training.metrics import DOCKQ_STD_BINS
from confidock.dockq_predictor.training.wandb_logging import (
    DockQScatterCallback,
    file_sha256,
    train_std_bin_histogram,
)


class _FakeExperiment:
    """Records what would have been sent to W&B."""

    def __init__(self) -> None:
        self.logged: List[dict] = []

    def log(self, payload: dict) -> None:
        self.logged.append(payload)

    @property
    def keys(self) -> List[str]:
        return [k for payload in self.logged for k in payload if k != "epoch"]


class _FakeLogger:
    def __init__(self) -> None:
        self.experiment = _FakeExperiment()


class _FakeTrainer:
    """Minimal stand-in for the two Trainer attributes the callback reads."""

    def __init__(self, current_epoch: int = 4, sanity_checking: bool = False, metrics: dict | None = None) -> None:
        self.current_epoch = current_epoch
        self.sanity_checking = sanity_checking
        self.callback_metrics = metrics if metrics is not None else {}


def _outputs(n: int = 6, natives: List[str] | None = None) -> dict:
    """Return a validation_step output dict shaped like SimpleGNN's."""
    return {
        "predictions_dockq": torch.linspace(0.1, 0.9, n),
        "targets_dockq": torch.linspace(0.15, 0.85, n),
        "model_source": ["boltz2"] * n,
        "native_id": natives if natives is not None else [f"c{i // 2}" for i in range(n)],
    }


def _feed(callback: DockQScatterCallback, trainer: _FakeTrainer, outputs: dict) -> None:
    callback.on_validation_epoch_start(trainer, None)
    callback.on_validation_batch_end(trainer, None, outputs, None, 0)
    callback.on_validation_epoch_end(trainer, None)


# ---------------------------------------------------------------------------
# file_sha256: pinning the manifest content
# ---------------------------------------------------------------------------


def test_file_sha256_matches_the_real_digest_truncated(tmp_path: Path) -> None:
    """A short prefix of the true SHA-256, so it can be eyeballed in the W&B UI."""
    path = tmp_path / "manifest.csv"
    path.write_bytes(b"pdb_path,dockq\na.pdb,0.5\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    short = file_sha256(path)
    assert digest.startswith(short)
    assert len(short) == 12  # prefix AND length: together these pin it to digest[:12]
    assert file_sha256(path, length=64) == digest


def test_file_sha256_distinguishes_an_edited_manifest(tmp_path: Path) -> None:
    """The whole point: a regenerated CSV at the same path must not look identical."""
    path = tmp_path / "m.csv"
    path.write_text("a\n")
    before = file_sha256(path)
    path.write_text("b\n")
    assert file_sha256(path) != before


def test_file_sha256_returns_none_rather_than_raising(tmp_path: Path) -> None:
    """Provenance is best-effort; a missing file must not abort a training run."""
    assert file_sha256(tmp_path / "absent.csv") is None
    assert file_sha256(None) is None


# ---------------------------------------------------------------------------
# train_std_bin_histogram: the training pool's DockQ-std footprint
# ---------------------------------------------------------------------------


def _entry(native: str, dockq: float) -> dict:
    # _native_id_from_path derives the native from the filename stem's first token.
    return {"pdb_path": f"/data/{native}_sample_0.pdb", "dockq": dockq}


def test_std_histogram_counts_every_decoy_and_complex_exactly_once() -> None:
    """Bins partition the pool: the totals must equal the inputs."""
    entries = [_entry("aaaa", 0.1), _entry("aaaa", 0.9), _entry("bbbb", 0.5), _entry("cccc", 0.5), _entry("cccc", 0.5)]
    hist = train_std_bin_histogram(entries)
    assert sum(v for k, v in hist.items() if "n_decoys" in k) == len(entries)
    assert sum(v for k, v in hist.items() if "n_complexes" in k) == 3


def test_std_histogram_puts_a_flat_complex_in_the_lowest_bin() -> None:
    """Identical DockQ across decoys is std 0 -- the regime perturbed decoys target."""
    hist = train_std_bin_histogram([_entry("flat", 0.4), _entry("flat", 0.4)])
    assert hist["train/stdbin_n_complexes/0.00_0.02"] == pytest.approx(1.0)
    assert hist["train/stdbin_n_decoys/0.00_0.02"] == pytest.approx(2.0)


def test_std_histogram_puts_a_wide_complex_in_the_highest_bin() -> None:
    """A 0.1-to-0.9 spread is std ~0.4, well inside the top bin."""
    hist = train_std_bin_histogram([_entry("wide", 0.1), _entry("wide", 0.9)])
    assert hist["train/stdbin_n_complexes/0.20_1.01"] == pytest.approx(1.0)


def test_std_histogram_uses_the_same_bins_as_the_eval_metrics() -> None:
    """Train- and val-side breakdowns are only comparable on identical edges."""
    keys = set(train_std_bin_histogram([_entry("a", 0.5), _entry("a", 0.6)]))
    expected = {
        f"train/stdbin_n_{what}/{lo:.2f}_{hi:.2f}"
        for what in ("decoys", "complexes")
        for lo, hi in zip(DOCKQ_STD_BINS[:-1], DOCKQ_STD_BINS[1:])
    }
    assert keys == expected


def test_std_histogram_groups_decoys_of_one_complex_together() -> None:
    """Grouping is by native id, not by row: 4 decoys of one complex are 1 complex."""
    hist = train_std_bin_histogram([_entry("same", d) for d in (0.1, 0.4, 0.6, 0.9)])
    assert sum(v for k, v in hist.items() if "n_complexes" in k) == pytest.approx(1.0)
    assert sum(v for k, v in hist.items() if "n_decoys" in k) == pytest.approx(4.0)


def test_std_histogram_of_an_empty_pool_is_all_zeros() -> None:
    """Guards a config that filters every row away."""
    assert set(train_std_bin_histogram([]).values()) == {0.0}


# ---------------------------------------------------------------------------
# DockQScatterCallback: cadence
# ---------------------------------------------------------------------------


def test_scatter_is_logged_on_its_cadence_epoch() -> None:
    """every_n_epochs=5 fires on epoch index 4 (the 5th epoch)."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=5, table_every_n_epochs=1000, n_native_scatters=0)
    _feed(cb, _FakeTrainer(current_epoch=4), _outputs())
    assert "val/pred_vs_true" in logger.experiment.keys


def test_nothing_is_logged_off_cadence() -> None:
    """An off-cadence epoch must not render figures -- that is the cost saving."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=5, table_every_n_epochs=1000)
    _feed(cb, _FakeTrainer(current_epoch=1), _outputs())
    assert logger.experiment.logged == []


def test_nothing_is_logged_during_the_sanity_check() -> None:
    """The sanity check runs on 2 batches of an untrained model: not a data point."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1, table_every_n_epochs=1)
    _feed(cb, _FakeTrainer(current_epoch=0, sanity_checking=True), _outputs())
    assert logger.experiment.logged == []


def test_the_table_key_is_unique_per_epoch() -> None:
    """A shared key would make each epoch's table overwrite the previous one."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1000, table_every_n_epochs=1, n_native_scatters=0)
    for epoch in (0, 1):
        _feed(cb, _FakeTrainer(current_epoch=epoch), _outputs())
    assert logger.experiment.keys == ["val/pred_vs_true_table_epoch_0", "val/pred_vs_true_table_epoch_1"]


def test_the_table_carries_one_row_per_decoy() -> None:
    """The table is the queryable record of the raw points, so none may be dropped."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1000, table_every_n_epochs=1, n_native_scatters=0)
    _feed(cb, _FakeTrainer(current_epoch=0), _outputs(n=6))
    table = logger.experiment.logged[0]["val/pred_vs_true_table_epoch_0"]
    assert len(table.data) == 6
    assert table.columns == ["epoch", "native_id", "model_source", "pred_dockq", "true_dockq"]


# ---------------------------------------------------------------------------
# DockQScatterCallback: content
# ---------------------------------------------------------------------------


def test_per_native_grid_is_logged_alongside_the_global_scatter() -> None:
    """Both figures come from the same accumulated epoch."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1, table_every_n_epochs=1000, n_native_scatters=3)
    _feed(cb, _FakeTrainer(current_epoch=0), _outputs(n=6))
    assert set(logger.experiment.keys) == {"val/pred_vs_true", "val/per_native_scatters"}


def test_the_same_complexes_are_shown_every_epoch() -> None:
    """The grid tracks complexes over training, so resampling would break the read."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1, table_every_n_epochs=1000, n_native_scatters=2)
    natives = [f"c{i // 2}" for i in range(8)]
    _feed(cb, _FakeTrainer(current_epoch=0), _outputs(n=8, natives=natives))
    first = list(cb._sampled_natives)
    _feed(cb, _FakeTrainer(current_epoch=1), _outputs(n=8, natives=natives))
    assert cb._sampled_natives == first


def test_a_complex_with_one_decoy_is_never_chosen_for_the_grid() -> None:
    """A single decoy has no ranking to show; its Spearman would be undefined."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1, table_every_n_epochs=1000, n_native_scatters=5)
    natives = ["pair", "pair", "solo1", "solo2"]
    _feed(cb, _FakeTrainer(current_epoch=0), _outputs(n=4, natives=natives))
    assert cb._sampled_natives == ["pair"]


def test_accumulators_reset_between_epochs() -> None:
    """Otherwise epoch N's figure would include every earlier epoch's predictions."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1, table_every_n_epochs=1, n_native_scatters=0)
    _feed(cb, _FakeTrainer(current_epoch=0), _outputs(n=6))
    _feed(cb, _FakeTrainer(current_epoch=1), _outputs(n=6))
    assert len(logger.experiment.logged[-1]["val/pred_vs_true_table_epoch_1"].data) == 6


def test_an_epoch_with_no_batches_logs_nothing() -> None:
    """Guards an empty val loader: torch.cat on an empty list would raise."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1, table_every_n_epochs=1)
    trainer = _FakeTrainer(current_epoch=0)
    cb.on_validation_epoch_start(trainer, None)
    cb.on_validation_epoch_end(trainer, None)
    assert logger.experiment.logged == []


def test_outputs_without_predictions_are_ignored() -> None:
    """A model whose validation_step returns only a loss must not crash the callback."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1, table_every_n_epochs=1)
    trainer = _FakeTrainer(current_epoch=0)
    cb.on_validation_epoch_start(trainer, None)
    cb.on_validation_batch_end(trainer, None, {"loss": torch.tensor(0.1)}, None, 0)
    cb.on_validation_batch_end(trainer, None, torch.tensor(0.1), None, 1)  # not a dict at all
    cb.on_validation_epoch_end(trainer, None)
    assert logger.experiment.logged == []


def test_max_points_subsamples_the_global_scatter() -> None:
    """A 90k-decoy val set would otherwise render an unreadable multi-MB PNG."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1, table_every_n_epochs=1000, max_points=10, n_native_scatters=0)
    _feed(cb, _FakeTrainer(current_epoch=0), _outputs(n=100))
    assert "val/pred_vs_true" in logger.experiment.keys  # rendered, not skipped


def test_cadences_below_one_are_clamped_not_zero() -> None:
    """every_n_epochs=0 would be a modulo-by-zero on the first validation."""
    cb = DockQScatterCallback(_FakeLogger(), every_n_epochs=0, table_every_n_epochs=0)
    assert cb.every_n_epochs == 1
    assert cb.table_every_n_epochs == 1


def test_scatter_annotation_reads_the_epochs_logged_metrics() -> None:
    """The figure quotes the same numbers as the scalars, so it cannot disagree."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1, table_every_n_epochs=1000, n_native_scatters=0)
    trainer = _FakeTrainer(current_epoch=0, metrics={"val/per_native_spearman_mean": torch.tensor(0.63)})
    _feed(cb, trainer, _outputs())
    assert "val/pred_vs_true" in logger.experiment.keys


@pytest.mark.parametrize("missing", ["model_source", "native_id"])
def test_missing_metadata_degrades_to_an_uncoloured_scatter(missing: str) -> None:
    """Metadata that does not line up 1:1 is dropped, not mis-joined."""
    logger = _FakeLogger()
    cb = DockQScatterCallback(logger, every_n_epochs=1, table_every_n_epochs=1000, n_native_scatters=3)
    outputs = _outputs(n=6)
    outputs[missing] = []
    _feed(cb, _FakeTrainer(current_epoch=0), outputs)
    assert "val/pred_vs_true" in logger.experiment.keys
    if missing == "native_id":
        assert "val/per_native_scatters" not in logger.experiment.keys
