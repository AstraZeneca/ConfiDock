"""Tests for the vendored ablation metric suite and its adapter.

The fine-tune logs two metric families over the same predictions: this trainer's
own :func:`~.trainer.correlation_report` and the SimpleGNN ablation's suite, copied
verbatim into :mod:`confidock.boltz.finetune.gnn_metrics`.  The point of the
copy is that a Boltz confidence run and a GNN ablation arm can be read off one axis,
which only holds while the key names and the group conventions stay identical.  So
these tests pin the produced key set and pin the one place the two families are
*expected* to disagree (degenerate groups).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "scripts" / "training"))

from confidock.boltz.finetune import gnn_metrics
from confidock.boltz.finetune.cache_dataset import EntryGroupedCacheDataset
from confidock.boltz.finetune.metric_suite import (
    external_suite_prefix,
    suite_from_rows,
    train_std_bin_histogram,
)
from confidock.boltz.finetune.trainer import correlation_report

SOURCES = ("boltz2", "esmfold2", "protenix")


def _rows(
    n_entries: int = 5,
    n_per_source: int = 3,
    *,
    flat_entry: bool = False,
) -> list[dict[str, object]]:
    """Build deterministic per-decoy rows shaped like ``predictions_*.csv``.

    Entry ``i`` gets DockQ spread ``0.02 * (i + 1)``, so the natives land in several
    of :data:`gnn_metrics.DOCKQ_STD_BINS` rather than piling into one -- otherwise
    the ``stdbin_*`` keys would only ever be exercised for a single bin.  The
    prediction is the label plus a fixed, entry-dependent wobble, so correlations
    are high but not degenerate and no metric hits an exactly-tied branch.

    With ``flat_entry`` a sixth native is appended whose DockQ is constant, which is
    the case the two per-native conventions treat differently.
    """
    rows: list[dict[str, object]] = []
    for i in range(n_entries):
        spread = 0.02 * (i + 1)
        for s, source in enumerate(SOURCES):
            for k in range(n_per_source):
                rank = s * n_per_source + k
                dockq = 0.3 + spread * rank
                rows.append(
                    {
                        "entry": f"e{i}",
                        "source": source,
                        "dockq": round(dockq, 6),
                        "dockq_pred": round(dockq + 0.01 * ((rank % 3) - 1), 6),
                    }
                )
    if flat_entry:
        for s, source in enumerate(SOURCES):
            for k in range(n_per_source):
                rows.append(
                    {
                        "entry": "flat",
                        "source": source,
                        "dockq": 0.5,
                        "dockq_pred": round(0.4 + 0.01 * (s * n_per_source + k), 6),
                    }
                )
    return rows


# Frozen key set for the fixture above.  Regenerating this by hand is the intended
# cost: if the vendored copy or the adapter starts emitting different names, the
# ablation's W&B queries stop finding this run's numbers, and that must surface here
# rather than as an empty chart three days into a 40-hour job.
EXPECTED_KEYS = frozenset(
    {
        "val/pearson_r",
        "val/spearman_r",
        "val/kendall_tau",
        "val/ndcg",
        "val/rmse",
        "val/mae",
        "val/tier_accuracy",
        "val/auc_0.23",
        "val/auc_0.49",
        "val/pred_mean",
        "val/true_mean",
        "val/pred_std",
        "val/true_std",
        "val/mean_shift",
        "val/std_ratio",
        "val/ks_stat",
        "val/ks_pvalue",
        "val/wasserstein",
        "val/per_native_spearman_mean",
        "val/per_native_spearman_std",
        "val/per_native_spearman_mean_rankable",
        "val/per_native_spearman_std_rankable",
        "val/per_native_n_rankable",
        "val/per_native_kendall_mean",
        "val/per_native_kendall_std",
        "val/per_native_ndcg_mean",
        "val/per_native_ndcg_std",
        "val/per_native_ks_mean",
        "val/per_native_ks_std",
        "val/per_native_wasserstein_mean",
        "val/per_native_wasserstein_std",
        "val/per_native_std_ratio_mean",
        "val/per_native_std_ratio_std",
        "val/per_native_ks_mean_rankable",
        "val/per_native_ks_std_rankable",
        "val/per_native_wasserstein_mean_rankable",
        "val/per_native_wasserstein_std_rankable",
        "val/per_native_spearman_mean_permodel",
        "val/per_native_spearman_min_permodel",
        "val/per_native_spearman_mean_permodel_rankable",
    }
    | {f"val/top{k}_success_{t}" for k in (1, 3, 5, 10, 15, 20) for t in ("0_23", "0_49", "0_80")}
    | {
        f"val/model_{m}_{suffix}"
        for m in SOURCES
        for suffix in (
            "spearman",
            "kendall",
            "n_samples",
            "per_native_spearman_mean",
            "per_native_spearman_std",
            "per_native_spearman_mean_rankable",
            "per_native_kendall_mean",
            "per_native_kendall_std",
            "per_native_ndcg_mean",
            "per_native_ndcg_std",
        )
    }
    | {
        f"val/stdbin_{stat}/{tag}"
        for tag in ("0.00_0.02", "0.02_0.05", "0.05_0.10", "0.10_0.20", "0.20_1.01")
        for stat in ("n_complexes", "n_decoys")
    }
    | {
        f"val/stdbin_{stat}/{tag}"
        # Only bins that hold at least one native get correlation keys; the
        # fixture's per-entry spreads land in the top three bins, and the two
        # flattest bins stay empty and so carry counts only.
        for tag in ("0.05_0.10", "0.10_0.20", "0.20_1.01")
        for stat in ("global_spearman", "pernative_spearman", "pernative_spearman_std")
    }
)


def test_suite_key_set_is_the_ablation_key_set() -> None:
    """The adapter emits exactly the ablation's names under the given prefix."""
    suite = suite_from_rows(_rows(), "val")

    assert set(suite) == EXPECTED_KEYS
    assert all(isinstance(v, float) for v in suite.values())


def test_prefix_needs_no_trailing_slash() -> None:
    """``"val_easy"`` and ``"val_easy/"`` are the same request."""
    rows = _rows()

    assert set(suite_from_rows(rows, "val_easy")) == set(suite_from_rows(rows, "val_easy/"))
    assert all(key.startswith("val_easy/") for key in suite_from_rows(rows, "val_easy"))


def test_empty_rows_log_nothing() -> None:
    """A skipped evaluation must not log a wall of NaN keys."""
    assert suite_from_rows([], "val") == {}


def test_per_entry_and_per_native_agree_without_degenerate_groups() -> None:
    """The two conventions coincide when every native is rankable.

    This is the claim the fine-tune's logs rest on: ``val/per_entry/dockq_pred``
    (checkpoint selection, and what the earlier BCE run logged) and
    ``val/per_native_spearman_mean`` (the ablation table) are the same quantity
    whenever no group is too small or too flat to correlate.
    """
    rows = _rows()
    report = correlation_report(rows)
    suite = suite_from_rows(rows, "val")

    assert report["per_entry_n"]["dockq_pred"] == len({r["entry"] for r in rows})
    assert suite["val/per_native_spearman_mean"] == pytest.approx(report["per_entry"]["dockq_pred"], abs=1e-6)


def test_flat_native_drags_the_ablation_number_down() -> None:
    """A constant-DockQ native scores 0.0 in the suite and is dropped by the report.

    Both numbers are logged precisely because of this, so the gap has to be a known
    quantity rather than a surprise: the suite's mean is the report's mean diluted by
    one zero.
    """
    rows = _rows(flat_entry=True)
    report = correlation_report(rows)
    suite = suite_from_rows(rows, "val")
    n_entries = len({r["entry"] for r in rows})

    # The report excludes the flat native outright; the suite keeps it at 0.0.
    assert report["per_entry_n"]["dockq_pred"] == n_entries - 1
    assert suite["val/per_native_spearman_mean"] < report["per_entry"]["dockq_pred"]
    assert suite["val/per_native_spearman_mean"] == pytest.approx(
        report["per_entry"]["dockq_pred"] * (n_entries - 1) / n_entries, abs=1e-6
    )
    # The rankable subsets use the same threshold on the same side (true DockQ), so
    # they drop the same native and do agree.
    assert suite["val/per_native_spearman_mean_rankable"] == pytest.approx(
        report["per_entry_rankable"]["dockq_pred"], abs=1e-6
    )
    assert int(suite["val/per_native_n_rankable"]) == report["n_entries_rankable"]


def test_std_bin_histogram_counts_complexes_and_decoys() -> None:
    """The training footprint splits natives by true-DockQ spread, GNN key names."""
    records = {
        "flat": [{"dockq": 0.5}, {"dockq": 0.5}, {"dockq": 0.5}],  # std 0.0   -> bin 0
        "wide": [{"dockq": 0.0}, {"dockq": 0.5}, {"dockq": 1.0}],  # std 0.408 -> bin 4
        "mid": [{"dockq": 0.40}, {"dockq": 0.52}],  # std 0.06  -> bin 2
    }

    hist = train_std_bin_histogram(records)

    # Bin contents are counts, floats only because the metrics backend takes floats,
    # so they are compared as the integers they are rather than with a tolerance.
    assert int(hist["train/stdbin_n_complexes/0.00_0.02"]) == 1
    assert int(hist["train/stdbin_n_decoys/0.00_0.02"]) == 3
    assert int(hist["train/stdbin_n_complexes/0.05_0.10"]) == 1
    assert int(hist["train/stdbin_n_decoys/0.05_0.10"]) == 2
    assert int(hist["train/stdbin_n_complexes/0.20_1.01"]) == 1
    assert int(hist["train/stdbin_n_decoys/0.20_1.01"]) == 3
    assert int(hist["train/stdbin_n_complexes/0.02_0.05"]) == 0
    assert int(hist["train/stdbin_n_decoys/0.02_0.05"]) == 0
    # Every complex and decoy lands in exactly one bin.
    assert int(sum(v for k, v in hist.items() if "n_complexes" in k)) == len(records)
    assert int(sum(v for k, v in hist.items() if "n_decoys" in k)) == 8
    # Same bins, same tags as the eval-side slice, so the two can be read together.
    suite_tags = {k.rsplit("/", 1)[1] for k in suite_from_rows(_rows(), "val") if "stdbin_" in k}
    assert {k.rsplit("/", 1)[1] for k in hist} == suite_tags


def test_std_bin_histogram_is_empty_for_no_records() -> None:
    """Bins are still emitted, at zero, so the chart exists from step 0."""
    hist = train_std_bin_histogram({})

    assert len(hist) == 2 * (len(gnn_metrics.DOCKQ_STD_BINS) - 1)
    assert all(int(count) == 0 for count in hist.values())


class _StubDataset(EntryGroupedCacheDataset):
    """A shard carrying just the state the footprint logging reads off it.

    A real subclass, so the call below is given the type ``_log_training_footprint``
    declares, but ``__init__`` is not chained: the real one scans a cache directory
    for entries that do not exist here.  Only the two private fields are set, and
    ``records``/``n_decoys``/``item_sizes`` then come from the real properties --
    which is the point, since those are what the footprint reads.
    """

    def __init__(self, records: dict[str, list[dict[str, float]]]):
        self._records = records
        # One item per entry, holding all of that entry's decoys.
        self._items = [(entry, tuple(range(len(rows)))) for entry, rows in records.items()]


def test_training_footprint_logs_bins_and_split_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run records its own training footprint under the ablation's key names.

    Single rank here, so ``sum_scalar`` is the identity and this checks the keys and
    the arithmetic; the multi-rank sum is the same call the epoch loop already uses
    for its step counters.
    """
    import finetune_confidence_ddp as ddp

    train = _StubDataset(
        {
            "wide": [{"dockq": 0.0}, {"dockq": 0.5}, {"dockq": 1.0}],
            "flat": [{"dockq": 0.5}, {"dockq": 0.5}],
        }
    )
    val = _StubDataset({"v": [{"dockq": 0.1}, {"dockq": 0.9}]})
    logged: dict[str, float] = {}
    monkeypatch.setattr(ddp, "log_metrics", lambda metrics, step=None: logged.update(metrics))

    ddp._log_training_footprint(train, val, device="cpu")

    # Counts again, so integers; only ``decoys_per_item_planned`` is a real float.
    assert int(logged["n_train_samples"]) == 5
    assert int(logged["n_val_samples"]) == 2
    assert int(logged["n_train_complexes"]) == 2
    assert int(logged["n_val_complexes"]) == 1
    assert int(logged["train/stdbin_n_complexes/0.20_1.01"]) == 1
    assert int(logged["train/stdbin_n_complexes/0.00_0.02"]) == 1
    assert int(sum(v for k, v in logged.items() if "stdbin_n_decoys" in k)) == train.n_decoys
    # The planned decoys-per-step figure the per-step counters are read against.
    assert int(logged["n_train_items"]) == 2
    assert logged["train/decoys_per_item_planned"] == pytest.approx(2.5)
    assert int(logged["train/single_decoy_items_planned"]) == 0


@pytest.mark.parametrize(
    ("cache_dir", "expected"),
    [
        ("/data/sabdab_val_easy/confidence_cache", "val_easy"),
        ("/data/sabdab_val_easy", "val_easy"),
        ("/data/boltz_trunk_cache/confidence_cache", "external"),
        ("/data/some_other_set", "external"),
    ],
)
def test_external_suite_prefix(cache_dir: str, expected: str) -> None:
    """Only a val_easy cache may borrow the name the ablation table reports under."""
    assert external_suite_prefix(Path(cache_dir)) == expected


def test_suite_numbers_are_the_vendored_functions() -> None:
    """The adapter is plumbing only -- no re-derivation of the metrics themselves."""
    rows = _rows()
    pred = np.array([r["dockq_pred"] for r in rows], dtype=float)
    true = np.array([r["dockq"] for r in rows], dtype=float)
    suite = suite_from_rows(rows, "val")

    assert suite["val/spearman_r"] == pytest.approx(
        gnn_metrics.spearman_r(torch.tensor(pred), torch.tensor(true)), abs=1e-6
    )
    assert suite["val/rmse"] == pytest.approx(float(np.sqrt(np.mean((pred - true) ** 2))), abs=1e-5)
