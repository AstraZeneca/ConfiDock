"""Tests for confidock.dockq_predictor.training.metrics module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from confidock.dockq_predictor.training import metrics as m


def _t(*values: float) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Correlation metrics and their degenerate conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [m.spearman_r, m.kendall_tau])
def test_rank_correlations_are_one_for_any_monotone_relation(fn) -> None:
    """Rank-based, so a non-linear but order-preserving map still scores 1."""
    assert fn(_t(0.1, 0.2, 0.3, 0.4), _t(0.05, 0.3, 0.31, 0.9)) == pytest.approx(1.0)


def test_pearson_needs_a_linear_relation_not_just_the_right_order() -> None:
    """The complement: pearson_r is 1 only under an affine map.

    Worth pinning because the two are logged side by side -- a pearson well below
    the Spearman means the ranking is right but the SCALE is not.
    """
    true = _t(0.05, 0.3, 0.31, 0.9)
    assert m.pearson_r(true * 0.5 + 0.2, true) == pytest.approx(1.0)
    assert m.pearson_r(_t(0.1, 0.2, 0.3, 0.4), true) < 0.95


@pytest.mark.parametrize("fn", [m.spearman_r, m.kendall_tau])
def test_rank_correlations_are_minus_one_when_reversed(fn) -> None:
    """A perfectly wrong ordering is -1, not 0."""
    assert fn(_t(0.4, 0.3, 0.2, 0.1), _t(0.1, 0.2, 0.3, 0.4)) == pytest.approx(-1.0)


@pytest.mark.parametrize("fn", [m.spearman_r, m.kendall_tau, m.pearson_r])
def test_correlations_return_zero_rather_than_nan_on_degenerate_input(fn) -> None:
    """A constant group must not poison an aggregate mean with NaN.

    Every per-group metric here is averaged over complexes, so one NaN would make
    the whole logged number NaN.
    """
    assert fn(_t(0.5, 0.5, 0.5), _t(0.1, 0.2, 0.3)) == pytest.approx(0.0)
    assert not np.isnan(fn(_t(0.5), _t(0.1)))


@pytest.mark.parametrize("fn", [m.spearman_r, m.kendall_tau, m.ndcg_score])
def test_metrics_needing_two_points_return_zero_for_one(fn) -> None:
    """A singleton group carries no ranking information."""
    assert fn(_t(0.5), _t(0.5)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def test_rmse_and_mae_measure_absolute_error() -> None:
    """The Huber term's yardstick: these must be in DockQ units."""
    pred, true = _t(0.5, 0.5), _t(0.1, 0.9)
    assert m.mae(pred, true) == pytest.approx(0.4)
    assert m.rmse(pred, true) == pytest.approx(0.4)
    assert m.rmse(_t(0.0, 0.6), _t(0.0, 0.0)) == pytest.approx(0.6 / np.sqrt(2), rel=1e-5)


def test_rmse_and_mae_are_zero_for_exact_predictions() -> None:
    """Sanity floor."""
    exact = _t(0.1, 0.5, 0.9)
    assert m.rmse(exact, exact) == pytest.approx(0.0)
    assert m.mae(exact, exact) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Distribution-shape metrics
# ---------------------------------------------------------------------------


def test_ks_statistic_is_zero_for_identical_clouds_and_one_when_disjoint() -> None:
    """sup|F_pred - F_true|, so identical samples give 0 and disjoint ranges 1."""
    same = _t(0.1, 0.2, 0.3, 0.4)
    assert m.ks_statistic(same, same) == pytest.approx(0.0)
    assert m.ks_statistic(_t(0.0, 0.1, 0.2), _t(0.8, 0.9, 1.0)) == pytest.approx(1.0)


def test_ks_metrics_use_no_difference_defaults_when_undefined() -> None:
    """Statistic -> 0.0 and p-value -> 1.0, i.e. "no detected difference"."""
    assert m.ks_statistic(_t(0.5), _t(0.5)) == pytest.approx(0.0)
    assert m.ks_pvalue(_t(0.5), _t(0.5)) == pytest.approx(1.0)


def test_ks_is_unpaired_so_a_shuffle_does_not_move_it() -> None:
    """Documented caveat: KS ignores the pred[i] <-> true[i] pairing entirely."""
    pred, true = _t(0.1, 0.4, 0.7), _t(0.2, 0.5, 0.8)
    assert m.ks_statistic(pred, true) == pytest.approx(m.ks_statistic(pred.flip(0), true))


def test_wasserstein_is_the_mean_transport_distance() -> None:
    """Reported in DockQ units: a uniform 0.2 shift costs exactly 0.2."""
    true = _t(0.1, 0.3, 0.5, 0.7)
    assert m.wasserstein_dist(true + 0.2, true) == pytest.approx(0.2, abs=1e-6)
    assert m.wasserstein_dist(true, true) == pytest.approx(0.0)


def test_distribution_metrics_name_the_failure_mode() -> None:
    """mean_shift and std_ratio say WHETHER it is a bias or a variance collapse."""
    out = m.distribution_metrics(_t(0.4, 0.5, 0.6), _t(0.1, 0.5, 0.9))
    assert out["mean_shift"] == pytest.approx(0.0, abs=1e-6)  # unbiased...
    assert out["std_ratio"] == pytest.approx(0.25, rel=1e-5)  # ...but squashed 4x
    assert set(out) == {
        "ks_stat",
        "ks_pvalue",
        "wasserstein",
        "pred_mean",
        "true_mean",
        "pred_std",
        "true_std",
        "mean_shift",
        "std_ratio",
    }


def test_distribution_metrics_guard_std_ratio_against_a_constant_target() -> None:
    """A single-complex group can have zero true spread; that must not divide by 0."""
    assert m.distribution_metrics(_t(0.4, 0.6), _t(0.5, 0.5))["std_ratio"] == pytest.approx(0.0)


def test_distribution_metrics_applies_the_prefix() -> None:
    """Keys land under val/ or test/ as the caller asks."""
    assert all(k.startswith("val/") for k in m.distribution_metrics(_t(0.1, 0.2), _t(0.3, 0.4), prefix="val/"))


# ---------------------------------------------------------------------------
# NDCG
# ---------------------------------------------------------------------------


def test_ndcg_is_one_for_a_correct_ranking_and_lower_when_wrong() -> None:
    """Ranking by pred, gains from true."""
    true = _t(0.1, 0.4, 0.9)
    assert m.ndcg_score(true, true) == pytest.approx(1.0)
    assert m.ndcg_score(true.flip(0), true) < 1.0


def test_ndcg_at_k_only_scores_the_top_k() -> None:
    """NDCG@1 sees only the single top-predicted decoy."""
    pred, true = _t(0.9, 0.1, 0.5), _t(0.2, 0.9, 0.5)
    assert m.ndcg_score(pred, true, k=1) < m.ndcg_score(true, true, k=1)


def test_ndcg_is_zero_when_every_gain_is_zero() -> None:
    """An all-zero-DockQ complex has no ideal DCG to normalise against."""
    assert m.ndcg_score(_t(0.3, 0.7), _t(0.0, 0.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CAPRI tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dockq", "tier"),
    [
        (0.0, "Unacceptable"),
        (0.229, "Unacceptable"),
        (0.23, "Acceptable"),  # CAPRI acceptable boundary
        (0.489, "Acceptable"),
        (0.49, "Medium"),  # CAPRI medium boundary
        (0.799, "Medium"),
        (0.8, "High"),  # CAPRI high boundary
        (1.0, "High"),
    ],
)
def test_dockq_tier_boundaries_are_the_capri_thresholds(dockq: float, tier: str) -> None:
    """0.23 / 0.49 / 0.8, inclusive at the lower edge."""
    assert m.dockq_tier(dockq) == tier


def test_dockq_tier_accuracy_counts_matching_tiers_not_matching_values() -> None:
    """A prediction may be off numerically and still land in the right tier."""
    # 0.30 vs 0.40 -> both Acceptable (match); 0.90 vs 0.10 -> High vs Unacceptable.
    assert m.dockq_tier_accuracy(_t(0.30, 0.90), _t(0.40, 0.10)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# top_k_success_rate
# ---------------------------------------------------------------------------


def test_top_k_selects_by_prediction_and_judges_on_truth() -> None:
    """Select by prediction, judge on truth -- the deployment question."""
    pred = _t(0.9, 0.1)  # decoy 0 looks best...
    true = _t(0.1, 0.8)  # ...but decoy 1 actually is
    assert m.top_k_success_rate(pred, true, threshold=0.49, k=1, native_ids=["a", "a"]) == pytest.approx(0.0)
    assert m.top_k_success_rate(pred, true, threshold=0.49, k=2, native_ids=["a", "a"]) == pytest.approx(1.0)


def test_top_k_is_monotonic_in_k() -> None:
    """The regression this fixed: top-3 used to be identical to top-1.

    Thresholding ``pred`` itself made ``any(pred_top3 >= t)`` == ``max(pred) >= t``,
    so k never mattered. The top-k set is a superset of top-1, so top-3 >= top-1.
    """
    pred = _t(0.9, 0.5, 0.2, 0.8, 0.4, 0.1)
    true = _t(0.1, 0.6, 0.2, 0.1, 0.9, 0.0)
    native_ids = ["a"] * 3 + ["b"] * 3
    rates = [m.top_k_success_rate(pred, true, threshold=0.49, k=k, native_ids=native_ids) for k in (1, 2, 3)]
    assert rates == sorted(rates)
    assert rates[0] < rates[-1]  # k genuinely changes the answer


def test_top_k_averages_over_complexes_not_decoys() -> None:
    """One complex succeeds, one fails -> 0.5, regardless of decoy counts."""
    pred = _t(0.9, 0.1, 0.9, 0.1, 0.5)
    true = _t(0.9, 0.1, 0.1, 0.1, 0.1)
    assert m.top_k_success_rate(pred, true, 0.49, k=1, native_ids=["a", "a", "b", "b", "b"]) == pytest.approx(0.5)


def test_top_k_handles_k_larger_than_a_complexs_decoy_count() -> None:
    """k=20 on a 2-decoy complex keeps both rather than erroring."""
    assert m.top_k_success_rate(_t(0.3, 0.7), _t(0.1, 0.9), 0.49, k=20, native_ids=["a", "a"]) == pytest.approx(1.0)


def test_top_k_treats_missing_native_ids_as_one_global_group() -> None:
    """Still well-defined without grouping, though the caller always groups."""
    assert m.top_k_success_rate(_t(0.3, 0.7), _t(0.1, 0.9), 0.49, k=1) == pytest.approx(1.0)


def test_top_k_is_zero_for_an_empty_input() -> None:
    """No complexes means no successes, not a division by zero."""
    assert m.top_k_success_rate(_t(), _t(), 0.49, k=1, native_ids=[]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# auc_at_threshold
# ---------------------------------------------------------------------------


def test_auc_is_one_when_predictions_separate_the_classes() -> None:
    """Binary labels come from TRUE DockQ >= threshold, scored by pred."""
    assert m.auc_at_threshold(_t(0.1, 0.2, 0.8, 0.9), _t(0.0, 0.1, 0.6, 0.7), 0.49) == pytest.approx(1.0)


def test_auc_is_zero_when_only_one_class_is_present() -> None:
    """AUC is undefined with one class; 0.0 flags "not measurable" in the logs."""
    assert m.auc_at_threshold(_t(0.1, 0.9), _t(0.9, 0.95), 0.49) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# per_group_metric
# ---------------------------------------------------------------------------


def test_per_group_metric_averages_over_groups() -> None:
    """Group a ranks perfectly, group b is reversed -> mean 0."""
    pred = _t(0.1, 0.9, 0.9, 0.1)
    true = _t(0.2, 0.8, 0.2, 0.8)
    mean, std = m.per_group_metric(pred, true, ["a", "a", "b", "b"], m.spearman_r)
    assert mean == pytest.approx(0.0)
    assert std == pytest.approx(1.0)


def test_per_group_metric_drops_groups_below_min_samples() -> None:
    """Singleton complexes contribute nothing rather than a spurious 0."""
    pred = _t(0.1, 0.9, 0.5)
    true = _t(0.2, 0.8, 0.5)
    mean, _ = m.per_group_metric(pred, true, ["a", "a", "singleton"], m.spearman_r)
    assert mean == pytest.approx(1.0)


def test_per_group_metric_min_true_std_excludes_near_degenerate_complexes() -> None:
    """The rankable subset: complexes whose TRUE DockQ barely varies are excluded.

    Selection is on TRUE spread only, so the subset is model-independent and
    comparable across runs.
    """
    pred = _t(0.1, 0.9, 0.9, 0.1)
    true = _t(0.10, 0.90, 0.50, 0.51)  # group b spans 0.01 DockQ
    plain, _ = m.per_group_metric(pred, true, ["a", "a", "b", "b"], m.spearman_r)
    rankable, _ = m.per_group_metric(pred, true, ["a", "a", "b", "b"], m.spearman_r, min_true_std=0.05)
    assert plain == pytest.approx(0.0)  # b's arbitrary -1 cancels a's +1
    assert rankable == pytest.approx(1.0)  # b excluded, only the real signal left


def test_per_group_metric_returns_zeros_when_no_group_qualifies() -> None:
    """An empty selection must be loggable."""
    assert m.per_group_metric(_t(0.1, 0.2), _t(0.1, 0.2), ["a", "b"], m.spearman_r) == pytest.approx((0.0, 0.0))


def test_per_group_metric_skips_nan_metric_values() -> None:
    """A metric_fn returning NaN for one group must not NaN the mean."""

    def nan_for_b(pred, true):
        return float("nan") if len(pred) == 3 else 1.0

    mean, _ = m.per_group_metric(
        _t(0.1, 0.9, 0.2, 0.4, 0.6), _t(0.1, 0.9, 0.2, 0.4, 0.6), ["a", "a", "b", "b", "b"], nan_for_b
    )
    assert mean == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# per_std_bin_metrics
# ---------------------------------------------------------------------------


def test_per_std_bin_metrics_places_complexes_by_true_dockq_spread() -> None:
    """Bin membership is set by TRUE std, so it is comparable across runs."""
    # Complex 'flat' spans 0.01 (std ~0.005 -> the 0.00-0.02 bin); 'wide' spans 0.8
    # (std ~0.4 -> the 0.20-1.01 bin).
    pred = _t(0.50, 0.51, 0.1, 0.9)
    true = _t(0.50, 0.51, 0.1, 0.9)
    out = m.per_std_bin_metrics(pred, true, ["flat", "flat", "wide", "wide"])

    assert out["stdbin_n_complexes/0.00_0.02"] == pytest.approx(1.0)
    assert out["stdbin_n_decoys/0.00_0.02"] == pytest.approx(2.0)
    assert out["stdbin_n_complexes/0.20_1.01"] == pytest.approx(1.0)
    assert out["stdbin_pernative_spearman/0.20_1.01"] == pytest.approx(1.0)


def test_per_std_bin_metrics_reports_empty_bins_without_a_score() -> None:
    """A thin bin must never be mistaken for a real signal."""
    out = m.per_std_bin_metrics(_t(0.1, 0.9), _t(0.1, 0.9), ["wide", "wide"])
    assert out["stdbin_n_complexes/0.02_0.05"] == pytest.approx(0.0)
    assert out["stdbin_n_decoys/0.02_0.05"] == pytest.approx(0.0)
    assert "stdbin_pernative_spearman/0.02_0.05" not in out
    assert "stdbin_global_spearman/0.02_0.05" not in out


def test_per_std_bin_metrics_covers_every_bin_edge_pair() -> None:
    """One count key per bin, whatever the data -- panels expect a fixed key set."""
    out = m.per_std_bin_metrics(_t(0.1, 0.9), _t(0.1, 0.9), ["a", "a"], prefix="val/")
    tags = [f"{lo:.2f}_{hi:.2f}" for lo, hi in zip(m.DOCKQ_STD_BINS[:-1], m.DOCKQ_STD_BINS[1:])]
    assert [f"val/stdbin_n_complexes/{tag}" in out for tag in tags] == [True] * len(tags)


# ---------------------------------------------------------------------------
# compute_comprehensive_metrics
# ---------------------------------------------------------------------------


def _comprehensive_inputs():
    """Two complexes x two folding models, ranked correctly within each complex."""
    pred = _t(0.2, 0.8, 0.3, 0.7, 0.25, 0.75, 0.35, 0.65)
    true = _t(0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75)
    native_ids = ["a", "a", "b", "b", "a", "a", "b", "b"]
    model_sources = ["boltz2"] * 4 + ["protenix"] * 4
    return (
        {"dockq": pred},
        {"dockq": true},
        {"native_id": native_ids, "model_source": model_sources},
    )


def test_comprehensive_metrics_reports_the_global_block_without_metadata() -> None:
    """Metadata is optional; the global metrics must still come out."""
    predictions, targets, _ = _comprehensive_inputs()
    out = m.compute_comprehensive_metrics(predictions, targets)
    for key in ("pearson_r", "spearman_r", "kendall_tau", "ndcg", "rmse", "mae", "tier_accuracy", "ks_stat"):
        assert key in out
    assert "per_native_spearman_mean" not in out


def test_comprehensive_metrics_adds_the_per_native_block_with_metadata() -> None:
    """The checkpoint monitor lives in this block, so pin its exact keys."""
    predictions, targets, metadata = _comprehensive_inputs()
    out = m.compute_comprehensive_metrics(predictions, targets, metadata, prefix="val/")

    assert out["val/per_native_spearman_mean"] == pytest.approx(1.0)
    for key in (
        "val/per_native_spearman_std",
        "val/per_native_spearman_mean_rankable",
        "val/per_native_n_rankable",
        "val/per_native_kendall_mean",
        "val/per_native_ndcg_mean",
        "val/per_native_ks_mean",
        "val/per_native_wasserstein_mean",
        "val/per_native_std_ratio_mean",
        "val/stdbin_n_complexes/0.20_1.01",
    ):
        assert key in out
    assert out["val/per_native_n_rankable"] == pytest.approx(2.0)  # both complexes span >= 0.05


def test_comprehensive_metrics_adds_per_model_and_aggregate_blocks() -> None:
    """Per-model within-model ranking, plus the two monitorable aggregates."""
    predictions, targets, metadata = _comprehensive_inputs()
    out = m.compute_comprehensive_metrics(predictions, targets, metadata)

    for model in ("boltz2", "protenix"):
        assert out[f"model_{model}_n_samples"] == pytest.approx(4.0)
        assert f"model_{model}_spearman" in out
        assert f"model_{model}_per_native_spearman_mean" in out
        assert f"model_{model}_per_native_spearman_mean_rankable" in out

    assert out["per_native_spearman_mean_permodel"] == pytest.approx(
        np.mean([out["model_boltz2_per_native_spearman_mean"], out["model_protenix_per_native_spearman_mean"]])
    )
    assert out["per_native_spearman_min_permodel"] == pytest.approx(
        min(out["model_boltz2_per_native_spearman_mean"], out["model_protenix_per_native_spearman_mean"])
    )
    assert "per_native_spearman_mean_permodel_rankable" in out


def test_comprehensive_metrics_per_native_groups_do_not_mix_folding_models() -> None:
    """The model-mixing mean is inflated by cross-model tier separation.

    Here every model ranks its own decoys backwards, but pooling the two models'
    decoys of a complex makes the mixed ordering look correct -- so the per-model
    number must be negative while the mixing one is not.
    """
    predictions = {"dockq": _t(0.2, 0.1, 0.8, 0.7)}
    targets = {"dockq": _t(0.1, 0.2, 0.7, 0.8)}
    metadata = {"native_id": ["a"] * 4, "model_source": ["boltz2", "boltz2", "protenix", "protenix"]}
    out = m.compute_comprehensive_metrics(predictions, targets, metadata)

    assert out["per_native_spearman_mean"] > 0
    assert out["model_boltz2_per_native_spearman_mean"] == pytest.approx(-1.0)
    assert out["per_native_spearman_min_permodel"] == pytest.approx(-1.0)


def test_comprehensive_metrics_prefixes_every_key() -> None:
    """Nothing may leak into the root namespace of the logger."""
    predictions, targets, metadata = _comprehensive_inputs()
    out = m.compute_comprehensive_metrics(predictions, targets, metadata, prefix="test/")
    assert [k for k in out if not k.startswith("test/")] == []


def test_comprehensive_metrics_includes_every_top_k_threshold_combination() -> None:
    """The CAPRI grid the results tables read from."""
    predictions, targets, metadata = _comprehensive_inputs()
    out = m.compute_comprehensive_metrics(predictions, targets, metadata)
    for k in (1, 3, 5, 10, 15, 20):
        for threshold in ("0_23", "0_49", "0_80"):
            assert f"top{k}_success_{threshold}" in out


def test_comprehensive_metrics_skips_a_model_with_a_single_decoy() -> None:
    """A model with <2 decoys has no correlation to report."""
    predictions = {"dockq": _t(0.2, 0.8, 0.5)}
    targets = {"dockq": _t(0.1, 0.9, 0.5)}
    metadata = {"native_id": ["a", "a", "b"], "model_source": ["boltz2", "boltz2", "esmfold2"]}
    out = m.compute_comprehensive_metrics(predictions, targets, metadata)
    assert "model_boltz2_spearman" in out
    assert "model_esmfold2_spearman" not in out
