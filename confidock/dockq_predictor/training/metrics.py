"""Evaluation metrics for DockQ prediction."""

import numpy as np
import torch
from scipy.stats import kendalltau, ks_2samp, spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score


def spearman_r(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute Spearman rank correlation coefficient.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    true : torch.Tensor
        True values

    Returns
    -------
    float
        Spearman correlation coefficient
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    if len(pred) < 2:
        return 0.0

    try:
        rho, _ = spearmanr(pred, true)
        return float(rho) if not np.isnan(rho) else 0.0
    except Exception:
        return 0.0


def kendall_tau(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute Kendall's tau rank correlation coefficient.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    true : torch.Tensor
        True values

    Returns
    -------
    float
        Kendall's tau correlation coefficient
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    if len(pred) < 2:
        return 0.0

    try:
        tau, _ = kendalltau(pred, true)
        return float(tau) if not np.isnan(tau) else 0.0
    except Exception:
        return 0.0


# --- Distribution-shape metrics ----------------------------------------------
# Everything above scores RANKING or per-point error, so a model can rank decoys
# perfectly while its predictions live on the wrong scale (e.g. all squashed into
# 0.3-0.6). These compare the predicted and true DockQ DISTRIBUTIONS as a whole,
# which is exactly the failure ranking metrics are blind to.
#
# Both are two-sample / unpaired: they ignore the pred[i] <-> true[i] pairing and
# ask only whether the two value CLOUDS have the same shape. That is deliberate --
# it is the complement to Spearman, not a substitute. A model can score a perfect
# KS of 0 with terrible per-decoy predictions (right distribution, wrong
# assignment), so read these ALONGSIDE the ranking metrics, never instead of them.


def ks_statistic(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov statistic between predicted and true DockQ.

    The KS statistic is sup|F_pred(x) - F_true(x)|: the largest gap between the two
    empirical CDFs, in [0, 1] where 0 means the distributions are indistinguishable.
    Being CDF-based it is sensitive to a shift OR a squashed/stretched spread, and
    it is unitless, so it is comparable across variants and across val/val_easy.

    Returns 0.0 (the "no detected difference" value) when either side has <2 points
    or SciPy fails, matching the convention of spearman_r/kendall_tau above so a
    degenerate group never poisons an aggregate mean with NaN.
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    if len(pred) < 2 or len(true) < 2:
        return 0.0

    try:
        stat, _ = ks_2samp(pred, true)
        return float(stat) if not np.isnan(stat) else 0.0
    except Exception:
        return 0.0


def ks_pvalue(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Two-sample KS p-value: P(a gap this large | same underlying distribution).

    Small p = the predicted distribution is detectably NOT the true one. Logged
    next to ks_statistic because the statistic alone doesn't say whether a gap is
    resolvable at the current sample size -- though note that on tens of thousands
    of decoys almost any real difference is "significant", so the STATISTIC is the
    effect size to compare across variants and this is mainly a sanity check.

    Returns 1.0 ("no evidence of a difference") when undefined, so a degenerate
    group reads as not-significant rather than as a false detection.
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    if len(pred) < 2 or len(true) < 2:
        return 1.0

    try:
        _, p = ks_2samp(pred, true)
        return float(p) if not np.isnan(p) else 1.0
    except Exception:
        return 1.0


def wasserstein_dist(pred: torch.Tensor, true: torch.Tensor) -> float:
    """1-D Wasserstein (earth-mover) distance between predicted and true DockQ.

    The mean horizontal distance between the two empirical CDFs -- i.e. the average
    amount of probability mass you must move, and how far, to turn one distribution
    into the other. Complements KS: KS reports the single WORST CDF gap, while this
    integrates the whole discrepancy, so a broad-but-shallow mismatch that barely
    moves KS still shows up here.

    Reported in DockQ units (0-1 scale), so ~0.05 means the prediction cloud sits
    about 0.05 DockQ from the truth on average -- directly interpretable, unlike KS.
    Returns 0.0 when undefined, consistent with the other metrics here.
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    if len(pred) < 1 or len(true) < 1:
        return 0.0

    try:
        return float(wasserstein_distance(pred, true))
    except Exception:
        return 0.0


def distribution_metrics(pred: torch.Tensor, true: torch.Tensor, prefix: str = "") -> dict:
    """Bundle the global distribution-shape metrics under ``prefix``.

    Also logs the raw first two moments of BOTH clouds and their signed gaps: a KS
    or Wasserstein value says the distributions differ but not HOW, whereas
    ``pred_mean - true_mean`` and ``pred_std / true_std`` name the failure mode
    directly (a bias vs the variance collapse that regression-to-the-mean produces,
    which is the characteristic way a DockQ regressor goes wrong).
    """
    p = pred.detach().cpu().numpy().flatten()
    t = true.detach().cpu().numpy().flatten()
    metrics = {
        f"{prefix}ks_stat": ks_statistic(pred, true),
        f"{prefix}ks_pvalue": ks_pvalue(pred, true),
        f"{prefix}wasserstein": wasserstein_dist(pred, true),
    }
    if len(p) and len(t):
        p_std, t_std = float(p.std()), float(t.std())
        metrics[f"{prefix}pred_mean"] = float(p.mean())
        metrics[f"{prefix}true_mean"] = float(t.mean())
        metrics[f"{prefix}pred_std"] = p_std
        metrics[f"{prefix}true_std"] = t_std
        metrics[f"{prefix}mean_shift"] = float(p.mean() - t.mean())
        # >1 = over-dispersed, <1 = the predictions are squashed toward the mean.
        # Guarded: a constant true set (possible in a single-complex group) would
        # otherwise divide by zero.
        metrics[f"{prefix}std_ratio"] = float(p_std / t_std) if t_std > 1e-12 else 0.0
    return metrics


def ndcg_score(pred: torch.Tensor, true: torch.Tensor, k: int = None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values (used for ranking)
    true : torch.Tensor
        True relevance scores
    k : int, optional
        Compute NDCG@k (if None, uses all items)

    Returns
    -------
    float
        NDCG score
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    if len(pred) < 2:
        return 0.0

    # Sort by predicted scores (descending)
    order = np.argsort(pred)[::-1]
    if k is not None:
        order = order[:k]

    # Compute DCG
    true_sorted = true[order]
    gains = 2**true_sorted - 1
    discounts = np.log2(np.arange(len(true_sorted)) + 2)
    dcg = np.sum(gains / discounts)

    # Compute ideal DCG
    ideal_order = np.argsort(true)[::-1]
    if k is not None:
        ideal_order = ideal_order[:k]
    true_ideal = true[ideal_order]
    ideal_gains = 2**true_ideal - 1
    ideal_discounts = np.log2(np.arange(len(true_ideal)) + 2)
    idcg = np.sum(ideal_gains / ideal_discounts)

    if idcg < 1e-8:
        return 0.0

    return float(dcg / idcg)


def pearson_r(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute Pearson correlation coefficient.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    true : torch.Tensor
        True values

    Returns
    -------
    float
        Pearson correlation coefficient
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    # Center the data
    pred_centered = pred - np.mean(pred)
    true_centered = true - np.mean(true)

    # Compute correlation
    numerator = np.sum(pred_centered * true_centered)
    denominator = np.sqrt(np.sum(pred_centered**2) * np.sum(true_centered**2))

    if denominator < 1e-8:
        return 0.0

    r = numerator / denominator

    return float(r)


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    true : torch.Tensor
        True values

    Returns
    -------
    float
        RMSE
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    mse = np.mean((pred - true) ** 2)
    return float(np.sqrt(mse))


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    true : torch.Tensor
        True values

    Returns
    -------
    float
        MAE
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    return float(np.mean(np.abs(pred - true)))


# Per-native DockQ-std bins, matching the bucket analysis
# (docs/figures/prismatic/dockq_std_buckets*.png). Each complex is placed by the
# std of its TRUE DockQ across decoys: the low bins are the flat/near-degenerate
# natives that carry almost no ranking signal; the high bins are the rankable
# ones. Slicing the ranking metrics by bin shows WHERE, in std regime, the model
# ranks well -- and hence where added (e.g. perturbed) data actually helps.
DOCKQ_STD_BINS = [0.0, 0.02, 0.05, 0.10, 0.20, 1.01]


def per_std_bin_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    native_ids: list,
    prefix: str = "",
    bins: list = DOCKQ_STD_BINS,
) -> dict:
    """Spearman ranking sliced by each complex's TRUE-DockQ std bin.

    A complex is assigned to a bin by the std of its TRUE DockQ across its decoys
    (model-independent, so the bin membership is comparable across runs). For each
    bin we report:

    - ``.../per_native_spearman_mean`` -- mean per-complex Spearman over the
      complexes in that bin (the intra-complex ranking quality at that std level);
    - ``.../spearman_r`` -- the regular/global Spearman over ALL decoys of those
      complexes pooled (cross-complex + intra mixed, the "how does the whole slice
      correlate" number);
    - ``.../n_complexes`` and ``.../n_decoys`` -- coverage, so an empty/thin bin
      is never mistaken for a real signal.

    Keys are grouped by metric (``stdbin_pernative_spearman/<lo>_<hi>`` etc.) so a
    single W&B panel shows all bins of one metric side by side. Note the existing
    ``per_native_spearman_mean_rankable`` (true-std >= 0.05) is exactly the union
    of the >= 0.05 bins here, so this breakdown is the finer-grained version of the
    rankable/mean split -- the low bins are what "rankable" drops.
    """
    pred_np = pred.detach().cpu().numpy().flatten()
    true_np = true.detach().cpu().numpy().flatten()
    nids = np.asarray(native_ids)

    metrics: dict = {}
    uniq = np.unique(nids)
    # Per-complex TRUE-DockQ std -> bin assignment.
    complex_std = {g: float(true_np[nids == g].std()) for g in uniq}

    for lo, hi in zip(bins[:-1], bins[1:]):
        tag = f"{lo:.2f}_{hi:.2f}"
        in_bin = [g for g in uniq if lo <= complex_std[g] < hi]
        n_complexes = len(in_bin)
        metrics[f"{prefix}stdbin_n_complexes/{tag}"] = float(n_complexes)

        if n_complexes == 0:
            metrics[f"{prefix}stdbin_n_decoys/{tag}"] = 0.0
            continue

        mask = np.isin(nids, in_bin)
        p_bin = torch.from_numpy(pred_np[mask])
        t_bin = torch.from_numpy(true_np[mask])
        metrics[f"{prefix}stdbin_n_decoys/{tag}"] = float(mask.sum())

        # Regular/global Spearman: all decoys of the bin's complexes pooled.
        if mask.sum() >= 2:
            metrics[f"{prefix}stdbin_global_spearman/{tag}"] = spearman_r(p_bin, t_bin)

        # Per-native mean Spearman over the bin's complexes (>= 2 decoys each).
        m_mean, m_std = per_group_metric(p_bin, t_bin, nids[mask].tolist(), spearman_r)
        metrics[f"{prefix}stdbin_pernative_spearman/{tag}"] = m_mean
        metrics[f"{prefix}stdbin_pernative_spearman_std/{tag}"] = m_std

    return metrics


def dockq_tier(dockq: float) -> str:
    """
    Classify DockQ score into quality tier.

    Parameters
    ----------
    dockq : float
        DockQ score

    Returns
    -------
    str
        Quality tier: 'Unacceptable', 'Acceptable', 'Medium', 'High'
    """
    if dockq < 0.23:
        return "Unacceptable"
    elif dockq < 0.49:
        return "Acceptable"
    elif dockq < 0.8:
        return "Medium"
    else:
        return "High"


def dockq_tier_accuracy(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute accuracy of DockQ tier classification.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted DockQ scores
    true : torch.Tensor
        True DockQ scores

    Returns
    -------
    float
        Tier classification accuracy (0-1)
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    # Classify into tiers
    pred_tiers = np.array([dockq_tier(p) for p in pred])
    true_tiers = np.array([dockq_tier(t) for t in true])

    # Compute accuracy
    accuracy = np.mean(pred_tiers == true_tiers)

    return float(accuracy)


def top_k_success_rate(
    pred: torch.Tensor, true: torch.Tensor, threshold: float, k: int = 1, native_ids: list = None
) -> float:
    """
    Compute the top-k selection success rate.

    This answers the deployment question: *if we rank a complex's decoys by the
    model's PREDICTED DockQ and keep the k highest-scoring ones, does at least one
    of those picks have a TRUE DockQ at or above ``threshold``?* Returns the
    fraction of complexes for which that holds.

    The selection is by ``pred``; success is judged on ``true``. (The previous
    implementation thresholded ``pred`` itself and never looked at ``true``, which
    made top-3 identical to top-1 for every input -- the top-3 set always contains
    the single max prediction, so ``any(pred_top3 >= t)`` == ``max(pred) >= t`` ==
    the top-1 condition.) With this definition top-k is monotonic in k (the top-k
    set is a superset of top-1), so top-3 >= top-1 always, as expected.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted DockQ scores.
    true : torch.Tensor
        True DockQ scores.
    threshold : float
        DockQ threshold for a decoy to count as a success (e.g. 0.23 acceptable,
        0.49 medium).
    k : int
        Number of top-predicted decoys to select per complex (default: 1).
    native_ids : list, optional
        Per-decoy complex ids for grouping. Required for a meaningful per-complex
        metric; if None, all decoys are treated as one group (a single global
        top-k selection).

    Returns
    -------
    float
        Success rate (0-1).
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    # Without grouping, treat the whole set as a single complex so the metric is
    # still well-defined (rank all decoys, take global top-k). In practice the
    # caller always supplies native_ids, giving the intended per-complex metric.
    if native_ids is None:
        native_ids = np.zeros(len(pred))

    native_ids = np.asarray(native_ids)
    unique_natives = np.unique(native_ids)
    if len(unique_natives) == 0:
        return 0.0

    success_count = 0
    for native_id in unique_natives:
        mask = native_ids == native_id
        pred_native = pred[mask]
        true_native = true[mask]

        # Indices of this complex's top-k decoys by PREDICTED score. argsort is
        # ascending, so the last k entries are the highest predictions; when a
        # complex has fewer than k decoys this simply keeps all of them.
        top_k_idx = np.argsort(pred_native)[-k:]

        # Success if any selected decoy's TRUE DockQ clears the threshold.
        if np.any(true_native[top_k_idx] >= threshold):
            success_count += 1

    return float(success_count / len(unique_natives))


def auc_at_threshold(pred: torch.Tensor, true: torch.Tensor, threshold: float) -> float:
    """
    Compute AUC for binary classification at a DockQ threshold.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted DockQ scores
    true : torch.Tensor
        True DockQ scores
    threshold : float
        DockQ threshold for binary classification

    Returns
    -------
    float
        AUC score
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    # Binary labels: 1 if true DockQ >= threshold, 0 otherwise
    binary_labels = (true >= threshold).astype(int)

    # Check if we have both classes
    if len(np.unique(binary_labels)) < 2:
        return 0.0

    try:
        auc = roc_auc_score(binary_labels, pred)
        return float(auc)
    except Exception:
        return 0.0


def per_group_metric(
    pred: torch.Tensor,
    true: torch.Tensor,
    group_ids: list,
    metric_fn,
    min_samples: int = 2,
    min_true_std: float = 0.0,
) -> tuple:
    """
    Compute a metric per group (e.g., per native structure or per model).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    true : torch.Tensor
        True values
    group_ids : list
        Group identifier for each sample
    metric_fn : callable
        Metric function that takes (pred, true) and returns float
    min_samples : int
        Minimum samples required per group to compute metric
    min_true_std : float
        Minimum std of the group's TRUE values for it to count. Groups below this
        are "near-degenerate": their decoys span almost no DockQ, so their ranking
        is dominated by label noise (a perfect model still scores ~random rho).
        0.0 (default) keeps every group -- the historical behaviour. Set e.g. 0.05
        for the "rankable subset" metric that isn't diluted (or poisoned) by
        complexes with no rankable signal. Uses TRUE std only, so the subset is
        model-independent and comparable across runs.

    Returns
    -------
    tuple
        (mean, std) of metric across groups
    """
    pred = pred.detach().cpu().numpy().flatten()
    true = true.detach().cpu().numpy().flatten()

    # Group by ID
    unique_groups = np.unique(group_ids)
    group_metrics = []

    for group_id in unique_groups:
        mask = np.array([g == group_id for g in group_ids])
        if np.sum(mask) < min_samples:
            continue

        true_group_np = true[mask]
        if min_true_std > 0.0 and true_group_np.std() < min_true_std:
            continue  # near-degenerate: no rankable signal, exclude

        pred_group = torch.from_numpy(pred[mask])
        true_group = torch.from_numpy(true_group_np)

        metric_val = metric_fn(pred_group, true_group)
        if not np.isnan(metric_val):
            group_metrics.append(metric_val)

    if len(group_metrics) == 0:
        return 0.0, 0.0

    return float(np.mean(group_metrics)), float(np.std(group_metrics))


# Minimum TRUE-DockQ std for a complex to count as "rankable". Below this a
# complex's decoys span almost no DockQ, so its ordering is label noise and a
# perfect model still scores ~random rho. Shared by the global per-native block and
# the per-model one so the two rankable subsets are defined identically.
RANKABLE_MIN_STD = 0.05


def _global_dockq_metrics(pred: torch.Tensor, true: torch.Tensor, native_ids, prefix: str) -> dict:
    """Global (all-decoys-pooled) DockQ metrics: correlation, error, top-k, AUC, shape."""
    metrics = {
        f"{prefix}pearson_r": pearson_r(pred, true),
        f"{prefix}spearman_r": spearman_r(pred, true),
        f"{prefix}kendall_tau": kendall_tau(pred, true),
        f"{prefix}ndcg": ndcg_score(pred, true),
        f"{prefix}rmse": rmse(pred, true),
        f"{prefix}mae": mae(pred, true),
        f"{prefix}tier_accuracy": dockq_tier_accuracy(pred, true),
    }

    # Top-k success rates at CAPRI thresholds
    # Compute for k=1,3,5,10,15,20 at thresholds 0.23 (Acceptable), 0.49 (Medium), 0.8 (High)
    k_values = [1, 3, 5, 10, 15, 20]
    thresholds = [0.23, 0.49, 0.8]

    for k in k_values:
        for threshold in thresholds:
            threshold_str = f"{threshold:.2f}".replace(".", "_")
            metrics[f"{prefix}top{k}_success_{threshold_str}"] = top_k_success_rate(
                pred, true, threshold, k=k, native_ids=native_ids
            )

    # AUC at thresholds
    metrics[f"{prefix}auc_0.23"] = auc_at_threshold(pred, true, 0.23)
    metrics[f"{prefix}auc_0.49"] = auc_at_threshold(pred, true, 0.49)

    # Global distribution shape: does the predicted DockQ cloud LOOK like the true
    # one (KS / Wasserstein), and if not, is it shifted or squashed (mean_shift /
    # std_ratio)? Every metric above is ranking- or error-based and cannot see a
    # model that ranks well on a compressed scale.
    metrics.update(distribution_metrics(pred, true, prefix=prefix))
    return metrics


def _count_rankable_natives(true: torch.Tensor, native_ids, min_true_std: float) -> int:
    """Count complexes with >=2 decoys AND true-DockQ std >= ``min_true_std``.

    Same qualifying condition ``per_group_metric(..., min_true_std=...)`` applies,
    reported separately so a rankable-subset mean is never read without its n.
    """
    nids = np.asarray(native_ids)
    true_np = true.detach().cpu().numpy().flatten()
    n_rankable = 0
    for gid in np.unique(nids):
        mask = nids == gid
        if mask.sum() >= 2 and true_np[mask].std() >= min_true_std:
            n_rankable += 1
    return n_rankable


def _per_native_metrics(pred: torch.Tensor, true: torch.Tensor, native_ids, prefix: str) -> dict:
    """Ranking and distribution metrics averaged WITHIN each complex."""
    metrics = {}

    # Per-native Spearman
    mean_spearman, std_spearman = per_group_metric(pred, true, native_ids, spearman_r)
    metrics[f"{prefix}per_native_spearman_mean"] = mean_spearman
    metrics[f"{prefix}per_native_spearman_std"] = std_spearman

    # Rankable-subset per-native Spearman: exclude near-degenerate complexes
    # (TRUE DockQ std < RANKABLE_MIN_STD) whose decoys carry no rankable signal.
    # On val_easy ~33% of complexes fall below this and score ~random (even
    # slightly NEGATIVE) rho, so they don't just dilute but actively depress
    # the plain mean above. This subset is the honest "how well does the model
    # rank complexes that CAN be ranked" number, comparable across runs since
    # the subset is defined by TRUE spread only. Also log the qualifying count.
    mean_rankable, std_rankable = per_group_metric(pred, true, native_ids, spearman_r, min_true_std=RANKABLE_MIN_STD)
    metrics[f"{prefix}per_native_spearman_mean_rankable"] = mean_rankable
    metrics[f"{prefix}per_native_spearman_std_rankable"] = std_rankable
    metrics[f"{prefix}per_native_n_rankable"] = float(_count_rankable_natives(true, native_ids, RANKABLE_MIN_STD))

    # Per-native Kendall tau
    mean_kendall, std_kendall = per_group_metric(pred, true, native_ids, kendall_tau)
    metrics[f"{prefix}per_native_kendall_mean"] = mean_kendall
    metrics[f"{prefix}per_native_kendall_std"] = std_kendall

    # Per-native NDCG
    mean_ndcg, std_ndcg = per_group_metric(pred, true, native_ids, ndcg_score)
    metrics[f"{prefix}per_native_ndcg_mean"] = mean_ndcg
    metrics[f"{prefix}per_native_ndcg_std"] = std_ndcg

    # Per-native distribution shape. The GLOBAL KS/Wasserstein above can look
    # healthy purely by averaging out -- one complex predicted too high and
    # another too low cancel in the pooled cloud. Computing them WITHIN each
    # complex and averaging removes that cancellation, so this answers "does the
    # model reproduce the DockQ spread of a typical complex?" -- the distribution
    # analogue of per-native Spearman, and the one to compare across variants.
    #
    # Caveat worth remembering when reading these: per-complex decoy counts here
    # are small (often <20 per side), so a single complex's KS is noisy and its
    # p-value weak. The MEAN over complexes is the usable signal, and the std
    # says how consistent the behaviour is.
    for label, fn in (("ks", ks_statistic), ("wasserstein", wasserstein_dist)):
        mean_v, std_v = per_group_metric(pred, true, native_ids, fn)
        metrics[f"{prefix}per_native_{label}_mean"] = mean_v
        metrics[f"{prefix}per_native_{label}_std"] = std_v

    # Per-native std_ratio: the cleanest read on variance collapse. Averaged over
    # complexes it shows whether the model compresses each complex's decoy spread
    # (<1) even when the pooled spread looks right -- the classic regression-to-
    # the-mean failure that still permits a decent global Pearson.
    mean_sr, std_sr = per_group_metric(
        pred,
        true,
        native_ids,
        lambda p, t: distribution_metrics(p, t)["std_ratio"],
    )
    metrics[f"{prefix}per_native_std_ratio_mean"] = mean_sr
    metrics[f"{prefix}per_native_std_ratio_std"] = std_sr

    # Same rankable-subset restriction the Spearman block uses: complexes whose
    # TRUE DockQ barely varies have a near-degenerate target distribution, so
    # their KS/std_ratio is dominated by label noise. This is the honest
    # distribution number on complexes that actually span a range.
    for label, fn in (("ks", ks_statistic), ("wasserstein", wasserstein_dist)):
        mean_v, std_v = per_group_metric(pred, true, native_ids, fn, min_true_std=RANKABLE_MIN_STD)
        metrics[f"{prefix}per_native_{label}_mean_rankable"] = mean_v
        metrics[f"{prefix}per_native_{label}_std_rankable"] = std_v

    # Ranking sliced by each complex's TRUE-DockQ std bin -- both the regular
    # (global, pooled) Spearman and the per-native mean, per bin. Shows WHERE
    # in the std regime the model ranks well, so added (e.g. perturbed) data
    # can be judged by which bins it moves.
    metrics.update(per_std_bin_metrics(pred, true, native_ids, prefix=prefix))
    return metrics


def _per_model_metrics(pred: torch.Tensor, true: torch.Tensor, model_sources, prefix: str) -> dict:
    """Pooled (across-complex) ranking metrics for each folding model's decoys."""
    metrics = {}
    for model in np.unique(model_sources):
        mask = np.array([m == model for m in model_sources])
        if np.sum(mask) < 2:
            continue

        pred_model = pred[torch.from_numpy(mask)]
        true_model = true[torch.from_numpy(mask)]

        metrics[f"{prefix}model_{model}_spearman"] = spearman_r(pred_model, true_model)
        metrics[f"{prefix}model_{model}_kendall"] = kendall_tau(pred_model, true_model)
        metrics[f"{prefix}model_{model}_n_samples"] = float(np.sum(mask))
    return metrics


def _one_model_per_native_metrics(
    pred_model: torch.Tensor, true_model: torch.Tensor, native_model: list, model, prefix: str
) -> dict:
    """Per-native ranking metrics restricted to one folding model's decoys."""
    metrics = {}

    mean_spearman, std_spearman = per_group_metric(pred_model, true_model, native_model, spearman_r)
    metrics[f"{prefix}model_{model}_per_native_spearman_mean"] = mean_spearman
    metrics[f"{prefix}model_{model}_per_native_spearman_std"] = std_spearman

    # Rankable subset WITHIN this model: exclude near-degenerate complexes
    # (true DockQ std < RANKABLE_MIN_STD) that carry no rankable signal. This is
    # the ceiling number -- within-model ranking on complexes that CAN be
    # ranked -- isolating a features/model limit from a data (flat-group)
    # dilution problem.
    mean_rankable_m, _ = per_group_metric(
        pred_model, true_model, native_model, spearman_r, min_true_std=RANKABLE_MIN_STD
    )
    metrics[f"{prefix}model_{model}_per_native_spearman_mean_rankable"] = mean_rankable_m

    mean_kendall, std_kendall = per_group_metric(pred_model, true_model, native_model, kendall_tau)
    metrics[f"{prefix}model_{model}_per_native_kendall_mean"] = mean_kendall
    metrics[f"{prefix}model_{model}_per_native_kendall_std"] = std_kendall

    mean_ndcg, std_ndcg = per_group_metric(pred_model, true_model, native_model, ndcg_score)
    metrics[f"{prefix}model_{model}_per_native_ndcg_mean"] = mean_ndcg
    metrics[f"{prefix}model_{model}_per_native_ndcg_std"] = std_ndcg

    return metrics


def _per_model_per_native_metrics(pred: torch.Tensor, true: torch.Tensor, metadata: dict, prefix: str) -> dict:
    """Per-native ranking computed WITHIN each folding model, plus cross-model aggregates.

    Grouping per (model, native) means each native group holds a single folding
    model's decoys -- the honest within-model ranking, without the cross-model tier
    separation that inflates the plain (model-mixing) per_native_spearman_mean.
    """
    model_sources = np.array(metadata["model_source"])
    native_ids = np.array(metadata["native_id"])
    unique_models = np.unique(model_sources)

    metrics = {}
    # Each folding model's within-model per-native Spearman, aggregated into ONE
    # monitorable scalar below.
    permodel_spearman: list[float] = []

    for model in unique_models:
        mask = model_sources == model
        if np.sum(mask) < 2:
            continue

        model_metrics = _one_model_per_native_metrics(
            pred[torch.from_numpy(mask)],
            true[torch.from_numpy(mask)],
            native_ids[mask].tolist(),
            model,
            prefix,
        )
        metrics.update(model_metrics)
        permodel_spearman.append(model_metrics[f"{prefix}model_{model}_per_native_spearman_mean"])

    # Aggregate the per-model within-model Spearmans into ONE scalar so it can
    # be the checkpoint-selection monitor. This is the honest within-model
    # ranking target (equal weight per folding model), as opposed to the
    # model-mixing per_native_spearman_mean. Mean and min are both logged:
    # _permodel = mean across models; _permodel_min = worst model (boltz2 is
    # hardest), useful if you want to select for balanced cross-model skill.
    if permodel_spearman:
        metrics[f"{prefix}per_native_spearman_mean_permodel"] = float(np.mean(permodel_spearman))
        metrics[f"{prefix}per_native_spearman_min_permodel"] = float(np.min(permodel_spearman))
        # Aggregate of the per-model RANKABLE-subset Spearmans (the ceiling:
        # within-model ranking on rankable complexes only). If this is much
        # higher than the plain permodel mean, flat groups are diluting and
        # should be masked; if it's still low, features/model are the limit.
        rankable_vals = [
            metrics[f"{prefix}model_{m}_per_native_spearman_mean_rankable"]
            for m in unique_models
            if f"{prefix}model_{m}_per_native_spearman_mean_rankable" in metrics
        ]
        if rankable_vals:
            metrics[f"{prefix}per_native_spearman_mean_permodel_rankable"] = float(np.mean(rankable_vals))

    return metrics


def compute_comprehensive_metrics(predictions: dict, targets: dict, metadata: dict = None, prefix: str = "") -> dict:
    """
    Compute comprehensive evaluation metrics including ranking and per-group metrics.

    Assembled from four blocks, each keyed under ``prefix``: global (all decoys
    pooled), per-native, per-model, and per-model-per-native. The last three need
    the matching ``metadata`` key and are skipped without it, so the returned key
    set depends on what metadata the caller supplied.

    Parameters
    ----------
    predictions : dict
        Dictionary with a "dockq" key: predicted DockQ [N]
    targets : dict
        Dictionary with a "dockq" key: true DockQ [N]
    metadata : dict, optional
        Dictionary with keys: native_id, model_source for per-group metrics
    prefix : str
        Prefix for metric names (e.g., 'val/' or 'test/')

    Returns
    -------
    dict
        Dictionary of all computed metrics
    """
    pred = predictions["dockq"]
    true = targets["dockq"]
    meta = metadata if metadata is not None else {}

    metrics = _global_dockq_metrics(pred, true, meta.get("native_id"), prefix)

    if "native_id" in meta:
        metrics.update(_per_native_metrics(pred, true, meta["native_id"], prefix))

    if "model_source" in meta:
        metrics.update(_per_model_metrics(pred, true, meta["model_source"], prefix))

    # For each folding model, average the per-native ranking over that model's
    # own complexes. This measures decoy selection within a single model, without
    # mixing decoys from different models into the same native group.
    if "model_source" in meta and "native_id" in meta:
        metrics.update(_per_model_per_native_metrics(pred, true, meta, prefix))

    return metrics
