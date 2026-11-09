"""Tests for confidock.dockq_predictor.losses.losses module."""

from __future__ import annotations

import pytest
import torch

from confidock.dockq_predictor.losses import losses

# ---------------------------------------------------------------------------
# soft_rank
# ---------------------------------------------------------------------------


def test_soft_rank_recovers_hard_ranks_at_small_tau() -> None:
    """A well-separated set maps to the ascending ranks 1..n."""
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    ranks = losses.soft_rank(x, tau=0.01)
    assert torch.allclose(ranks, torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-4)


def test_soft_rank_is_order_equivariant() -> None:
    """Permuting the input permutes the ranks the same way."""
    x = torch.tensor([0.3, 0.9, 0.1, 0.5])
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(losses.soft_rank(x, 0.05)[perm], losses.soft_rank(x[perm], 0.05))


def test_soft_rank_all_equal_gives_flat_ranks() -> None:
    """All-equal inputs get identical ranks (every pairwise sigmoid is 0.5)."""
    ranks = losses.soft_rank(torch.zeros(4), tau=0.05)
    assert torch.allclose(ranks, torch.full((4,), 1.0 + 1.5))


def test_soft_rank_is_differentiable() -> None:
    """The point of the soft rank: a non-zero gradient where argsort has none."""
    x = torch.tensor([0.2, 0.6, 0.4], requires_grad=True)
    losses.soft_rank(x, tau=0.1).sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# spearman_term
# ---------------------------------------------------------------------------


def test_spearman_term_zero_when_order_is_perfect() -> None:
    """Correctly ordered predictions give (1 - rho) ~ 0."""
    p = torch.tensor([0.1, 0.4, 0.7, 0.9])
    t = torch.tensor([0.2, 0.3, 0.6, 0.8])
    assert losses.spearman_term(p, t, tau=0.01) == pytest.approx(0.0, abs=1e-4)


def test_spearman_term_two_when_order_is_reversed() -> None:
    """Anti-correlated ranks give (1 - -1) = 2, the maximum."""
    p = torch.tensor([0.9, 0.7, 0.4, 0.1])
    t = torch.tensor([0.2, 0.3, 0.6, 0.8])
    assert losses.spearman_term(p, t, tau=0.01) == pytest.approx(2.0, abs=1e-4)


def test_spearman_term_none_when_predictions_are_degenerate() -> None:
    """No variance in the soft PREDICTED ranks -> no signal -> None, not 0."""
    p = torch.zeros(4)
    t = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert losses.spearman_term(p, t, tau=0.05) is None


def test_spearman_term_is_shift_invariant() -> None:
    """Shifting every prediction by a constant leaves the term unchanged.

    The soft ranks depend only on pairwise DIFFERENCES, so a per-complex offset is
    invisible to this term -- which is why the Huber term is the only thing
    anchoring the absolute DockQ level. (Scale invariance holds only as tau -> 0:
    scaling the predictions rescales the differences that tau is compared against.)
    """
    p = torch.tensor([0.1, 0.5, 0.3, 0.9])
    t = torch.tensor([0.2, 0.4, 0.3, 0.8])
    base = float(losses.spearman_term(p, t, tau=0.05))
    shifted = float(losses.spearman_term(p + 5.0, t, tau=0.05))
    assert shifted == pytest.approx(base, abs=1e-6)


def test_spearman_term_breaks_target_ties_arbitrarily() -> None:
    """Documented behaviour: all-equal TARGETS still get a spurious ordering.

    ``t.argsort().argsort()`` hands out distinct ranks to any input, so the guard
    only fires on degenerate predictions -- an all-equal-target complex yields a
    non-zero term rather than contributing nothing. Deliberately not patched (it
    would change trained numbers), so pin it.
    """
    p = torch.tensor([0.1, 0.9, 0.5])
    term = losses.spearman_term(p, torch.full((3,), 0.5), tau=0.05)
    assert term is not None
    assert abs(float(term)) > 1e-6


# ---------------------------------------------------------------------------
# listnet_term
# ---------------------------------------------------------------------------


def test_listnet_term_minimal_when_predictions_match_targets() -> None:
    """Matching distributions minimise the cross-entropy."""
    t = torch.tensor([0.1, 0.5, 0.9])
    exact = losses.listnet_term(t, t, tau=0.1)
    wrong = losses.listnet_term(t.flip(0), t, tau=0.1)
    assert float(exact) < float(wrong)


def test_listnet_term_is_shift_invariant_but_not_scale_invariant() -> None:
    """softmax((p+c)/tau) == softmax(p/tau): the term cannot anchor the scale."""
    p = torch.tensor([0.2, 0.5, 0.8])
    t = torch.tensor([0.1, 0.4, 0.9])
    assert float(losses.listnet_term(p + 0.3, t, 0.1)) == pytest.approx(float(losses.listnet_term(p, t, 0.1)), abs=1e-5)
    assert float(losses.listnet_term(p * 2.0, t, 0.1)) != pytest.approx(float(losses.listnet_term(p, t, 0.1)), abs=1e-5)


def test_listnet_gradient_scales_inversely_with_tau() -> None:
    """The property that makes lambda_rank/tau, not lambda_rank, the real weight.

    Comparing taus at fixed inputs conflates two effects, since tau also sharpens
    both distributions. Scaling the inputs by tau/tau0 holds both softmax arguments
    -- and therefore the loss VALUE -- fixed, isolating the 1/tau factor in
    d(loss)/dp = (P_pred - P_true)/tau: 5x smaller tau, exactly 5x the gradient.
    """
    base_p, base_t = torch.tensor([0.5, 0.4, 0.6]), torch.tensor([0.1, 0.4, 0.9])
    value, grad = {}, {}
    for tau in (0.05, 0.25):
        scale = tau / 0.05
        p = (base_p * scale).clone().requires_grad_(True)
        term = losses.listnet_term(p, base_t * scale, tau)
        term.backward()
        value[tau], grad[tau] = float(term), float(p.grad.abs().sum())

    assert value[0.05] == pytest.approx(value[0.25], abs=1e-6)
    assert grad[0.05] / grad[0.25] == pytest.approx(5.0, rel=1e-4)


def test_listnet_term_no_gradient_when_uniform_targets_are_already_matched() -> None:
    """All-equal targets -> uniform P_true -> nothing to learn once preds are equal."""
    p = torch.full((3,), 0.5, requires_grad=True)
    losses.listnet_term(p, torch.full((3,), 0.5), tau=0.1).backward()
    assert float(p.grad.abs().sum()) < 1e-5


# ---------------------------------------------------------------------------
# within_complex_ranking_loss
# ---------------------------------------------------------------------------


def test_ranking_loss_is_zero_without_repeated_natives() -> None:
    """Random batching gives every decoy its own native -> exactly 0.0.

    This is why ``DockQDataModule(group_by_complex=True)`` is required and not
    optional: without it the ranking term is a silent no-op.
    """
    preds = torch.tensor([0.1, 0.5, 0.9])
    targets = torch.tensor([0.2, 0.4, 0.8])
    out = losses.within_complex_ranking_loss(preds, targets, ["a", "b", "c"])
    assert float(out["rank"]) == pytest.approx(0.0)


def test_ranking_loss_averages_over_complexes_only() -> None:
    """Groups of >=2 decoys contribute; singletons are dropped from the mean."""
    preds = torch.tensor([0.1, 0.9, 0.4, 0.6, 0.5])
    targets = torch.tensor([0.2, 0.8, 0.3, 0.7, 0.5])
    native_ids = ["a", "a", "b", "b", "c"]

    pooled = losses.within_complex_ranking_loss(preds, targets, native_ids, rank_type="listnet", tau=0.1)["rank"]
    per_complex = [
        losses.listnet_term(preds[[0, 1]], targets[[0, 1]], 0.1),
        losses.listnet_term(preds[[2, 3]], targets[[2, 3]], 0.1),
    ]
    expected = torch.stack(per_complex).mean()
    assert float(pooled) == pytest.approx(float(expected), abs=1e-6)


def test_ranking_loss_defaults_to_spearman() -> None:
    """An unspecified rank_type is the incumbent soft-Spearman objective."""
    preds = torch.tensor([0.1, 0.9, 0.5, 0.4])
    targets = torch.tensor([0.2, 0.8, 0.6, 0.3])
    native_ids = ["a", "a", "b", "b"]
    default = losses.within_complex_ranking_loss(preds, targets, native_ids)["rank"]
    explicit = losses.within_complex_ranking_loss(preds, targets, native_ids, rank_type="spearman")["rank"]
    assert float(default) == pytest.approx(float(explicit))


def test_ranking_loss_keeps_gradient_to_predictions() -> None:
    """The whole point: the ordering term must reach the predictions."""
    preds = torch.tensor([0.1, 0.9, 0.4, 0.6], requires_grad=True)
    targets = torch.tensor([0.2, 0.8, 0.3, 0.7])
    losses.within_complex_ranking_loss(preds, targets, ["a", "a", "a", "a"], rank_type="listnet")["rank"].backward()
    assert preds.grad is not None
    assert float(preds.grad.abs().sum()) > 0


def test_ranking_loss_returns_zero_tensor_on_matching_device_and_dtype() -> None:
    """The empty case must still be addable to the Huber term."""
    preds = torch.tensor([0.4, 0.6], dtype=torch.float32)
    out = losses.within_complex_ranking_loss(preds, torch.tensor([0.3, 0.7]), ["a", "b"])["rank"]
    assert out.dtype == preds.dtype
    assert out.shape == torch.Size([])


# ---------------------------------------------------------------------------
# validate_loss_options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank_type", losses.RANK_TYPES)
def test_validate_loss_options_accepts_supported_types(rank_type: str) -> None:
    """Both shipped ranking objectives validate."""
    assert losses.validate_loss_options(rank_type=rank_type) is None


def test_validate_loss_options_rejects_unknown_type() -> None:
    """A typo must fail fast rather than silently training soft-Spearman."""
    with pytest.raises(ValueError, match="Unknown loss.rank_type"):
        losses.validate_loss_options(rank_type="listmle")
