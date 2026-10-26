"""Loss functions for DockQ prediction.

Everything here is a stateless TERM FUNCTION: tensors and plain args in, a 0-d
tensor out, no state and no registered parameters, so each is unit-testable
without building a LightningModule. The objective itself is assembled in
``SimpleGNN.training_step`` out of these terms:

    loss = huber(pred, dockq) + lambda_rank * within_complex_ranking_loss(...)

Huber optimises absolute error but gives no gradient for ORDERING the decoys of a
single complex, which is what ``val/per_native_spearman_mean`` measures. The
ranking term supplies that gradient. Two ranking objectives are provided --
``spearman`` and ``listnet`` (see ``within_complex_ranking_loss``).

There is no loss ``nn.Module`` here, by design: nothing in the objective owns
parameters. (An unrelated, live ``DockQLoss`` exists at
``boltz/finetune/dockq_head.py`` for the Boltz-2 confidence head -- same name,
different objective.)
"""

from typing import Dict

import torch

RANK_TYPES = ("spearman", "listnet")


# ---------------------------------------------------------------------------
# PER-COMPLEX TERM FUNCTIONS
#
# All stateless: tensors in, 0-d tensor out, no parameters, no self. Each operates on
# ONE complex's decoys (p = predictions, t = targets, both 1-D of length n) except
# within_complex_ranking_loss, which does the grouping and calls the rest.
#
# Terms that can find no signal return None (rather than 0.0) so the caller can drop
# them from the mean instead of diluting it -- an all-equal complex must contribute
# NOTHING, not a spurious zero.
# ---------------------------------------------------------------------------


def soft_rank(x, tau):
    """Differentiable rank of each element of a 1-D tensor via pairwise sigmoids.

    Hard rank uses argsort, which has zero gradient a.e. The soft rank
    ``r_i = 1 + sum_{j != i} sigmoid((x_j - x_i) / tau)`` counts, smoothly,
    how many other elements exceed x_i, so it -> the true ascending rank as
    tau -> 0 while staying differentiable in x. tau trades off fidelity
    (small) vs. gradient smoothness (large).
    """
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    # Exclude the diagonal (i vs itself contributes sigmoid(0)=0.5) so a
    # perfectly-ordered set maps to ranks 1..n.
    p = torch.sigmoid(diff / tau)
    p = p - torch.diag(torch.diagonal(p))
    return 1.0 + p.sum(dim=0)


def spearman_term(p, t, tau):
    """(1 - soft-Spearman) for one complex. Scale/shift-invariant.

    Returns None when the SOFT PREDICTED ranks have no variance. Note this is NOT the
    same as "no ranking signal in the targets": ``t.argsort().argsort()`` hands out
    the distinct ranks 0..n-1 to ANY input, including all-equal targets, so ``tr``
    always has variance and the guard fires on degenerate PREDICTIONS only. Two
    consequences, both measured against the full training manifest:

    * An all-equal-target chunk yields an arbitrary non-zero term (the target order is
      whatever argsort's tie-break produced) instead of contributing nothing. Rare
      here -- 0/3247 complexes are all-equal, and simulated chunking gives 0.002%
      (dpc 8) to 0.004% (dpc 28) of chunks -- so it does not materially move any
      trained number.
    * PARTIAL ties are common (10.5% of chunks at dpc 8, 18.0% at dpc 28) and are also
      broken arbitrarily, so tied decoys get a spurious target ordering the model is
      pushed to reproduce.

    Averaging tied ranks would be the principled fix, and it WOULD change trained
    numbers -- which is why the trained results this code produced were measured with
    the behaviour as written, and the fix belongs in a fresh sweep rather than as a
    silent patch. Documented, deliberately not patched.
    """
    pr = soft_rank(p, tau)
    tr = t.argsort().argsort().float()
    pr = pr - pr.mean()
    tr = tr - tr.mean()
    denom = pr.norm() * tr.norm()
    if denom < 1e-8:  # no rank variance -> no ranking signal
        return None
    return 1.0 - (pr * tr).sum() / denom


def listnet_term(p, t, tau):
    """ListNet top-one cross-entropy for one complex.

    Turn the TRUE scores into a target "top-one" probability distribution over
    the complex's decoys, P_true = softmax(t / tau), and the PREDICTIONS into
    P_pred = softmax(p / tau); the loss is the cross-entropy H(P_true, P_pred)
    = -sum_i P_true_i * log P_pred_i. This is a genuinely LISTWISE objective --
    it scores the whole ordering of all decoys of the complex jointly, not just
    pairs -- which fits our ~28-decoy-per-complex data better than pairwise
    alternatives. tau sharpens the distributions (small -> concentrate on the best
    decoy(s)). All-equal targets give a uniform P_true, which asks only that the
    predictions be equally spread -- so the gradient vanishes once they are equal.

    TWO PROPERTIES THAT MAKE tau LOAD-BEARING, both measured:

    1. The gradient scales as 1/tau -- both softmaxes divide by the same tau, so
       d(loss)/dp ~ (P_pred - P_true)/tau. tau=0.05 delivers 5x the gradient of
       tau=0.25 at the same lambda_rank, i.e. the EFFECTIVE ranking weight is
       lambda_rank/tau, not lambda_rank.
    2. It cannot anchor the prediction SCALE. softmax((p+c)/tau) == softmax(p/tau),
       so this term places exactly zero constraint on a complex's absolute level;
       only the Huber term does. Push lambda_rank/tau high enough and the
       scale-blind term overwhelms the only scale-aware one: predictions compress
       into a thin band (measured std 0.102 vs 0.243, MAE 0.198 vs 0.112) and the
       GLOBAL Spearman -- which pools decoys across complexes, so per-complex
       offsets dominate it -- drops even while per-complex ranking improves.
    """
    pt = torch.softmax(t / tau, dim=0)
    log_pp = torch.log_softmax(p / tau, dim=0)
    return -(pt * log_pp).sum()


def within_complex_ranking_loss(
    preds,
    targets,
    native_ids,
    *,
    rank_type: str = "spearman",
    tau: float = 0.05,
):
    """Per-complex within-complex ranking loss.

    Groups a batch's decoys by ``native_id`` and averages a per-complex ranking
    term over the groups. The objective is selected by ``rank_type``:

    - "spearman": differentiable soft-Spearman (Pearson on soft ranks vs the
      fixed target ranks). Scale- and shift-invariant.
    - "listnet": listwise top-one cross-entropy over ALL decoys of the complex
      jointly (softmax of preds vs softmax of targets). Shift-invariant.

    Both are invariant to a per-complex offset, so neither constrains the absolute
    DockQ level -- the Huber term is what anchors that, and the balance between
    them is the thing to tune (see ``listnet_term``).

    Returns a dict with a single 0-d tensor, ``{"rank": ...}``. Complexes with <2
    decoys, or with no ranking signal, contribute nothing -- so this is exactly 0.0
    under random batching. It only bites when the grouped sampler packs multiple
    decoys of a complex into one batch, which is why
    ``DockQDataModule(group_by_complex=True)`` is required rather than optional.
    """
    groups: Dict[object, list] = {}
    for i, nid in enumerate(native_ids):
        groups.setdefault(nid, []).append(i)

    rank_terms = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        idx = torch.as_tensor(idxs, device=preds.device)
        p = preds[idx]
        t = targets[idx]

        if rank_type == "listnet":
            rt = listnet_term(p, t, tau)
        else:
            rt = spearman_term(p, t, tau)
        if rt is not None:
            rank_terms.append(rt)

    rank = torch.stack(rank_terms).mean() if rank_terms else preds.new_zeros(())
    return {"rank": rank}


def validate_loss_options(*, rank_type: str) -> None:
    """Fail fast on an unknown ``loss.rank_type``.

    A typo would otherwise fall through to the ``else`` branch in
    ``within_complex_ranking_loss`` and silently train soft-Spearman under a
    config that says listnet.
    """
    if rank_type not in RANK_TYPES:
        raise ValueError(f"Unknown loss.rank_type '{rank_type}' ({'|'.join(RANK_TYPES)})")
