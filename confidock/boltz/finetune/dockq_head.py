"""A DockQ readout on top of Boltz-2's frozen confidence module.

Why a new head rather than retraining ``iptm``
----------------------------------------------
``iptm`` is a *derived* quantity: ``compute_ptms`` reduces the PAE head's logits
to a scalar (``confidence_utils.py``).  Regressing it against DockQ would
overwrite the meaning of a metric the rest of the pipeline still reports, and
would force the DockQ signal through a 64-bin PAE discretisation.  A separate
head leaves every pretrained output intact and reads the confidence pairformer's
own representations directly.

What it reads
-------------
With ``confidence_module.return_latent_feats = True`` the module additionally
returns the post-pairformer representations (``confidencev2.py:215-217``):

* ``s_conf`` -- ``(B, L, token_s)`` per-token
* ``z_conf`` -- ``(B, L, L, token_z)`` per-token-pair

DockQ measures **interface** quality, so a plain mean over all tokens/pairs
dilutes the signal with intra-chain content that DockQ ignores.  The pooling here
is therefore interface-aware: pair features are pooled over cross-chain pairs
(``asym_id[i] != asym_id[j]``) separately from all valid pairs, and both are
concatenated with the pooled token features.  ``asym_id`` and ``token_pad_mask``
are already in the cached ``feats``, so this needs nothing new on disk.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["DockQHead", "DockQLoss", "pool_confidence_latents", "component_targets"]


# DockQ distance-scaling constants (Basu & Wallner 2016).
DOCKQ_D1_IRMSD = 1.5  # Angstrom
DOCKQ_D2_LRMSD = 8.5  # Angstrom


def scale_irmsd(irmsd: torch.Tensor) -> torch.Tensor:
    """Map iRMSD (Angstrom) to its DockQ sub-score in [0, 1]. Monotonically DEcreasing."""
    return 1.0 / (1.0 + (irmsd / DOCKQ_D1_IRMSD) ** 2)


def scale_lrmsd(lrmsd: torch.Tensor) -> torch.Tensor:
    """Map LRMSD (Angstrom) to its DockQ sub-score in [0, 1]. Monotonically DEcreasing."""
    return 1.0 / (1.0 + (lrmsd / DOCKQ_D2_LRMSD) ** 2)


def component_targets(y_fnat: torch.Tensor, y_irmsd: torch.Tensor, y_lrmsd: torch.Tensor) -> torch.Tensor:
    """Ground-truth DockQ sub-scores as [B, 3] in [0, 1]: (fnat, s(iRMSD), s(LRMSD))."""
    return torch.stack([y_fnat, scale_irmsd(y_irmsd), scale_lrmsd(y_lrmsd)], dim=-1)


def pool_confidence_latents(
    s_conf: torch.Tensor,
    z_conf: torch.Tensor,
    feats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Pool per-token and per-pair latents into one vector per decoy.

    Returns ``(B, 2 * token_z + token_s)``: cross-chain pair mean, all-pair mean,
    and token mean.  Keeping the cross-chain and all-pair means separate lets the
    head weigh interface evidence against global fold quality rather than having
    to disentangle a single blended average.

    Masking uses ``token_pad_mask``, so padded tokens contribute nothing.  Every
    denominator is clamped: an entry whose chains are all one ``asym_id`` has no
    cross-chain pairs at all, and an unclamped divide would emit NaN and poison
    the backward pass.
    """
    batch = s_conf.shape[0]
    pad = feats["token_pad_mask"]
    if pad.shape[0] != batch:
        pad = pad.expand(batch, -1)
    pad = pad.to(dtype=s_conf.dtype)

    asym = feats["asym_id"]
    if asym.shape[0] != batch:
        asym = asym.expand(batch, -1)

    # Token pooling.
    token_mean = (s_conf * pad.unsqueeze(-1)).sum(dim=1) / pad.sum(dim=1, keepdim=True).clamp_min(1.0)

    # Pair pooling.
    pair_mask = pad.unsqueeze(-1) * pad.unsqueeze(-2)  # (B, L, L)
    cross = (asym.unsqueeze(-1) != asym.unsqueeze(-2)).to(dtype=s_conf.dtype) * pair_mask

    pair_den = pair_mask.sum(dim=(1, 2)).clamp_min(1.0).unsqueeze(-1)
    cross_den = cross.sum(dim=(1, 2)).clamp_min(1.0).unsqueeze(-1)

    all_pair_mean = (z_conf * pair_mask.unsqueeze(-1)).sum(dim=(1, 2)) / pair_den
    cross_pair_mean = (z_conf * cross.unsqueeze(-1)).sum(dim=(1, 2)) / cross_den

    return torch.cat([cross_pair_mean, all_pair_mean, token_mean], dim=-1)


class DockQHead(nn.Module):
    """MLP predicting DockQ components from pooled confidence latents.

    Outputs **3 values in [0, 1]**: fnat, scaled iRMSD, scaled lRMSD (the three
    DockQ sub-scores). DockQ is computed as their exact mean, matching the Basu &
    Wallner formula. Training directly on the components (rather than a single
    logit) matches the GNN's ``comp_sup`` ablation winner, where per-complex
    sub-score Spearman vs DockQ is ~0.89-0.92 and WHICH component orders best
    varies by complex. Best-of-three rho: 0.958 vs best fixed component 0.915.

    Parameters
    ----------
    token_s, token_z
        Latent widths, read off the model rather than hardcoded (384 / 128 for the
        released Boltz-2 checkpoint).
    hidden
        Width of the first hidden layer; the second is ``hidden // 2``.
    dropout
        Applied after each hidden layer.

    Note on width: the GNN's composite head is ``64 -> 32 -> 3`` because its graph
    embedding is 64-wide.  The pooled input here is ``2 * token_z + token_s`` = 640,
    so the trunk keeps the two hidden layers the scalar head used and only the
    output layer changes (1 logit -> 3 sigmoids).  Copying the GNN's literal widths
    would have compressed 640 -> 32 in one step and thrown away capacity that has
    nothing to do with the loss change.
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        *,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = 2 * token_z + token_s
        self.in_dim = in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        s_conf: torch.Tensor,
        z_conf: torch.Tensor,
        feats: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return dict with ``components`` (B, 3) and ``dockq`` (B,)."""
        return self.from_pooled(pool_confidence_latents(s_conf, z_conf, feats))

    def from_pooled(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return dict from an already-pooled ``(B, in_dim)`` tensor.

        Split out from :meth:`forward` so the caller can pool decoys in chunks --
        ``z_conf`` is ``(B, L, L, token_z)`` and does not fit for a full entry --
        and then run the MLP once over the concatenated vectors.

        Returns
        -------
        dict
            - ``components``: (B, 3) in [0, 1] -- fnat, scaled iRMSD, scaled lRMSD
            - ``dockq``: (B,) in [0, 1] -- exact mean of the three components
        """
        components = self.mlp(self.norm(pooled))  # (B, 3)
        return {"components": components, "dockq": components.mean(dim=-1)}


class DockQLoss(nn.Module):
    """Composite loss: Huber on components + Huber on DockQ + a within-entry ranking term.

    Matches the GNN's ``comp_sup`` ablation winner. Three terms:

    1. **Component supervision**: Huber on each of the three scaled sub-scores (fnat,
       scaled iRMSD, scaled lRMSD), all in [0, 1], against their ground-truth values.
       Disabled by ``lambda_component=0``, which leaves the components supervised only
       through the DockQ mean they feed.
    2. **DockQ regression**: Huber on the final DockQ (mean of the three components).
       This is the **only** scale-aware term -- see ``rank_type``.
    3. **Ranking**: within-entry ordering of the DockQ predictions, either soft-Spearman
       or ListNet (``rank_type``).

    The effective ranking weight is ``lambda_rank / rank_tau``, not ``lambda_rank``
    ------------------------------------------------------------------------------
    Both ranking terms divide by ``rank_tau`` before a sigmoid/softmax, so their
    gradient scales as ``1 / rank_tau``: halving tau at fixed ``lambda_rank`` doubles
    the ranking pressure *on the entries whose order is still wrong*. Where the order is
    already right the opposite happens -- both softmaxes saturate onto the same one-hot,
    ``P_pred - P_true`` vanishes faster than ``1 / tau`` grows, and the term stops
    contributing (measured for ListNet on a 3-decoy entry: gradient norm 14.1 -> 28.6 as
    tau goes 0.1 -> 0.05 when inverted, but 0.186 -> 0.045 when correctly ordered). So
    small tau does not merely turn the term up; it concentrates it on the failures.
    Both are also invariant to a per-entry offset (soft ranks and
    ``softmax((p + c) / tau)`` alike), so neither constrains a given native's absolute
    DockQ level -- only the Huber term does. Push the ratio high enough and the
    scale-blind term overwhelms the only scale-aware one: predictions compress into a
    thin band and *global* Spearman, which pools decoys across natives and so is
    dominated by per-native offsets, falls even while within-native ranking improves.

    This is measured, on both models. The GNN sweep at ratio 10 (``rank_tau=0.05``,
    ``lambda_rank=0.5``) got the best high-variance-bin ranking of any arm (0.6023 vs
    0.5762 baseline) but ``pred_std`` 0.102 against a true spread of 0.282, and global
    Spearman 0.6352 vs 0.6795; it settled on ratio 2 (``rank_tau=0.10``,
    ``lambda_rank=0.2``). ``runs/confidence_ft_composite`` ran this loss at ratio 10 and
    reproduced the signature -- global Spearman 0.5869 on ``val_easy`` against 0.6221
    for the older BCE+margin run it replaced, with within-native ranking no better.

    Parameters
    ----------
    lambda_component
        Weight on component supervision (Huber on each of fnat, s_irmsd, s_lrmsd).
    lambda_rank
        Weight on the ranking term. Read it together with ``rank_tau``.
    huber_delta
        Delta for all Huber losses (components + DockQ). Note that DockQ and the
        components all live in [0, 1], so the absolute error can never exceed 1: any
        ``huber_delta >= 1`` keeps every sample in the quadratic regime and makes this
        plain (halved) MSE. The GNN uses 0.1, which is where the linear regime -- and
        hence any robustness to outlier labels -- actually starts.
    rank_type
        ``"spearman"`` for (1 - soft-Spearman) on soft ranks, or ``"listnet"`` for
        listwise top-one cross-entropy over all the entry's decoys jointly. ListNet
        scores the whole ordering at once rather than reducing it to ranks, which suits
        the ~16-33 decoys an entry contributes here; it is the GNN's final choice.
    rank_tau
        Temperature. For soft-Spearman, smaller = closer to true rank, larger =
        smoother. For ListNet, it sharpens both softmaxes (smaller = concentrate on the
        best decoy). Load-bearing as a weight either way -- see above.
    """

    #: Accepted ``rank_type`` values, mirroring ``dockq_predictor.losses.RANK_TYPES``.
    RANK_TYPES = ("spearman", "listnet")

    def __init__(
        self,
        *,
        lambda_component: float = 1.0,
        lambda_rank: float = 0.5,
        huber_delta: float = 1.0,
        rank_type: str = "spearman",
        rank_tau: float = 0.05,
    ) -> None:
        super().__init__()
        # Fail fast rather than falling through to the soft-Spearman branch: a typo
        # would otherwise train the incumbent objective under a config that says
        # listnet, and nothing downstream would look wrong.
        if rank_type not in self.RANK_TYPES:
            raise ValueError(f"Unknown rank_type {rank_type!r} ({'|'.join(self.RANK_TYPES)})")
        self.lambda_component = float(lambda_component)
        self.lambda_rank = float(lambda_rank)
        self.huber_delta = float(huber_delta)
        self.rank_type = str(rank_type)
        self.rank_tau = float(rank_tau)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(loss, parts)`` where ``parts`` holds the scalar components.

        Parameters
        ----------
        predictions
            - ``components``: (B, 3) in [0, 1] -- fnat, scaled iRMSD, scaled lRMSD
            - ``dockq``: (B,) in [0, 1]
        targets
            - ``components``: (B, 3) in [0, 1] -- ground-truth scaled sub-scores
            - ``dockq``: (B,) in [0, 1] -- ground-truth DockQ
        """
        pred_dockq = predictions["dockq"]
        targ_dockq = targets["dockq"].clamp(0.0, 1.0)

        # 1. DockQ Huber
        huber_dockq = nn.functional.huber_loss(pred_dockq, targ_dockq, delta=self.huber_delta)

        # 2. Component supervision
        if self.lambda_component > 0:
            pred_comp = predictions["components"]
            targ_comp = targets["components"]
            huber_comp = (
                sum(
                    nn.functional.huber_loss(pred_comp[:, k], targ_comp[:, k], delta=self.huber_delta) for k in range(3)
                )
                / 3.0
            )
        else:
            huber_comp = pred_dockq.new_zeros(())

        # 3. Within-entry ranking.  Both variants need a pair to compare, so a
        # single-decoy item contributes nothing rather than a spurious zero-gradient
        # term -- see ``cache_dataset._chunk_indices``, which exists to make those rare.
        rank = pred_dockq.new_zeros(())
        if self.lambda_rank > 0 and pred_dockq.numel() > 1:
            if self.rank_type == "listnet":
                rank = self._listnet(pred_dockq, targ_dockq)
            else:
                rank_term = self._soft_spearman(pred_dockq, targ_dockq)
                if rank_term is not None:
                    rank = rank_term

        loss = huber_dockq + self.lambda_component * huber_comp + self.lambda_rank * rank

        return loss, {
            "loss": float(loss.detach()),
            "huber_dockq": float(huber_dockq.detach()),
            "huber_comp": float(huber_comp.detach()),
            "rank": float(rank.detach()),
        }

    def _listnet(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """ListNet top-one cross-entropy for one entry: ``H(softmax(t/tau), softmax(p/tau))``.

        The true scores become a top-one probability distribution over the entry's
        decoys and the predictions likewise; the loss is their cross-entropy. This is
        genuinely listwise -- the whole ordering is scored jointly rather than
        decomposed into pairs or reduced to ranks.

        Unlike :meth:`_soft_spearman` this never returns ``None``: all-equal targets
        give a uniform target distribution, which asks only that the predictions be
        equally spread, so the gradient vanishes once they are. No degenerate-input
        guard is needed, and none of the tied-target tie-breaking that soft-Spearman
        inherits from ``argsort`` applies here -- ties stay ties.

        Byte-for-byte the same computation as
        ``dockq_predictor.losses.losses.listnet_term``, so the two models' ranking
        objectives are comparable rather than merely similarly named.
        """
        target_p = torch.softmax(targ / self.rank_tau, dim=0)
        pred_logp = torch.log_softmax(pred / self.rank_tau, dim=0)
        return -(target_p * pred_logp).sum()

    def _soft_spearman(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor | None:
        """(1 - soft-Spearman) for one entry. Returns None if no rank variance."""
        pred_rank = self._soft_rank(pred)
        targ_rank = targ.argsort().argsort().float()

        pred_rank = pred_rank - pred_rank.mean()
        targ_rank = targ_rank - targ_rank.mean()

        denom = pred_rank.norm() * targ_rank.norm()
        if denom < 1e-8:
            return None

        return 1.0 - (pred_rank * targ_rank).sum() / denom

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """Soft ranks via sigmoid, counting how many others exceed each element."""
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        p = torch.sigmoid(diff / self.rank_tau)
        p = p - torch.diag(torch.diagonal(p))  # zero out diagonal
        return 1.0 + p.sum(dim=0)
