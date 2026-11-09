"""Readout (interface pooling) for :class:`SimpleGNN`.

The readout turns a graph's per-node embeddings ``[n, hidden]`` into ONE vector
``[hidden]`` that the prediction head consumes. It is the bottleneck this model's
per-native ranking has repeatedly been traced back to: mean-pooling produces a
global quality summary, so decoys of the same complex map to nearly the same
vector and within-complex ORDERING is poorly encoded (the G-Vendi gradient PCA
collapses onto a ~1-D DockQ-magnitude arc). Replacing mean-pooling with the gated
set readout below is what moved per-native Spearman.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedSetReadout(nn.Module):
    """Multi-head gated attention pooling (attention-MIL, Ilse et al. 2018).

    Each head scores a node as ``w . (tanh(V h) * sigmoid(U h))`` -- the sigmoid
    gate lets a head suppress nodes that the ``tanh`` branch would otherwise score
    highly, which is what makes gated attention able to represent "relevant only in
    a certain context". Heads are pooled independently and their concatenation is
    projected back to ``hidden``, so multiple heads can attend to different parts
    of the interface simultaneously (a single-softmax readout must pick one).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        attn_dim = max(hidden_dim // 2, num_heads)
        self.value_branch = nn.Linear(hidden_dim, attn_dim)
        self.gate_branch = nn.Linear(hidden_dim, attn_dim)
        self.head_scores = nn.Linear(attn_dim, num_heads)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_sel: torch.Tensor) -> torch.Tensor:
        """Pool ``x_sel`` ``[n, hidden]`` into ``[hidden]``."""
        a = torch.tanh(self.value_branch(x_sel)) * torch.sigmoid(self.gate_branch(x_sel))
        w = torch.softmax(self.head_scores(a), dim=0)  # [n, heads], softmax over nodes
        pooled = torch.einsum("nh,nd->hd", w, x_sel).reshape(-1)  # [heads * hidden]
        return self.output_proj(pooled)
