"""The message-passing stack used by :class:`SimpleGNN`.

:class:`MPNNStack` consumes the already-encoded scalar node features ``x``
``[N, hidden]`` plus ``edge_index`` and encoded ``edge_attr`` ``[E, edge_dim]``,
and returns updated scalars ``[N, hidden]``. Everything downstream (interface
pooling, prediction head, losses, metrics) sees only that shape.

This is the classic Gilmer-style message-passing operator: each message is an MLP
of ``[x_i, x_j, e_ij]`` and messages are MEAN-aggregated with NO attention. It was
selected over attention-based alternatives (GAT, edge-aware graph transformer,
interface pairformer) on measured per-native Spearman -- with the rich edge
features in this feature set, attention bought nothing over equal-weight
aggregation at a fraction of the parameter count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MPNNConv(MessagePassing):
    """Edge-conditioned message-passing convolution (Gilmer et al. 2017 style).

    For each edge ``j -> i`` the message is ``MLP([x_i ; x_j ; e_ij])``, and a
    node's messages are MEAN-aggregated (mean, not sum, so a node's update does
    not scale with its degree -- our k-NN graphs have near-uniform degree but the
    interface nodes sit at the density boundary, where sum-aggregation would give
    them systematically larger activations purely for being well-connected).
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0):
        # "mean" aggregation, messages flow source -> target (PyG default).
        super().__init__(aggr="mean")
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Update MLP applied to [x_i ; aggregated message], mirroring the GRU-style
        # update of the original paper with a plain MLP (cheaper, and the residual
        # in the enclosing stack already carries the "keep old state" path).
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        """Propagate messages and return updated node features ``[N, hidden]``."""
        aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.update_mlp(torch.cat([x, aggregated], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        """Build the per-edge message from both endpoints and the edge features."""
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class MPNNStack(nn.Module):
    """A stack of :class:`MPNNConv` layers with per-layer LayerNorm + ReLU."""

    def __init__(self, hidden_dim: int, edge_dim: int, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.convs = nn.ModuleList(MPNNConv(hidden_dim, edge_dim, dropout) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))

    def forward(self, x, edge_index, edge_attr):
        """Run the stack; returns updated scalar node features ``[N, hidden]``."""
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_new = torch.relu(norm(conv(x, edge_index, edge_attr)))
            # Residual from the second layer onward; layer 1 has no same-width
            # predecessor to add (its input is the encoder output).
            x = x + x_new if i > 0 else x_new
        return x
