"""The DockQ-prediction GNN.

Architecture, end to end:

1. Node / edge encoders -- project the fixed 18-dim node and 96-dim edge features
   (see ``data/feature_utils.py``) to ``hidden_dim`` / ``edge_dim``.
2. :class:`MPNNStack` -- 3 layers of edge-conditioned, attention-free message
   passing over the interface graph.
3. :class:`GatedSetReadout` -- multi-head gated attention-MIL pooling over the
   graph's INTERFACE nodes into one vector per graph.
4. Scalar head -- a sigmoid regressor onto DockQ in [0, 1].

Objective (assembled in :meth:`training_step`):

    loss = huber(pred, dockq) + lambda_rank * within_complex_ranking_loss(...)

The Huber term fits absolute DockQ; the ranking term supplies the gradient for
ORDERING the decoys of one complex, which is what ``val/per_native_spearman_mean``
measures. Both are needed -- the ranking objectives are shift-invariant, so the
Huber term is the only thing anchoring the absolute scale.

Each of these four choices was selected against alternatives on measured per-native
Spearman; the alternatives (GAT / graph-transformer / GVP / pairformer conv stacks,
mean / single-softmax-attention / hierarchical readouts, binned and composite heads,
pairwise-margin ranking, variance and distribution guards, MRL and multitask
auxiliaries) are not carried here. Config keys that name one of them are REJECTED at
construction rather than silently ignored, so a stale config cannot quietly train a
different model than it describes.

The class name is historical -- it started as a stripped-down baseline. It is the
production model.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..data.feature_utils import EDGE_FEATURE_DIM, NODE_FEATURE_DIM

# Objective TERM functions live in losses/losses.py: stateless, parameter-free, and
# unit-testable without a LightningModule. This module keeps only the pieces that own
# parameters and the wiring that reads self.lambda_* to assemble them in training_step.
from ..losses.losses import validate_loss_options, within_complex_ranking_loss
from .conv_stacks import MPNNStack
from .readouts import GatedSetReadout


def _print_topk_success_rates(metrics: dict) -> None:
    """Print top-k success rates, one block per DockQ tier."""
    for threshold, label in (("0_23", "Acceptable"), ("0_49", "Medium"), ("0_80", "High")):
        print(f"  @ DockQ tier {label}:")
        for k in [1, 3, 5, 10, 15, 20]:
            key = f"test/top{k}_success_{threshold}"
            if key in metrics:
                print(f"    Top-{k:2d}: {metrics[key]:.4f}")


def _print_per_native_metrics(metrics: dict) -> None:
    """Print the per-native mean +/- std ranking metrics."""
    print(
        f"  Spearman:   {metrics['test/per_native_spearman_mean']:.4f} ± {metrics['test/per_native_spearman_std']:.4f}"
    )
    print(f"  Kendall:    {metrics['test/per_native_kendall_mean']:.4f} ± {metrics['test/per_native_kendall_std']:.4f}")
    print(f"  NDCG:       {metrics['test/per_native_ndcg_mean']:.4f} ± {metrics['test/per_native_ndcg_std']:.4f}")
    # Rankable subset (true-DockQ std >= 0.05): honest ranking number, not
    # diluted by near-degenerate complexes whose ordering is label noise.
    if "test/per_native_spearman_mean_rankable" in metrics:
        n_rank = int(metrics.get("test/per_native_n_rankable", 0))
        print(
            f"  Spearman (rankable, std>=0.05): "
            f"{metrics['test/per_native_spearman_mean_rankable']:.4f} "
            f"± {metrics['test/per_native_spearman_std_rankable']:.4f}  (n={n_rank} complexes)"
        )


def _print_per_model_spearman(metrics: dict) -> None:
    """Print each folding model's pooled (across-complex) Spearman."""
    for key, value in sorted(metrics.items()):
        if "model_" in key and "_spearman" in key and "_n_samples" not in key and "per_native" not in key:
            model_name = key.split("model_")[1].split("_spearman")[0]
            n_samples = int(metrics.get(f"test/model_{model_name}_n_samples", 0))
            print(f"  {model_name:12s}: {value:.4f} (n={n_samples})")


def _print_per_model_per_native_spearman(metrics: dict) -> None:
    """Print each folding model's per-native Spearman mean +/- std."""
    for key, value in sorted(metrics.items()):
        if "model_" in key and "_per_native_spearman_mean" in key:
            model_name = key.split("model_")[1].split("_per_native_spearman_mean")[0]
            std = metrics.get(f"test/model_{model_name}_per_native_spearman_std", 0.0)
            print(f"  {model_name:12s}: {value:.4f} ± {std:.4f}")


def _print_test_summary(metrics: dict) -> None:
    """Print the whole test-metric report to stdout, section by section.

    Layout only -- every number comes straight from ``metrics`` (keyed with the
    ``test/`` prefix). Sections whose keys are absent are skipped by the helpers
    rather than raising, since the metric set depends on how many decoys per
    complex the test split actually had.
    """
    print("\n" + "=" * 80)
    print("Test Results:")
    print("=" * 80)

    print("\n=== Global Metrics ===")
    print(f"  Pearson R:             {metrics['test/pearson_r']:.4f}")
    print(f"  Spearman R:            {metrics['test/spearman_r']:.4f}")
    print(f"  Kendall Tau:           {metrics['test/kendall_tau']:.4f}")
    print(f"  NDCG:                  {metrics['test/ndcg']:.4f}")
    print(f"  RMSE:                  {metrics['test/rmse']:.4f}")
    print(f"  MAE:                   {metrics['test/mae']:.4f}")
    print(f"  Tier Accuracy:         {metrics['test/tier_accuracy']:.4f}")

    print("\n=== Top-k Success Rates ===")
    _print_topk_success_rates(metrics)

    print("\n=== AUC Metrics ===")
    print(f"  AUC @ threshold 0.23:  {metrics['test/auc_0.23']:.4f}")
    print(f"  AUC @ threshold 0.49:  {metrics['test/auc_0.49']:.4f}")

    print("\n=== Per-Native Metrics ===")
    _print_per_native_metrics(metrics)

    print("\n=== Per-Model Spearman ===")
    _print_per_model_spearman(metrics)

    print("\n=== Per-Model, Per-Native Spearman ===")
    _print_per_model_per_native_spearman(metrics)

    print("=" * 80)


class SimpleGNN(pl.LightningModule):
    """MPNN + gated-set-readout + scalar-head DockQ regressor. See module docstring."""

    def __init__(self, config: dict):
        """
        Initialize the model.

        Parameters
        ----------
        config : dict
            Configuration with keys:
            - model.node_input_dim / edge_input_dim (default: the fixed 18 / 96)
            - model.hidden_dim (default: 128)
            - model.edge_dim (default: 64)
            - model.num_gnn_layers (default: 3)
            - model.readout_num_heads (default: 4)
            - model.dropout (default: 0.1)
            - loss.lambda_rank, loss.rank_type, loss.rank_tau, loss.huber_delta_dockq
            - training.lr, training.weight_decay, training.n_epochs
        """
        super().__init__()

        # Save hyperparameters for Lightning checkpointing
        self.save_hyperparameters(config)
        self.config = config

        # Storage for validation outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Extract model config
        model_cfg = config.get("model", {})
        self.node_input_dim = int(model_cfg.get("node_input_dim", NODE_FEATURE_DIM))
        self.edge_input_dim = int(model_cfg.get("edge_input_dim", EDGE_FEATURE_DIM))
        # The feature set is FIXED (feature_utils.NODE_BLOCK_DIMS / EDGE_BLOCK_DIMS),
        # so a config that disagrees with it is a config for a different feature set.
        # Caught here rather than as a Linear shape error on the first batch, after
        # the graph cache has warmed.
        if self.node_input_dim != NODE_FEATURE_DIM or self.edge_input_dim != EDGE_FEATURE_DIM:
            raise ValueError(
                f"model.node_input_dim/edge_input_dim = {self.node_input_dim}/{self.edge_input_dim} "
                f"but the feature set is fixed at {NODE_FEATURE_DIM}/{EDGE_FEATURE_DIM} "
                "(see data/feature_utils.py)."
            )
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))
        self.edge_dim = int(model_cfg.get("edge_dim", 64))
        self.num_gnn_layers = int(model_cfg.get("num_gnn_layers", 3))
        # Dropout applied inside the conv stack and the pooling MLP.
        self.dropout = float(model_cfg.get("dropout", 0.1))

        self._reject_removed_variants(model_cfg)

        # 1. Feature encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_dim),
            nn.LayerNorm(self.edge_dim),
            nn.ReLU(),
        )

        # 2. Message passing: attention-free edge-conditioned MPNN.
        self.conv_stack = MPNNStack(
            hidden_dim=self.hidden_dim,
            edge_dim=self.edge_dim,
            num_layers=self.num_gnn_layers,
            dropout=self.dropout,
        )

        # 3. Interface pooling: multi-head gated attention-MIL. Several heads can
        # attend to different interface regions at once, which is what lets decoys
        # of the SAME complex map to different pooled vectors -- an unweighted mean
        # collapses them to a near-identical global quality summary and loses the
        # within-complex ordering the ranking loss is trying to teach.
        self.readout = GatedSetReadout(
            self.hidden_dim,
            num_heads=int(model_cfg.get("readout_num_heads", 4)),
            dropout=self.dropout,
        )
        self.pool = nn.Sequential(nn.Linear(self.hidden_dim, 64), nn.ReLU(), nn.Dropout(self.dropout))

        # 4. Prediction head: single sigmoid regressor onto DockQ in [0, 1].
        self.dockq_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

        # ---- Objective knobs --------------------------------------------------
        loss_cfg = config.get("loss", {})
        self.huber_delta_dockq = float(loss_cfg.get("huber_delta_dockq", 0.1))

        # Within-complex ranking loss weight. The Huber term only optimises absolute
        # error (global fit); it gives no gradient for ordering decoys WITHIN a
        # complex, which is exactly what val/per_native_spearman_mean measures.
        # lambda_rank = 0 reproduces pure-Huber training exactly.
        # NOTE: this only has an effect when batches contain multiple decoys of the
        # same complex -- use data.group_by_complex to guarantee that.
        self.lambda_rank = float(loss_cfg.get("lambda_rank", 0.0))
        # Which within-complex ranking objective to use:
        #   "listnet"  -- listwise top-one cross-entropy over ALL decoys of a complex
        #                 jointly (softmax(pred) vs softmax(true)); scores the whole
        #                 ordering at once. Best fit for our multi-decoy complexes.
        #   "spearman" -- differentiable soft-Spearman (Pearson on soft ranks).
        self.rank_type = str(loss_cfg.get("rank_type", "spearman")).lower()
        # Temperature of the ranking term. For listnet this is load-bearing well
        # beyond "sharpness": the gradient scales as 1/tau, so the EFFECTIVE ranking
        # weight is lambda_rank/tau -- see losses.listnet_term.
        self.rank_tau = float(loss_cfg.get("rank_tau", 0.05))

        # Declared-in-one-place validation of the string loss options (losses.py owns
        # RANK_TYPES, next to the code that branches on it).
        validate_loss_options(rank_type=self.rank_type)

        if self.lambda_rank > 0:
            # The ranking term is exactly 0 unless batches actually contain multiple
            # decoys of the same complex. That is a DATA-side flag, so it cannot be
            # asserted here -- but it is the single easiest way to run a whole sweep
            # whose ranking term never fired.
            print(
                "Note: the within-complex ranking term is active (loss.lambda_rank > 0). It "
                "contributes exactly 0 unless data.group_by_complex=true packs >=2 decoys of "
                "the same native per batch -- verify that flag is set."
            )

    @staticmethod
    def _reject_removed_variants(model_cfg: dict) -> None:
        """Reject configs naming a conv stack / readout / head this package dropped.

        These keys used to select between architecture variants during the ablations.
        Only the selected combination is implemented here, so silently ignoring e.g.
        ``architecture: gat`` would train an MPNN under a config that claims
        otherwise -- and the run would look legitimate. Absent keys are fine; the
        keys are accepted only at their one supported value.
        """
        supported = {"architecture": "mpnn", "pooling_type": "gated_set", "head_type": "scalar"}
        for key, value in supported.items():
            got = model_cfg.get(key)
            if got is not None and str(got).lower() != value:
                raise ValueError(
                    f"model.{key}='{got}' is not implemented; this package implements only "
                    f"model.{key}='{value}'. (The other variants were ablation arms and are "
                    "not carried here.)"
                )

    def forward(self, data):
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data or Batch
            Graph data with:
            - x: [N, 18] node features
            - edge_index: [2, E] edge connectivity
            - edge_attr: [E, 96] edge features
            - is_interface: [N] bool mask for interface residues
            - batch: [N] batch assignment (for batched graphs)

        Returns
        -------
        dict
            Predictions with a 'dockq' key: [batch_size] in [0, 1].
        """
        # 1. Encode features.
        x = self.node_encoder(data.x)  # [N, hidden_dim]
        edge_attr = self.edge_encoder(data.edge_attr)  # [E, edge_dim]

        # 2. Message passing -> updated scalar node features [N, hidden_dim].
        x = self.conv_stack(x, data.edge_index, edge_attr)

        # 3. Pool each graph's interface nodes into one vector. Pooling is per-graph
        # (the readout softmaxes over a graph's nodes), so a batch is looped rather
        # than pooled with a scatter op.
        # A single unbatched ``Data`` still exposes a ``batch`` attribute in recent
        # PyG versions -- it is just None -- so test the value, not the attribute.
        if getattr(data, "batch", None) is not None:
            batch_size = data.batch.max().item() + 1
            graph_embeddings = []
            for b in range(batch_size):
                mask = data.batch == b
                x_graph = x[mask]  # [N_graph, hidden_dim]
                is_interface_graph = data.is_interface[mask]  # [N_graph]
                # Interface nodes if any, else fall back to all nodes -- a graph with
                # no detected interface must still produce a prediction.
                x_sel = x_graph[is_interface_graph] if is_interface_graph.any() else x_graph
                graph_embeddings.append(self.readout(x_sel))
            graph_embedding = torch.stack(graph_embeddings)  # [batch_size, hidden_dim]
        else:
            x_sel = x[data.is_interface] if data.is_interface.any() else x
            graph_embedding = self.readout(x_sel).unsqueeze(0)  # [1, hidden_dim]

        # Project to the 64-d prediction representation.
        graph_embedding = self.pool(graph_embedding)  # [batch_size, 64]

        # 4. Predict DockQ.
        return {"dockq": self.dockq_head(graph_embedding).squeeze(-1)}

    def training_step(self, batch, batch_idx):
        """Training step.

        ``loss = huber + lambda_rank * rank``, with the ranking term omitted
        (contributing exactly 0) when ``lambda_rank == 0``. Every component is logged
        so each contribution can be tracked independently:

        - ``train/huber``              -- primary DockQ Huber (always present)
        - ``train/rank_loss_{type}``   -- the raw ranking term
        - ``train/rank_weighted``      -- the ranking term AFTER lambda_rank, i.e. its
          actual additive contribution to ``train/loss``
        - ``train/pred_std``           -- realised prediction spread, so the
          collapse-to-a-thin-band failure mode of a shift-invariant ranking term is
          visible directly rather than inferred from a drop in global Spearman
        - ``train/loss``               -- the total optimised loss
        """
        predictions = self(batch)
        target_dockq = batch.y_dockq

        # Helper so every component logs with identical epoch-mean settings.
        def _log(name, value):
            self.log(name, value, on_step=False, on_epoch=True, batch_size=batch.num_graphs)

        huber = torch.nn.functional.huber_loss(predictions["dockq"], target_dockq, delta=self.huber_delta_dockq)
        loss = huber
        _log("train/huber", huber)

        if self.lambda_rank > 0:
            rank = within_complex_ranking_loss(
                predictions["dockq"],
                target_dockq,
                batch.native_id,
                rank_type=self.rank_type,
                tau=self.rank_tau,
            )["rank"]
            rank_contribution = self.lambda_rank * rank
            loss = loss + rank_contribution
            _log(f"train/rank_loss_{self.rank_type}", rank)
            _log("train/rank_weighted", rank_contribution)
            _log("train/pred_std", predictions["dockq"].std(unbiased=False).detach())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def validation_step(self, batch, batch_idx):
        """Run a single validation step."""
        predictions = self(batch)

        loss = torch.nn.functional.huber_loss(predictions["dockq"], batch.y_dockq, delta=self.huber_delta_dockq)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        # Collect metadata for comprehensive metrics
        output = {
            "predictions_dockq": predictions["dockq"].detach(),
            "targets_dockq": batch.y_dockq.detach(),
            "model_source": [batch[i].model_source for i in range(batch.num_graphs)],
            "native_id": [batch[i].native_id for i in range(batch.num_graphs)],
        }
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        """Compute validation metrics."""
        if len(self.validation_step_outputs) == 0:
            return

        all_preds = torch.cat([x["predictions_dockq"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets_dockq"] for x in self.validation_step_outputs])

        # Collect metadata
        all_model_sources = []
        all_native_ids = []
        for x in self.validation_step_outputs:
            all_model_sources.extend(x["model_source"])
            all_native_ids.extend(x["native_id"])

        from ..training.metrics import compute_comprehensive_metrics

        metrics = compute_comprehensive_metrics(
            {"dockq": all_preds},
            {"dockq": all_targets},
            {"model_source": all_model_sources, "native_id": all_native_ids},
            prefix="val/",
        )

        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, prog_bar=(metric_name in ["val/pearson_r", "val/spearman_r"]))

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step."""
        predictions = self(batch)

        loss = torch.nn.functional.huber_loss(predictions["dockq"], batch.y_dockq, delta=self.huber_delta_dockq)

        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_graphs)

        output = {
            "predictions_dockq": predictions["dockq"].detach(),
            "targets_dockq": batch.y_dockq.detach(),
            "model_source": [batch[i].model_source for i in range(batch.num_graphs)],
            "native_id": [batch[i].native_id for i in range(batch.num_graphs)],
        }
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        """Compute and print test metrics."""
        if len(self.test_step_outputs) == 0:
            return

        all_preds = torch.cat([x["predictions_dockq"] for x in self.test_step_outputs])
        all_targets = torch.cat([x["targets_dockq"] for x in self.test_step_outputs])

        all_model_sources = []
        all_native_ids = []
        for x in self.test_step_outputs:
            all_model_sources.extend(x["model_source"])
            all_native_ids.extend(x["native_id"])

        from ..training.metrics import compute_comprehensive_metrics

        metrics = compute_comprehensive_metrics(
            {"dockq": all_preds},
            {"dockq": all_targets},
            {"model_source": all_model_sources, "native_id": all_native_ids},
            prefix="test/",
        )

        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value)

        _print_test_summary(metrics)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure the AdamW optimizer and cosine LR schedule."""
        train_cfg = self.config.get("training", {})
        lr = float(train_cfg.get("lr", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-5))

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        n_epochs = train_cfg.get("n_epochs", 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def count_parameters(self) -> dict:
        """Count trainable parameters, grouped by component."""
        pool_params = list(self.pool.parameters()) + list(self.readout.parameters())
        counts = {
            "node_encoder": sum(p.numel() for p in self.node_encoder.parameters() if p.requires_grad),
            "edge_encoder": sum(p.numel() for p in self.edge_encoder.parameters() if p.requires_grad),
            "gnn_layers": sum(p.numel() for p in self.conv_stack.parameters() if p.requires_grad),
            "pooling": sum(p.numel() for p in pool_params if p.requires_grad),
            "prediction_head": sum(p.numel() for p in self.dockq_head.parameters() if p.requires_grad),
        }
        counts["total"] = sum(counts.values())
        return counts
