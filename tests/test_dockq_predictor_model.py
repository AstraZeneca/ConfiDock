"""Tests for the DockQ-predictor GNN: conv stack, readout and SimpleGNN wiring."""

from __future__ import annotations

import copy

import pytest
import torch
from torch_geometric.data import Batch, Data

from confidock.dockq_predictor.data.feature_utils import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from confidock.dockq_predictor.models.conv_stacks import MPNNConv, MPNNStack
from confidock.dockq_predictor.models.readouts import GatedSetReadout
from confidock.dockq_predictor.models.simple_gnn import SimpleGNN

CONFIG = {
    "model": {
        "node_input_dim": NODE_FEATURE_DIM,
        "edge_input_dim": EDGE_FEATURE_DIM,
        "hidden_dim": 32,
        "edge_dim": 16,
        "num_gnn_layers": 2,
        "readout_num_heads": 2,
        "dropout": 0.0,
    },
    "loss": {"huber_delta_dockq": 0.1, "rank_type": "listnet", "rank_tau": 0.1, "lambda_rank": 0.2},
    "training": {"lr": 1e-4, "weight_decay": 1e-5, "n_epochs": 10},
}


def _graph(native_id: str, dockq: float, n_nodes: int = 8, seed: int = 0) -> Data:
    """Build a small random graph shaped exactly like a real featurised decoy."""
    generator = torch.Generator().manual_seed(seed)
    n_edges = 20
    data = Data(
        x=torch.randn(n_nodes, NODE_FEATURE_DIM, generator=generator),
        edge_index=torch.randint(0, n_nodes, (2, n_edges), generator=generator),
        edge_attr=torch.randn(n_edges, EDGE_FEATURE_DIM, generator=generator),
    )
    data.is_interface = torch.tensor([i % 2 == 0 for i in range(n_nodes)])
    data.y_dockq = torch.tensor([dockq])
    data.native_id = native_id
    data.model_source = "boltz2"
    return data


def _batch(*specs) -> Batch:
    """Batch of (native_id, dockq) decoy specs."""
    return Batch.from_data_list([_graph(nid, dockq, seed=i) for i, (nid, dockq) in enumerate(specs)])


def _config(**model_overrides) -> dict:
    """Deep-copied CONFIG with model-block overrides applied."""
    config = copy.deepcopy(CONFIG)
    config["model"].update(model_overrides)
    return config


# ---------------------------------------------------------------------------
# MPNN conv stack
# ---------------------------------------------------------------------------


def test_mpnn_conv_preserves_the_node_dimension() -> None:
    """The stack contract: scalars in, same-width scalars out."""
    conv = MPNNConv(hidden_dim=8, edge_dim=4)
    x = torch.randn(5, 8)
    out = conv(x, torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.randn(3, 4))
    assert out.shape == (5, 8)


def test_mpnn_conv_message_uses_both_endpoints_and_the_edge() -> None:
    """Edge conditioning is the point: changing edge_attr must change the output."""
    torch.manual_seed(0)
    conv = MPNNConv(hidden_dim=8, edge_dim=4)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    first = conv(x, edge_index, torch.zeros(2, 4))
    second = conv(x, edge_index, torch.ones(2, 4))
    assert not torch.allclose(first, second)


def test_mpnn_conv_mean_aggregation_ignores_duplicate_degree() -> None:
    """Mean, not sum: repeating an identical incoming edge must not scale the update."""
    torch.manual_seed(0)
    conv = MPNNConv(hidden_dim=8, edge_dim=4)
    x = torch.randn(3, 8)
    edge_attr = torch.randn(1, 4)
    single = conv(x, torch.tensor([[1], [0]]), edge_attr)
    doubled = conv(x, torch.tensor([[1, 1], [0, 0]]), edge_attr.repeat(2, 1))
    assert torch.allclose(single[0], doubled[0], atol=1e-6)


def test_mpnn_stack_has_one_conv_and_norm_per_layer() -> None:
    """num_gnn_layers is honoured, not silently clamped."""
    stack = MPNNStack(hidden_dim=8, edge_dim=4, num_layers=3)
    assert len(stack.convs) == 3
    assert len(stack.norms) == 3


def test_mpnn_stack_output_shape_and_gradient() -> None:
    """End-to-end shape plus a gradient reaching every layer."""
    stack = MPNNStack(hidden_dim=8, edge_dim=4, num_layers=3)
    x = torch.randn(6, 8, requires_grad=True)
    out = stack(x, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]), torch.randn(4, 4))
    assert out.shape == (6, 8)
    out.sum().backward()
    assert all(p.grad is not None for p in stack.parameters())


# ---------------------------------------------------------------------------
# Gated set readout
# ---------------------------------------------------------------------------


def test_gated_set_readout_pools_to_one_vector() -> None:
    """[n, hidden] -> [hidden], whatever n is."""
    readout = GatedSetReadout(hidden_dim=16, num_heads=4)
    assert readout(torch.randn(7, 16)).shape == (16,)
    assert readout(torch.randn(1, 16)).shape == (16,)


def test_gated_set_readout_is_permutation_invariant() -> None:
    """It pools a SET, so node order must not matter."""
    torch.manual_seed(0)
    readout = GatedSetReadout(hidden_dim=16, num_heads=4).eval()
    x = torch.randn(6, 16)
    with torch.no_grad():
        assert torch.allclose(readout(x), readout(x[torch.randperm(6)]), atol=1e-5)


def test_gated_set_readout_is_not_a_mean_pool() -> None:
    """The measured failure mode it replaces: an unweighted mean over the nodes."""
    torch.manual_seed(0)
    readout = GatedSetReadout(hidden_dim=16, num_heads=4).eval()
    x = torch.randn(6, 16)
    with torch.no_grad():
        assert not torch.allclose(readout(x), readout(x.mean(dim=0, keepdim=True).repeat(6, 1)), atol=1e-4)


def test_gated_set_readout_attention_weights_sum_over_nodes() -> None:
    """Each head's weights are a softmax over NODES, not over heads."""
    readout = GatedSetReadout(hidden_dim=16, num_heads=3)
    x = torch.randn(5, 16)
    a = torch.tanh(readout.value_branch(x)) * torch.sigmoid(readout.gate_branch(x))
    weights = torch.softmax(readout.head_scores(a), dim=0)
    assert weights.shape == (5, 3)
    assert torch.allclose(weights.sum(dim=0), torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# SimpleGNN: construction guards
# ---------------------------------------------------------------------------


def test_simple_gnn_defaults_to_the_fixed_feature_widths() -> None:
    """An omitted node/edge dim resolves to the feature set's actual width."""
    model = SimpleGNN({"model": {"hidden_dim": 16}, "loss": {}, "training": {}})
    assert model.node_input_dim == NODE_FEATURE_DIM
    assert model.edge_input_dim == EDGE_FEATURE_DIM


@pytest.mark.parametrize(
    "overrides",
    [{"node_input_dim": 24}, {"edge_input_dim": 35}, {"node_input_dim": 1033, "edge_input_dim": 96}],
)
def test_simple_gnn_rejects_a_config_for_a_different_feature_set(overrides: dict) -> None:
    """Caught at construction, not as a Linear shape error after the cache warms."""
    with pytest.raises(ValueError, match="the feature set is fixed"):
        SimpleGNN(_config(**overrides))


@pytest.mark.parametrize(
    ("key", "value"),
    [("architecture", "gat"), ("architecture", "gvp"), ("pooling_type", "mean"), ("head_type", "binned")],
)
def test_simple_gnn_rejects_dropped_ablation_variants(key: str, value: str) -> None:
    """A stale config must not silently train a different model than it names."""
    with pytest.raises(ValueError, match="is not implemented"):
        SimpleGNN(_config(**{key: value}))


@pytest.mark.parametrize(
    ("key", "value"),
    [("architecture", "mpnn"), ("architecture", "MPNN"), ("pooling_type", "gated_set"), ("head_type", "scalar")],
)
def test_simple_gnn_accepts_the_supported_variant_values(key: str, value: str) -> None:
    """The keys are allowed at their one implemented value, case-insensitively.

    Constructing must not raise, AND the accepted value must resolve to the one
    implemented component -- an MPNN stack, a gated-set readout and a scalar head.
    """
    model = SimpleGNN(_config(**{key: value}))
    assert isinstance(model.conv_stack, MPNNStack)
    assert isinstance(model.readout, GatedSetReadout)
    with torch.no_grad():
        assert model.dockq_head(torch.zeros(1, 64)).shape == (1, 1)


def test_simple_gnn_rejects_an_unknown_rank_type() -> None:
    """loss.rank_type is validated by losses.py at construction."""
    config = copy.deepcopy(CONFIG)
    config["loss"]["rank_type"] = "listmle"
    with pytest.raises(ValueError, match="Unknown loss.rank_type"):
        SimpleGNN(config)


def test_count_parameters_totals_its_components() -> None:
    """The reported breakdown must add up and cover every trainable parameter."""
    model = SimpleGNN(CONFIG)
    counts = model.count_parameters()
    assert set(counts) == {"node_encoder", "edge_encoder", "gnn_layers", "pooling", "prediction_head", "total"}
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total")
    assert counts["total"] == sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# SimpleGNN: forward
# ---------------------------------------------------------------------------


def test_forward_returns_one_bounded_dockq_per_graph() -> None:
    """A sigmoid head, so predictions are in [0, 1] by construction."""
    model = SimpleGNN(CONFIG).eval()
    batch = _batch(("a", 0.2), ("a", 0.8), ("b", 0.5))
    with torch.no_grad():
        out = model(batch)
    assert set(out) == {"dockq"}
    assert out["dockq"].shape == (3,)
    assert bool(((out["dockq"] >= 0) & (out["dockq"] <= 1)).all())


def test_forward_handles_a_single_unbatched_graph() -> None:
    """Inference on one structure must not require a Batch wrapper."""
    model = SimpleGNN(CONFIG).eval()
    with torch.no_grad():
        out = model(_graph("a", 0.5))
    assert out["dockq"].shape == (1,)


def test_forward_pools_interface_nodes_only() -> None:
    """Changing a NON-interface node must not change the prediction.

    Run on an edgeless graph so message passing cannot carry the perturbation into
    an interface node; then only the pooling selection is under test.
    """
    model = SimpleGNN(CONFIG).eval()
    data = _graph("a", 0.5)
    data.edge_index = torch.zeros(2, 0, dtype=torch.long)
    data.edge_attr = torch.zeros(0, EDGE_FEATURE_DIM)

    perturbed = data.clone()
    non_interface = int((~data.is_interface).nonzero()[0])
    perturbed.x[non_interface] += 10.0

    with torch.no_grad():
        assert torch.allclose(model(perturbed)["dockq"], model(data)["dockq"], atol=1e-6)

    # ... whereas perturbing an INTERFACE node does move the prediction.
    interface = data.clone()
    interface.x[int(data.is_interface.nonzero()[0])] += 10.0
    with torch.no_grad():
        assert not torch.allclose(model(interface)["dockq"], model(data)["dockq"], atol=1e-6)


def test_forward_falls_back_to_all_nodes_without_an_interface() -> None:
    """A graph with no detected interface must still produce a prediction."""
    model = SimpleGNN(CONFIG).eval()
    data = _graph("a", 0.5)
    data.is_interface = torch.zeros_like(data.is_interface)
    with torch.no_grad():
        out = model(data)
    assert out["dockq"].shape == (1,)
    assert torch.isfinite(out["dockq"]).all()


# ---------------------------------------------------------------------------
# SimpleGNN: objective assembly
# ---------------------------------------------------------------------------


def test_training_step_adds_the_weighted_ranking_term_to_huber() -> None:
    """Loss = huber + lambda_rank * rank, with both terms actually present."""
    model = SimpleGNN(CONFIG)
    model.eval()  # dropout off so the two forward passes agree
    batch = _batch(("a", 0.2), ("a", 0.9), ("b", 0.4), ("b", 0.6))

    loss = model.training_step(batch, 0)

    with torch.no_grad():
        preds = model(batch)["dockq"]
    huber = torch.nn.functional.huber_loss(preds, batch.y_dockq, delta=0.1)
    from confidock.dockq_predictor.losses.losses import within_complex_ranking_loss

    rank = within_complex_ranking_loss(preds, batch.y_dockq, batch.native_id, rank_type="listnet", tau=0.1)["rank"]
    assert float(loss) == pytest.approx(float(huber + 0.2 * rank), abs=1e-5)


def test_training_step_is_pure_huber_when_lambda_rank_is_zero() -> None:
    """lambda_rank = 0 reproduces pure-Huber training exactly."""
    config = copy.deepcopy(CONFIG)
    config["loss"]["lambda_rank"] = 0.0
    model = SimpleGNN(config).eval()
    batch = _batch(("a", 0.2), ("a", 0.9))

    loss = model.training_step(batch, 0)
    with torch.no_grad():
        preds = model(batch)["dockq"]
    assert float(loss) == pytest.approx(float(torch.nn.functional.huber_loss(preds, batch.y_dockq, delta=0.1)))


def test_training_step_gradient_reaches_every_parameter() -> None:
    """No component may be silently detached from the objective."""
    model = SimpleGNN(CONFIG)
    model.training_step(_batch(("a", 0.2), ("a", 0.9), ("b", 0.4)), 0).backward()
    ungrown = [
        name
        for name, p in model.named_parameters()
        if p.grad is None or float(p.grad.abs().sum()) == pytest.approx(0.0)
    ]
    assert ungrown == []


def test_validation_step_collects_metadata_for_per_native_metrics() -> None:
    """Val metrics group by native_id, so the step must carry it per graph."""
    model = SimpleGNN(CONFIG).eval()
    batch = _batch(("a", 0.2), ("a", 0.9), ("b", 0.4))
    output = model.validation_step(batch, 0)
    assert output["native_id"] == ["a", "a", "b"]
    assert output["model_source"] == ["boltz2"] * 3
    assert output["predictions_dockq"].shape == (3,)
    assert len(model.validation_step_outputs) == 1


def test_on_validation_epoch_end_clears_its_buffer() -> None:
    """Otherwise every epoch's metrics would pool the previous epochs' outputs."""
    model = SimpleGNN(CONFIG).eval()
    model.validation_step(_batch(("a", 0.2), ("a", 0.9), ("b", 0.4), ("b", 0.7)), 0)
    model.on_validation_epoch_end()
    assert model.validation_step_outputs == []


def test_on_validation_epoch_end_is_a_noop_without_outputs() -> None:
    """Sanity-check hooks that fire before any batch has run."""
    model = SimpleGNN(CONFIG)
    assert model.on_validation_epoch_end() is None


def test_test_step_collects_the_same_metadata_as_validation() -> None:
    """``trainer.test(ckpt_path="best")`` runs after fit, into a separate buffer."""
    model = SimpleGNN(CONFIG).eval()
    output = model.test_step(_batch(("a", 0.2), ("a", 0.9), ("b", 0.4)), 0)
    assert output["native_id"] == ["a", "a", "b"]
    assert output["model_source"] == ["boltz2"] * 3
    assert output["targets_dockq"].tolist() == pytest.approx([0.2, 0.9, 0.4])
    assert len(model.test_step_outputs) == 1
    assert model.validation_step_outputs == []  # buffers must not be shared


def test_on_test_epoch_end_prints_the_summary_and_clears_its_buffer(capsys) -> None:
    """The final report is read off stdout in the job log, so it must not crash.

    It indexes ``metrics`` directly for the per-native and AUC keys, which only
    exist once metadata is present and a complex has >= 2 decoys -- a regression
    here would surface as a KeyError at the very end of a multi-hour run.
    """
    model = SimpleGNN(CONFIG).eval()
    model.test_step(_batch(("a", 0.2), ("a", 0.9), ("b", 0.4), ("b", 0.7)), 0)
    model.on_test_epoch_end()

    printed = capsys.readouterr().out
    assert "Test Results:" in printed
    for section in ("Global Metrics", "Top-k Success Rates", "AUC Metrics", "Per-Native Metrics"):
        assert section in printed
    assert "boltz2" in printed  # the per-model blocks found the one source
    assert model.test_step_outputs == []


def test_on_test_epoch_end_is_a_noop_without_outputs() -> None:
    """Guards a bare ``trainer.test`` with an empty loader."""
    model = SimpleGNN(CONFIG)
    assert model.on_test_epoch_end() is None


def test_configure_optimizers_uses_the_config_lr_and_a_cosine_schedule() -> None:
    """The final recipe is AdamW at 1e-4 with cosine annealing per epoch."""
    model = SimpleGNN(CONFIG)
    spec = model.configure_optimizers()
    optimizer = spec["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)
    assert isinstance(spec["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)
    assert spec["lr_scheduler"]["interval"] == "epoch"
