"""Tests for confidock.dockq_predictor.data.feature_utils module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from confidock.dockq_predictor.data import feature_utils as fu


def _residue(
    chain_id: str = "H",
    chain_type: str = "VH",
    residue_name: str = "ALA",
    cdr_label: int = 0,
    ca=(0.0, 0.0, 0.0),
) -> dict:
    """Build a residue dict of the shape ``graph_builder`` produces."""
    ca = np.asarray(ca, dtype=np.float32)
    return {
        "chain_id": chain_id,
        "chain_type": chain_type,
        "residue_name": residue_name,
        "cdr_label": cdr_label,
        "backbone_coords": {
            "N": ca + np.array([-1.0, 0.5, 0.0], dtype=np.float32),
            "CA": ca,
            "C": ca + np.array([1.0, 0.5, 0.0], dtype=np.float32),
        },
        "heavy_atom_coords": np.stack([ca, ca + np.array([1.5, 0.0, 0.0], dtype=np.float32)]),
    }


# ---------------------------------------------------------------------------
# Block layout: the two tables are the single source of truth
# ---------------------------------------------------------------------------


def test_feature_dims_are_the_documented_18_and_96() -> None:
    """The widths the model and the config both assert against."""
    assert fu.NODE_FEATURE_DIM == 18
    assert fu.EDGE_FEATURE_DIM == 96


def test_feature_dims_equal_the_sum_of_their_blocks() -> None:
    """A block added to a table must move the reported width with it."""
    assert fu.node_feature_dim() == sum(fu.NODE_BLOCK_DIMS.values())
    assert fu.edge_feature_dim() == sum(fu.EDGE_BLOCK_DIMS.values())


@pytest.mark.parametrize(
    ("layout_fn", "block_dims", "total"),
    [
        (fu.node_feature_layout, fu.NODE_BLOCK_DIMS, fu.NODE_FEATURE_DIM),
        (fu.edge_feature_layout, fu.EDGE_BLOCK_DIMS, fu.EDGE_FEATURE_DIM),
    ],
)
def test_layout_spans_are_contiguous_and_cover_every_column(layout_fn, block_dims: dict, total: int) -> None:
    """Spans tile [0, total) in declaration order, so a slice-by-name is exact."""
    layout = layout_fn()
    assert [name for name, _, _ in layout] == list(block_dims)
    cursor = 0
    for name, start, end in layout:
        assert start == cursor
        assert end - start == block_dims[name]
        cursor = end
    assert cursor == total


# ---------------------------------------------------------------------------
# Individual encoders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", range(7))
def test_one_hot_cdr_sets_exactly_the_labelled_position(label: int) -> None:
    """Label scheme 0=none, 1..3=H1..H3, 4..6=L1..L3."""
    encoding = fu.one_hot_cdr(label)
    assert encoding.shape == (7,)
    assert encoding.sum() == pytest.approx(1.0)
    assert encoding[label] == pytest.approx(1.0)


def test_rbf_encode_distance_peaks_at_the_nearest_centre() -> None:
    """The RBF block is unnormalised Gaussians centred on a distance grid."""
    centers = np.linspace(0, 20.0, 16)
    encoded = fu.rbf_encode_distance(float(centers[3]), num_rbf=16, max_dist=20.0)
    assert encoded.shape == (16,)
    assert int(encoded.argmax()) == 3  # distance sits exactly on centre 3
    assert encoded[3] == pytest.approx(1.0)  # unnormalised Gaussian, peak 1.0


def test_rbf_encode_distance_decays_with_distance() -> None:
    """A far distance excites the far bins, not the near ones."""
    near = fu.rbf_encode_distance(2.0)
    far = fu.rbf_encode_distance(18.0)
    assert near[0] > far[0]
    assert far[-1] > near[-1]


def test_soft_distogram_encode_is_a_normalised_distribution() -> None:
    """Unlike the RBF block, the distogram sums to 1."""
    probs = fu.soft_distogram_encode(8.0)
    assert probs.shape == (fu.DISTOGRAM_NUM_BINS,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert int(probs.argmax()) == int(np.abs(fu.DISTOGRAM_BINS - 8.0).argmin())


def test_soft_distogram_saturates_outside_the_binned_range() -> None:
    """Distances beyond 2-22A pile into the end bins rather than wrapping."""
    assert int(fu.soft_distogram_encode(0.5).argmax()) == 0
    assert int(fu.soft_distogram_encode(50.0).argmax()) == fu.DISTOGRAM_NUM_BINS - 1


def test_compute_relative_orientation_is_a_unit_direction_outer_product() -> None:
    """Flattened 3x3 outer product of the unit CA->CA direction with itself."""
    out = fu.compute_relative_orientation(np.zeros(3), np.array([3.0, 0.0, 0.0]))
    assert out.shape == (9,)
    expected = np.zeros(9, dtype=np.float32)
    expected[0] = 1.0  # direction is +x, so only the xx term is non-zero
    assert np.allclose(out, expected)


def test_compute_relative_orientation_handles_coincident_positions() -> None:
    """Zero displacement must not divide by zero."""
    assert np.allclose(fu.compute_relative_orientation(np.zeros(3), np.zeros(3)), np.zeros(9))


@pytest.mark.parametrize(
    ("res_i", "res_j", "distance", "expected_index"),
    [
        ("SER", "THR", 3.0, 1),  # polar-polar in H-bond range
        ("LEU", "VAL", 4.0, 2),  # nonpolar-nonpolar in hydrophobic range
        ("LYS", "ASP", 5.0, 3),  # opposite charges in ionic range
        ("SER", "THR", 3.9, 4),  # too far for an H-bond, still vdW
        ("SER", "THR", 12.0, 0),  # nothing at all
    ],
)
def test_compute_interaction_type_classifies_by_chemistry_and_distance(
    res_i: str, res_j: str, distance: float, expected_index: int
) -> None:
    """Order: none / H-bond / hydrophobic / ionic / vdW."""
    out = fu.compute_interaction_type({"residue_name": res_i}, {"residue_name": res_j}, distance)
    assert out.shape == (5,)
    assert out.sum() == pytest.approx(1.0)
    assert out[expected_index] == pytest.approx(1.0)


def test_local_frame_is_orthonormal() -> None:
    """The AlphaFold-style frame: Gram-Schmidt on CA->N and CA->C."""
    rotation, origin = fu._local_frame(_residue()["backbone_coords"])
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    assert np.allclose(origin, np.zeros(3))


def test_local_frame_returns_none_without_a_full_backbone() -> None:
    """A residue missing N/CA/C gets no frame (features fall back to zeros)."""
    assert fu._local_frame({"CA": np.zeros(3)}) == (None, None)
    assert fu._local_frame(None) == (None, None)


def test_opposite_chain_proximity_only_counts_the_other_chain() -> None:
    """Same-chain neighbours must not register as interface proximity."""
    residues = [
        _residue(chain_id="H", ca=(0.0, 0.0, 0.0)),
        _residue(chain_id="H", ca=(3.0, 0.0, 0.0)),  # same chain: ignored
        _residue(chain_id="A", chain_type="antigen", ca=(5.0, 0.0, 0.0)),
    ]
    nearest, neighbors = fu._opposite_chain_proximity(residues)

    # Residue 0's nearest opposite-chain CA is 5A away, encoded as 1 - 5/30. The
    # same-chain neighbour 3A away is invisible to both blocks.
    assert nearest[0] == pytest.approx(1.0 - 5.0 / 30.0, abs=1e-6)
    assert neighbors[0, 0] == pytest.approx(0.1)  # one opposite CA within 6A, /10 cap
    assert neighbors[0, 1] == pytest.approx(0.1)  # and within 8A
    # The antigen sees BOTH H residues (2A and 5A away), so its counts are 2/10.
    assert nearest[2] == pytest.approx(1.0 - 2.0 / 30.0, abs=1e-6)
    assert neighbors[2, 0] == pytest.approx(0.2)


def test_opposite_chain_proximity_is_zero_for_a_single_chain() -> None:
    """With no opposite chain there is no proximity signal at all."""
    nearest, neighbors = fu._opposite_chain_proximity([_residue(), _residue(ca=(4.0, 0.0, 0.0))])
    assert np.allclose(nearest, 0.0)
    assert np.allclose(neighbors, 0.0)


def test_opposite_chain_proximity_handles_no_residues() -> None:
    """Empty input returns empty arrays rather than raising."""
    nearest, neighbors = fu._opposite_chain_proximity([])
    assert nearest.shape == (0,)
    assert neighbors.shape == (0, 2)


# ---------------------------------------------------------------------------
# compute_node_features
# ---------------------------------------------------------------------------


def test_compute_node_features_fills_the_documented_blocks() -> None:
    """Check each node block lands in its declared span."""
    residues = [
        _residue(chain_id="H", chain_type="VH", cdr_label=3, ca=(0.0, 0.0, 0.0)),
        _residue(chain_id="A", chain_type="antigen", cdr_label=0, ca=(4.0, 0.0, 0.0)),
    ]
    cdr_labels = torch.tensor([3, 0])
    plddt = torch.tensor([90.0, 50.0])

    x = fu.compute_node_features(residues, cdr_labels, plddt)
    spans = dict((name, (start, end)) for name, start, end in fu.node_feature_layout())
    assert x.shape == (2, fu.NODE_FEATURE_DIM)

    start, end = spans["cdr"]
    assert x[0, start + 3] == pytest.approx(1.0)  # CDR-H3
    assert x[1, start] == pytest.approx(1.0)  # no CDR

    assert x[0, spans["is_antibody"][0]] == pytest.approx(1.0)
    assert x[1, spans["is_antibody"][0]] == pytest.approx(0.0)
    assert x[0, spans["is_heavy"][0]] == pytest.approx(1.0)
    assert x[0, spans["plddt"][0]] == pytest.approx(0.9)
    assert x[1, spans["plddt"][0]] == pytest.approx(0.5)

    start, end = spans["chain_type_onehot"]
    assert torch.equal(x[0, start:end], torch.tensor([0.0, 1.0, 0.0]))  # VH
    assert torch.equal(x[1, start:end], torch.tensor([1.0, 0.0, 0.0]))  # antigen

    assert x[0, spans["is_cdr"][0]] == pytest.approx(1.0)
    assert x[0, spans["is_cdr_h3"][0]] == pytest.approx(1.0)
    assert x[1, spans["is_cdr"][0]] == pytest.approx(0.0)
    assert x[0, spans["nearest_opposite_dist"][0]] > 0.0  # 4A apart, opposite chains


def test_compute_node_features_defaults_plddt_to_a_constant() -> None:
    """Without pLDDT the block is a constant 0.7 -- 17 informative dims, not 18."""
    residues = [_residue(), _residue(chain_id="A", chain_type="antigen", ca=(4.0, 0.0, 0.0))]
    x = fu.compute_node_features(residues, torch.tensor([0, 0]), plddt=None)
    col = dict((name, start) for name, start, _ in fu.node_feature_layout())["plddt"]
    assert torch.allclose(x[:, col], torch.full((2,), 0.7))


# ---------------------------------------------------------------------------
# compute_edge_features
# ---------------------------------------------------------------------------


def _two_residue_graph():
    """Build a single cross-chain edge pair 4A apart, one CDR-H3 and one antigen."""
    residues = [
        _residue(chain_id="H", chain_type="VH", residue_name="SER", cdr_label=3, ca=(0.0, 0.0, 0.0)),
        _residue(chain_id="A", chain_type="antigen", residue_name="THR", cdr_label=0, ca=(4.0, 0.0, 0.0)),
    ]
    pos = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    chain_ids = torch.tensor([0, 1])
    return residues, pos, edge_index, chain_ids


def test_compute_edge_features_fills_the_documented_blocks() -> None:
    """Check each edge block lands in its declared span."""
    residues, pos, edge_index, chain_ids = _two_residue_graph()
    edge_attr = fu.compute_edge_features(edge_index, pos, residues, model_source="boltz2", chain_ids=chain_ids)
    spans = dict((name, (start, end)) for name, start, end in fu.edge_feature_layout())
    assert edge_attr.shape == (2, fu.EDGE_FEATURE_DIM)

    assert edge_attr[0, spans["is_cross_chain"][0]] == pytest.approx(1.0)

    start, end = spans["model_source"]
    assert torch.equal(edge_attr[0, start:end], torch.tensor([1.0, 0.0, 0.0]))

    # Ordered chain-pair one-hot: source VH (index 1) x target antigen (index 0).
    start, end = spans["chain_pair_onehot"]
    assert int(edge_attr[0, start:end].argmax()) == 1 * 3 + 0
    assert int(edge_attr[1, start:end].argmax()) == 0 * 3 + 1  # reversed edge

    assert edge_attr[0, spans["source_is_cdr"][0]] == pytest.approx(1.0)
    assert edge_attr[0, spans["target_is_cdr"][0]] == pytest.approx(0.0)
    assert edge_attr[0, spans["source_is_cdr_h3"][0]] == pytest.approx(1.0)
    assert edge_attr[1, spans["source_is_cdr_h3"][0]] == pytest.approx(0.0)  # direction matters

    start, end = spans["contact_bins"]
    assert torch.equal(edge_attr[0, start:end], torch.tensor([1.0, 1.0]))  # 4A < 5A and < 8A

    start, end = spans["relative_rotation"]
    rel = edge_attr[0, start:end].reshape(3, 3)
    assert torch.allclose(rel.T @ rel, torch.eye(3), atol=1e-4)  # rotation, so orthonormal

    start, end = spans["ca_distogram"]
    assert edge_attr[0, start:end].sum() == pytest.approx(1.0, abs=1e-5)


def test_compute_edge_features_unknown_model_source_is_all_zero_not_an_error() -> None:
    """An unrecognised folding model must not crash featurisation."""
    residues, pos, edge_index, chain_ids = _two_residue_graph()
    edge_attr = fu.compute_edge_features(edge_index, pos, residues, model_source="af3", chain_ids=chain_ids)
    start, end = dict((n, (s, e)) for n, s, e in fu.edge_feature_layout())["model_source"]
    assert torch.equal(edge_attr[0, start:end], torch.zeros(3))


def test_compute_edge_features_without_chain_ids_reports_no_cross_chain() -> None:
    """The cross-chain flag needs chain ids; absent, it is 0 rather than guessed."""
    residues, pos, edge_index, _ = _two_residue_graph()
    edge_attr = fu.compute_edge_features(edge_index, pos, residues, model_source="boltz2", chain_ids=None)
    col = dict((n, s) for n, s, _ in fu.edge_feature_layout())["is_cross_chain"]
    assert float(edge_attr[:, col].abs().sum()) == pytest.approx(0.0)


def test_compute_edge_features_min_heavy_atom_distance_beats_the_ca_distance() -> None:
    """The block is the true closest approach, so it differs from the CA-CA RBF."""
    residues, pos, edge_index, chain_ids = _two_residue_graph()
    edge_attr = fu.compute_edge_features(edge_index, pos, residues, model_source="boltz2", chain_ids=chain_ids)
    spans = dict((name, (start, end)) for name, start, end in fu.edge_feature_layout())
    ca_rbf = edge_attr[0, slice(*spans["ca_distance_rbf"])]
    heavy_rbf = edge_attr[0, slice(*spans["min_heavy_atom_dist"])]
    # Side atoms reach 1.5A toward the partner, so min heavy dist (2.5A) < CA dist (4A).
    assert torch.allclose(heavy_rbf, torch.tensor(fu.rbf_encode_distance(2.5)))
    assert not torch.allclose(heavy_rbf, ca_rbf)


def test_compute_edge_features_falls_back_to_the_ca_distance_without_heavy_atoms() -> None:
    """A residue with no heavy-atom array must still get a finite feature."""
    residues, pos, edge_index, chain_ids = _two_residue_graph()
    for res in residues:
        res["heavy_atom_coords"] = None
    edge_attr = fu.compute_edge_features(edge_index, pos, residues, model_source="boltz2", chain_ids=chain_ids)
    spans = dict((name, (start, end)) for name, start, end in fu.edge_feature_layout())
    heavy_rbf = edge_attr[0, slice(*spans["min_heavy_atom_dist"])]
    assert torch.allclose(heavy_rbf, torch.tensor(fu.rbf_encode_distance(4.0)))
