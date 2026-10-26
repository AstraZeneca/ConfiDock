"""Node and edge feature computation for the interface graph.

The feature set here is FIXED: 18 node dims and 96 edge dims, in the block order
declared by NODE_BLOCK_DIMS / EDGE_BLOCK_DIMS below. Those two tables are the
single source of truth -- ``compute_node_features`` / ``compute_edge_features``
concatenate blocks in exactly that order, and ``node_feature_layout`` /
``edge_feature_layout`` report the resulting column spans so a consumer can slice
``data.x`` / ``data.edge_attr`` per block by name.

Every block is computed from the predicted structure's coordinates and the CDR
labels alone. Three consequences worth knowing:

* There is no sequence-identity block (no amino-acid one-hot) and no language-model
  embedding. Residue identity enters only through the interaction-type heuristic.
* There is no DSSP / SASA / dihedral pass, so no ``mkdssp`` binary is required and
  features never depend on an external tool's availability.
* There is no PAE block, so a structure needs no companion PAE file to be featurised.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

# --- Node feature blocks ----------------------------------------------------
#
# Fixed widths, in concatenation order. compute_node_features() appends these
# blocks in exactly this sequence, so the order IS the column layout.
NODE_BLOCK_DIMS = {
    "cdr": 7,  # CDR label one-hot: none/H1/H2/H3/L1/L2/L3
    "is_antibody": 1,  # 1.0 if the residue is on VH or VL
    "is_heavy": 1,  # 1.0 if the residue is on VH
    "plddt": 1,  # per-residue pLDDT, scaled to [0, 1]
    "chain_type_onehot": 3,  # antigen / VH / VL, explicit 3-way one-hot
    "is_cdr": 1,  # any CDR (cdr_label != 0)
    "is_cdr_h3": 1,  # CDR-H3 specifically (cdr_label == 3)
    "nearest_opposite_dist": 1,  # min CA distance to any opposite chain
    "opposite_neighbors": 2,  # # opposite-chain CA within 6A / within 8A
}

# --- Edge feature blocks ----------------------------------------------------
EDGE_BLOCK_DIMS = {
    "ca_distance_rbf": 16,  # CA-CA distance, RBF-encoded
    "is_cross_chain": 1,  # 1.0 if the two nodes are on different chains
    "model_source": 3,  # folding-model one-hot (boltz2/esmfold2/protenix)
    "relative_orientation": 9,  # CA-CA direction outer-product proxy
    "interaction_type": 5,  # heuristic H-bond/hydrophobic/ionic/vdW/none
    "chain_pair_onehot": 9,  # ordered source_type x target_type one-hot
    "source_is_cdr": 1,  # residues[i] is any CDR
    "target_is_cdr": 1,  # residues[j] is any CDR
    "source_is_cdr_h3": 1,  # residues[i] is CDR-H3
    "target_is_cdr_h3": 1,  # residues[j] is CDR-H3
    "contact_bins": 2,  # [CA dist < 5A, CA dist < 8A] binaries
    "min_heavy_atom_dist": 16,  # min heavy-atom distance, RBF-encoded
    "translation_source_frame": 3,  # t_ij in source residue's local frame
    "translation_target_frame": 3,  # t_ij in target residue's local frame
    "relative_rotation": 9,  # R_i^T R_j between the two backbone frames (flat)
    "ca_distogram": 16,  # soft-histogram of the CA-CA distance (2-22A bins)
}


def _layout(block_dims: Dict[str, int]) -> "list[tuple[str, int, int]]":
    """Turn a block-width table into ``(name, start, end)`` spans; ``end`` exclusive."""
    layout = []
    cursor = 0
    for name, width in block_dims.items():
        layout.append((name, cursor, cursor + width))
        cursor += width
    return layout


def node_feature_layout() -> "list[tuple[str, int, int]]":
    """Node blocks as ``(name, start, end)`` column spans into ``data.x``."""
    return _layout(NODE_BLOCK_DIMS)


def node_feature_dim() -> int:
    """Width of the node feature vector (18)."""
    return sum(NODE_BLOCK_DIMS.values())


def edge_feature_layout() -> "list[tuple[str, int, int]]":
    """Edge blocks as ``(name, start, end)`` column spans into ``data.edge_attr``."""
    return _layout(EDGE_BLOCK_DIMS)


def edge_feature_dim() -> int:
    """Width of the edge feature vector (96)."""
    return sum(EDGE_BLOCK_DIMS.values())


NODE_FEATURE_DIM = node_feature_dim()
EDGE_FEATURE_DIM = edge_feature_dim()


def one_hot_cdr(cdr_label: int) -> np.ndarray:
    """Convert a CDR label to a one-hot encoding (7-dim).

    Label scheme: 0=none, 1=H1, 2=H2, 3=H3, 4=L1, 5=L2, 6=L3.
    """
    encoding = np.zeros(7, dtype=np.float32)
    encoding[cdr_label] = 1.0
    return encoding


def rbf_encode_distance(distance: float, num_rbf: int = 16, max_dist: float = 20.0) -> np.ndarray:
    """Radial basis function encoding of a distance.

    Parameters
    ----------
    distance : float
        Distance in Angstroms.
    num_rbf : int
        Number of RBF functions (== the block width).
    max_dist : float
        Distance at which the last RBF centre sits.

    Returns
    -------
    np.ndarray
        RBF-encoded distance (num_rbf-dim), unnormalised Gaussians.
    """
    centers = np.linspace(0, max_dist, num_rbf)
    gamma = 1.0 / ((max_dist / num_rbf) ** 2)
    rbf = np.exp(-gamma * (distance - centers) ** 2)
    return rbf.astype(np.float32)


# Distogram binning: AlphaFold-style edges from 2A to 22A. The soft distogram is
# NOT a folding model's predicted distance distribution (those aren't saved by the
# models used here) -- it's a normalised soft-histogram encoding of the OBSERVED
# distance in the predicted structure, a sharper complement to the RBF block.
# DISTOGRAM_BINS gives bin CENTRES; width = number of bins.
DISTOGRAM_MIN = 2.0
DISTOGRAM_MAX = 22.0
DISTOGRAM_NUM_BINS = 16
DISTOGRAM_BINS = np.linspace(DISTOGRAM_MIN, DISTOGRAM_MAX, DISTOGRAM_NUM_BINS, dtype=np.float32)


def soft_distogram_encode(distance: float, num_bins: int = DISTOGRAM_NUM_BINS) -> np.ndarray:
    """Soft one-hot ("distogram") encoding of a distance over distance bins.

    Assigns the distance to the bins via a softmax over negative squared distance
    to each bin centre, so the encoding is a smooth probability-like vector that
    sums to 1 (sharper / more localised than the RBF block, which is unnormalised
    Gaussians). Distances beyond the range saturate into the end bins.
    """
    centers = (
        DISTOGRAM_BINS
        if num_bins == DISTOGRAM_NUM_BINS
        else np.linspace(DISTOGRAM_MIN, DISTOGRAM_MAX, num_bins, dtype=np.float32)
    )
    bin_width = (DISTOGRAM_MAX - DISTOGRAM_MIN) / max(1, num_bins - 1)
    logits = -((distance - centers) ** 2) / (2.0 * bin_width**2)
    logits -= logits.max()  # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum() + 1e-8
    return probs.astype(np.float32)


def compute_relative_orientation(pos_i: np.ndarray, pos_j: np.ndarray) -> np.ndarray:
    """Relative backbone orientation between two residues.

    Simplified: the outer product of the unit CA->CA direction with itself, as a
    rotation-like proxy. Returns a flattened 3x3 matrix (9-dim).
    """
    displacement = pos_j - pos_i
    dist = np.linalg.norm(displacement)

    if dist < 1e-6:
        return np.zeros(9, dtype=np.float32)

    direction = displacement / dist
    outer = np.outer(direction, direction)
    return outer.flatten().astype(np.float32)


def _local_frame(bb: Optional[Dict]) -> "tuple[Optional[np.ndarray], Optional[np.ndarray]]":
    """Build a residue's local backbone frame from N, CA, C.

    Returns ``(rotation, origin)`` where rotation is a 3x3 matrix whose columns are an
    orthonormal basis (Gram-Schmidt on the CA->N and CA->C directions, matching
    the AlphaFold rigid-frame convention) and origin is the CA position. Returns
    ``(None, None)`` when N/CA/C are not all available.
    """
    if not bb or not all(k in bb for k in ("N", "CA", "C")):
        return None, None
    n, ca, c = np.asarray(bb["N"]), np.asarray(bb["CA"]), np.asarray(bb["C"])
    v1 = n - ca
    v2 = c - ca
    e1 = v1 / (np.linalg.norm(v1) + 1e-8)
    # Orthogonalise v2 against e1.
    u2 = v2 - np.dot(v2, e1) * e1
    e2 = u2 / (np.linalg.norm(u2) + 1e-8)
    e3 = np.cross(e1, e2)
    rotation = np.stack([e1, e2, e3], axis=1).astype(np.float32)
    return rotation, ca.astype(np.float32)


def compute_interaction_type(residue_i: Dict, residue_j: Dict, distance: float) -> np.ndarray:
    """Classify the interaction between two residues as a one-hot (5-dim).

    Order: none / H-bond / hydrophobic / ionic / vdW. Simplified heuristics:

    - H-bond: polar-polar, dist 2.5-3.5A
    - Hydrophobic: nonpolar-nonpolar, dist 3-5A
    - Ionic: charged-charged, dist 3-6A
    - vdW: any, dist < 4A
    - None: otherwise

    This is the only block that reads residue identity; there is no amino-acid
    one-hot in the node features.
    """
    interaction = np.zeros(5, dtype=np.float32)

    # Residue property classifications
    polar = ["SER", "THR", "ASN", "GLN", "TYR", "CYS"]
    charged_pos = ["LYS", "ARG", "HIS"]
    charged_neg = ["ASP", "GLU"]
    hydrophobic = ["ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "GLY"]

    res_i_name = residue_i["residue_name"]
    res_j_name = residue_j["residue_name"]

    # H-bond
    if (
        (res_i_name in polar or res_i_name in charged_pos or res_i_name in charged_neg)
        and (res_j_name in polar or res_j_name in charged_pos or res_j_name in charged_neg)
        and 2.5 <= distance <= 3.5
    ):
        interaction[1] = 1.0
    # Hydrophobic
    elif res_i_name in hydrophobic and res_j_name in hydrophobic and 3.0 <= distance <= 5.0:
        interaction[2] = 1.0
    # Ionic
    elif (
        (res_i_name in charged_pos and res_j_name in charged_neg)
        or (res_i_name in charged_neg and res_j_name in charged_pos)
    ) and 3.0 <= distance <= 6.0:
        interaction[3] = 1.0
    # Van der Waals
    elif distance < 4.0:
        interaction[4] = 1.0
    else:
        interaction[0] = 1.0  # None

    return interaction


def _opposite_chain_proximity(residues: List[Dict]) -> "tuple[np.ndarray, np.ndarray]":
    """Per-residue opposite-chain CA proximity.

    Returns
    -------
    (nearest, neighbors)
        nearest [N]: min CA distance to any residue on a DIFFERENT chain,
        encoded as 1 - d/30 clamped to [0,1] (1 => touching, 0 => >=30A / none).
        neighbors [N, 2]: count of opposite-chain CA within 6A and within 8A,
        each divided by a soft cap (10) and clamped to [0,1].
    """
    n = len(residues)
    nearest = np.zeros(n, dtype=np.float32)
    neighbors = np.zeros((n, 2), dtype=np.float32)
    if n == 0:
        return nearest, neighbors

    # CA coords + chain ids from the backbone block stored at build time.
    ca = np.zeros((n, 3), dtype=np.float32)
    have_ca = np.zeros(n, dtype=bool)
    for i, res in enumerate(residues):
        bb = res.get("backbone_coords") or {}
        if "CA" in bb:
            ca[i] = bb["CA"]
            have_ca[i] = True
    chain = np.array([res.get("chain_id") for res in residues], dtype=object)

    for i in range(n):
        if not have_ca[i]:
            continue
        opp = have_ca & (chain != chain[i])
        if not opp.any():
            continue
        d = np.linalg.norm(ca[opp] - ca[i], axis=1)
        dmin = float(d.min())
        nearest[i] = max(0.0, 1.0 - dmin / 30.0)
        neighbors[i, 0] = min(float((d < 6.0).sum()) / 10.0, 1.0)
        neighbors[i, 1] = min(float((d < 8.0).sum()) / 10.0, 1.0)

    return nearest, neighbors


def _node_feature_row(
    res: Dict,
    cdr_label: int,
    plddt_scaled: float,
    opp_nearest: float,
    opp_neighbors: np.ndarray,
) -> np.ndarray:
    """Build one residue's 18-dim feature row, blocks in ``NODE_BLOCK_DIMS`` order."""
    ct = res["chain_type"]
    return np.concatenate(
        [
            # 1. CDR label one-hot (7)
            one_hot_cdr(cdr_label),
            # 2. Is antibody chain (1)
            np.array([1.0 if ct in ["VH", "VL"] else 0.0], dtype=np.float32),
            # 3. Is heavy chain (1)
            np.array([1.0 if ct == "VH" else 0.0], dtype=np.float32),
            # 4. Per-residue pLDDT (1), already scaled to [0, 1]
            np.array([plddt_scaled], dtype=np.float32),
            # 5. Chain-type 3-way one-hot: [antigen, VH, VL] (3). Complementary to
            # the is_antibody/is_heavy binaries above (both can be on at once).
            np.array(
                [1.0 if ct == "antigen" else 0.0, 1.0 if ct == "VH" else 0.0, 1.0 if ct == "VL" else 0.0],
                dtype=np.float32,
            ),
            # 6. is_CDR (1): any CDR residue (label != 0)
            np.array([1.0 if cdr_label != 0 else 0.0], dtype=np.float32),
            # 7. is_CDR_H3 (1): CDR-H3 specifically (label == 3)
            np.array([1.0 if cdr_label == 3 else 0.0], dtype=np.float32),
            # 8. Nearest opposite-chain distance (1), normalised over a 30A cap
            np.array([opp_nearest], dtype=np.float32),
            # 9. Opposite-chain neighbour counts (2): CA within 6A and within 8A
            opp_neighbors,
        ]
    )


def compute_node_features(
    residues: List[Dict],
    cdr_labels: torch.Tensor,
    plddt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the 18-dim node feature matrix.

    Parameters
    ----------
    residues : List[Dict]
        Residue dictionaries as built by ``graph_builder``; must carry
        ``chain_type``, ``chain_id`` and ``backbone_coords``.
    cdr_labels : torch.Tensor
        CDR labels [N], scheme 0=none, 1=H1, 2=H2, 3=H3, 4=L1, 5=L2, 6=L3.
    plddt : torch.Tensor, optional
        Per-residue pLDDT [N] on the 0-100 scale. When absent every residue gets
        0.7, i.e. the block is a constant and carries no signal -- so a manifest
        without pLDDT trains on 17 informative dims, not 18.

    Returns
    -------
    torch.Tensor
        Node feature matrix [N, 18]; columns are ``node_feature_layout()``.
    """
    # Opposite-chain proximity: one vectorised pass shared by both blocks.
    opp_nearest, opp_neighbors = _opposite_chain_proximity(residues)

    labels = [int(v) for v in cdr_labels.tolist()]
    # A manifest without pLDDT gets a constant 0.7 (see the docstring).
    plddt_scaled = [v / 100.0 for v in plddt.tolist()] if plddt is not None else [0.7] * len(residues)

    features = [
        _node_feature_row(res, labels[i], plddt_scaled[i], float(opp_nearest[i]), opp_neighbors[i])
        for i, res in enumerate(residues)
    ]

    feature_matrix = np.stack(features, axis=0)
    return torch.tensor(feature_matrix, dtype=torch.float32)


# Folding-model one-hot; an unrecognised source yields all zeros.
_MODEL_SOURCE_ONEHOT = {"boltz2": [1, 0, 0], "esmfold2": [0, 1, 0], "protenix": [0, 0, 1]}

# Chain-type -> index for the ordered chain-pair one-hot (3x3 = 9).
_CHAIN_TYPE_INDEX = {"antigen": 0, "VH": 1, "VL": 2}


def _cdr_label(res: Dict) -> int:
    """Read a residue dict's CDR label, tolerant of residues without the key."""
    return int(res.get("cdr_label", 0))


def _chain_pair_onehot(res_i: Dict, res_j: Dict) -> np.ndarray:
    """Encode the ordered source_type x target_type chain pair as a one-hot (9)."""
    pair = np.zeros(9, dtype=np.float32)
    si = _CHAIN_TYPE_INDEX.get(res_i.get("chain_type", "antigen"), 0)
    ti = _CHAIN_TYPE_INDEX.get(res_j.get("chain_type", "antigen"), 0)
    pair[si * 3 + ti] = 1.0
    return pair


def _min_heavy_atom_distance(res_i: Dict, res_j: Dict, ca_distance: float) -> float:
    """Find the closest approach of any heavy atom of i to any heavy atom of j.

    This is the true contact distance. Falls back to the CA-CA distance when
    either residue carries no heavy-atom coordinates.
    """
    ha_i = res_i.get("heavy_atom_coords")
    ha_j = res_j.get("heavy_atom_coords")
    if ha_i is None or ha_j is None or not len(ha_i) or not len(ha_j):
        return ca_distance
    diff = np.asarray(ha_i)[:, None, :] - np.asarray(ha_j)[None, :, :]
    return float(np.sqrt((diff**2).sum(-1)).min())


def _frame_geometry(res_i: Dict, res_j: Dict) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Compute local-frame geometry between two residues.

    Returns ``(t_source_frame, t_target_frame, relative_rotation)``: the
    translation of j expressed in i's frame, the translation of i in j's frame
    (both scaled by 20A), and the relative rotation R_i^T R_j (flattened). Any
    component whose residue lacks a complete N/CA/C backbone is zeroed.
    """
    rot_i, origin_i = _local_frame(res_i.get("backbone_coords"))
    rot_j, origin_j = _local_frame(res_j.get("backbone_coords"))

    if rot_i is not None and origin_j is not None:
        t_source = ((rot_i.T @ (origin_j - origin_i)) / 20.0).astype(np.float32)
    else:
        t_source = np.zeros(3, dtype=np.float32)

    if rot_j is not None and origin_i is not None:
        t_target = ((rot_j.T @ (origin_i - origin_j)) / 20.0).astype(np.float32)
    else:
        t_target = np.zeros(3, dtype=np.float32)

    if rot_i is not None and rot_j is not None:
        relative_rotation = (rot_i.T @ rot_j).flatten().astype(np.float32)
    else:
        relative_rotation = np.zeros(9, dtype=np.float32)

    return t_source, t_target, relative_rotation


def _edge_feature_row(
    res_i: Dict,
    res_j: Dict,
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    distance: float,
    model_vec: np.ndarray,
    is_cross_chain: float,
) -> np.ndarray:
    """Build one edge's 96-dim feature row, blocks in ``EDGE_BLOCK_DIMS`` order."""
    t_source, t_target, relative_rotation = _frame_geometry(res_i, res_j)
    return np.concatenate(
        [
            # 1. CA-CA distance, RBF encoded (16)
            rbf_encode_distance(distance, num_rbf=16, max_dist=20.0),
            # 2. Is cross-chain edge (1)
            np.array([is_cross_chain], dtype=np.float32),
            # 3. Model source one-hot (3)
            model_vec,
            # 4. Relative backbone orientation (9)
            compute_relative_orientation(pos_i, pos_j),
            # 5. Interaction type one-hot (5)
            compute_interaction_type(res_i, res_j, distance),
            # 6. Chain-pair one-hot: ordered source_type x target_type (9)
            _chain_pair_onehot(res_i, res_j),
            # 7-10. CDR membership of source / target (1 each)
            np.array([1.0 if _cdr_label(res_i) != 0 else 0.0], dtype=np.float32),
            np.array([1.0 if _cdr_label(res_j) != 0 else 0.0], dtype=np.float32),
            np.array([1.0 if _cdr_label(res_i) == 3 else 0.0], dtype=np.float32),
            np.array([1.0 if _cdr_label(res_j) == 3 else 0.0], dtype=np.float32),
            # 11. Contact-threshold binaries: CA dist < 5A, < 8A (2)
            np.array([1.0 if distance < 5.0 else 0.0, 1.0 if distance < 8.0 else 0.0], dtype=np.float32),
            # 12. Min heavy-atom distance RBF (16)
            rbf_encode_distance(_min_heavy_atom_distance(res_i, res_j, distance), num_rbf=16, max_dist=20.0),
            # 13-15. Local-frame geometry (3 + 3 + 9)
            t_source,
            t_target,
            relative_rotation,
            # 16. CA distogram: soft-histogram of the CA-CA distance (16 bins)
            soft_distogram_encode(distance),
        ]
    )


def compute_edge_features(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    residues: List[Dict],
    model_source: str = "af3",
    chain_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the 96-dim edge feature matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge connectivity [2, E].
    pos : torch.Tensor
        CA positions [N, 3].
    residues : List[Dict]
        Residue information, carrying ``chain_type``, ``cdr_label``,
        ``backbone_coords`` and ``heavy_atom_coords``.
    model_source : str
        Folding model that produced the structure (boltz2, esmfold2, protenix).
        An unrecognised value yields an all-zero one-hot rather than an error.
    chain_ids : torch.Tensor, optional
        Chain IDs [N] for cross-chain detection. Absent -> the cross-chain flag
        is 0 everywhere.

    Returns
    -------
    torch.Tensor
        Edge feature matrix [E, 96]; columns are ``edge_feature_layout()``.
    """
    n_edges = edge_index.shape[1]

    model_vec = np.array(_MODEL_SOURCE_ONEHOT.get(model_source, [0, 0, 0]), dtype=np.float32)

    # Cross-chain flags for every edge in one pass; all-zero when chain ids are
    # absent (see the docstring).
    src, dst = edge_index[0], edge_index[1]
    if chain_ids is not None:
        cross_chain = (chain_ids[src] != chain_ids[dst]).cpu().numpy().astype(np.float32)
    else:
        cross_chain = np.zeros(n_edges, dtype=np.float32)

    # CA-CA distance of every edge in one pass. dim=-1 reduces over the coordinate
    # axis only, giving one distance per edge ([E], not a single global norm).
    distances = torch.linalg.vector_norm(pos[src] - pos[dst], dim=-1).tolist()

    pos_np = pos.cpu().numpy()
    edge_features = []

    for e in range(n_edges):
        i, j = src[e].item(), dst[e].item()
        edge_features.append(
            _edge_feature_row(
                res_i=residues[i],
                res_j=residues[j],
                pos_i=pos_np[i],
                pos_j=pos_np[j],
                distance=distances[e],
                model_vec=model_vec,
                is_cross_chain=cross_chain[e],
            )
        )

    edge_feature_matrix = np.stack(edge_features, axis=0)
    return torch.tensor(edge_feature_matrix, dtype=torch.float32)
