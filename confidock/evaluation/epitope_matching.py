"""Epitope matching for multi-binding-mode complexes.

This module provides utilities for matching predicted structures to the correct
native binding mode when multiple epitopes exist for the same antibody-antigen pair.

Uses a 2-stage alignment strategy:
1. Align on antibody framework regions (conserved, rigid)
2. Measure epitope divergence via interface position metrics

This addresses the multiplicity problem where the same PDB contains multiple
entries with identical antibody and antigen sequences but different epitopes
or binding modes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

# Fallback CDR ranges, used ONLY when no SAbDab JSON is found for a structure.
# These are approximate residue-NUMBER ranges and do not match the true CDRs on
# structures numbered sequentially from 1 (the usual case). The preferred source
# is the SAbDab JSON h_cdr/l_cdr, which defines CDRs by sequence position and is
# the same source the DockQ-predictor graph pipeline uses -- see
# _json_cdr_keys_for_chain() below and
# dockq_predictor.data.chain_mapping_cif.compute_cdr_labels_from_json.
DEFAULT_CDR_RANGES_HEAVY = [
    (26, 32),  # H-CDR1
    (52, 56),  # H-CDR2
    (95, 102),  # H-CDR3
]

DEFAULT_CDR_RANGES_LIGHT = [
    (24, 34),  # L-CDR1
    (50, 56),  # L-CDR2
    (89, 97),  # L-CDR3
]


def _json_cdr_keys_for_chain(native_path, chain, chain_type: str) -> set[tuple[int, str]] | None:
    """Return JSON-derived CDR residue keys for a binder chain, or None.

    Reads the SAbDab JSON ``h_cdr`` / ``l_cdr`` (1-based sequence positions) for
    ``native_path`` and maps them onto ``chain``'s standard residues *by order*
    -- the same mapping the graph pipeline uses, so CDR identity is consistent
    across training and evaluation. This is robust to structures numbered 1..N
    (which the hard-coded ranges are not).

    Returns None -- signalling the caller to fall back to the range heuristic --
    when there is no ``native_path``, no matching JSON, the sequence field is
    empty, or the chain's residue count does not equal the JSON sequence length
    (an indel would shift every position, so we decline rather than mislabel).
    """
    if native_path is None:
        return None
    # Reuse the graph pipeline's JSON locator so the path stays single-sourced.
    from confidock.dockq_predictor.data.chain_mapping_cif import load_sabdab_json
    from confidock.evaluation.structures import residue_key, standard_residues

    json_data = load_sabdab_json(str(native_path))
    if json_data is None:
        return None

    seq_key, cdr_key = ("h_seq", "h_cdr") if chain_type == "heavy" else ("l_seq", "l_cdr")
    seq = json_data.get(seq_key, "") or ""
    cdr = json_data.get(cdr_key, {}) or {}
    if not seq or not cdr:
        return None

    residues = standard_residues(chain)
    if len(residues) != len(seq):
        LOGGER.warning(
            "CDR keys: chain %s has %d residues but %s length is %d; falling back to range heuristic",
            getattr(chain, "id", "?"),
            len(residues),
            seq_key,
            len(seq),
        )
        return None

    cdr_positions = set()
    for positions in cdr.values():
        cdr_positions.update(int(p) for p in positions)

    return {residue_key(res) for pos1, res in enumerate(residues, start=1) if pos1 in cdr_positions}


def infer_antibody_chain_type(chain) -> str:
    """Infer if an antibody chain is heavy or light based on sequence length.

    Heavy chains are typically longer (~110-130 residues) than light chains (~105-115).
    This is a heuristic - for production use, consider using ANARCI or similar tools.

    Parameters
    ----------
    chain
        BioPython Chain object

    Returns
    -------
    str
        "heavy" or "light"
    """
    from confidock.evaluation.structures import standard_residues

    # Count standard residues
    num_residues = len(list(standard_residues(chain)))

    # Heuristic threshold: chains with >115 residues are likely heavy
    # This is based on typical antibody structure:
    # - Light chains: ~105-115 residues
    # - Heavy chains: ~110-130+ residues
    # Using 115 as a midpoint threshold
    return "heavy" if num_residues > 115 else "light"


def get_framework_residues(
    chain,
    cdr_ranges: list[tuple[int, int]],
    native_path=None,
    chain_type: str | None = None,
) -> set[tuple[int, str]]:
    """Extract framework residue keys (non-CDR regions) from a chain.

    The CDR set is taken from the SAbDab JSON when available (via
    ``native_path`` + ``chain_type``), else from the ``cdr_ranges`` heuristic.
    Framework = all standard residues minus the CDR set.

    Parameters
    ----------
    chain
        BioPython Chain object
    cdr_ranges : list of (start, end) tuples
        Fallback CDR ranges in PDB residue numbering.
    native_path : Path or str, optional
        Native structure path, for JSON CDR lookup.
    chain_type : str, optional
        "heavy" or "light" -- selects h_cdr vs l_cdr in the JSON.

    Returns
    -------
    set of (seqnum, icode) tuples
        Framework residue keys
    """
    from confidock.evaluation.structures import residue_key, standard_residues

    cdr_keys = _cdr_keys_for_chain(chain, cdr_ranges, native_path, chain_type)
    return {residue_key(residue) for residue in standard_residues(chain)} - cdr_keys


def _cdr_keys_for_chain(
    chain,
    cdr_ranges: list[tuple[int, int]],
    native_path=None,
    chain_type: str | None = None,
) -> set[tuple[int, str]]:
    """Return CDR residue keys for a chain: JSON if available, else range heuristic."""
    from confidock.evaluation.structures import residue_key, standard_residues

    if chain_type is not None:
        json_keys = _json_cdr_keys_for_chain(native_path, chain, chain_type)
        if json_keys is not None:
            return json_keys

    return {
        residue_key(residue)
        for residue in standard_residues(chain)
        if any(start <= residue_key(residue)[0] <= end for start, end in cdr_ranges)
    }


def _determine_chain_type(native_chain_id: str, native_chain, chain_types: dict[str, str] | None) -> str:
    """Determine chain type using metadata or sequence-based inference."""
    if chain_types and native_chain_id in chain_types:
        return chain_types[native_chain_id]
    return infer_antibody_chain_type(native_chain)


def _validate_chain_exists(chain_id: str, available_ids: set[str], structure_type: str) -> bool:
    """Validate that a chain exists in the structure."""
    if chain_id not in available_ids:
        LOGGER.warning(f"{structure_type} chain {chain_id} not found in structure")
        return False
    return True


def _collect_framework_atoms_for_chain(
    native_chain,
    model_chain,
    framework_keys: set,
) -> tuple[list, list]:
    """Collect framework CA atoms for a single chain pair."""
    from confidock.evaluation.structures import backbone_ca_atoms

    native_ca = backbone_ca_atoms(native_chain, framework_keys)
    model_ca = backbone_ca_atoms(model_chain, framework_keys)

    common_keys = sorted(native_ca.keys() & model_ca.keys())
    native_atoms = [native_ca[k] for k in common_keys]
    model_atoms = [model_ca[k] for k in common_keys]

    return native_atoms, model_atoms


def align_on_framework(
    model_struct,
    native_struct,
    chain_map: dict[str, str],
    native_chain_roles: dict[str, str],
    heavy_cdr_ranges: list[tuple[int, int]] | None = None,
    light_cdr_ranges: list[tuple[int, int]] | None = None,
    chain_types: dict[str, str] | None = None,
    native_path=None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Align model to native using antibody framework regions only.

    Parameters
    ----------
    model_struct, native_struct
        BioPython Structure objects
    chain_map : dict
        Mapping {native_chain_id: model_chain_id}
    native_chain_roles : dict
        Mapping {native_chain_id: role} where role in {"binder", "antigen"}
    heavy_cdr_ranges : list of (start, end), optional
        Fallback CDR ranges for heavy chain (used only when no SAbDab JSON found)
    light_cdr_ranges : list of (start, end), optional
        Fallback CDR ranges for light chain (used only when no SAbDab JSON found)
    chain_types : dict, optional
        Mapping {chain_id: chain_type} where chain_type in {"heavy", "light", "antigen"}
        from SAbDab metadata. If provided, this overrides sequence-based inference.
    native_path : Path or str, optional
        Native structure path, enabling JSON-derived CDR definitions (preferred
        over the range heuristic).

    Returns
    -------
    tuple or None
        (rotation_matrix, translation_vector, rmsd) or None if alignment fails
    """
    from confidock.evaluation.structures import fit_superposition

    heavy_cdr_ranges = heavy_cdr_ranges or DEFAULT_CDR_RANGES_HEAVY
    light_cdr_ranges = light_cdr_ranges or DEFAULT_CDR_RANGES_LIGHT

    # Get available chains in both structures
    native_chain_ids = {chain.id for chain in native_struct.get_chains()}
    model_chain_ids = {chain.id for chain in model_struct.get_chains()}

    # Collect framework CA atoms from binder chains only
    native_framework_atoms = []
    model_framework_atoms = []

    for native_chain_id, model_chain_id in chain_map.items():
        role = native_chain_roles.get(native_chain_id, "antigen")
        if role != "binder":
            continue

        # Validate chains exist in structures
        if not _validate_chain_exists(native_chain_id, native_chain_ids, "Native"):
            continue
        if not _validate_chain_exists(model_chain_id, model_chain_ids, "Model"):
            continue

        native_chain = native_struct[native_chain_id]
        model_chain = model_struct[model_chain_id]

        chain_type = _determine_chain_type(native_chain_id, native_chain, chain_types)
        cdr_ranges = heavy_cdr_ranges if chain_type == "heavy" else light_cdr_ranges
        framework_keys = get_framework_residues(
            native_chain, cdr_ranges, native_path=native_path, chain_type=chain_type
        )

        # Collect framework atoms for this chain pair
        native_atoms, model_atoms = _collect_framework_atoms_for_chain(native_chain, model_chain, framework_keys)
        native_framework_atoms.extend(native_atoms)
        model_framework_atoms.extend(model_atoms)

    if not native_framework_atoms:
        LOGGER.warning("No framework atoms found for alignment")
        return None

    return fit_superposition(native_framework_atoms, model_framework_atoms)


def compute_interface_ca_rmsd(
    model_struct,
    native_struct,
    chain_map: dict[str, str],
    native_chain_roles: dict[str, str],
    rot: np.ndarray,
    tran: np.ndarray,
    contact_cutoff: float = 8.0,
) -> float | str:
    """Compute interface CA-RMSD after applying a framework alignment.

    Measures only CA atoms at the binding interface (within contact_cutoff
    of the binding partner).

    Parameters
    ----------
    model_struct, native_struct
        BioPython Structure objects
    chain_map : dict
        Mapping {native_chain_id: model_chain_id}
    native_chain_roles : dict
        Mapping {native_chain_id: role}
    rot, tran
        Rotation and translation from framework alignment
    contact_cutoff : float
        Distance threshold for interface definition (Ångströms)

    Returns
    -------
    float or ""
        Interface CA-RMSD in Ångströms
    """
    from confidock.evaluation.structures import (
        backbone_ca_atoms,
        residue_contacts,
        transform_coord,
    )

    # Identify interface residues in native structure
    binder_chains = [
        (nat_id, mod_id) for nat_id, mod_id in chain_map.items() if native_chain_roles.get(nat_id) == "binder"
    ]
    antigen_chains = [
        (nat_id, mod_id) for nat_id, mod_id in chain_map.items() if native_chain_roles.get(nat_id) == "antigen"
    ]

    # Collect all interface residues
    interface_residues_binder = set()
    interface_residues_antigen = set()

    for nat_binder_id, _ in binder_chains:
        for nat_antigen_id, _ in antigen_chains:
            _, binder_res, antigen_res = residue_contacts(
                native_struct[nat_binder_id],
                native_struct[nat_antigen_id],
                cutoff=contact_cutoff,
            )
            interface_residues_binder.update((nat_binder_id, key) for key in binder_res)
            interface_residues_antigen.update((nat_antigen_id, key) for key in antigen_res)

    # Collect CA atoms at interface
    native_coords = []
    model_coords = []

    for nat_id, mod_id in chain_map.items():
        # Get interface residues for this chain
        if native_chain_roles.get(nat_id) == "binder":
            chain_interface_keys = {key for chain_id, key in interface_residues_binder if chain_id == nat_id}
        else:
            chain_interface_keys = {key for chain_id, key in interface_residues_antigen if chain_id == nat_id}

        if not chain_interface_keys:
            continue

        native_ca = backbone_ca_atoms(native_struct[nat_id], chain_interface_keys)
        model_ca = backbone_ca_atoms(model_struct[mod_id], chain_interface_keys)

        common_keys = sorted(native_ca.keys() & model_ca.keys())
        for key in common_keys:
            native_coords.append(native_ca[key].coord)
            model_coords.append(transform_coord(model_ca[key].coord, rot, tran))

    if not native_coords:
        return ""

    from confidock.evaluation.structures import rmsd_from_pairs

    return rmsd_from_pairs(native_coords, model_coords)


def _extract_cdr_residue_keys(
    chain,
    cdr_ranges: list[tuple[int, int]],
    native_path=None,
    chain_type: str | None = None,
) -> set[tuple[int, str]]:
    """Extract CDR residue keys from a chain.

    Uses SAbDab JSON CDRs when available (``native_path`` + ``chain_type``), else
    the ``cdr_ranges`` heuristic.

    Parameters
    ----------
    chain
        BioPython Chain object
    cdr_ranges : list of (start, end) tuples
        Fallback CDR ranges in PDB residue numbering.
    native_path : Path or str, optional
        Native structure path, for JSON CDR lookup.
    chain_type : str, optional
        "heavy" or "light" -- selects h_cdr vs l_cdr in the JSON.

    Returns
    -------
    set of (seqnum, icode) tuples
        CDR residue keys
    """
    return _cdr_keys_for_chain(chain, cdr_ranges, native_path, chain_type)


def _collect_cdr_coords_for_chain(
    native_chain,
    model_chain,
    cdr_ranges: list[tuple[int, int]],
    rot: np.ndarray,
    tran: np.ndarray,
    native_path=None,
    chain_type: str | None = None,
) -> tuple[list, list]:
    """Collect CDR coordinates for a single chain pair.

    Returns
    -------
    tuple of (list, list)
        (native_coords, transformed_model_coords) or ([], []) if no CDRs found
    """
    from confidock.evaluation.structures import (
        backbone_ca_atoms,
        transform_coord,
    )

    cdr_keys = _extract_cdr_residue_keys(native_chain, cdr_ranges, native_path, chain_type)
    if not cdr_keys:
        return [], []

    native_ca = backbone_ca_atoms(native_chain, cdr_keys)
    model_ca = backbone_ca_atoms(model_chain, cdr_keys)

    common_keys = native_ca.keys() & model_ca.keys()
    native_coords = [native_ca[key].coord for key in common_keys]
    model_coords = [transform_coord(model_ca[key].coord, rot, tran) for key in common_keys]

    return native_coords, model_coords


def compute_cdr_center_distance(
    model_struct,
    native_struct,
    chain_map: dict[str, str],
    native_chain_roles: dict[str, str],
    rot: np.ndarray,
    tran: np.ndarray,
    heavy_cdr_ranges: list[tuple[int, int]] | None = None,
    light_cdr_ranges: list[tuple[int, int]] | None = None,
    chain_types: dict[str, str] | None = None,
    native_path=None,
) -> float | str:
    """Compute distance between CDR centers of mass after framework alignment.

    Parameters
    ----------
    model_struct, native_struct
        BioPython Structure objects
    chain_map : dict
        Mapping {native_chain_id: model_chain_id}
    native_chain_roles : dict
        Mapping {native_chain_id: role}
    rot, tran
        Rotation and translation from framework alignment
    heavy_cdr_ranges : list of (start, end), optional
        Fallback CDR ranges for heavy chain (used only when no SAbDab JSON found)
    light_cdr_ranges : list of (start, end), optional
        Fallback CDR ranges for light chain (used only when no SAbDab JSON found)
    native_path : Path or str, optional
        Native structure path, enabling JSON-derived CDR definitions.

    Returns
    -------
    float or ""
        Distance between CDR centers in Ångströms
    """
    heavy_cdr_ranges = heavy_cdr_ranges or DEFAULT_CDR_RANGES_HEAVY
    light_cdr_ranges = light_cdr_ranges or DEFAULT_CDR_RANGES_LIGHT

    # Collect CDR CA atoms
    native_cdr_coords = []
    model_cdr_coords = []

    for native_chain_id, model_chain_id in chain_map.items():
        role = native_chain_roles.get(native_chain_id, "antigen")
        if role != "binder":
            continue

        native_chain = native_struct[native_chain_id]
        model_chain = model_struct[model_chain_id]

        chain_type = _determine_chain_type(native_chain_id, native_chain, chain_types)
        cdr_ranges = heavy_cdr_ranges if chain_type == "heavy" else light_cdr_ranges

        # Collect CDR coords for this chain pair
        native_coords, model_coords = _collect_cdr_coords_for_chain(
            native_chain, model_chain, cdr_ranges, rot, tran, native_path=native_path, chain_type=chain_type
        )
        native_cdr_coords.extend(native_coords)
        model_cdr_coords.extend(model_coords)

    if not native_cdr_coords or not model_cdr_coords:
        return ""

    # Compute centers of mass
    native_center = np.mean(native_cdr_coords, axis=0)
    model_center = np.mean(model_cdr_coords, axis=0)

    return float(np.linalg.norm(native_center - model_center))


def _compute_distance_by_method(
    method: str,
    model_struct,
    native_struct,
    remapped_chain_map: dict[str, str],
    chain_roles: dict[str, str],
    rot: np.ndarray,
    tran: np.ndarray,
    contact_cutoff: float,
    native_path=None,
) -> float | str:
    """Compute distance using the specified method.

    Parameters
    ----------
    method : str
        Matching method: "interface_rmsd" or "cdr_distance"
    model_struct, native_struct
        BioPython Structure objects
    remapped_chain_map : dict
        Mapping {native_chain_id: model_chain_id}
    chain_roles : dict
        Mapping {native_chain_id: role}
    rot, tran
        Rotation and translation from framework alignment
    contact_cutoff : float
        Distance threshold for interface definition
    native_path : Path or str, optional
        Native structure path, for JSON-derived CDR definitions (cdr_distance).

    Returns
    -------
    float or ""
        Computed distance or empty string if computation fails
    """
    if method == "interface_rmsd":
        return compute_interface_ca_rmsd(
            model_struct,
            native_struct,
            remapped_chain_map,
            chain_roles,
            rot,
            tran,
            contact_cutoff=contact_cutoff,
        )
    elif method == "cdr_distance":
        return compute_cdr_center_distance(
            model_struct,
            native_struct,
            remapped_chain_map,
            chain_roles,
            rot,
            tran,
            native_path=native_path,
        )
    else:
        LOGGER.error(f"Unknown matching method: {method}")
        return ""


def _classify_match_quality(distance: float) -> str:
    """Classify match quality based on distance threshold.

    Parameters
    ----------
    distance : float
        Distance in Ångströms

    Returns
    -------
    str
        Match quality: "good_match", "ambiguous", or "poor_match"
    """
    if distance < 8.0:
        return "good_match"
    elif distance < 15.0:
        return "ambiguous"
    else:
        return "poor_match"


def match_epitope_by_interface_position(
    model_path: Path,
    native_candidates: list[tuple[Path, dict[str, str], dict[str, str]]],
    method: str = "interface_rmsd",
    contact_cutoff: float = 8.0,
) -> tuple[Path, dict[str, str], dict[str, str], float, str] | None:
    """Match a predicted structure to the best native binding mode.

    Uses 2-stage alignment:
    1. Align on antibody framework
    2. Measure interface divergence

    Parameters
    ----------
    model_path : Path
        Path to predicted structure
    native_candidates : list of (native_path, chain_map, chain_roles)
        List of native structures with same Ab+Ag sequences but different epitopes
    method : str
        Matching method: "interface_rmsd", "cdr_distance", or "contact_overlap"
    contact_cutoff : float
        Distance threshold for interface definition

    Returns
    -------
    tuple or None
        (best_native_path, chain_map, chain_roles, distance, match_quality)
        or None if matching fails

    match_quality categories:
    - "good_match": distance < 8Å (clear match)
    - "ambiguous": 8Å ≤ distance < 15Å (uncertain)
    - "poor_match": distance ≥ 15Å (likely true negative)
    """
    from confidock.evaluation.structures import load_structure_pair

    if not native_candidates:
        LOGGER.warning("No native candidates provided for epitope matching")
        return None

    results = []

    for native_path, chain_map, chain_roles in native_candidates:
        try:
            model_struct, native_struct, chain_remap = load_structure_pair(model_path, native_path)

            # Apply chain remapping to chain_map
            remapped_chain_map = {nat: chain_remap.get(mod, mod) for nat, mod in chain_map.items()}

            # Stage 1: Framework alignment
            alignment = align_on_framework(
                model_struct,
                native_struct,
                remapped_chain_map,
                chain_roles,
                native_path=native_path,
            )

            if alignment is None:
                LOGGER.warning(f"Framework alignment failed for {native_path.name}")
                continue

            rot, tran, framework_rmsd = alignment

            # Stage 2: Interface divergence measurement
            distance = _compute_distance_by_method(
                method,
                model_struct,
                native_struct,
                remapped_chain_map,
                chain_roles,
                rot,
                tran,
                contact_cutoff,
                native_path=native_path,
            )

            if distance == "":
                LOGGER.warning(f"Distance computation failed for {native_path.name}")
                continue

            results.append((native_path, chain_map, chain_roles, float(distance), framework_rmsd))

        except Exception as exc:
            LOGGER.error(f"Epitope matching failed for {native_path.name}: {exc}")
            continue

    if not results:
        LOGGER.warning("All epitope matching attempts failed")
        return None

    # Sort by distance (closest first)
    results.sort(key=lambda x: x[3])
    best_native_path, best_chain_map, best_chain_roles, best_distance, framework_rmsd = results[0]

    # Classify match quality
    match_quality = _classify_match_quality(best_distance)

    LOGGER.info(
        f"Matched {model_path.name} to {best_native_path.name}: "
        f"distance={best_distance:.2f}Å, quality={match_quality}"
    )

    return best_native_path, best_chain_map, best_chain_roles, best_distance, match_quality


def detect_multiple_binding_modes(
    entry: str,
    native_dir: Path,
    multiplicity_data: dict[str, list[dict]],
) -> list[tuple[Path, dict[str, str], dict[str, str]]] | None:
    """Detect if an entry has multiple binding modes and return all candidates.

    Parameters
    ----------
    entry : str
        Entry name (e.g., "1jp5_HcLcC")
    native_dir : Path
        Directory containing native PDB files
    multiplicity_data : dict
        Mapping {pdb_id: [list of entry dicts with sequences and chain roles]}

    Returns
    -------
    list of (native_path, chain_map, chain_roles) or None
        None if only one binding mode exists
    """
    pdb_id = entry.split("_")[0].lower()

    if pdb_id not in multiplicity_data:
        return None

    entries_for_pdb = multiplicity_data[pdb_id]

    if len(entries_for_pdb) <= 1:
        return None

    # Check if this is "Same Ab + Same Ag" case by sequence uniqueness
    h_seqs = {e["h_seq"] for e in entries_for_pdb}
    l_seqs = {e["l_seq"] for e in entries_for_pdb}
    t_seqs = {e["t_seq"] for e in entries_for_pdb}

    same_ab = len(h_seqs) == 1 and len(l_seqs) == 1
    same_ag = len(t_seqs) == 1

    if not (same_ab and same_ag):
        # Not a "Same Ab + Same Ag" case - no need for epitope matching
        return None

    LOGGER.info(f"Detected {len(entries_for_pdb)} binding modes for {pdb_id} (same Ab+Ag)")

    # Build candidate list
    candidates = []
    for entry_data in entries_for_pdb:
        native_path = native_dir / f"{entry_data['full_entry']}.pdb"
        if not native_path.is_file():
            continue

        # Extract chain_map and chain_roles from entry_data
        # This assumes multiplicity_data contains sufficient info
        # You may need to adapt this based on your actual data structure
        chain_map = entry_data.get("chain_map", {})
        chain_roles = entry_data.get("chain_roles", {})

        candidates.append((native_path, chain_map, chain_roles))

    return candidates if len(candidates) > 1 else None
