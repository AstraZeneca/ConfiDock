"""Structure quality and interface metrics for protein complex evaluation.

This module provides higher-level metrics for evaluating predicted protein
structures against native references:

- RMSD metrics (backbone, sidechain, role-specific)
- Interface contact precision/recall/F1
- Role-aware placement metrics for antibody-antigen complexes
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from confidock.evaluation.structures import (
    chain_local_ca_rmsd,
    collect_common_ca_atom_pairs,
    fit_superposition,
    heavy_atoms,
    residue_contacts,
    residue_key,
    rmsd_from_pairs,
    standard_residues,
    transform_coord,
)

LOGGER = logging.getLogger(__name__)


def set_metrics(native_items: set, pred_items: set) -> dict[str, float | str]:
    """Compute precision, recall, and F1 for two sets.

    Parameters
    ----------
    native_items : set
        Reference/ground truth items
    pred_items : set
        Predicted items

    Returns
    -------
    dict
        {"precision": float|"", "recall": float|"", "f1": float|""}
    """
    if not native_items and not pred_items:
        return {"precision": "", "recall": "", "f1": ""}

    tp = len(native_items & pred_items)
    precision = tp / len(pred_items) if pred_items else ""
    recall = tp / len(native_items) if native_items else ""

    if precision == "" or recall == "":
        f1 = ""
    else:
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom else ""

    return {"precision": precision, "recall": recall, "f1": f1}


def sidechain_rmsd(
    native_chain,
    model_chain,
    residue_keys: set[tuple[int, str]],
    rot: np.ndarray,
    tran: np.ndarray,
) -> float | str:
    """Compute sidechain RMSD for specified residues after applying a transform.

    Parameters
    ----------
    native_chain, model_chain
        BioPython Chain objects
    residue_keys : set of (int, str)
        Residue keys to include in the RMSD calculation
    rot, tran
        Rotation matrix and translation vector to apply to model coordinates

    Returns
    -------
    float or ""
        Sidechain RMSD in Ångströms, or empty string if no common atoms
    """
    native_residues = {residue_key(res): res for res in standard_residues(native_chain)}
    model_residues = {residue_key(res): res for res in standard_residues(model_chain)}

    native_coords: list[np.ndarray] = []
    model_coords: list[np.ndarray] = []

    for key in residue_keys:
        native_res = native_residues.get(key)
        model_res = model_residues.get(key)
        if native_res is None or model_res is None:
            continue

        native_atoms_dict = {atom.get_name(): atom for atom in heavy_atoms(native_res)}
        model_atoms_dict = {atom.get_name(): atom for atom in heavy_atoms(model_res)}
        common_atom_names = native_atoms_dict.keys() & model_atoms_dict.keys()

        for atom_name in common_atom_names:
            native_coords.append(native_atoms_dict[atom_name].coord)
            model_coords.append(transform_coord(model_atoms_dict[atom_name].coord, rot, tran))

    return rmsd_from_pairs(native_coords, model_coords)


def sidechain_rmsd_for_chain_pairs(
    native,
    model,
    chain_pairs: list[tuple[str, str]],
    residue_keys_by_native_chain: dict[str, set[tuple[int, str]]],
    rot: np.ndarray,
    tran: np.ndarray,
) -> float | str:
    """Side-chain RMSD across multiple chain pairs under one alignment frame.

    Parameters
    ----------
    native, model
        BioPython Structure objects
    chain_pairs : list of (native_chain_id, model_chain_id)
        Chains to include
    residue_keys_by_native_chain : dict
        Mapping {native_chain_id: set of residue keys to include}
    rot, tran
        Alignment transform to apply to model coordinates

    Returns
    -------
    float or ""
        Pooled sidechain RMSD across all specified chain pairs and residues
    """
    native_coords: list[np.ndarray] = []
    model_coords: list[np.ndarray] = []

    for native_chain_id, model_chain_id in chain_pairs:
        residue_keys = residue_keys_by_native_chain.get(native_chain_id, set())
        if not residue_keys:
            continue

        native_chain = native[native_chain_id]
        model_chain = model[model_chain_id]
        native_residues = {residue_key(res): res for res in standard_residues(native_chain)}
        model_residues = {residue_key(res): res for res in standard_residues(model_chain)}

        for key in residue_keys:
            native_res = native_residues.get(key)
            model_res = model_residues.get(key)
            if native_res is None or model_res is None:
                continue

            native_atoms_dict = {atom.get_name(): atom for atom in heavy_atoms(native_res)}
            model_atoms_dict = {atom.get_name(): atom for atom in heavy_atoms(model_res)}

            for atom_name in native_atoms_dict.keys() & model_atoms_dict.keys():
                native_coords.append(native_atoms_dict[atom_name].coord)
                model_coords.append(transform_coord(model_atoms_dict[atom_name].coord, rot, tran))

    return rmsd_from_pairs(native_coords, model_coords)


def ca_rmsd_for_chain_pairs_in_frame(
    native,
    model,
    chain_pairs: list[tuple[str, str]],
    rot: np.ndarray,
    tran: np.ndarray,
) -> float | str:
    """CA RMSD for selected chain pairs after applying a provided frame transform.

    This is used to measure placement of one group of chains (e.g., binder)
    after aligning on a different group (e.g., antigen).

    Parameters
    ----------
    native, model
        BioPython Structure objects
    chain_pairs : list of (native_chain_id, model_chain_id)
        Chains to measure
    rot, tran
        Alignment transform (from a different subset of chains)

    Returns
    -------
    float or ""
        CA RMSD in Ångströms
    """
    native_coords: list[np.ndarray] = []
    model_coords: list[np.ndarray] = []

    for native_chain_id, model_chain_id in chain_pairs:
        native_atoms, model_atoms = collect_common_ca_atom_pairs(
            native[native_chain_id],
            model[model_chain_id],
        )
        for native_atom, model_atom in zip(native_atoms, model_atoms):
            native_coords.append(native_atom.coord)
            model_coords.append(transform_coord(model_atom.coord, rot, tran))

    return rmsd_from_pairs(native_coords, model_coords)


def backbone_rmsd(
    native_chain,
    model_chain,
    residue_keys: set[tuple[int, str]] | None,
    rot: np.ndarray,
    tran: np.ndarray,
) -> float | str:
    """Compute CA RMSD for specified residues after applying a transform.

    Parameters
    ----------
    native_chain, model_chain
        BioPython Chain objects
    residue_keys : set of (int, str) or None
        Residue keys to include; if None, use all common residues
    rot, tran
        Transform to apply to model coordinates

    Returns
    -------
    float or ""
        CA RMSD in Ångströms
    """
    native_residues = {residue_key(res): res for res in standard_residues(native_chain)}
    model_residues = {residue_key(res): res for res in standard_residues(model_chain)}

    native_coords: list[np.ndarray] = []
    model_coords: list[np.ndarray] = []

    keys = residue_keys if residue_keys is not None else (native_residues.keys() & model_residues.keys())

    for key in keys:
        native_res = native_residues.get(key)
        model_res = model_residues.get(key)
        if native_res is None or model_res is None or "CA" not in native_res or "CA" not in model_res:
            continue
        native_coords.append(native_res["CA"].coord)
        model_coords.append(transform_coord(model_res["CA"].coord, rot, tran))

    return rmsd_from_pairs(native_coords, model_coords)


def _compute_role_specific_fits(
    native,
    model,
    role_to_pairs: dict[str, list[tuple[str, str]]],
) -> dict[str, tuple[np.ndarray, np.ndarray, float] | None]:
    """Compute role-specific alignment fits (binder and antigen).

    Parameters
    ----------
    native, model
        BioPython Structure objects
    role_to_pairs : dict
        Mapping {role: [(native_chain_id, model_chain_id), ...]}

    Returns
    -------
    dict
        Mapping {role: (rot, tran, rmsd) or None}
    """
    role_fits: dict[str, tuple[np.ndarray, np.ndarray, float] | None] = {"binder": None, "antigen": None}
    for role in ("binder", "antigen"):
        role_native_atoms: list = []
        role_model_atoms: list = []
        for native_chain_id, model_chain_id in role_to_pairs.get(role, []):
            native_atoms, model_atoms = collect_common_ca_atom_pairs(
                native[native_chain_id],
                model[model_chain_id],
            )
            role_native_atoms.extend(native_atoms)
            role_model_atoms.extend(model_atoms)
        role_fit = fit_superposition(role_native_atoms, role_model_atoms)
        if role_fit is not None:
            role_fits[role] = role_fit
    return role_fits


def _collect_interface_contacts(
    native,
    model,
    binder_native_chains: list[str],
    antigen_native_chains: list[str],
    native_to_model: dict[str, str],
    native_chain_ids: list[str],
    model_chain_ids: list[str],
) -> tuple[
    set[tuple[tuple[str, tuple[int, str]], tuple[str, tuple[int, str]]]],
    set[tuple[tuple[str, tuple[int, str]], tuple[str, tuple[int, str]]]],
    set[tuple[str, tuple[int, str]]],
    set[tuple[str, tuple[int, str]]],
    set[tuple[str, tuple[int, str]]],
    set[tuple[str, tuple[int, str]]],
]:
    """Collect native and predicted interface contact residues.

    Parameters
    ----------
    native, model
        BioPython Structure objects
    binder_native_chains : list of str
        Native binder chain IDs
    antigen_native_chains : list of str
        Native antigen chain IDs
    native_to_model : dict
        Mapping {native_chain_id: model_chain_id}
    native_chain_ids, model_chain_ids : list of str
        Available chain IDs in native and model structures

    Returns
    -------
    tuple
        (native_contact_pairs, pred_contact_pairs,
         native_binder_contact_residues, native_antigen_contact_residues,
         pred_binder_contact_residues, pred_antigen_contact_residues)
    """
    native_binder_contact_residues: set[tuple[str, tuple[int, str]]] = set()
    native_antigen_contact_residues: set[tuple[str, tuple[int, str]]] = set()
    native_contact_pairs: set[tuple[tuple[str, tuple[int, str]], tuple[str, tuple[int, str]]]] = set()
    pred_binder_contact_residues: set[tuple[str, tuple[int, str]]] = set()
    pred_antigen_contact_residues: set[tuple[str, tuple[int, str]]] = set()
    pred_contact_pairs: set[tuple[tuple[str, tuple[int, str]], tuple[str, tuple[int, str]]]] = set()

    for native_binder_id in binder_native_chains:
        for native_antigen_id in antigen_native_chains:
            model_binder_id = native_to_model.get(native_binder_id)
            model_antigen_id = native_to_model.get(native_antigen_id)
            if (
                native_binder_id not in native_chain_ids
                or native_antigen_id not in native_chain_ids
                or model_binder_id is None
                or model_antigen_id is None
                or model_binder_id not in model_chain_ids
                or model_antigen_id not in model_chain_ids
            ):
                continue

            native_pair, native_a_res, native_b_res = residue_contacts(
                native[native_binder_id],
                native[native_antigen_id],
            )
            pred_pair, pred_a_res, pred_b_res = residue_contacts(
                model[model_binder_id],
                model[model_antigen_id],
            )

            native_contact_pairs.update({((native_binder_id, a), (native_antigen_id, b)) for a, b in native_pair})
            pred_contact_pairs.update({((native_binder_id, a), (native_antigen_id, b)) for a, b in pred_pair})
            native_binder_contact_residues.update({(native_binder_id, key) for key in native_a_res})
            native_antigen_contact_residues.update({(native_antigen_id, key) for key in native_b_res})
            pred_binder_contact_residues.update({(native_binder_id, key) for key in pred_a_res})
            pred_antigen_contact_residues.update({(native_antigen_id, key) for key in pred_b_res})

    return (
        native_contact_pairs,
        pred_contact_pairs,
        native_binder_contact_residues,
        native_antigen_contact_residues,
        pred_binder_contact_residues,
        pred_antigen_contact_residues,
    )


def _populate_sidechain_rmsd_metrics(
    structure_metrics: dict[str, Any],
    native,
    model,
    role_to_pairs: dict[str, list[tuple[str, str]]],
    role_fits: dict[str, tuple[np.ndarray, np.ndarray, float] | None],
    native_binder_contact_residues: set[tuple[str, tuple[int, str]]],
    native_antigen_contact_residues: set[tuple[str, tuple[int, str]]],
) -> None:
    """Populate side-chain RMSD metrics for binder and antigen interfaces.

    Parameters
    ----------
    structure_metrics : dict
        Metrics dictionary to populate (modified in-place)
    native, model
        BioPython Structure objects
    role_to_pairs : dict
        Mapping {role: [(native_chain_id, model_chain_id), ...]}
    role_fits : dict
        Mapping {role: (rot, tran, rmsd) or None}
    native_binder_contact_residues, native_antigen_contact_residues : set
        Native contact residue sets
    """
    binder_native_contact_keys_by_chain: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for chain_id, key in native_binder_contact_residues:
        binder_native_contact_keys_by_chain[chain_id].add(key)
    antigen_native_contact_keys_by_chain: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for chain_id, key in native_antigen_contact_residues:
        antigen_native_contact_keys_by_chain[chain_id].add(key)

    binder_fit = role_fits.get("binder")
    if binder_fit is not None and role_to_pairs.get("binder"):
        structure_metrics["binder_interface_sidechain_rmsd"] = sidechain_rmsd_for_chain_pairs(
            native,
            model,
            role_to_pairs["binder"],
            binder_native_contact_keys_by_chain,
            binder_fit[0],
            binder_fit[1],
        )

    antigen_fit = role_fits.get("antigen")
    if antigen_fit is not None and role_to_pairs.get("antigen"):
        structure_metrics["antigen_interface_sidechain_rmsd"] = sidechain_rmsd_for_chain_pairs(
            native,
            model,
            role_to_pairs["antigen"],
            antigen_native_contact_keys_by_chain,
            antigen_fit[0],
            antigen_fit[1],
        )


def compute_structure_metrics(
    model,
    native,
    chain_map: dict[str, str],
    native_chain_roles: dict[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compute role-aware placement/contact/side-chain metrics for one sample.

    Alignment strategy:
    - binder_ca_rmsd uses a joint binder fit (e.g., H+L together) before
      computing binder CA RMSD.
    - antigen_ca_rmsd uses an antigen-only fit before computing antigen CA RMSD.
    - complex_ca_rmsd is the mean per-chain local CA RMSD (each chain fitted to
      itself), which is useful for "overall structure" disentangled from complex
      placement.
    - binder/antigen interface side-chain RMSD are computed in their matching
      role-specific fitted frames.
    - antigen_aligned_binder_ca_rmsd quantifies binder placement after aligning
      on antigen chains only; this is a direct wrong-epitope/docking signal.

    Parameters
    ----------
    model, native
        BioPython Structure objects
    chain_map : dict
        Mapping {native_chain_id: model_chain_id}
    native_chain_roles : dict
        Mapping {native_chain_id: role} where role in {"binder", "antigen"}

    Returns
    -------
    tuple of (structure_metrics, interface_metrics)
        - structure_metrics: dict with keys like "complex_ca_rmsd", "binder_ca_rmsd", etc.
        - interface_metrics: dict of {interface_key: metrics_dict} (currently empty)
    """
    native_chain_ids = [chain.id for chain in native.get_chains()]
    model_chain_ids = [chain.id for chain in model.get_chains()]

    mapped_chain_pairs: list[tuple[str, str]] = [
        (native_chain_id, model_chain_id)
        for native_chain_id, model_chain_id in chain_map.items()
        if native_chain_id in native_chain_ids and model_chain_id in model_chain_ids
    ]

    structure_metrics: dict[str, Any] = {
        "complex_ca_rmsd": "",
        "binder_ca_rmsd": "",
        "antigen_ca_rmsd": "",
        "interface_contact_precision": "",
        "interface_contact_recall": "",
        "interface_contact_f1": "",
        "binder_residue_precision": "",
        "binder_residue_recall": "",
        "antigen_residue_precision": "",
        "antigen_residue_recall": "",
        "binder_interface_sidechain_rmsd": "",
        "antigen_interface_sidechain_rmsd": "",
        "antigen_aligned_binder_ca_rmsd": "",
    }
    interface_metrics: dict[str, dict[str, Any]] = {}

    if not mapped_chain_pairs:
        return structure_metrics, interface_metrics

    # Role-aware chain groups, mapped to model chain ids.
    role_to_pairs: dict[str, list[tuple[str, str]]] = {"binder": [], "antigen": []}
    for native_chain_id, model_chain_id in mapped_chain_pairs:
        role = native_chain_roles.get(native_chain_id, "antigen")
        role_to_pairs.setdefault(role, []).append((native_chain_id, model_chain_id))

    # Per-chain local structure quality (decoupled from inter-chain placement).
    chain_local_rmsds: list[float] = []
    for native_chain_id, model_chain_id in mapped_chain_pairs:
        rmsd_val = chain_local_ca_rmsd(native[native_chain_id], model[model_chain_id])
        if rmsd_val != "":
            chain_local_rmsds.append(float(rmsd_val))
    if chain_local_rmsds:
        structure_metrics["complex_ca_rmsd"] = float(np.mean(chain_local_rmsds))

    # Role-specific frames for interface-sidechain RMSD.
    role_fits = _compute_role_specific_fits(native, model, role_to_pairs)
    for role in ("binder", "antigen"):
        if role_fits.get(role) is not None:
            structure_metrics[f"{role}_ca_rmsd"] = role_fits[role][2]

    # Native binder-antigen residue contact sets across the whole complex.
    native_to_model = {native_chain_id: model_chain_id for native_chain_id, model_chain_id in mapped_chain_pairs}
    binder_native_chains = [native_chain_id for native_chain_id, _ in role_to_pairs.get("binder", [])]
    antigen_native_chains = [native_chain_id for native_chain_id, _ in role_to_pairs.get("antigen", [])]

    (
        native_contact_pairs,
        pred_contact_pairs,
        native_binder_contact_residues,
        native_antigen_contact_residues,
        pred_binder_contact_residues,
        pred_antigen_contact_residues,
    ) = _collect_interface_contacts(
        native,
        model,
        binder_native_chains,
        antigen_native_chains,
        native_to_model,
        native_chain_ids,
        model_chain_ids,
    )

    contact_metrics = set_metrics(native_contact_pairs, pred_contact_pairs)
    structure_metrics["interface_contact_precision"] = contact_metrics["precision"]
    structure_metrics["interface_contact_recall"] = contact_metrics["recall"]
    structure_metrics["interface_contact_f1"] = contact_metrics["f1"]

    binder_res_metrics = set_metrics(native_binder_contact_residues, pred_binder_contact_residues)
    antigen_res_metrics = set_metrics(native_antigen_contact_residues, pred_antigen_contact_residues)
    structure_metrics["binder_residue_precision"] = binder_res_metrics["precision"]
    structure_metrics["binder_residue_recall"] = binder_res_metrics["recall"]
    structure_metrics["antigen_residue_precision"] = antigen_res_metrics["precision"]
    structure_metrics["antigen_residue_recall"] = antigen_res_metrics["recall"]

    # Side-chain RMSD on the native contact residues for each role.
    _populate_sidechain_rmsd_metrics(
        structure_metrics,
        native,
        model,
        role_to_pairs,
        role_fits,
        native_binder_contact_residues,
        native_antigen_contact_residues,
    )

    # Interface placement proxy: binder RMSD after antigen-only alignment.
    antigen_fit = role_fits.get("antigen")
    if antigen_fit is not None and role_to_pairs.get("binder"):
        structure_metrics["antigen_aligned_binder_ca_rmsd"] = ca_rmsd_for_chain_pairs_in_frame(
            native,
            model,
            role_to_pairs["binder"],
            antigen_fit[0],
            antigen_fit[1],
        )

    # Interface role labels for the per-interface DockQ rows.
    for iface_key, metrics in interface_metrics.items():
        chain_a, chain_b = iface_key[:1], iface_key[1:2]
        role_a = native_chain_roles.get(chain_a, "unknown")
        role_b = native_chain_roles.get(chain_b, "unknown")
        metrics["interface_role_pair"] = f"{role_a}-{role_b}"

    return structure_metrics, interface_metrics
