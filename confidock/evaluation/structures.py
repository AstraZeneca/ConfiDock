"""Structure loading and atomic utilities for protein structure evaluation.

This module provides low-level utilities for working with BioPython structure
objects, including atom selection, residue filtering, and structure loading.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import NeighborSearch, Superimposer

LOGGER = logging.getLogger(__name__)

# Constants
BACKBONE_ATOMS = {"N", "CA", "C", "O"}
CONTACT_CUTOFF = 5.0


def load_structure_pair(model_path: Path, native_path: Path):
    """Load sanitised model/native structures using the same path as DockQ.

    Parameters
    ----------
    model_path : Path
        Path to predicted structure (PDB or CIF format)
    native_path : Path
        Path to native/reference structure (PDB format)

    Returns
    -------
    tuple
        (model_structure, native_structure, chain_remap)
        where chain_remap maps any non-standard chain IDs in the model to
        single-character IDs
    """
    # Import locally to avoid heavy dependencies
    from DockQ.DockQ import load_PDB  # noqa: PLC0415

    from confidock.evaluation.dockq import (  # noqa: PLC0415
        _cif_to_pdb_text,
        _normalise_pdb_chains,
        _strip_hydrogens,
    )

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        # Use fixed, distinct filenames. Deriving names from the input stems can
        # collide when the model and native share a stem (e.g. ESMFold2 no_msa
        # single-sample predictions named "{entry}.cif" against "{entry}.pdb"),
        # in which case the native write clobbers the model and both handles
        # end up pointing at the native structure.
        sanitised_model = tmp / "model.pdb"
        sanitised_native = tmp / "native.pdb"

        if model_path.suffix == ".cif":
            model_text, cif_chain_remap = _cif_to_pdb_text(model_path)
        else:
            model_text = model_path.read_text()
            cif_chain_remap = {}

        model_text, normalize_chain_remap = _normalise_pdb_chains(model_text)

        # Combine the two remappings: cif_chain_remap (single->multi) + normalize (canonical->single)
        # If cif_chain_remap exists, chain the mappings
        chain_remap = {}
        for norm_id, single_id in normalize_chain_remap.items():
            # norm_id is the final normalized ID, single_id is the intermediate ID
            # Check if single_id came from a multi-char ID
            original_id = cif_chain_remap.get(single_id, single_id)
            chain_remap[norm_id] = original_id
        sanitised_model.write_text(_strip_hydrogens(model_text))
        sanitised_native.write_text(_strip_hydrogens(native_path.read_text()))

        return load_PDB(str(sanitised_model)), load_PDB(str(sanitised_native)), chain_remap


def chain_ids_in_order(structure) -> list[str]:
    """Extract chain IDs from a structure in order."""
    return [chain.id for chain in structure.get_chains()]


def standard_residues(chain) -> list:
    """Return only standard residues (no HETATM) from a chain."""
    return [residue for residue in chain if residue.id[0] == " "]


def residue_key(residue) -> tuple[int, str]:
    """Return a (sequence_number, insertion_code) tuple for a residue."""
    return (int(residue.id[1]), str(residue.id[2]).strip())


def heavy_atoms(residue) -> list:
    """Return non-hydrogen sidechain atoms from a residue (excludes backbone)."""
    return [atom for atom in residue if atom.element != "H" and atom.get_name() not in BACKBONE_ATOMS]


def backbone_ca_atoms(chain, residue_keys: set[tuple[int, str]] | None = None) -> dict[tuple[int, str], Any]:
    """Extract CA atoms from a chain, optionally filtered by residue keys.

    Parameters
    ----------
    chain
        BioPython Chain object
    residue_keys : set of (int, str), optional
        If provided, only return CA atoms for these (seqnum, icode) keys

    Returns
    -------
    dict
        Mapping {(seqnum, icode): CA_atom}
    """
    atoms: dict[tuple[int, str], Any] = {}
    for residue in standard_residues(chain):
        key = residue_key(residue)
        if residue_keys is not None and key not in residue_keys:
            continue
        if "CA" in residue:
            atoms[key] = residue["CA"]
    return atoms


def common_residue_keys(chain_a, chain_b) -> set[tuple[int, str]]:
    """Return the intersection of residue keys between two chains."""
    keys_a = {residue_key(res) for res in standard_residues(chain_a)}
    keys_b = {residue_key(res) for res in standard_residues(chain_b)}
    return keys_a & keys_b


def transform_coord(coord: np.ndarray, rot: np.ndarray, tran: np.ndarray) -> np.ndarray:
    """Apply a rotation and translation to a coordinate."""
    return np.dot(coord, rot) + tran


def rmsd_from_pairs(native_coords: list[np.ndarray], model_coords: list[np.ndarray]) -> float | str:
    """Compute RMSD between two lists of coordinates.

    Returns
    -------
    float or ""
        RMSD value, or empty string if inputs are invalid
    """
    if not native_coords or len(native_coords) != len(model_coords):
        return ""
    diffs = np.asarray(native_coords, dtype=float) - np.asarray(model_coords, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))


def fit_superposition(native_atoms: list, model_atoms: list) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Compute best-fit superposition between two atom lists.

    Parameters
    ----------
    native_atoms : list
        List of BioPython Atom objects (reference)
    model_atoms : list
        List of BioPython Atom objects (to be aligned)

    Returns
    -------
    tuple or None
        (rotation_matrix, translation_vector, rmsd) or None if alignment fails
    """
    if not native_atoms or len(native_atoms) != len(model_atoms):
        return None

    superimposer = Superimposer()
    superimposer.set_atoms(native_atoms, model_atoms)
    rot, tran = superimposer.rotran
    return rot, tran, float(superimposer.rms)


def collect_common_ca_atom_pairs(native_chain, model_chain) -> tuple[list, list]:
    """Collect aligned CA atom pairs by residue identity for one chain pair.

    Returns
    -------
    tuple of (native_atoms, model_atoms)
        Lists of CA atoms in matching order
    """
    common_keys = common_residue_keys(native_chain, model_chain)
    native_ca = backbone_ca_atoms(native_chain, common_keys)
    model_ca = backbone_ca_atoms(model_chain, common_keys)
    keys = sorted(native_ca.keys() & model_ca.keys())
    return [native_ca[k] for k in keys], [model_ca[k] for k in keys]


def chain_local_ca_rmsd(native_chain, model_chain) -> float | str:
    """CA RMSD for one chain after fitting that chain to itself.

    This metric measures local chain structure quality independent of
    global placement or other chains.
    """
    native_atoms, model_atoms = collect_common_ca_atom_pairs(native_chain, model_chain)
    fit = fit_superposition(native_atoms, model_atoms)
    if fit is None:
        return ""
    return fit[2]


def residue_contacts(
    chain_a,
    chain_b,
    cutoff: float = CONTACT_CUTOFF,
) -> tuple[
    set[tuple[tuple[int, str], tuple[int, str]]],
    set[tuple[int, str]],
    set[tuple[int, str]],
]:
    """Find inter-chain residue contacts within a distance cutoff.

    Parameters
    ----------
    chain_a, chain_b
        BioPython Chain objects
    cutoff : float
        Distance cutoff in Ångströms (default: 5.0)

    Returns
    -------
    tuple of (contact_pairs, residues_a, residues_b)
        - contact_pairs: set of ((key_a, key_b)) tuples for contacting residue pairs
        - residues_a: set of residue keys from chain_a that contact chain_b
        - residues_b: set of residue keys from chain_b that contact chain_a
    """
    atoms_b = [atom for residue in standard_residues(chain_b) for atom in heavy_atoms(residue)]
    if not atoms_b:
        return set(), set(), set()

    ns = NeighborSearch(atoms_b)
    pair_contacts: set[tuple[tuple[int, str], tuple[int, str]]] = set()
    residues_a: set[tuple[int, str]] = set()
    residues_b: set[tuple[int, str]] = set()

    for residue_a in standard_residues(chain_a):
        atoms_a = heavy_atoms(residue_a)
        if not atoms_a:
            continue
        key_a = residue_key(residue_a)
        for atom in atoms_a:
            for residue_b in ns.search(atom.coord, cutoff, level="R"):
                if residue_b.id[0] != " ":
                    continue
                key_b = residue_key(residue_b)
                pair_contacts.add((key_a, key_b))
                residues_a.add(key_a)
                residues_b.add(key_b)

    return pair_contacts, residues_a, residues_b
