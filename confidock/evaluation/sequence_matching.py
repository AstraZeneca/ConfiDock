"""Sequence-based chain matching for structure alignment.

This module provides utilities to match chains between predicted and native
structures based on sequence similarity rather than chain IDs, which is more
robust for handling:
- Different chain labeling conventions
- Multi-entry PDBs with same sequences but different chain IDs
- Cases where chain IDs don't correspond between model and native
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def extract_sequence_from_structure(structure, chain_id: str) -> str:
    """Extract amino acid sequence from a structure chain.

    Parameters
    ----------
    structure
        BioPython Structure object or DockQ-loaded structure
    chain_id : str
        Chain identifier

    Returns
    -------
    str
        Single-letter amino acid sequence
    """
    from Bio.PDB.Polypeptide import PPBuilder

    ppb = PPBuilder()

    # Handle both BioPython and DockQ structures
    try:
        # BioPython structure: structure[model_id][chain_id]
        chain = structure[0][chain_id]
    except (KeyError, TypeError):
        # DockQ structure: structure[chain_id]
        chain = structure[chain_id]

    peptides = ppb.build_peptides(chain)
    return "".join(str(pep.get_sequence()) for pep in peptides)


def sequence_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity between two sequences.

    Parameters
    ----------
    seq1, seq2 : str
        Amino acid sequences (single-letter code)

    Returns
    -------
    float
        Identity score between 0.0 and 1.0
    """
    if not seq1 or not seq2:
        return 0.0

    # Simple identity: exact matches / length of shorter sequence
    min_len = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
    return matches / min_len if min_len > 0 else 0.0


def _get_chain_ids(structure) -> list[str]:
    """Extract chain IDs from a structure (BioPython or DockQ format).

    Parameters
    ----------
    structure
        BioPython Structure object or DockQ-loaded structure (dict-like)

    Returns
    -------
    list of str
        Chain identifiers
    """
    try:
        return [c.id for c in structure.get_chains()]
    except AttributeError:
        # DockQ structure (dict-like)
        return list(structure.keys())


def _extract_sequences_from_chains(structure, chain_ids: list[str]) -> dict[str, str]:
    """Extract sequences for all chains in a structure.

    Parameters
    ----------
    structure
        BioPython Structure object or DockQ-loaded structure
    chain_ids : list of str
        Chain identifiers to extract

    Returns
    -------
    dict
        Mapping {chain_id: sequence}
    """
    sequences = {}
    for chain_id in chain_ids:
        try:
            sequences[chain_id] = extract_sequence_from_structure(structure, chain_id)
        except Exception as exc:
            LOGGER.debug(f"Could not extract sequence for chain {chain_id}: {exc}")
    return sequences


def match_chains_by_sequence(
    model_structure,
    native_structure,
    min_identity: float = 0.9,
) -> dict[str, str]:
    """Match chains between model and native structures by sequence similarity.

    Parameters
    ----------
    model_structure
        Predicted structure (BioPython or DockQ format)
    native_structure
        Native reference structure (BioPython or DockQ format)
    min_identity : float
        Minimum sequence identity threshold (default: 0.9 = 90%)

    Returns
    -------
    dict
        Chain mapping {native_chain_id: model_chain_id}
    """
    # Get chain IDs and extract sequences
    model_chains = _get_chain_ids(model_structure)
    native_chains = _get_chain_ids(native_structure)
    model_seqs = _extract_sequences_from_chains(model_structure, model_chains)
    native_seqs = _extract_sequences_from_chains(native_structure, native_chains)

    # Match chains by sequence identity
    chain_map = {}
    used_model_chains = set()

    for nat_id, nat_seq in native_seqs.items():
        best_match = None
        best_identity = min_identity

        for mod_id, mod_seq in model_seqs.items():
            if mod_id in used_model_chains:
                continue

            identity = sequence_identity(nat_seq, mod_seq)

            if identity > best_identity:
                best_identity = identity
                best_match = mod_id

        if best_match:
            chain_map[nat_id] = best_match
            used_model_chains.add(best_match)
            LOGGER.debug(f"Matched chains by sequence: {nat_id} -> {best_match} (identity={best_identity:.3f})")

    return chain_map


def load_sequences_from_json(json_path: Path) -> dict[str, str]:
    """Load sequences from SAbDab JSON file.

    Parameters
    ----------
    json_path : Path
        Path to JSON file (e.g., 1jp5_HcLcC.json)

    Returns
    -------
    dict
        Mapping {chain_type: sequence} where chain_type in {"h_seq", "l_seq", "t_seq"}
    """
    with json_path.open() as f:
        data = json.load(f)

    sequences = {}
    for key in ["h_seq", "l_seq", "t_seq"]:
        if key in data:
            sequences[key] = data[key]

    return sequences


def _get_canonical_candidates(role: str, canonical_seqs: dict[str, str]) -> list[tuple[str, str]]:
    """Get canonical sequence candidates based on chain role.

    Parameters
    ----------
    role : str
        Chain role: "binder" or "antigen"
    canonical_seqs : dict
        Canonical sequences from JSON {seq_type: sequence}

    Returns
    -------
    list of (chain_type, sequence) tuples
        Candidate sequences to match against
    """
    if role == "binder":
        candidates = []
        if "h_seq" in canonical_seqs:
            candidates.append(("h", canonical_seqs["h_seq"]))
        if "l_seq" in canonical_seqs:
            candidates.append(("l", canonical_seqs["l_seq"]))
        return candidates
    else:  # antigen
        return [("t", canonical_seqs.get("t_seq", ""))]


def _find_best_sequence_match(
    candidates: list[tuple[str, str]],
    model_seqs: dict[str, str],
    used_model_chains: set[str],
    min_identity: float = 0.9,
) -> tuple[str | None, float]:
    """Find the best matching model chain for canonical sequence candidates.

    Parameters
    ----------
    candidates : list of (chain_type, sequence) tuples
        Canonical sequences to match against
    model_seqs : dict
        Model chain sequences {chain_id: sequence}
    used_model_chains : set
        Set of already-matched model chain IDs
    min_identity : float
        Minimum identity threshold

    Returns
    -------
    tuple of (best_match_chain_id or None, best_identity_score)
    """
    best_match = None
    best_identity = min_identity

    for chain_type, canon_seq in candidates:
        if not canon_seq:
            continue

        for mod_id, mod_seq in model_seqs.items():
            if mod_id in used_model_chains:
                continue

            identity = sequence_identity(canon_seq, mod_seq)

            if identity > best_identity:
                best_identity = identity
                best_match = mod_id

    return best_match, best_identity


def match_chains_using_json(
    model_structure,
    native_structure,
    json_path: Path,
    native_chain_roles: dict[str, str],
) -> dict[str, str]:
    """Match chains using canonical sequences from JSON file.

    This is more accurate than structure-based sequence extraction as it uses
    the pre-processed sequences from the dataset.

    Parameters
    ----------
    model_structure
        Predicted structure
    native_structure
        Native reference structure
    json_path : Path
        Path to JSON file with canonical sequences
    native_chain_roles : dict
        {native_chain_id: role} mapping where role in {"binder", "antigen"}

    Returns
    -------
    dict
        Chain mapping {native_chain_id: model_chain_id}
    """
    # Load canonical sequences
    canonical_seqs = load_sequences_from_json(json_path)

    # Get model chain sequences
    model_chains = _get_chain_ids(model_structure)
    model_seqs = _extract_sequences_from_chains(model_structure, model_chains)

    # Match based on role and canonical sequences
    chain_map = {}
    used_model_chains = set()

    for nat_id, role in native_chain_roles.items():
        # Determine which canonical sequence to use
        candidates = _get_canonical_candidates(role, canonical_seqs)

        # Find best matching model chain
        best_match, best_identity = _find_best_sequence_match(candidates, model_seqs, used_model_chains)

        if best_match:
            chain_map[nat_id] = best_match
            used_model_chains.add(best_match)
            LOGGER.debug(f"Matched {nat_id} ({role}) -> {best_match} (identity={best_identity:.3f})")

    return chain_map


def get_chain_map_with_sequence_matching(
    model_path: Path,
    native_path: Path,
    entry: str,
    native_chain_roles: dict[str, str],
    json_dir: Path | None = None,
) -> dict[str, str]:
    """Get chain mapping using sequence-based matching.

    This is the main entry point for sequence-based chain matching. It will:
    1. Try to use JSON sequences if available (most accurate)
    2. Fall back to structure-based sequence extraction

    Parameters
    ----------
    model_path : Path
        Predicted structure file
    native_path : Path
        Native reference structure file
    entry : str
        Entry name (e.g., "1jp5_HcLcC")
    native_chain_roles : dict
        {native_chain_id: role} mapping
    json_dir : Path, optional
        Directory containing JSON files with sequences

    Returns
    -------
    dict
        Chain mapping {native_chain_id: model_chain_id}
    """
    from confidock.evaluation.structures import load_structure_pair

    # Load structures
    model_struct, native_struct, _ = load_structure_pair(model_path, native_path)

    # Try JSON-based matching first
    if json_dir:
        json_path = json_dir / f"{entry}.json"
        if json_path.exists():
            try:
                chain_map = match_chains_using_json(
                    model_struct,
                    native_struct,
                    json_path,
                    native_chain_roles,
                )
                if chain_map:
                    LOGGER.debug(f"Used JSON sequences for chain matching: {entry}")
                    return chain_map
            except Exception as exc:
                LOGGER.warning(f"JSON-based matching failed for {entry}: {exc}")

    # Fall back to structure-based sequence matching
    LOGGER.debug(f"Using structure-based sequence matching (no JSON): {entry}")
    return match_chains_by_sequence(model_struct, native_struct)
