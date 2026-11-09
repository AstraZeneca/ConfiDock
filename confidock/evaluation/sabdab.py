"""SAbDab-specific utilities for antibody-antigen structure evaluation.

This module provides utilities for working with antibody-antigen complexes from
the Structural Antibody Database (SAbDab), including chain role inference and
data parsing.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def split_chain_tokens(value: str) -> list[str]:
    """Parse a pipe-delimited chain token string from SAbDab TSV.

    Parameters
    ----------
    value : str
        Pipe-delimited string like "H|L" or "A|B|C" or "NA"

    Returns
    -------
    list of str
        Non-empty, non-NA tokens
    """
    tokens = []
    for token in value.split("|"):
        token = token.strip()
        if token and token.upper() != "NA":
            tokens.append(token)
    return tokens


def load_sabdab_role_counts(summary_tsv: Path | None) -> dict[str, tuple[int, int]]:
    """Return ``{pdb_id: (binder_chain_count, antigen_chain_count)}``.

    The SAbDab summary can contain multiple rows per PDB id; for the analysis
    here we only need the coarse binder/antigen split and the chain ordering.
    Binder chains are inferred from the ``Hchain``/``Lchain`` fields and antigen
    chains from ``antigen_chain``.

    Parameters
    ----------
    summary_tsv : Path or None
        Path to SAbDab summary TSV file (typically `*_summary.tsv`)

    Returns
    -------
    dict
        Mapping {pdb_id: (n_binder_chains, n_antigen_chains)}
    """
    if summary_tsv is None or not summary_tsv.is_file():
        LOGGER.warning("SAbDab summary TSV not found: %s", summary_tsv)
        return {}

    role_counts: dict[str, tuple[int, int]] = {}
    with summary_tsv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            pdb_id = (row.get("pdb") or "").strip().lower()
            if not pdb_id or pdb_id in role_counts:
                continue

            # Count binder chains (heavy and/or light)
            binder_count = sum(
                1
                for field in (row.get("Hchain", ""), row.get("Lchain", ""))
                if field.strip() and field.strip().upper() != "NA"
            )

            # Count antigen chains (pipe-delimited in antigen_chain column)
            antigen_count = len(split_chain_tokens(row.get("antigen_chain", "")))

            role_counts[pdb_id] = (binder_count, antigen_count)

    return role_counts


def role_sequence(n_binder: int, n_antigen: int, n_chains: int) -> list[str]:
    """Generate a role assignment sequence for a complex.

    Parameters
    ----------
    n_binder : int
        Number of binder chains (e.g., 2 for H+L)
    n_antigen : int
        Number of antigen chains
    n_chains : int
        Total number of chains in the structure

    Returns
    -------
    list of str
        Sequence of roles ["binder", "binder", "antigen", ...] of length n_chains

    Notes
    -----
    If n_binder + n_antigen < n_chains, extra chains are assigned "antigen".
    If n_binder + n_antigen == 0, default to 1 binder + rest antigen.
    """
    roles = ["binder"] * max(n_binder, 0) + ["antigen"] * max(n_antigen, 0)

    # Default: if no roles specified, assume 1 binder + rest antigen
    if not roles:
        roles = ["binder"] + ["antigen"] * max(n_chains - 1, 0)

    # Pad with antigen if needed
    if len(roles) < n_chains:
        roles.extend(["antigen"] * (n_chains - len(roles)))

    return roles[:n_chains]


def load_sabdab_chain_types(summary_tsv: Path | None) -> dict[str, dict[str, str]]:
    """Load detailed chain type information (heavy/light/antigen) from SAbDab TSV.

    Parameters
    ----------
    summary_tsv : Path or None
        Path to SAbDab summary TSV file

    Returns
    -------
    dict
        Mapping {pdb_id: {chain_id: chain_type}} where chain_type in {"heavy", "light", "antigen"}
    """
    if summary_tsv is None or not summary_tsv.is_file():
        LOGGER.warning("SAbDab summary TSV not found: %s", summary_tsv)
        return {}

    chain_types: dict[str, dict[str, str]] = {}
    with summary_tsv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            pdb_id = (row.get("pdb") or "").strip().lower()
            if not pdb_id:
                continue

            if pdb_id not in chain_types:
                chain_types[pdb_id] = {}

            # Parse heavy chains
            h_chains = split_chain_tokens(row.get("Hchain", ""))
            for chain_id in h_chains:
                chain_types[pdb_id][chain_id] = "heavy"

            # Parse light chains
            l_chains = split_chain_tokens(row.get("Lchain", ""))
            for chain_id in l_chains:
                chain_types[pdb_id][chain_id] = "light"

            # Parse antigen chains
            ag_chains = split_chain_tokens(row.get("antigen_chain", ""))
            for chain_id in ag_chains:
                chain_types[pdb_id][chain_id] = "antigen"

    return chain_types


def infer_chain_roles(
    native_chain_ids: list[str],
    pdb_id: str,
    role_counts_lookup: dict[str, tuple[int, int]],
) -> dict[str, str]:
    """Infer chain roles (binder/antigen) for a complex using SAbDab metadata.

    Parameters
    ----------
    native_chain_ids : list of str
        Chain IDs from the native structure, in order
    pdb_id : str
        PDB identifier (lowercased for lookup)
    role_counts_lookup : dict
        Output of load_sabdab_role_counts()

    Returns
    -------
    dict
        Mapping {chain_id: role} where role in {"binder", "antigen"}

    Notes
    -----
    Chains are assigned roles positionally: first n_binder chains become "binder",
    remaining chains become "antigen".
    """
    binder_count, antigen_count = role_counts_lookup.get(
        pdb_id.lower(),
        (1 if len(native_chain_ids) >= 2 else 0, max(len(native_chain_ids) - 1, 0)),
    )

    role_seq = role_sequence(binder_count, antigen_count, len(native_chain_ids))
    return {chain_id: role for chain_id, role in zip(native_chain_ids, role_seq)}


def interface_role_pair(native_interface: str, native_chain_roles: dict[str, str]) -> str:
    """Determine the role-pair label for an interface.

    Parameters
    ----------
    native_interface : str
        Interface identifier (e.g., "AB" for chains A-B)
    native_chain_roles : dict
        Mapping {chain_id: role}

    Returns
    -------
    str
        Role pair label like "binder-antigen", "binder-binder", etc.

    Notes
    -----
    If either chain in the interface is not found in native_chain_roles, this
    indicates a potential chain mapping issue (e.g., DockQ returned model chain
    IDs instead of native chain IDs, or chains were filtered due to missing
    structure data). A warning is logged and "unknown-unknown" is returned.
    """
    chain_ids = list(native_interface[:2])
    if len(chain_ids) < 2:
        LOGGER.warning("Interface key '%s' has fewer than 2 chain IDs (expected format: 'AB')", native_interface)
        return "unknown-unknown"

    role_a = native_chain_roles.get(chain_ids[0])
    role_b = native_chain_roles.get(chain_ids[1])

    # Validate that both chains are found in the roles dict
    if role_a is None or role_b is None:
        missing = [c for c in chain_ids if c not in native_chain_roles]
        LOGGER.warning(
            "Interface '%s' has unmapped chains: %s (available native chains: %s). "
            "This may indicate a chain mapping mismatch or filtered chains.",
            native_interface,
            missing,
            list(native_chain_roles.keys()),
        )
        return "unknown-unknown"

    return f"{role_a}-{role_b}"
