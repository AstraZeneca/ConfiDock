"""Chain identification from manifest metadata."""

from typing import Dict


def parse_chain_string(chain_str: str, pdb_path: str) -> Dict[str, str]:
    """
    Parse chain string from SAbDab manifest to identify chain types.

    SAbDab convention:
    - H or Hc = Heavy chain (VH)
    - L or Lc = Light chain (VL)
    - K or Kc = Kappa light chain (VL)
    - Everything else = Antigen

    Examples
    --------
    - "HLA" -> {H: VH, L: VL, A: antigen}
    - "HcLcA" -> {Hc: VH, Lc: VL, A: antigen}
    - "AxB" -> {A: antigen, x: antigen, B: antigen}

    Parameters
    ----------
    chain_str : str
        Chain string from manifest (e.g., "HcLcA")
    pdb_path : str
        Path to PDB (for debugging/logging)

    Returns
    -------
    Dict[str, str]
        Mapping from chain_id to chain_type (VH, VL, antigen)
    """
    chain_types = {}

    # Parse chain string
    # SAbDab uses H/L/K for antibody chains, everything else is antigen
    i = 0
    while i < len(chain_str):
        # Check for multi-char chain IDs (Hc, Lc, Kc)
        if i + 1 < len(chain_str) and chain_str[i : i + 2] in ["Hc", "Lc", "Kc"]:
            chain_id = chain_str[i : i + 2]
            if chain_id == "Hc":
                chain_types[chain_id] = "VH"
            elif chain_id in ["Lc", "Kc"]:
                chain_types[chain_id] = "VL"
            i += 2
        # Single char chain IDs
        elif chain_str[i] in ["H", "L", "K"]:
            chain_id = chain_str[i]
            if chain_id == "H":
                chain_types[chain_id] = "VH"
            elif chain_id in ["L", "K"]:
                chain_types[chain_id] = "VL"
            i += 1
        # Everything else is antigen
        else:
            chain_id = chain_str[i]
            chain_types[chain_id] = "antigen"
            i += 1

    return chain_types
