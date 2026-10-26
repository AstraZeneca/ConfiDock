"""Analysis utilities for DockQ results and bias characterization.

This module provides utilities for analyzing DockQ results, including:
- Grid condition inference
- Chain topology classification
- Bias analysis between binder and antigen predictions
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Required columns for bias analysis
BIAS_REQUIRED_COLS = {
    "DockQ",
    "binder_ca_rmsd_mean",
    "antigen_ca_rmsd_mean",
    "binder_interface_sidechain_rmsd_mean",
    "antigen_interface_sidechain_rmsd_mean",
    "binder_residue_recall_mean",
    "antigen_residue_recall_mean",
    "interface_contact_f1_mean",
}


def infer_condition_cols(rows: list[dict]) -> list[str]:
    """Infer condition columns from CSV rows for Boltz or Protenix grids.

    Parameters
    ----------
    rows : list of dict
        CSV rows as dictionaries

    Returns
    -------
    list of str
        Condition column names (e.g., ["msa_mode", "recycling_steps", "step_scale"])
    """
    if not rows:
        return []
    candidates = [
        ["msa_mode", "recycling_steps", "step_scale"],
        ["msa_mode", "cycle", "step"],
    ]
    for cols in candidates:
        if all(any(cols_i in row for row in rows) for cols_i in cols):
            return cols
    return ["msa_mode"] if any("msa_mode" in row for row in rows) else []


def get_grouping_axes(df: pd.DataFrame) -> list[str]:
    """Return condition axes available for grouped bias summaries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DockQ results

    Returns
    -------
    list of str
        Available grouping axes (e.g., ["recycling_steps", "step_scale"])
    """
    if "recycling_steps" in df.columns:
        return ["recycling_steps", "step_scale"]
    if "cycle" in df.columns:
        return ["cycle", "step"]
    return []


def infer_chain_topology(df: pd.DataFrame) -> pd.Series:
    """Infer a coarse chain-topology class from model chain text.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'model_chains' column

    Returns
    -------
    pd.Series
        Series with chain topology labels:
        - "paired_HcLc": Has both Hc and Lc chains
        - "paired_HL": Has HL chain designation
        - "single_H_or_L": Has only H or L (not both)
        - "contains_x_chain": Contains x or X chain
        - "other_chain_style": None of the above
    """
    text = df.get("model_chains", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)

    out = pd.Series("other_chain_style", index=df.index, dtype="object")
    has_hc = text.str.contains("Hc")
    has_lc = text.str.contains("Lc")
    has_hl = text.str.contains("HL")
    has_x = text.str.contains("x|X", regex=True)

    out[has_hc & has_lc] = "paired_HcLc"
    out[has_hl] = "paired_HL"
    out[(has_hc ^ has_lc)] = "single_H_or_L"
    out[(out == "other_chain_style") & has_x] = "contains_x_chain"
    return out


def check_bias_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check if DataFrame has required columns for bias analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check

    Returns
    -------
    tuple
        (has_all_columns, missing_columns)
    """
    missing = sorted(BIAS_REQUIRED_COLS - set(df.columns))
    return len(missing) == 0, missing


def csv_has_bias_metrics(csv_path: Path) -> bool:
    """Check if a CSV contains usable structural bias metrics.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file

    Returns
    -------
    bool
        True if CSV has at least 3 non-null structural metric columns
    """
    if not csv_path.is_file():
        return False
    df = pd.read_csv(csv_path)
    if not BIAS_REQUIRED_COLS.issubset(df.columns):
        return False
    structural_cols = sorted(BIAS_REQUIRED_COLS - {"DockQ"})
    non_null_struct_cols = 0
    for col in structural_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            non_null_struct_cols += 1
    return non_null_struct_cols >= 3


def compute_bias_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute bias delta columns for analysis.

    Adds three delta columns to the DataFrame:
    - delta_ca_rmsd_antigen_minus_binder
    - delta_sc_rmsd_antigen_minus_binder
    - delta_recall_binder_minus_antigen

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with bias metric columns

    Returns
    -------
    pd.DataFrame
        DataFrame with added delta columns
    """
    # Ensure numeric types
    numeric_cols = [
        "DockQ",
        "binder_ca_rmsd_mean",
        "antigen_ca_rmsd_mean",
        "binder_interface_sidechain_rmsd_mean",
        "antigen_interface_sidechain_rmsd_mean",
        "binder_residue_recall_mean",
        "antigen_residue_recall_mean",
        "interface_contact_f1_mean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute deltas
    df["delta_ca_rmsd_antigen_minus_binder"] = df["antigen_ca_rmsd_mean"] - df["binder_ca_rmsd_mean"]
    df["delta_sc_rmsd_antigen_minus_binder"] = (
        df["antigen_interface_sidechain_rmsd_mean"] - df["binder_interface_sidechain_rmsd_mean"]
    )
    df["delta_recall_binder_minus_antigen"] = df["binder_residue_recall_mean"] - df["antigen_residue_recall_mean"]

    return df


def build_condition_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'condition' column to DataFrame if grid columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential grid condition columns

    Returns
    -------
    pd.DataFrame
        DataFrame with 'condition' column added if applicable
    """
    if "condition" in df.columns:
        return df

    # Boltz-style grid
    if {
        "msa_mode",
        "recycling_steps",
        "step_scale",
    }.issubset(df.columns):
        df["condition"] = (
            "msa="
            + df["msa_mode"].astype(str)
            + "__recycling="
            + df["recycling_steps"].astype(str)
            + "__scale="
            + df["step_scale"].astype(str)
        )
    # Protenix-style grid
    elif {
        "msa_mode",
        "cycle",
        "step",
    }.issubset(df.columns):
        df["condition"] = (
            "msa="
            + df["msa_mode"].astype(str)
            + "__cycle="
            + df["cycle"].astype(str)
            + "__step="
            + df["step"].astype(str)
        )

    return df
