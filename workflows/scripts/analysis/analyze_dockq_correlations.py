#!/usr/bin/env python3
"""
Comprehensive DockQ correlation analysis for Boltz2, Protenix, and ESMFold2.

Analyzes:
1. Complex-level DockQ vs confidence scores (pTM, iPTM, plDDT, pDE, etc.)
2. Complex-level DockQ vs chain-specific pTM scores (H/L/Ag)

Uses antibody-antigen conventions: H (heavy), L (light), Ag (antigen).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Import from project modules
from confidock.boltz.inference.inputs import _parse_chain_labels
from confidock.plot import plot_chain_ptm_distributions, plot_correlation_scatter

# Constants
MSG_ADDING_MODEL_CHAINS = "  Adding model_chains column from entry names..."
DOCKQ_CSV_FILENAME = "dockq_scores.csv"


def parse_entry_chains(entry_name: str, n_chains: int = None) -> Dict[str, List[int]]:
    """
    Parse entry name to identify Heavy, Light, and Antigen chain positions.

    Entry naming convention: {pdb_id}_{chains}
    where chains is a string like "HLA" meaning position 0=H, 1=L, 2=Antigen

    Returns dict with 'H', 'L', and 'Ag' keys mapping to chain position indices.
    """
    parts = entry_name.split("_")
    if len(parts) < 2:
        return {"H": [], "L": [], "Ag": []}

    chain_string = parts[1]

    # Parse the chain string using the same logic as the input preparation
    if n_chains is not None:
        chain_labels = _parse_chain_labels(chain_string, n_chains)
    else:
        # Fallback: just use the string as-is
        chain_labels = list(chain_string)

    # Map positions to roles based on H/L presence
    chains = {"H": [], "L": [], "Ag": []}

    for pos, label in enumerate(chain_labels):
        # Check if this is Heavy chain (H or Hc)
        if label[0] in ["H", "h"]:
            chains["H"].append(pos)
        # Check if this is Light chain (L or Lc)
        elif label[0] in ["L", "l"]:
            chains["L"].append(pos)
        # Everything else is antigen
        else:
            chains["Ag"].append(pos)

    return chains


def extract_model_chains_from_entry(entry_name: str) -> Optional[str]:
    """Extract chain string from entry name.

    Example: "1a14_HLN" -> "HLN"
    """
    parts = entry_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def _is_sequential_chain_format(model_chains: str) -> bool:
    """Check if model_chains uses sequential format (A, B, C, ...)."""
    return all(ord(ch.upper()) == ord("A") + i for i, ch in enumerate(sorted(model_chains.upper())))


def _map_sequential_chain_to_role(model_ch: str, entry_chains: Dict[str, list]) -> str:
    """Map a sequential chain ID to its role (Protenix style)."""
    entry_pos = ord(model_ch.upper()) - ord("A")

    if entry_pos in entry_chains["H"]:
        return "H"
    elif entry_pos in entry_chains["L"]:
        return "L"
    else:
        return "Ag"


def _map_original_chain_to_role(model_ch: str) -> str:
    """Map an original chain ID to its role (Boltz2 style)."""
    if model_ch[0] in ["H", "h"]:
        return "H"
    elif model_ch[0] in ["L", "l"]:
        return "L"
    else:
        return "Ag"


def map_chain_to_role(entry_name: str, model_chains: str) -> Dict[int, str]:
    """
    Map chain index in model to role (H, L, or Ag) based on entry name.

    For Boltz2: model_chains contains the original chain IDs (e.g., "HL", "HP")
    For Protenix: model_chains contains sequential IDs (e.g., "AB", "AC")
                  which map positionally to entry chains

    Args:
        entry_name: Entry name like "1ai1_HLP" or "1k4c_ABC"
        model_chains: Chain string from model like "HL" or "AB"

    Returns
    -------
        Dict mapping model chain index (0, 1, 2, ...) to role ('H', 'L', or 'Ag')
    """
    entry_chains = parse_entry_chains(entry_name, n_chains=len(model_chains))
    role_map = {}

    if _is_sequential_chain_format(model_chains):
        # Protenix style: sequential chain IDs (A, B, C...)
        for model_idx, model_ch in enumerate(model_chains):
            role_map[model_idx] = _map_sequential_chain_to_role(model_ch, entry_chains)
    else:
        # Boltz2 style: original chain IDs (H, L, P, etc.)
        for model_idx, model_ch in enumerate(model_chains):
            role_map[model_idx] = _map_original_chain_to_role(model_ch)

    return role_map


def _ensure_model_chains_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add model_chains column if not present in dataframe."""
    if "model_chains" not in df.columns:
        print(MSG_ADDING_MODEL_CHAINS)
        df["model_chains"] = df["entry"].apply(extract_model_chains_from_entry)
    return df


def _convert_ag_list_to_mean(chain_ptm_dict: Dict) -> None:
    """Convert antigen pTM list to mean value."""
    if "ptm_Ag" in chain_ptm_dict and isinstance(chain_ptm_dict["ptm_Ag"], list):
        if len(chain_ptm_dict["ptm_Ag"]) > 0:
            chain_ptm_dict["ptm_Ag"] = np.mean(chain_ptm_dict["ptm_Ag"])
        else:
            chain_ptm_dict["ptm_Ag"] = None


def _store_chain_ptm_by_role(chain_ptm_dict: Dict, role: str, chain_ptm: float) -> None:
    """Store chain pTM score by role (H, L, or Ag)."""
    if role in ["H", "L"]:
        chain_ptm_dict[f"ptm_{role}"] = chain_ptm
    else:  # Ag
        ag_key = "ptm_Ag"
        if ag_key not in chain_ptm_dict:
            chain_ptm_dict[ag_key] = []
        chain_ptm_dict[ag_key].append(chain_ptm)


def _load_json_confidence_file(confidence_file: Path) -> Optional[Dict]:
    """Load confidence JSON file, return None on error."""
    import json

    try:
        with open(confidence_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Failed to load {confidence_file}: {e}")
        return None


def _process_boltz2_confidence_file(confidence_file: Path, row: pd.Series, idx: int) -> Dict[str, any]:
    """Process a Boltz2 confidence file and extract chain pTM data."""
    chain_ptm_dict = {"row_idx": idx}

    if not confidence_file.exists():
        if idx < 10:
            print(f"  Warning: Confidence file not found: {confidence_file}")
        return chain_ptm_dict

    full_data = _load_json_confidence_file(confidence_file)
    if full_data is None:
        return chain_ptm_dict

    chains_ptm = full_data.get("chains_ptm", {})
    model_chains = row["model_chains"]

    if model_chains and isinstance(model_chains, str):
        role_map = map_chain_to_role(row["entry"], model_chains)
        for chain_idx, role in role_map.items():
            chain_ptm = chains_ptm.get(str(chain_idx))
            if chain_ptm is not None:
                _store_chain_ptm_by_role(chain_ptm_dict, role, chain_ptm)

    _convert_ag_list_to_mean(chain_ptm_dict)
    return chain_ptm_dict


def _process_protenix_confidence_file(confidence_file: Path, row: pd.Series, idx: int) -> Dict[str, any]:
    """Process a Protenix confidence file and extract chain pTM data."""
    chain_ptm_dict = {"row_idx": idx}

    if not confidence_file.exists():
        if idx < 10:
            print(f"  Warning: Confidence file not found: {confidence_file}")
        return chain_ptm_dict

    full_data = _load_json_confidence_file(confidence_file)
    if full_data is None:
        return chain_ptm_dict

    chain_ptm_list = full_data.get("chain_ptm", [])
    model_chains = row["model_chains"]

    if model_chains and isinstance(model_chains, str):
        role_map = map_chain_to_role(row["entry"], model_chains)
        for chain_idx, role in role_map.items():
            if chain_idx < len(chain_ptm_list):
                chain_ptm = chain_ptm_list[chain_idx]
                if chain_ptm is not None:
                    _store_chain_ptm_by_role(chain_ptm_dict, role, chain_ptm)

    _convert_ag_list_to_mean(chain_ptm_dict)
    return chain_ptm_dict


def _build_boltz2_confidence_path(base_dir: Path, row: pd.Series) -> Path:
    """Build path to Boltz2 confidence file."""
    condition_dir = f"msa={row['msa_mode']}__recycling={row['recycling_steps']}__scale={row['step_scale']}"
    entry = row["entry"]
    seed = row["seed"]
    sample = row["sample"]
    return (
        base_dir
        / condition_dir
        / "boltz_results__staged_inputs"
        / "predictions"
        / entry
        / f"confidence_{entry}_seed{seed}_sample{sample}.json"
    )


def _build_protenix_confidence_path(base_dir: Path, row: pd.Series) -> Path:
    """Build path to Protenix confidence file."""
    condition_dir = f"msa={row['msa_mode']}__cycle={row['cycle']}__step={row['step']}"
    entry = row["entry"]
    seed = row["seed"]
    sample = row["sample"]
    return (
        base_dir
        / condition_dir
        / entry
        / f"seed_{seed}"
        / "predictions"
        / f"{entry}_seed_{seed}_summary_confidence_sample_{sample}.json"
    )


def parse_boltz2_predictions(base_dir: Path, dockq_csv: Path) -> pd.DataFrame:
    """Parse Boltz2 predictions and extract chain-specific pTM scores.

    Maps chains to H/L/Ag roles based on entry naming convention.
    Note: Complex-level confidence scores (ptm, iptm, etc.) are already in the CSV.
    """
    print(f"Loading Boltz2 DockQ scores from {dockq_csv}...")
    df = pd.read_csv(dockq_csv)
    df = _ensure_model_chains_column(df)

    chain_ptm_data = []
    print(f"Processing {len(df)} Boltz2 predictions for chain-specific pTM...")

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} rows...")

        confidence_file = _build_boltz2_confidence_path(base_dir, row)
        chain_ptm_dict = _process_boltz2_confidence_file(confidence_file, row, idx)
        chain_ptm_data.append(chain_ptm_dict)

    chain_ptm_df = pd.DataFrame(chain_ptm_data)
    df = df.join(chain_ptm_df.set_index("row_idx"), how="left")
    df["model"] = "Boltz2"
    return df


def parse_protenix_predictions(base_dir: Path, dockq_csv: Path) -> pd.DataFrame:
    """Parse Protenix predictions and extract chain-specific pTM scores.

    Maps chains to H/L/Ag roles based on entry naming convention.
    Note: Complex-level confidence scores are already in the CSV.
    """
    print(f"Loading Protenix DockQ scores from {dockq_csv}...")
    df = pd.read_csv(dockq_csv)
    df = _ensure_model_chains_column(df)

    chain_ptm_data = []
    print(f"Processing {len(df)} Protenix predictions for chain-specific pTM...")

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} rows...")

        confidence_file = _build_protenix_confidence_path(base_dir, row)
        chain_ptm_dict = _process_protenix_confidence_file(confidence_file, row, idx)
        chain_ptm_data.append(chain_ptm_dict)

    chain_ptm_df = pd.DataFrame(chain_ptm_data)
    df = df.join(chain_ptm_df.set_index("row_idx"), how="left")
    df["model"] = "Protenix"
    return df


def parse_esmfold2_predictions(dockq_csv: Path) -> pd.DataFrame:
    """Parse ESMFold2 predictions (complex-level metrics only - no chain-specific pTM).

    ESMFold2 only outputs complex-level ptm/iptm, not per-chain scores.
    Complex-level confidence scores are already in the CSV.

    Parameters
    ----------
    dockq_csv : Path
        Path to the DockQ CSV file containing pre-computed metrics

    Returns
    -------
    pd.DataFrame
        DataFrame with ESMFold2 predictions and model identifier
    """
    print(f"Loading ESMFold2 DockQ scores from {dockq_csv}...")
    df = pd.read_csv(dockq_csv)
    df = _ensure_model_chains_column(df)

    print(f"Loaded {len(df)} ESMFold2 predictions")
    print("  Note: ESMFold2 does not output chain-specific pTM scores")

    df["model"] = "ESMFold2"
    return df


def compute_correlations(df: pd.DataFrame, metric_pairs: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations for metric pairs.

    Args:
        df: DataFrame with metrics
        metric_pairs: List of (metric1, metric2, description) tuples

    Returns
    -------
        DataFrame with correlation statistics
    """
    results = []

    for metric1, metric2, description in metric_pairs:
        if metric1 not in df.columns or metric2 not in df.columns:
            continue

        # Filter out rows with missing values
        mask = df[metric1].notna() & df[metric2].notna()
        data = df[mask]

        if len(data) < 10:  # Skip if too few samples
            continue

        x = data[metric1].values
        y = data[metric2].values

        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)

        results.append(
            {
                "metric1": metric1,
                "metric2": metric2,
                "description": description,
                "n_samples": len(data),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )

    return pd.DataFrame(results)


def _parse_arguments():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive DockQ correlation analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all three models
  python analyze_dockq_correlations_comprehensive.py \\
    --boltz2-dir /path/to/boltz2_r1 \\
    --protenix-dir /path/to/protenix_r1 \\
    --esmfold2-dir /path/to/esmfold2_r1 \\
    --output-dir ./comprehensive_analysis

  # Analyze just Boltz2
  python analyze_dockq_correlations_comprehensive.py \\
    --boltz2-dir /path/to/boltz2_r1 \\
    --output-dir ./boltz2_analysis
        """,
    )
    parser.add_argument("--boltz2-dir", type=Path, help="Path to Boltz2 results directory")
    parser.add_argument("--protenix-dir", type=Path, help="Path to Protenix results directory")
    parser.add_argument("--esmfold2-dir", type=Path, help="Path to ESMFold2 results directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    args = parser.parse_args()

    if not any([args.boltz2_dir, args.protenix_dir, args.esmfold2_dir]):
        parser.error("At least one model directory (--boltz2-dir, --protenix-dir, --esmfold2-dir) must be provided")

    return args


def _load_model_predictions(args) -> tuple[list, dict]:
    """Load predictions from all specified models."""
    all_dfs = []
    model_dfs = {}

    if args.boltz2_dir:
        boltz2_csv = args.boltz2_dir / DOCKQ_CSV_FILENAME
        if boltz2_csv.exists():
            print("\n" + "=" * 80)
            print("PARSING BOLTZ2 DATA")
            print("=" * 80)
            boltz2_df = parse_boltz2_predictions(args.boltz2_dir, boltz2_csv)
            model_dfs["boltz2"] = boltz2_df
            all_dfs.append(boltz2_df)
            print(f"Loaded {len(boltz2_df)} predictions with chain-specific pTM")
        else:
            print(f"Warning: Boltz2 CSV not found at {boltz2_csv}")

    if args.protenix_dir:
        protenix_csv = args.protenix_dir / DOCKQ_CSV_FILENAME
        if protenix_csv.exists():
            print("\n" + "=" * 80)
            print("PARSING PROTENIX DATA")
            print("=" * 80)
            protenix_df = parse_protenix_predictions(args.protenix_dir, protenix_csv)
            model_dfs["protenix"] = protenix_df
            all_dfs.append(protenix_df)
            print(f"Loaded {len(protenix_df)} predictions with chain-specific pTM")
        else:
            print(f"Warning: Protenix CSV not found at {protenix_csv}")

    if args.esmfold2_dir:
        esmfold2_csv = args.esmfold2_dir / DOCKQ_CSV_FILENAME
        if esmfold2_csv.exists():
            print("\n" + "=" * 80)
            print("PARSING ESMFOLD2 DATA")
            print("=" * 80)
            esmfold2_df = parse_esmfold2_predictions(esmfold2_csv)
            model_dfs["esmfold2"] = esmfold2_df
            all_dfs.append(esmfold2_df)
        else:
            print(f"Warning: ESMFold2 CSV not found at {esmfold2_csv}")

    return all_dfs, model_dfs


def _save_parsed_data(model_dfs: dict, combined_df: pd.DataFrame, output_dir: Path) -> None:
    """Save parsed data to CSV files."""
    print("\n" + "=" * 80)
    print("SAVING PARSED DATA")
    print("=" * 80)

    for model_name, df in model_dfs.items():
        output_file = output_dir / f"{model_name}_with_chain_ptm.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {model_name} data to {output_file}")

    combined_df.to_csv(output_dir / "combined_all_metrics.csv", index=False)
    print(f"Saved combined data to {output_dir / 'combined_all_metrics.csv'}")


def _get_metric_pairs() -> dict:
    """Define all metric pairs for correlation analysis."""
    common_pairs = [
        ("complex_dockq", "ptm", "Complex DockQ vs pTM"),
        ("complex_dockq", "iptm", "Complex DockQ vs iPTM"),
        ("ptm", "iptm", "pTM vs iPTM"),
    ]

    chain_pairs = [
        ("complex_dockq", "ptm_H", "Complex DockQ vs Heavy chain pTM"),
        ("complex_dockq", "ptm_L", "Complex DockQ vs Light chain pTM"),
        ("complex_dockq", "ptm_Ag", "Complex DockQ vs Antigen pTM"),
        ("iptm", "ptm_H", "iPTM vs Heavy chain pTM"),
        ("iptm", "ptm_L", "iPTM vs Light chain pTM"),
        ("iptm", "ptm_Ag", "iPTM vs Antigen pTM"),
        ("ptm", "ptm_H", "pTM vs Heavy chain pTM"),
        ("ptm", "ptm_L", "pTM vs Light chain pTM"),
        ("ptm", "ptm_Ag", "pTM vs Antigen pTM"),
    ]

    boltz2_conf_pairs = [
        ("complex_dockq", "confidence_score", "Complex DockQ vs Confidence Score"),
        ("complex_dockq", "complex_plddt", "Complex DockQ vs Complex plDDT"),
        ("complex_dockq", "complex_iplddt", "Complex DockQ vs Complex iplDDT"),
        ("complex_dockq", "complex_pde", "Complex DockQ vs Complex pDE"),
        ("complex_dockq", "complex_ipde", "Complex DockQ vs Complex ipDE"),
        ("iptm", "confidence_score", "iPTM vs Confidence Score"),
        ("ptm", "confidence_score", "pTM vs Confidence Score"),
        ("iptm", "complex_plddt", "iPTM vs Complex plDDT"),
        ("ptm", "complex_plddt", "pTM vs Complex plDDT"),
    ]

    other_conf_pairs = [
        ("complex_dockq", "plddt", "Complex DockQ vs plDDT"),
        ("complex_dockq", "gpde", "Complex DockQ vs gPDE"),
        ("iptm", "plddt", "iPTM vs plDDT"),
        ("ptm", "plddt", "pTM vs plDDT"),
        ("iptm", "gpde", "iPTM vs gPDE"),
        ("ptm", "gpde", "pTM vs gPDE"),
    ]

    protenix_conf_pairs = [
        ("complex_dockq", "ranking_score", "Complex DockQ vs Ranking Score"),
        ("iptm", "ranking_score", "iPTM vs Ranking Score"),
        ("ptm", "ranking_score", "pTM vs Ranking Score"),
    ]

    return {
        "common": common_pairs,
        "chain": chain_pairs,
        "boltz2": boltz2_conf_pairs,
        "other": other_conf_pairs,
        "protenix": protenix_conf_pairs,
    }


def _select_metric_pairs_for_model(model_name: str, metric_pairs: dict) -> list:
    """Select appropriate metric pairs based on model."""
    pairs = metric_pairs["common"].copy()

    if model_name == "boltz2":
        pairs.extend(metric_pairs["chain"])
        pairs.extend(metric_pairs["boltz2"])
    elif model_name == "protenix":
        pairs.extend(metric_pairs["chain"])
        pairs.extend(metric_pairs["other"])
        pairs.extend(metric_pairs["protenix"])
    elif model_name == "esmfold2":
        pairs.extend(metric_pairs["other"])

    return pairs


def _compute_and_save_model_correlations(model_dfs: dict, output_dir: Path, metric_pairs: dict) -> None:
    """Compute and save correlations for each model."""
    print("\n" + "=" * 80)
    print("COMPUTING CORRELATIONS")
    print("=" * 80)

    for model_name, df in model_dfs.items():
        print("\n" + "=" * 40)
        print(f"{model_name.upper()} CORRELATIONS")
        print("=" * 40)

        pairs = _select_metric_pairs_for_model(model_name, metric_pairs)
        model_corr = compute_correlations(df, pairs)

        if len(model_corr) > 0:
            print("\n  Complex-level confidence correlations:")
            complex_corr = model_corr[~model_corr["description"].str.contains("chain pTM")]
            if len(complex_corr) > 0:
                print(complex_corr[["description", "n_samples", "pearson_r", "spearman_r"]].to_string(index=False))

            chain_corr = model_corr[model_corr["description"].str.contains("chain pTM")]
            if len(chain_corr) > 0:
                print("\n  Chain-specific pTM correlations:")
                print(chain_corr[["description", "n_samples", "pearson_r", "spearman_r"]].to_string(index=False))

            model_corr.to_csv(output_dir / f"{model_name}_correlations_comprehensive.csv", index=False)
            print(f"\n  Saved to: {output_dir / f'{model_name}_correlations_comprehensive.csv'}")
        else:
            print("  No valid correlations computed")


def _compute_combined_correlations(
    model_dfs: dict, combined_df: pd.DataFrame, output_dir: Path, metric_pairs: dict
) -> None:
    """Compute correlations across all models."""
    if len(model_dfs) <= 1:
        return

    print("\n" + "=" * 40)
    print("COMBINED CORRELATIONS (all models)")
    print("=" * 40)

    combined_pairs = metric_pairs["common"].copy()
    if "boltz2" in model_dfs or "protenix" in model_dfs:
        combined_pairs.extend(metric_pairs["chain"])

    combined_corr = compute_correlations(combined_df, combined_pairs)
    if len(combined_corr) > 0:
        print(combined_corr[["description", "n_samples", "pearson_r", "spearman_r"]].to_string(index=False))
        combined_corr.to_csv(output_dir / "combined_correlations_comprehensive.csv", index=False)
        print(f"\nSaved to: {output_dir / 'combined_correlations_comprehensive.csv'}")


def _create_chain_ptm_plots(model_dfs: dict, output_dir: Path) -> None:
    """Create chain-specific pTM distribution plots."""
    print("\nCreating chain-specific pTM distribution plots...")

    for model_name, df in model_dfs.items():
        if model_name == "esmfold2":
            print(f"  Skipping {model_name} (no chain-specific pTM)")
            continue

        if "ptm_H" in df.columns or "ptm_L" in df.columns or "ptm_Ag" in df.columns:
            plot_chain_ptm_distributions(
                df.to_dict("records"),
                output_dir / f"{model_name}_chain_ptm_distributions_by_msa.png",
                title=f"{model_name.capitalize()}: Chain-specific pTM by MSA Mode",
            )

    chain_capable_dfs = [df for name, df in model_dfs.items() if name != "esmfold2"]
    if chain_capable_dfs:
        combined_chain_df = pd.concat(chain_capable_dfs, ignore_index=True)
        plot_chain_ptm_distributions(
            combined_chain_df.to_dict("records"),
            output_dir / "combined_chain_ptm_distributions_by_msa.png",
            title="Combined: Chain-specific pTM by MSA Mode",
        )


def _collect_all_metric_pairs(model_dfs: dict, metric_pairs: dict) -> set:
    """Collect all unique metric pairs across models."""
    all_pairs = set()

    for model_name in model_dfs.keys():
        pairs = _select_metric_pairs_for_model(model_name, metric_pairs)
        for pair in pairs:
            all_pairs.add(pair)

    return all_pairs


def _create_model_scatter_plot(
    model_name: str, df: pd.DataFrame, metric1: str, metric2: str, description: str, safe_title: str, output_dir: Path
) -> None:
    """Create scatter plots for a single model."""
    if metric1 not in df.columns or metric2 not in df.columns:
        return

    if model_name == "esmfold2" and any(x in description for x in ["Heavy", "Light", "Antigen"]):
        return

    plot_correlation_scatter(
        df.to_dict("records"),
        metric1,
        metric2,
        output_dir / f"{model_name}_scatter_{safe_title}.png",
        f"{model_name.capitalize()}: {description}",
    )

    if "msa_mode" in df.columns:
        plot_correlation_scatter(
            df.to_dict("records"),
            metric1,
            metric2,
            output_dir / f"{model_name}_scatter_{safe_title}_by_msa.png",
            f"{model_name.capitalize()}: {description} (by MSA mode)",
            hue_col="msa_mode",
            show_per_hue_stats=True,
        )


def _create_combined_scatter_plot(
    combined_df: pd.DataFrame, metric1: str, metric2: str, description: str, safe_title: str, output_dir: Path
) -> None:
    """Create combined scatter plots across models."""
    if metric1 not in combined_df.columns or metric2 not in combined_df.columns:
        return

    valid_combined = combined_df[combined_df[metric1].notna() & combined_df[metric2].notna()]
    if len(valid_combined) < 10:
        return

    plot_correlation_scatter(
        valid_combined.to_dict("records"),
        metric1,
        metric2,
        output_dir / f"combined_scatter_{safe_title}_by_model.png",
        f"Combined: {description} (by model)",
        hue_col="model",
        show_per_hue_stats=True,
    )

    if "msa_mode" in combined_df.columns:
        plot_correlation_scatter(
            valid_combined.to_dict("records"),
            metric1,
            metric2,
            output_dir / f"combined_scatter_{safe_title}_by_msa.png",
            f"Combined: {description} (by MSA mode)",
            hue_col="msa_mode",
            show_per_hue_stats=True,
        )


def _create_correlation_scatter_plots(
    model_dfs: dict, combined_df: pd.DataFrame, output_dir: Path, metric_pairs: dict
) -> None:
    """Create all correlation scatter plots."""
    print("\nCreating correlation scatter plots...")

    all_metric_pairs_set = _collect_all_metric_pairs(model_dfs, metric_pairs)

    for metric1, metric2, description in sorted(all_metric_pairs_set):
        safe_title = description.replace(" ", "_").replace("/", "_")

        for model_name, df in model_dfs.items():
            _create_model_scatter_plot(model_name, df, metric1, metric2, description, safe_title, output_dir)

        if len(model_dfs) > 1:
            _create_combined_scatter_plot(combined_df, metric1, metric2, description, safe_title, output_dir)


def _print_summary(model_dfs: dict, combined_df: pd.DataFrame, output_dir: Path) -> None:
    """Print analysis summary."""
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {output_dir}")
    print("\nSummary:")
    for model_name, df in model_dfs.items():
        print(f"  {model_name.capitalize()}: {len(df)} samples")
    print(f"  Combined: {len(combined_df)} samples")

    print("\nChain-specific pTM availability:")
    for model_name, df in model_dfs.items():
        h_count = df["ptm_H"].notna().sum() if "ptm_H" in df.columns else 0
        l_count = df["ptm_L"].notna().sum() if "ptm_L" in df.columns else 0
        ag_count = df["ptm_Ag"].notna().sum() if "ptm_Ag" in df.columns else 0
        if h_count > 0 or l_count > 0 or ag_count > 0:
            print(f"  {model_name.capitalize()} - H: {h_count}, L: {l_count}, Ag: {ag_count}")
        else:
            print(f"  {model_name.capitalize()} - No chain-specific pTM")

    print("\nOutput files:")
    for model_name in model_dfs.keys():
        print(f"  - {model_name}_with_chain_ptm.csv")
        print(f"  - {model_name}_correlations_comprehensive.csv")
    print("  - combined_all_metrics.csv")
    if len(model_dfs) > 1:
        print("  - combined_correlations_comprehensive.csv")
    print("  - *_scatter_*.png (scatter plots)")
    print("  - *_chain_ptm_distributions_by_msa.png (distribution plots)")


def main():
    """Comprehensive DockQ correlation analysis."""
    args = _parse_arguments()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs, model_dfs = _load_model_predictions(args)

    if not all_dfs:
        print("Error: No valid data found for any model!")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    _save_parsed_data(model_dfs, combined_df, output_dir)

    metric_pairs = _get_metric_pairs()
    _compute_and_save_model_correlations(model_dfs, output_dir, metric_pairs)
    _compute_combined_correlations(model_dfs, combined_df, output_dir, metric_pairs)

    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    _create_chain_ptm_plots(model_dfs, output_dir)
    _create_correlation_scatter_plots(model_dfs, combined_df, output_dir, metric_pairs)

    _print_summary(model_dfs, combined_df, output_dir)


if __name__ == "__main__":
    main()
