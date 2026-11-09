r"""Compute DockQ scores for Boltz-2 or Protenix predictions vs. native crystal structures.

For each prediction under ``--predictions_dir`` the script:

1. Locates the matching native PDB in ``--native_dir`` by entry name.
2. Derives the chain mapping (native sequential → model original IDs) from
   the filename stem (e.g. ``8qrg_H_L_A`` → ``{'A': 'H', 'B': 'L', 'C': 'A'}``).
3. Runs DockQ v2 for every interface in the complex (per-interface scoring).
4. Computes complex-level DockQ by merging binder chains (e.g., H+L) as a single
   unit using DockQ's recommended chain merging approach (GitHub issue #33).
5. Reads confidence metrics and joins them to each row.
6. Writes TWO CSV files:
   - ``*_per_interface.csv``: Detailed per-interface DockQ (multiple rows per sample)
   - ``*.csv`` (main output): Complex-level DockQ (one row per sample, for plotting)

Use ``--model boltz`` (default) for Boltz-2 predictions (PDB files) or
``--model protenix`` for Protenix predictions (CIF files).

**Complex vs Per-Interface DockQ:**
- Per-interface: Scores each chain pair separately (e.g., H-A, L-A)
- Complex-level: Merges binder chains (H+L) and scores as one unit vs antigen (A)
- Complex DockQ properly weights each chain's contribution (not a simple average)
- Plots use complex-level DockQ for cleaner visualization

Chain mapping
~~~~~~~~~~~~~
**Sequence-based matching (default, recommended):**
By default, the script uses sequence-based chain matching, which matches chains
between model and native structures by comparing their amino acid sequences. This
is the most robust method as it handles:
- Different chain labeling conventions (H/L/A vs A/B/C)
- Multi-entry PDBs with same sequences but different chain IDs
- Cases where chain IDs don't correspond between model and native

Provide ``--json_dir`` pointing to a directory with JSON files containing canonical
sequences (e.g., ``1jp5_HcLcC.json``) for the most accurate matching. If JSON files
are unavailable, the script falls back to extracting sequences directly from the
structure files.

**Traditional chain mapping (legacy):**
Use ``--no_sequence_matching`` to disable sequence-based matching and fall back to
traditional chain mapping from filenames or YAML files. If native PDBs use
non-sequential chain IDs (e.g. H, L, T for SAbDab structures), provide ``--input_dir``
pointing to the Boltz-2 input YAML directory for positional chain mapping. Without
``--input_dir``, the script assumes native PDBs have sequential chains (A, B, C, ...).

Typical usage::

    # Single condition mode with sequence-based chain matching (recommended)
    python workflows/scripts/analysis/compute_dockq_scores.py \
        --predictions_dir data/Unsal2026/boltz2_predictions/boltz_results_boltz2_inputs_postJune2023/predictions \
        --native_dir      data/Unsal2026/models/PDBs \
        --json_dir        data/Unsal2026/jsons \
        --output_csv      data/Unsal2026/boltz2_predictions/dockq_scores.csv

    # Grid mode with multiple conditions (e.g. msa=*__recycling=*__scale=*)
    python workflows/scripts/analysis/compute_dockq_scores.py \
        --model boltz \
        --grid_dir data/sabdab_boltz2_grid_pre2023_n200_v2 \
        --native_dir /data/sabdab/interfaces \
        --json_dir data/sabdab_jsons \
        --output_csv data/sabdab_boltz2_grid_pre2023_n200_v2/dockq_scores.csv

    # Legacy mode with traditional chain mapping (not recommended)
    python workflows/scripts/analysis/compute_dockq_scores.py \
        --model boltz \
        --grid_dir data/sabdab_boltz2_grid_pre2023_n200_v2 \
        --native_dir /data/sabdab/interfaces \
        --input_dir data/sabdab_boltz2_inputs_pre2023 \
        --no_sequence_matching \
        --output_csv data/sabdab_boltz2_grid_pre2023_n200_v2/dockq_scores.csv

    # Protenix predictions
    python workflows/scripts/analysis/compute_dockq_scores.py \
        --model protenix \
        --predictions_dir data/Unsal2026/protenix_predictions \
        --native_dir      data/Unsal2026/models/PDBs \
        --output_csv      data/Unsal2026/protenix_predictions/dockq_scores.csv

Requires the ``dockq`` package (``pip install dockq``).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from confidock.evaluation import (
    annotate_complex_diversity_metrics as _annotate_complex_diversity_metrics,
)
from confidock.evaluation import (
    collect_rows_boltz as _collect_rows_boltz,
)
from confidock.evaluation import (
    compute_diversity_metrics as _compute_diversity_metrics,
)
from confidock.evaluation import (
    compute_interface_mean as _compute_interface_mean,
)
from confidock.evaluation import (
    load_sabdab_role_counts as _load_sabdab_role_counts,
)
from confidock.evaluation import (
    resolve_boltz_input_dir as _resolve_boltz_input_dir,
)
from confidock.evaluation import (
    scan_grid_conditions as _scan_grid_conditions,
)
from confidock.evaluation.collection import (
    collect_rows_esmfold2 as _collect_rows_esmfold2,
)
from confidock.evaluation.collection import (
    collect_rows_protenix as _collect_rows_protenix,
)
from confidock.plot import (
    plot_condition_violins as _plot_condition_violins,
)
from confidock.plot import (
    plot_diversity as _plot_diversity,
)
from confidock.plot import (
    plot_dockq as _plot_dockq,
)
from confidock.plot import (
    plot_global_summary as _plot_global_summary,
)
from confidock.plot import (
    plot_heatmap as _plot_heatmap,
)
from confidock.plot import (
    plot_violin as _plot_violin,
)
from confidock.plot import (
    plot_violin_gif as _plot_violin_gif,
)

LOGGER = logging.getLogger(__name__)

# CSV columns written before any confidence fields
_FIXED_COLS = [
    "entry",
    "seed",
    "sample",
    "model_file",
    "native_file",
    "native_interface",
    "model_chains",
    "global_dockq",
    "DockQ",
    "iRMSD",
    "LRMSD",
    "fnat",
    "fnonnat",
    "F1",
    "clashes",
]

# Complex-level DockQ columns (added when binder chains can be grouped)
_COMPLEX_COLS = [
    "complex_dockq",
    "complex_irmsd",
    "complex_lrmsd",
    "complex_fnat",
    "complex_fnonnat",
    "complex_F1",
    "complex_clashes",
    "binder_chains",
    "antigen_chains",
    "selected_antigen_chain",
    "used_intended_antigen",
]

_BOLTZ_CONFIDENCE_COLS = [
    "confidence_score",
    "ptm",
    "iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
]

_PROTENIX_CONFIDENCE_COLS = [
    "plddt",
    "gpde",
    "ptm",
    "iptm",
]


# Diversity columns appended after confidence fields
_DIVERSITY_COLS = [
    "dockq_mean",
    "dockq_std",
    "irmsd_mean",
    "irmsd_std",
    "lrmsd_std",
    "irmsd_range",
    "n_conf_clusters",
    "complex_ca_rmsd_mean",
    "complex_ca_rmsd_std",
    "binder_ca_rmsd_mean",
    "binder_ca_rmsd_std",
    "antigen_ca_rmsd_mean",
    "antigen_ca_rmsd_std",
    "interface_contact_f1_mean",
    "interface_contact_f1_std",
    "binder_residue_recall_mean",
    "binder_residue_recall_std",
    "antigen_residue_recall_mean",
    "antigen_residue_recall_std",
    "binder_interface_sidechain_rmsd_mean",
    "binder_interface_sidechain_rmsd_std",
    "antigen_interface_sidechain_rmsd_mean",
    "antigen_interface_sidechain_rmsd_std",
    "antigen_aligned_binder_ca_rmsd_mean",
    "antigen_aligned_binder_ca_rmsd_std",
]

# Optional SAbDab summary TSV used to infer binder/antigen roles by chain order. The path
# is site-specific, so it comes from the environment (or --summary_tsv). Unset means None,
# which load_sabdab_role_counts() handles by inferring roles without it.
_SABDAB_SUMMARY_ENV = os.environ.get("CONFIDOCK_SABDAB_SUMMARY_TSV", "")
_SABDAB_SUMMARY_TSV = Path(_SABDAB_SUMMARY_ENV).expanduser() if _SABDAB_SUMMARY_ENV else None

# Structure-level metrics appended after the DockQ/interface fields.
_STRUCTURE_COLS = [
    "interface_role_pair",
    "complex_ca_rmsd",
    "binder_ca_rmsd",
    "antigen_ca_rmsd",
    "interface_contact_precision",
    "interface_contact_recall",
    "interface_contact_f1",
    "binder_residue_precision",
    "binder_residue_recall",
    "antigen_residue_precision",
    "antigen_residue_recall",
    "binder_interface_sidechain_rmsd",
    "antigen_interface_sidechain_rmsd",
    "antigen_aligned_binder_ca_rmsd",
]

# Epitope matching columns (for multi-binding-mode cases)
_EPITOPE_COLS = [
    "matched_native",
    "epitope_distance",
    "match_quality",
    "num_candidates",
    "matched_to_intended_target",
    "scored_against_intended",
]


def _build_multiplicity_data(json_dir: Path) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    """Build multiplicity data from JSON files to detect multi-epitope scenarios.

    Parameters
    ----------
    json_dir : Path
        Directory containing JSON files with sequences

    Returns
    -------
    tuple
        (multiplicity_data, multi_epitope_pdbs) where:
        - multiplicity_data: {pdb_id: [list of entry dicts with sequences]}
        - multi_epitope_pdbs: {pdb_id: [list of entry names]} for same Ab+Ag cases
    """
    import json
    from collections import defaultdict

    multiplicity_data = defaultdict(list)

    if not json_dir or not json_dir.is_dir():
        return {}, {}

    for json_path in json_dir.glob("*.json"):
        try:
            with json_path.open() as f:
                data = json.load(f)

            entry_name = json_path.stem
            pdb_id = entry_name.split("_")[0].lower()

            # Extract sequences
            entry_data = {
                "full_entry": entry_name,
                "pdb_id": pdb_id,
                "h_seq": data.get("h_seq", ""),
                "l_seq": data.get("l_seq", ""),
                "t_seq": data.get("t_seq", ""),
            }

            multiplicity_data[pdb_id].append(entry_data)

        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(f"Failed to load {json_path.name}: {exc}")
            continue

    # Identify multi-epitope scenarios
    multi_epitope_pdbs = {}
    multi_epitope_count = 0

    for pdb_id, entries in multiplicity_data.items():
        if len(entries) > 1:
            # Check if same Ab + same Ag
            h_seqs = {e["h_seq"] for e in entries if e["h_seq"]}
            l_seqs = {e["l_seq"] for e in entries if e["l_seq"]}
            t_seqs = {e["t_seq"] for e in entries if e["t_seq"]}

            same_ab = len(h_seqs) == 1 and len(l_seqs) == 1
            same_ag = len(t_seqs) == 1

            if same_ab and same_ag:
                entry_names = [e["full_entry"] for e in entries]
                multi_epitope_pdbs[pdb_id] = entry_names
                multi_epitope_count += 1
                LOGGER.info(
                    f"Multi-epitope PDB {pdb_id}: {len(entries)} entries "
                    f"(same Ab+Ag, different binding modes) - {entry_names}"
                )

    if multi_epitope_count > 0:
        LOGGER.info(f"Found {multi_epitope_count} PDBs with multiple binding modes (same Ab+Ag)")

    return dict(multiplicity_data), multi_epitope_pdbs


def _create_complex_rows(interface_rows: list[dict], condition_cols: list[str]) -> list[dict]:
    """Create complex-level rows (one per sample) from per-interface rows.

    For each unique sample, takes complex_dockq (same across all interfaces) and
    aggregates diversity metrics. Drops per-interface columns like native_interface,
    model_chains, DockQ, iRMSD, etc.

    Parameters
    ----------
    interface_rows : list of dict
        Per-interface rows with all DockQ metrics
    condition_cols : list of str
        Grid condition columns (e.g., msa_mode, recycling_steps)

    Returns
    -------
    list of dict
        Complex-level rows, one per sample
    """
    # Group rows by sample (entry + seed + sample + conditions)
    sample_groups = defaultdict(list)
    for row in interface_rows:
        key_parts = [row["entry"], row["seed"], row["sample"]]
        key_parts.extend(row.get(c, "") for c in condition_cols)
        key = tuple(key_parts)
        sample_groups[key].append(row)

    sample_metric_cols = [
        "complex_ca_rmsd",
        "binder_ca_rmsd",
        "antigen_ca_rmsd",
        "interface_contact_f1",
        "binder_residue_recall",
        "antigen_residue_recall",
        "binder_interface_sidechain_rmsd",
        "antigen_interface_sidechain_rmsd",
        "antigen_aligned_binder_ca_rmsd",
    ]

    complex_rows = []
    for key, group in sample_groups.items():
        # Take first row as template (complex_dockq is same across interfaces)
        first = group[0]

        # Build complex-level row
        complex_row = {
            "entry": first["entry"],
            "seed": first["seed"],
            "sample": first["sample"],
            "model_file": first["model_file"],
            "native_file": first["native_file"],
        }

        # Add condition columns
        for col in condition_cols:
            complex_row[col] = first.get(col, "")

        # Add complex-level DockQ metrics (same across all interfaces)
        for col in _COMPLEX_COLS:
            complex_row[col] = first.get(col, "")

        # Add confidence metrics (same across all interfaces)
        conf_cols = _BOLTZ_CONFIDENCE_COLS if "confidence_score" in first else _PROTENIX_CONFIDENCE_COLS
        for col in conf_cols:
            complex_row[col] = first.get(col, "")

        # Store per-sample interface-aggregated structure metrics for later
        # complex-level diversity aggregation across samples.
        for base_col in sample_metric_cols:
            complex_row[f"__sample_{base_col}"] = _compute_interface_mean(group, base_col)

        # Initialize diversity fields; values are populated after all samples
        # are available for each (condition, entry) group.
        for col in _DIVERSITY_COLS:
            complex_row[col] = ""

        complex_rows.append(complex_row)

    _annotate_complex_diversity_metrics(complex_rows, condition_cols, sample_metric_cols)

    return complex_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute DockQ scores for Boltz-2 or Protenix predictions vs. native PDBs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["boltz", "protenix", "esmfold2"],
        default="boltz",
        help="Prediction model type: 'boltz' expects PDB files, 'protenix'/'esmfold2' expect CIF files.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=Path("data/Unsal2026/boltz2_predictions/boltz_results_boltz2_inputs_postJune2023/predictions"),
        help="Directory containing per-entry prediction subdirectories",
    )
    parser.add_argument(
        "--native_dir",
        type=Path,
        default=Path("data/Unsal2026/models/PDBs"),
        help="Directory containing native (reference) PDB files",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help=(
            "Output CSV file path. Defaults to dockq_scores.csv inside "
            "--grid_dir (grid mode) or alongside --predictions_dir (single mode)."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing per-entry Boltz-2 input YAML files "
            "(e.g. 8jln_HcLcA.yaml).  When provided, chain IDs are read "
            "from the YAML rather than parsed from the filename stem."
        ),
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        default=None,
        metavar="ENTRY",
        help=(
            "Restrict to specific entries (e.g. 8qf5_H_A 8qrg_H_L_A). "
            "Defaults to all subdirectories found in --predictions_dir."
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Save a DockQ boxplot (one box per entry) to this path.  "
            "Format is inferred from the file extension (e.g. .png, .pdf, .svg).  "
            "Defaults to <output_csv stem>_dockq_boxplot.png alongside the CSV."
        ),
    )
    parser.add_argument(
        "--grid_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Grid output directory (parent of condition subdirs, e.g. "
            "msa=*__recycling=*__scale=* or msa=*__cycle=*__step=*). "
            "When provided, "
            "--predictions_dir is ignored and all matching conditions are scored "
            "into a single combined CSV with inferred condition columns."
        ),
    )
    parser.add_argument(
        "--msa_modes",
        nargs="+",
        default=None,
        metavar="MODE",
        help="When --grid_dir is used, restrict scoring to these MSA modes (e.g. no_msa).",
    )
    parser.add_argument(
        "--summary_tsv",
        type=Path,
        default=_SABDAB_SUMMARY_TSV,
        help="SAbDab summary TSV used to infer binder/antigen chain roles.",
    )
    parser.add_argument(
        "--json_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing JSON files with canonical sequences "
            "(e.g., 1jp5_HcLcC.json). Used for more accurate sequence-based "
            "chain matching. If not provided, falls back to structure-based "
            "sequence extraction."
        ),
    )
    parser.add_argument(
        "--no_sequence_matching",
        action="store_true",
        help=(
            "Disable sequence-based chain matching and use traditional "
            "chain mapping from YAML/filename. Not recommended unless you "
            "need to reproduce old results."
        ),
    )
    return parser.parse_args()


def _initialize_main(args) -> tuple:
    """Initialize main function variables and validate inputs."""
    if not args.native_dir.is_dir():
        LOGGER.error("Native PDB directory not found: %s", args.native_dir)
        sys.exit(1)

    confidence_cols = _PROTENIX_CONFIDENCE_COLS if args.model in ("protenix", "esmfold2") else _BOLTZ_CONFIDENCE_COLS
    model_label = {"protenix": "Protenix", "esmfold2": "ESMFold2", "boltz": "Boltz-2"}[args.model]
    role_counts_lookup = _load_sabdab_role_counts(args.summary_tsv)

    return confidence_cols, model_label, role_counts_lookup


def _setup_epitope_matching(args):
    """Set up epitope matching data structures."""
    multiplicity_data = None
    multi_epitope_pdbs = {}
    epitope_match_log = []

    if args.json_dir:
        multiplicity_data, multi_epitope_pdbs = _build_multiplicity_data(args.json_dir)

    return multiplicity_data, multi_epitope_pdbs, epitope_match_log


def _collect_rows_for_condition(
    preds_dir: Path,
    args,
    role_counts_lookup: dict,
    multiplicity_data,
    epitope_match_log: list,
) -> list[dict]:
    """Collect rows for a single condition based on model type."""
    if args.model == "boltz":
        condition_dir = preds_dir
        if preds_dir.name == "predictions":
            condition_dir = preds_dir.parent
            if condition_dir.name == "boltz_results__staged_inputs":
                condition_dir = condition_dir.parent

        input_dir = _resolve_boltz_input_dir(
            preds_dir,
            explicit_input_dir=args.input_dir,
            condition_dir=condition_dir,
            grid_dir=args.grid_dir,
        )

        return _collect_rows_boltz(
            preds_dir,
            args.native_dir,
            args.entries,
            input_dir,
            role_counts_lookup,
            json_dir=args.json_dir,
            use_sequence_matching=not args.no_sequence_matching,
            multiplicity_data=multiplicity_data,
            epitope_match_log=epitope_match_log,
        )
    elif args.model == "esmfold2":
        return _collect_rows_esmfold2(
            preds_dir,
            args.native_dir,
            args.entries,
            role_counts_lookup,
            json_dir=args.json_dir,
            use_sequence_matching=not args.no_sequence_matching,
            multiplicity_data=multiplicity_data,
            epitope_match_log=epitope_match_log,
        )
    else:  # protenix
        return _collect_rows_protenix(
            preds_dir,
            args.native_dir,
            args.entries,
            role_counts_lookup,
            json_dir=args.json_dir,
            use_sequence_matching=not args.no_sequence_matching,
            multiplicity_data=multiplicity_data,
            epitope_match_log=epitope_match_log,
        )


def _process_grid_mode(
    args,
    role_counts_lookup: dict,
    multiplicity_data,
    epitope_match_log: list,
) -> tuple[list[dict], list[str], Path]:
    """Process predictions in grid mode."""
    if not args.grid_dir.is_dir():
        LOGGER.error("Grid directory not found: %s", args.grid_dir)
        sys.exit(1)

    conditions, condition_cols, schema_name = _scan_grid_conditions(
        args.grid_dir,
        model=args.model,
        msa_modes=args.msa_modes,
    )

    if not conditions:
        LOGGER.error("No matching condition subdirs found in %s", args.grid_dir)
        sys.exit(1)

    LOGGER.info(
        "Scoring %d conditions in %s (schema=%s, condition_cols=%s)",
        len(conditions),
        args.grid_dir,
        schema_name or "unknown",
        ",".join(condition_cols),
    )

    rows: list[dict] = []
    with tqdm(total=len(conditions), desc="Processing conditions", unit="cond") as pbar_conditions:
        for cond_meta, preds_dir in conditions:
            cond_text = "  ".join(f"{col}={cond_meta[col]}" for col in condition_cols if col in cond_meta)
            pbar_conditions.set_postfix_str(cond_text)

            cond_rows = _collect_rows_for_condition(
                preds_dir, args, role_counts_lookup, multiplicity_data, epitope_match_log
            )

            for row in cond_rows:
                row.update(cond_meta)
            rows.extend(cond_rows)
            pbar_conditions.update(1)

    output_csv = args.output_csv or args.grid_dir / "dockq_scores.csv"

    return rows, condition_cols, output_csv


def _process_single_mode(
    args,
    role_counts_lookup: dict,
    multiplicity_data,
    epitope_match_log: list,
) -> tuple[list[dict], list[str], Path]:
    """Process predictions in single-condition mode."""
    if not args.predictions_dir.is_dir():
        LOGGER.error("Predictions directory not found: %s", args.predictions_dir)
        sys.exit(1)

    if args.model == "esmfold2":
        rows = _collect_rows_esmfold2(
            args.predictions_dir,
            args.native_dir,
            args.entries,
            role_counts_lookup,
            json_dir=args.json_dir,
            use_sequence_matching=not args.no_sequence_matching,
            multiplicity_data=multiplicity_data,
            epitope_match_log=epitope_match_log,
        )
    elif args.model == "protenix":
        rows = _collect_rows_protenix(
            args.predictions_dir,
            args.native_dir,
            args.entries,
            role_counts_lookup,
            json_dir=args.json_dir,
            use_sequence_matching=not args.no_sequence_matching,
            multiplicity_data=multiplicity_data,
            epitope_match_log=epitope_match_log,
        )
    else:  # boltz
        input_dir = _resolve_boltz_input_dir(
            args.predictions_dir,
            explicit_input_dir=args.input_dir,
        )
        if input_dir:
            LOGGER.info("Using Boltz input YAML dir: %s", input_dir)
        rows = _collect_rows_boltz(
            args.predictions_dir,
            args.native_dir,
            args.entries,
            input_dir,
            role_counts_lookup,
            json_dir=args.json_dir,
            use_sequence_matching=not args.no_sequence_matching,
            multiplicity_data=multiplicity_data,
            epitope_match_log=epitope_match_log,
        )

    output_csv = args.output_csv or args.predictions_dir.parent / "dockq_scores.csv"
    condition_cols: list[str] = []

    return rows, condition_cols, output_csv


def _write_interface_csv(rows: list[dict], output_csv: Path, fieldnames: list[str]) -> None:
    """Write per-interface CSV file."""
    interface_csv = output_csv.with_name(output_csv.stem + "_per_interface.csv")
    with interface_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info(
        "Wrote %d per-interface rows (%d unique entries × conditions × interfaces) → %s",
        len(rows),
        len({r["entry"] for r in rows}),
        interface_csv,
    )


def _write_complex_csv(
    rows: list[dict],
    condition_cols: list[str],
    output_csv: Path,
    confidence_cols: list[str],
    grid_mode: bool,
) -> list[dict]:
    """Write complex-level CSV file and return complex rows."""
    complex_rows = _create_complex_rows(rows, condition_cols if grid_mode else [])
    complex_fieldnames = (
        (condition_cols if grid_mode else [])
        + ["entry", "seed", "sample", "model_file", "native_file"]
        + _COMPLEX_COLS
        + confidence_cols
        + _DIVERSITY_COLS
    )

    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=complex_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(complex_rows)

    LOGGER.info(
        "Wrote %d complex-level rows (one per sample) → %s",
        len(complex_rows),
        output_csv,
    )

    return complex_rows


def _write_epitope_log(
    epitope_match_log: list,
    multi_epitope_pdbs: dict,
    complex_csv: Path,
) -> None:
    """Write epitope matching log file."""
    if not epitope_match_log:
        return

    epitope_log_path = complex_csv.with_name(complex_csv.stem + "_epitope_matching.log")
    with epitope_log_path.open("w", encoding="utf-8") as fh:
        fh.write("=" * 80 + "\n")
        fh.write("EPITOPE MATCHING LOG\n")
        fh.write("=" * 80 + "\n\n")
        fh.write(f"Multi-epitope PDBs detected: {len(multi_epitope_pdbs)}\n")
        fh.write(f"Total epitope matches performed: {len(epitope_match_log)}\n\n")

        # Summary by PDB
        fh.write("=" * 80 + "\n")
        fh.write("MULTI-EPITOPE PDB SUMMARY\n")
        fh.write("=" * 80 + "\n\n")
        for pdb_id, entries in sorted(multi_epitope_pdbs.items()):
            fh.write(f"PDB {pdb_id.upper()}: {len(entries)} binding modes\n")
            fh.write(f"  Entries: {', '.join(entries)}\n")
            pdb_matches = [m for m in epitope_match_log if m["pdb_id"] == pdb_id]
            fh.write(f"  Predictions matched: {len(pdb_matches)}\n\n")

        # Detailed match log
        fh.write("=" * 80 + "\n")
        fh.write("DETAILED EPITOPE MATCHES\n")
        fh.write("=" * 80 + "\n\n")
        for match in epitope_match_log:
            fh.write(f"PDB: {match['pdb_id'].upper()}\n")
            fh.write(f"  Model: {match['model_file']}\n")
            fh.write(f"  Original Native: {match['original_native']}\n")
            fh.write(f"  Matched Native: {match['matched_native']}\n")
            fh.write(f"  Distance: {match['epitope_distance']} Å\n")
            fh.write(f"  Quality: {match['match_quality']}\n")
            fh.write(f"  Candidates ({match['num_candidates']}): {match['all_candidates']}\n")
            fh.write("\n")

        # Statistics
        fh.write("=" * 80 + "\n")
        fh.write("MATCH QUALITY STATISTICS\n")
        fh.write("=" * 80 + "\n\n")
        quality_counts = {}
        for match in epitope_match_log:
            quality = match["match_quality"]
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        for quality in ["good_match", "ambiguous", "poor_match"]:
            count = quality_counts.get(quality, 0)
            pct = 100 * count / len(epitope_match_log) if epitope_match_log else 0
            fh.write(f"  {quality}: {count} ({pct:.1f}%)\n")

    LOGGER.info(f"Wrote epitope matching log ({len(epitope_match_log)} matches) → {epitope_log_path}")


def _generate_plots(
    args,
    rows: list[dict],
    complex_rows: list[dict],
    complex_csv: Path,
    model_label: str,
    grid_mode: bool,
) -> None:
    """Generate all visualization plots."""
    plot_path = args.plot or complex_csv.with_name(complex_csv.stem + "_dockq_boxplot.png")
    _plot_dockq(complex_rows, plot_path, title=model_label)

    stem = complex_csv.stem
    _plot_violin(complex_rows, complex_csv.with_name(stem + "_dockq_violin.png"), title=model_label)
    _plot_heatmap(complex_rows, complex_csv.with_name(stem + "_dockq_heatmap.png"), title=model_label)
    _plot_global_summary(complex_rows, complex_csv.with_name(stem + "_dockq_global_summary.png"), title=model_label)

    if grid_mode:
        _plot_condition_violins(complex_rows, complex_csv.with_name(stem + "_condition_violins.png"), title=model_label)
        try:
            _plot_violin_gif(
                complex_rows, complex_csv.with_name(stem + "_per_target_conditions.gif"), title=model_label
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("GIF generation failed: %s", exc, exc_info=True)

    _plot_diversity(rows, complex_csv.with_name(stem + "_diversity.png"), title=model_label)


def main() -> None:
    """Score DockQ and structural metrics for a prediction set or grid."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    confidence_cols, model_label, role_counts_lookup = _initialize_main(args)
    multiplicity_data, multi_epitope_pdbs, epitope_match_log = _setup_epitope_matching(args)

    grid_mode = bool(args.grid_dir)

    if grid_mode:
        rows, condition_cols, output_csv = _process_grid_mode(
            args, role_counts_lookup, multiplicity_data, epitope_match_log
        )
        fieldnames = (
            condition_cols
            + _FIXED_COLS
            + _COMPLEX_COLS
            + _STRUCTURE_COLS
            + _EPITOPE_COLS
            + confidence_cols
            + _DIVERSITY_COLS
        )
    else:
        rows, condition_cols, output_csv = _process_single_mode(
            args, role_counts_lookup, multiplicity_data, epitope_match_log
        )
        fieldnames = _FIXED_COLS + _COMPLEX_COLS + _STRUCTURE_COLS + _EPITOPE_COLS + confidence_cols + _DIVERSITY_COLS

    if not rows:
        LOGGER.error("No results collected – check paths and file names")
        sys.exit(1)

    _compute_diversity_metrics(rows, condition_cols if grid_mode else [])
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    _write_interface_csv(rows, output_csv, fieldnames)
    complex_rows = _write_complex_csv(rows, condition_cols, output_csv, confidence_cols, grid_mode)
    _write_epitope_log(epitope_match_log, multi_epitope_pdbs, output_csv)
    _generate_plots(args, rows, complex_rows, output_csv, model_label, grid_mode)


if __name__ == "__main__":
    main()
