"""Collection functions for DockQ scoring across different model types.

This module provides functions to collect DockQ rows from prediction directories
for Boltz-2, Protenix, and ESMFold2 models. These functions handle:
- Prediction file discovery and loading
- Chain mapping (sequence-based or traditional)
- Epitope matching for multi-binding-mode cases
- DockQ scoring (per-interface and complex-level)
- Structure metrics computation
- Confidence metric loading
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from tqdm import tqdm

import confidock.boltz.evaluate as boltz_eval
import confidock.esmfold2.evaluate as esmfold2_eval
import confidock.protenix.evaluate as protenix_eval
from confidock.evaluation import (
    chain_ids_in_order,
    compute_structure_metrics,
    get_chain_map_with_sequence_matching,
    infer_chain_roles,
    interface_role_pair,
    load_structure_pair,
    make_dockq_row,
    match_epitope_by_interface_position,
    run_complex_level_dockq,
    run_dockq,
    validate_interface_coverage,
)

LOGGER = logging.getLogger(__name__)

# Column definitions for different model types
BOLTZ_CONFIDENCE_COLS = [
    "confidence_score",
    "ptm",
    "iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
]

PROTENIX_CONFIDENCE_COLS = [
    "plddt",
    "gpde",
    "ptm",
    "iptm",
]

COMPLEX_COLS = [
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

# Progress bar and logging messages
TQDM_DESC_PROCESSING_ENTRIES = "Processing entries"
LOG_MSG_NATIVE_NOT_FOUND = "Native PDB not found for %s – skipping"


def _infer_intended_antigen_chain(entry: str) -> str | None:
    """Infer intended antigen chain ID from an entry name.

    Supported examples:
    - ``1a2y_BAC`` -> ``C``
    - ``1jp5_HcLcC`` -> ``C``
    - ``8qrg_H_L_A`` -> ``A``
    """
    if "_" not in entry:
        return None

    parts = entry.split("_")

    # Underscore-separated explicit chain tokens (e.g., 8qrg_H_L_A)
    if len(parts) >= 3:
        token = parts[-1].strip()
        return token[:1] if token else None

    # Compact single suffix forms (e.g., BAC or HcLcC)
    suffix = parts[-1].strip()
    if not suffix:
        return None

    # HcLcX style
    match = re.match(r"^[A-Za-z]c[A-Za-z]c([A-Za-z])$", suffix)
    if match:
        return match.group(1)

    # BAC style (H, L, Ag)
    if len(suffix) >= 3:
        return suffix[2]

    return None


def _resolve_intended_native_antigen_chain(
    intended_chain: str | None,
    chain_map: dict[str, str],
    native_chain_roles: dict[str, str],
) -> str | None:
    """Resolve intended antigen chain to native chain IDs used by DockQ.

    Entry names often encode model/original chain IDs (e.g., ``..._HLN``), while
    complex-level scoring expects native chain IDs. This helper maps
    model/original IDs back to native IDs via ``chain_map``.
    """
    if not intended_chain:
        return None

    # If already a native antigen chain ID, keep it.
    if native_chain_roles.get(intended_chain) == "antigen":
        return intended_chain

    # Reverse map model/original -> native and keep only antigen chains.
    native_candidates = [
        nat_chain
        for nat_chain, model_chain in chain_map.items()
        if model_chain == intended_chain and native_chain_roles.get(nat_chain) == "antigen"
    ]
    if len(native_candidates) == 1:
        return native_candidates[0]
    if len(native_candidates) > 1:
        return sorted(native_candidates)[0]

    return None


def collect_rows_boltz(
    predictions_dir: Path,
    native_dir: Path,
    entries: list[str] | None,
    input_dir: Path | None = None,
    role_counts_lookup: dict[str, tuple[int, int]] | None = None,
    json_dir: Path | None = None,
    use_sequence_matching: bool = True,
    multiplicity_data: dict[str, list[dict]] | None = None,
    epitope_match_log: list[dict] | None = None,
) -> list[dict]:
    """Collect DockQ rows for Boltz-2 predictions.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing per-entry prediction subdirectories
    native_dir : Path
        Directory containing native PDB files
    entries : list of str, optional
        Specific entries to process (None = all)
    input_dir : Path, optional
        Directory containing Boltz input YAML files (for traditional chain mapping)
    role_counts_lookup : dict, optional
        SAbDab role counts for chain role inference
    json_dir : Path, optional
        Directory containing JSON files with canonical sequences
    use_sequence_matching : bool
        Use sequence-based chain matching (default: True)
    multiplicity_data : dict, optional
        Multi-epitope PDB data for epitope matching
    epitope_match_log : list, optional
        List to append epitope match results to

    Returns
    -------
    list of dict
        DockQ rows (one per interface per sample)
    """
    entry_names = entries or sorted(d.name for d in predictions_dir.iterdir() if d.is_dir())
    rows: list[dict] = []

    with tqdm(total=len(entry_names), desc=TQDM_DESC_PROCESSING_ENTRIES, unit="entry") as pbar_entries:
        for entry in entry_names:
            pbar_entries.set_postfix_str(f"{entry}")
            native_path = native_dir / f"{entry}.pdb"
            if not native_path.is_file():
                LOGGER.warning(LOG_MSG_NATIVE_NOT_FOUND, entry)
                pbar_entries.update(1)
                continue

            samples = boltz_eval.prediction_samples(predictions_dir, entry)
            if not samples:
                LOGGER.warning("No prediction PDBs found for %s", entry)
                pbar_entries.update(1)
                continue

            for seed, sample_idx, model_path in samples:
                rows_for_sample = _process_sample_boltz(
                    entry,
                    seed,
                    sample_idx,
                    model_path,
                    native_path,
                    native_dir,
                    input_dir,
                    role_counts_lookup,
                    json_dir,
                    use_sequence_matching,
                    multiplicity_data,
                    epitope_match_log,
                )
                rows.extend(rows_for_sample)
            pbar_entries.update(1)
    return rows


def _get_chain_mapping_boltz(
    model_path: Path,
    native_path: Path,
    entry: str,
    native_chain_roles: dict[str, str],
    json_dir: Path | None,
    use_sequence_matching: bool,
    input_dir: Path | None,
) -> dict[str, str]:
    """Determine chain mapping for Boltz predictions."""
    if use_sequence_matching:
        chain_map = get_chain_map_with_sequence_matching(
            model_path,
            native_path,
            entry,
            native_chain_roles,
            json_dir=json_dir,
        )
        LOGGER.debug(f"Using sequence-based chain mapping for {entry}: {chain_map}")
        return chain_map

    # Fallback to traditional chain mapping from YAML/stem
    yaml_path = input_dir / f"{entry}.yaml" if input_dir else None
    if yaml_path and yaml_path.is_file():
        chain_map = boltz_eval.chain_map_from_yaml(yaml_path, native_path=native_path)
    else:
        chain_map = boltz_eval.chain_map_from_stem(entry)
    LOGGER.info(f"!  Using traditional chain mapping for {entry}: {chain_map}")
    return chain_map


def _add_epitope_info_to_row(row: dict, epitope_match_info: dict | None) -> None:
    """Add epitope matching info to a DockQ row."""
    if epitope_match_info:
        row.update(epitope_match_info)
    else:
        row.update(
            {
                "matched_native": "",
                "epitope_distance": "",
                "match_quality": "",
                "num_candidates": "",
                "matched_to_intended_target": "",
                "scored_against_intended": "",
            }
        )


def _apply_epitope_matching_result(
    epitope_match_info: dict | None,
) -> tuple[Path | None, dict[str, str] | None, dict[str, str] | None]:
    """Extract matched native path and chain info from epitope matching result."""
    if not epitope_match_info or "matched_native_path" not in epitope_match_info:
        return None, None, None

    native_path = epitope_match_info["matched_native_path"]
    chain_map = epitope_match_info["matched_chain_map"]
    native_chain_roles = epitope_match_info["matched_roles"]

    # Remove the path from epitope info (not for CSV)
    del epitope_match_info["matched_native_path"]
    del epitope_match_info["matched_chain_map"]
    del epitope_match_info["matched_roles"]

    return native_path, chain_map, native_chain_roles


def _process_sample_boltz(
    entry: str,
    seed: int,
    sample_idx: int,
    model_path: Path,
    native_path: Path,
    native_dir: Path,
    input_dir: Path | None,
    role_counts_lookup: dict[str, tuple[int, int]] | None,
    json_dir: Path | None,
    use_sequence_matching: bool,
    multiplicity_data: dict[str, list[dict]] | None,
    epitope_match_log: list[dict] | None,
) -> list[dict]:
    """Process a single Boltz prediction sample."""
    model_struct, native_struct, chain_remap = load_structure_pair(model_path, native_path)

    # Get chain roles for complex-level scoring
    pdb_id = entry.split("_")[0].lower()
    native_chain_roles = infer_chain_roles(
        chain_ids_in_order(native_struct),
        pdb_id,
        role_counts_lookup or {},
    )

    # Determine chain mapping strategy
    chain_map = _get_chain_mapping_boltz(
        model_path,
        native_path,
        entry,
        native_chain_roles,
        json_dir,
        use_sequence_matching,
        input_dir,
    )

    # Store original intended target info
    intended_native_path = native_path
    intended_chain_map = chain_map
    intended_chain_roles = native_chain_roles

    # Check for multi-epitope scenarios and perform epitope matching
    epitope_match_info = _perform_epitope_matching(
        entry,
        pdb_id,
        model_path,
        native_dir,
        native_chain_roles,
        json_dir,
        multiplicity_data,
        epitope_match_log,
    )

    # If epitope matching succeeded, use the matched native
    matched_native, matched_chain_map, matched_roles = _apply_epitope_matching_result(
        epitope_match_info,
    )

    # Select which native to use for scoring based on match quality
    native_path, chain_map, native_chain_roles, epitope_match_info = _select_native_for_scoring(
        entry,
        matched_native,
        matched_chain_map,
        matched_roles,
        epitope_match_info,
        intended_native_path,
        intended_chain_map,
        intended_chain_roles,
    )

    # Reload structures with selected native
    model_struct, native_struct, chain_remap = load_structure_pair(model_path, native_path)

    # Apply chain remapping to the chain_map for structure metrics
    remapped_chain_map = {nat: chain_remap.get(mod, mod) for nat, mod in chain_map.items()}

    # Run standard per-interface DockQ
    try:
        interfaces, global_dockq = run_dockq(model_path, native_path, chain_map)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("DockQ failed for %s: %s", model_path.name, exc)
        return []

    # Run complex-level DockQ
    try:
        intended_antigen_chain_raw = _infer_intended_antigen_chain(entry)
        intended_antigen_chain = _resolve_intended_native_antigen_chain(
            intended_antigen_chain_raw,
            chain_map,
            native_chain_roles,
        )
        antigen_chain_count = sum(1 for role in native_chain_roles.values() if role == "antigen")
        if antigen_chain_count > 1 and not intended_antigen_chain:
            LOGGER.warning(
                "Entry %s has %d antigen chains but intended antigen could not be inferred from name; "
                "complex scoring may fall back to best available antigen chain",
                entry,
                antigen_chain_count,
            )
        elif antigen_chain_count > 1 and intended_antigen_chain:
            LOGGER.info(
                "Entry %s inferred intended antigen chain: %s (native=%s)",
                entry,
                intended_antigen_chain_raw,
                intended_antigen_chain,
            )
        complex_metrics = run_complex_level_dockq(
            model_path,
            native_path,
            chain_map,
            native_chain_roles,
            intended_antigen_chain=intended_antigen_chain,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Complex-level DockQ not computed for %s: %s", model_path.name, exc)
        complex_metrics = None

    # Compute structure metrics
    metrics_chain_map = remapped_chain_map
    structure_metrics, _ = compute_structure_metrics(
        model_struct,
        native_struct,
        metrics_chain_map,
        native_chain_roles,
    )

    # Load confidence
    confidence = boltz_eval.load_confidence(boltz_eval.find_confidence_file(model_path.parent, model_path.stem))

    # Validate interface coverage
    validate_interface_coverage(interfaces, native_chain_roles, entry)

    # Create rows for each interface
    rows = []
    for iface_key, metrics in interfaces.items():
        row_structure_metrics = dict(structure_metrics)
        row_structure_metrics["interface_role_pair"] = interface_role_pair(
            iface_key,
            native_chain_roles,
        )
        row = make_dockq_row(
            entry,
            seed,
            sample_idx,
            model_path,
            native_path,
            iface_key,
            metrics,
            global_dockq,
            confidence,
            BOLTZ_CONFIDENCE_COLS,
            row_structure_metrics,
            complex_metrics,
            COMPLEX_COLS,
        )

        _add_epitope_info_to_row(row, epitope_match_info)
        rows.append(row)

    return rows


def _is_same_ab_same_ag_case(entries_for_pdb: list[dict]) -> bool:
    """Check if this is a 'Same Ab + Same Ag' multi-binding-mode case."""
    h_seqs = {e["h_seq"] for e in entries_for_pdb if e["h_seq"]}
    l_seqs = {e["l_seq"] for e in entries_for_pdb if e["l_seq"]}
    t_seqs = {e["t_seq"] for e in entries_for_pdb if e["t_seq"]}

    same_ab = len(h_seqs) == 1 and len(l_seqs) == 1
    same_ag = len(t_seqs) == 1

    return same_ab and same_ag


def _build_native_candidates(
    entries_for_pdb: list[dict],
    model_path: Path,
    native_dir: Path,
    native_chain_roles: dict[str, str],
    json_dir: Path | None,
) -> list[tuple[Path, dict[str, str], dict[str, str]]]:
    """Build list of native candidate structures for epitope matching."""
    native_candidates = []
    for entry_data in entries_for_pdb:
        candidate_entry = entry_data["full_entry"]
        candidate_path = native_dir / f"{candidate_entry}.pdb"
        if not candidate_path.is_file():
            continue

        # Get chain map for this candidate
        candidate_chain_map = get_chain_map_with_sequence_matching(
            model_path,
            candidate_path,
            candidate_entry,
            native_chain_roles,
            json_dir=json_dir,
        )

        native_candidates.append((candidate_path, candidate_chain_map, native_chain_roles))

    return native_candidates


def _log_epitope_match_details(
    epitope_match_log: list[dict],
    pdb_id: str,
    entry: str,
    model_path: Path,
    matched_native: Path,
    distance: float,
    quality: str,
    native_candidates: list,
) -> None:
    """Log epitope match details to the match log."""
    candidate_names = [c[0].stem for c in native_candidates]
    matched_to_intended = matched_native.stem == entry

    epitope_match_log.append(
        {
            "pdb_id": pdb_id,
            "entry": entry,
            "model_file": model_path.name,
            "seed": model_path.stem.split("_seed")[1].split("_")[0] if "_seed" in model_path.stem else "unknown",
            "sample": (
                model_path.stem.split("_sample")[1].split(".")[0] if "_sample" in model_path.stem else "unknown"
            ),
            "matched_native": matched_native.stem,
            "original_native": entry,
            "epitope_distance": f"{distance:.2f}",
            "match_quality": quality,
            "matched_to_intended_target": matched_to_intended,
            "num_candidates": len(native_candidates),
            "all_candidates": ", ".join(candidate_names),
        }
    )


def _perform_epitope_matching(
    entry: str,
    pdb_id: str,
    model_path: Path,
    native_dir: Path,
    native_chain_roles: dict[str, str],
    json_dir: Path | None,
    multiplicity_data: dict[str, list[dict]] | None,
    epitope_match_log: list[dict] | None,
) -> dict | None:
    """Perform epitope matching if multi-binding-mode case detected.

    Returns
    -------
    dict or None
        Epitope match info including matched_native_path, matched_chain_map, matched_roles
        Returns None if no epitope matching needed or failed
    """
    if not multiplicity_data or pdb_id not in multiplicity_data:
        return None

    entries_for_pdb = multiplicity_data[pdb_id]
    if len(entries_for_pdb) <= 1:
        return None

    # Check if this is "Same Ab + Same Ag" case
    if not _is_same_ab_same_ag_case(entries_for_pdb):
        return None

    # Build native candidates
    native_candidates = _build_native_candidates(
        entries_for_pdb,
        model_path,
        native_dir,
        native_chain_roles,
        json_dir,
    )

    if len(native_candidates) <= 1:
        return None

    LOGGER.info(f"Running epitope matching for {entry}: {len(native_candidates)} binding modes detected")

    # Perform epitope matching
    match_result = match_epitope_by_interface_position(
        model_path,
        native_candidates,
        method="interface_rmsd",
        contact_cutoff=8.0,
    )

    if not match_result:
        return None

    matched_native, matched_chain_map, matched_roles, distance, quality = match_result

    # Check if matched native is the intended target (from entry name)
    matched_to_intended = matched_native.stem == entry

    epitope_match_info = {
        "matched_native": matched_native.stem,
        "epitope_distance": distance,
        "match_quality": quality,
        "num_candidates": len(native_candidates),
        "matched_to_intended_target": matched_to_intended,
        "matched_native_path": matched_native,
        "matched_chain_map": matched_chain_map,
        "matched_roles": matched_roles,
    }

    LOGGER.info(
        f"Epitope match for {model_path.stem}: "
        f"best={matched_native.stem}, "
        f"distance={distance:.2f}Å, "
        f"quality={quality}, "
        f"intended_target={entry}, "
        f"matched_to_intended={matched_to_intended}"
    )

    # Log epitope match details
    if epitope_match_log is not None:
        _log_epitope_match_details(
            epitope_match_log,
            pdb_id,
            entry,
            model_path,
            matched_native,
            distance,
            quality,
            native_candidates,
        )

    return epitope_match_info


def collect_rows_protenix(
    predictions_dir: Path,
    native_dir: Path,
    entries: list[str] | None,
    role_counts_lookup: dict[str, tuple[int, int]] | None = None,
    json_dir: Path | None = None,
    use_sequence_matching: bool = True,
    multiplicity_data: dict[str, list[dict]] | None = None,
    epitope_match_log: list[dict] | None = None,
) -> list[dict]:
    """Collect DockQ rows for Protenix predictions (CIF files).

    Parameters
    ----------
    predictions_dir : Path
        Directory containing per-entry prediction subdirectories
    native_dir : Path
        Directory containing native PDB files
    entries : list of str, optional
        Specific entries to process (None = all)
    role_counts_lookup : dict, optional
        SAbDab role counts for chain role inference
    json_dir : Path, optional
        Directory containing JSON files with canonical sequences
    use_sequence_matching : bool
        Use sequence-based chain matching (default: True)
    multiplicity_data : dict, optional
        Multi-epitope PDB data for epitope matching
    epitope_match_log : list, optional
        List to append epitope match results to

    Returns
    -------
    list of dict
        DockQ rows (one per interface per sample)
    """
    entry_names = entries or sorted(
        d.name for d in predictions_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    rows: list[dict] = []

    with tqdm(total=len(entry_names), desc=TQDM_DESC_PROCESSING_ENTRIES, unit="entry") as pbar_entries:
        for entry in entry_names:
            pbar_entries.set_postfix_str(f"{entry}")
            native_path = native_dir / f"{entry}.pdb"
            if not native_path.is_file():
                LOGGER.warning(LOG_MSG_NATIVE_NOT_FOUND, entry)
                pbar_entries.update(1)
                continue

            samples = protenix_eval.prediction_samples(predictions_dir, entry)
            if not samples:
                LOGGER.warning("No prediction CIFs found for %s", entry)
                pbar_entries.update(1)
                continue

            for seed, sample_idx, cif_path in samples:
                rows_for_sample = _process_sample_cif(
                    entry,
                    seed,
                    sample_idx,
                    cif_path,
                    native_path,
                    native_dir,
                    role_counts_lookup,
                    json_dir,
                    use_sequence_matching,
                    multiplicity_data,
                    epitope_match_log,
                    model_eval_module=protenix_eval,
                    confidence_cols=PROTENIX_CONFIDENCE_COLS,
                )
                rows.extend(rows_for_sample)
            pbar_entries.update(1)
    return rows


def collect_rows_esmfold2(
    predictions_dir: Path,
    native_dir: Path,
    entries: list[str] | None,
    role_counts_lookup: dict[str, tuple[int, int]] | None = None,
    json_dir: Path | None = None,
    use_sequence_matching: bool = True,
    multiplicity_data: dict[str, list[dict]] | None = None,
    epitope_match_log: list[dict] | None = None,
) -> list[dict]:
    """Collect DockQ rows for ESMFold2 predictions (CIF files, flat directory).

    Parameters
    ----------
    predictions_dir : Path
        Directory containing per-entry prediction subdirectories
    native_dir : Path
        Directory containing native PDB files
    entries : list of str, optional
        Specific entries to process (None = all)
    role_counts_lookup : dict, optional
        SAbDab role counts for chain role inference
    json_dir : Path, optional
        Directory containing JSON files with canonical sequences
    use_sequence_matching : bool
        Use sequence-based chain matching (default: True)
    multiplicity_data : dict, optional
        Multi-epitope PDB data for epitope matching
    epitope_match_log : list, optional
        List to append epitope match results to

    Returns
    -------
    list of dict
        DockQ rows (one per interface per sample)
    """
    entry_names = entries or sorted(
        d.name for d in predictions_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    rows: list[dict] = []

    with tqdm(total=len(entry_names), desc=TQDM_DESC_PROCESSING_ENTRIES, unit="entry") as pbar_entries:
        for entry in entry_names:
            pbar_entries.set_postfix_str(f"{entry}")
            native_path = native_dir / f"{entry}.pdb"
            if not native_path.is_file():
                LOGGER.warning(LOG_MSG_NATIVE_NOT_FOUND, entry)
                pbar_entries.update(1)
                continue

            samples = esmfold2_eval.prediction_samples(predictions_dir, entry)
            if not samples:
                LOGGER.warning("No prediction CIFs found for %s", entry)
                pbar_entries.update(1)
                continue

            for seed, sample_idx, cif_path in samples:
                rows_for_sample = _process_sample_cif(
                    entry,
                    seed,
                    sample_idx,
                    cif_path,
                    native_path,
                    native_dir,
                    role_counts_lookup,
                    json_dir,
                    use_sequence_matching,
                    multiplicity_data,
                    epitope_match_log,
                    model_eval_module=esmfold2_eval,
                    confidence_cols=PROTENIX_CONFIDENCE_COLS,  # ESMFold2 uses same as Protenix
                )
                rows.extend(rows_for_sample)
            pbar_entries.update(1)
    return rows


def _select_native_for_scoring(
    entry: str,
    matched_native: Path | None,
    matched_chain_map: dict[str, str] | None,
    matched_roles: dict[str, str] | None,
    epitope_match_info: dict | None,
    intended_native_path: Path,
    intended_chain_map: dict[str, str],
    intended_chain_roles: dict[str, str],
) -> tuple[Path, dict[str, str], dict[str, str], dict | None]:
    """Select which native structure to use for scoring based on epitope match quality."""
    if not matched_native:
        return intended_native_path, intended_chain_map, intended_chain_roles, epitope_match_info

    if not epitope_match_info:
        return matched_native, matched_chain_map, matched_roles, epitope_match_info

    match_quality = epitope_match_info.get("match_quality", "")
    matched_to_intended = epitope_match_info.get("matched_to_intended_target", True)

    if match_quality == "poor_match":
        LOGGER.info(
            f"Entry {entry}: Poor match for all candidates (best={epitope_match_info.get('matched_native')}, "
            f"dist={epitope_match_info.get('epitope_distance')}Å). "
            f"Reverting to intended target {entry} for scoring."
        )
        epitope_match_info["scored_against_intended"] = True
        return intended_native_path, intended_chain_map, intended_chain_roles, epitope_match_info

    if not matched_to_intended:
        LOGGER.info(
            f"Entry {entry}: Matched to different binding mode {epitope_match_info.get('matched_native')} "
            f"(dist={epitope_match_info.get('epitope_distance')}Å, quality={match_quality}). "
            f"Scoring against matched native."
        )
    epitope_match_info["scored_against_intended"] = matched_to_intended
    return matched_native, matched_chain_map, matched_roles, epitope_match_info


def _process_sample_cif(
    entry: str,
    seed: int,
    sample_idx: int,
    cif_path: Path,
    native_path: Path,
    native_dir: Path,
    role_counts_lookup: dict[str, tuple[int, int]] | None,
    json_dir: Path | None,
    use_sequence_matching: bool,
    multiplicity_data: dict[str, list[dict]] | None,
    epitope_match_log: list[dict] | None,
    model_eval_module,
    confidence_cols: list[str],
) -> list[dict]:
    """Process a single CIF prediction sample (Protenix or ESMFold2)."""
    model_struct, native_struct, chain_remap = load_structure_pair(cif_path, native_path)

    # Get chain roles for complex-level scoring
    pdb_id = entry.split("_")[0].lower()
    native_chain_roles = infer_chain_roles(
        chain_ids_in_order(native_struct),
        pdb_id,
        role_counts_lookup or {},
    )

    # Determine chain mapping strategy
    if use_sequence_matching:
        chain_map = get_chain_map_with_sequence_matching(
            cif_path,
            native_path,
            entry,
            native_chain_roles,
            json_dir=json_dir,
        )
        LOGGER.debug(f"Using sequence-based chain mapping for {entry}: {chain_map}")
    else:
        # Fallback to traditional chain mapping from stem
        chain_map = model_eval_module.chain_map_from_stem(entry, native_path=native_path)
        LOGGER.info(f"⚠️  Using traditional chain mapping for {entry}: {chain_map}")

    # Store original intended target info
    intended_native_path = native_path
    intended_chain_map = chain_map
    intended_chain_roles = native_chain_roles

    # Check for multi-epitope scenarios and perform epitope matching
    epitope_match_info = _perform_epitope_matching_cif(
        entry,
        pdb_id,
        cif_path,
        native_dir,
        native_chain_roles,
        json_dir,
        multiplicity_data,
        epitope_match_log,
        seed,
        sample_idx,
    )

    # If epitope matching succeeded, use the matched native
    matched_native, matched_chain_map, matched_roles = _apply_epitope_matching_result(
        epitope_match_info,
    )

    # Select which native to use for scoring based on match quality
    native_path, chain_map, native_chain_roles, epitope_match_info = _select_native_for_scoring(
        entry,
        matched_native,
        matched_chain_map,
        matched_roles,
        epitope_match_info,
        intended_native_path,
        intended_chain_map,
        intended_chain_roles,
    )

    # Reload structures with selected native
    model_struct, native_struct, chain_remap = load_structure_pair(cif_path, native_path)

    # Apply chain remapping to the chain_map for structure metrics
    remapped_chain_map = {nat: chain_remap.get(mod, mod) for nat, mod in chain_map.items()}

    # Run standard per-interface DockQ
    try:
        interfaces, global_dockq = run_dockq(cif_path, native_path, chain_map)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("DockQ failed for %s: %s", cif_path.name, exc)
        return []

    # Run complex-level DockQ
    try:
        intended_antigen_chain_raw = _infer_intended_antigen_chain(entry)
        intended_antigen_chain = _resolve_intended_native_antigen_chain(
            intended_antigen_chain_raw,
            chain_map,
            native_chain_roles,
        )
        antigen_chain_count = sum(1 for role in native_chain_roles.values() if role == "antigen")
        if antigen_chain_count > 1 and not intended_antigen_chain:
            LOGGER.warning(
                "Entry %s has %d antigen chains but intended antigen could not be inferred from name; "
                "complex scoring may fall back to best available antigen chain",
                entry,
                antigen_chain_count,
            )
        elif antigen_chain_count > 1 and intended_antigen_chain:
            LOGGER.info(
                "Entry %s inferred intended antigen chain: %s (native=%s)",
                entry,
                intended_antigen_chain_raw,
                intended_antigen_chain,
            )
        complex_metrics = run_complex_level_dockq(
            cif_path,
            native_path,
            chain_map,
            native_chain_roles,
            intended_antigen_chain=intended_antigen_chain,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Complex-level DockQ not computed for %s: %s", cif_path.name, exc)
        complex_metrics = None

    # Compute structure metrics
    metrics_chain_map = remapped_chain_map
    structure_metrics, _ = compute_structure_metrics(
        model_struct,
        native_struct,
        metrics_chain_map,
        native_chain_roles,
    )

    # Load confidence
    # Both ESMFold2 and Protenix: find_confidence_file(cif_path, entry, sample)
    conf_file = model_eval_module.find_confidence_file(cif_path, entry, sample_idx)
    confidence = model_eval_module.load_confidence(conf_file)

    # Validate interface coverage
    validate_interface_coverage(interfaces, native_chain_roles, entry)

    # Create rows for each interface
    rows = []
    for iface_key, metrics in interfaces.items():
        row_structure_metrics = dict(structure_metrics)
        row_structure_metrics["interface_role_pair"] = interface_role_pair(
            iface_key,
            native_chain_roles,
        )
        row = make_dockq_row(
            entry,
            seed,
            sample_idx,
            cif_path,
            native_path,
            iface_key,
            metrics,
            global_dockq,
            confidence,
            confidence_cols,
            row_structure_metrics,
            complex_metrics,
            COMPLEX_COLS,
        )

        # Add epitope matching info if available
        if epitope_match_info:
            row.update(epitope_match_info)
        else:
            row.update(
                {
                    "matched_native": "",
                    "epitope_distance": "",
                    "match_quality": "",
                    "num_candidates": "",
                    "matched_to_intended_target": "",
                    "scored_against_intended": "",
                }
            )

        rows.append(row)

    return rows


def _perform_epitope_matching_cif(
    entry: str,
    pdb_id: str,
    cif_path: Path,
    native_dir: Path,
    native_chain_roles: dict[str, str],
    json_dir: Path | None,
    multiplicity_data: dict[str, list[dict]] | None,
    epitope_match_log: list[dict] | None,
    seed: int,
    sample_idx: int,
) -> dict | None:
    """Perform epitope matching for CIF files (Protenix/ESMFold2).

    Same as _perform_epitope_matching but handles seed/sample extraction for CIF files.
    """
    if not multiplicity_data or pdb_id not in multiplicity_data:
        return None

    entries_for_pdb = multiplicity_data[pdb_id]
    if len(entries_for_pdb) <= 1:
        return None

    # Check if this is "Same Ab + Same Ag" case
    h_seqs = {e["h_seq"] for e in entries_for_pdb if e["h_seq"]}
    l_seqs = {e["l_seq"] for e in entries_for_pdb if e["l_seq"]}
    t_seqs = {e["t_seq"] for e in entries_for_pdb if e["t_seq"]}

    same_ab = len(h_seqs) == 1 and len(l_seqs) == 1
    same_ag = len(t_seqs) == 1

    if not (same_ab and same_ag):
        return None

    # Build native candidates
    native_candidates = []
    for entry_data in entries_for_pdb:
        candidate_entry = entry_data["full_entry"]
        candidate_path = native_dir / f"{candidate_entry}.pdb"
        if not candidate_path.is_file():
            continue

        # Get chain map for this candidate
        candidate_chain_map = get_chain_map_with_sequence_matching(
            cif_path,
            candidate_path,
            candidate_entry,
            native_chain_roles,
            json_dir=json_dir,
        )

        native_candidates.append((candidate_path, candidate_chain_map, native_chain_roles))

    if len(native_candidates) <= 1:
        return None

    LOGGER.info(f"Running epitope matching for {entry}: {len(native_candidates)} binding modes detected")

    # Perform epitope matching
    match_result = match_epitope_by_interface_position(
        cif_path,
        native_candidates,
        method="interface_rmsd",
        contact_cutoff=8.0,
    )

    if not match_result:
        return None

    matched_native, matched_chain_map, matched_roles, distance, quality = match_result

    # Check if matched native is the intended target (from entry name)
    matched_to_intended = matched_native.stem == entry

    epitope_match_info = {
        "matched_native": matched_native.stem,
        "epitope_distance": distance,
        "match_quality": quality,
        "num_candidates": len(native_candidates),
        "matched_to_intended_target": matched_to_intended,
        "matched_native_path": matched_native,
        "matched_chain_map": matched_chain_map,
        "matched_roles": matched_roles,
    }

    LOGGER.info(
        f"Epitope match for {cif_path.stem}: "
        f"best={matched_native.stem}, "
        f"distance={distance:.2f}Å, "
        f"quality={quality}, "
        f"intended_target={entry}, "
        f"matched_to_intended={matched_to_intended}"
    )

    # Log epitope match details
    if epitope_match_log is not None:
        candidate_names = [c[0].stem for c in native_candidates]
        epitope_match_log.append(
            {
                "pdb_id": pdb_id,
                "entry": entry,
                "model_file": cif_path.name,
                "seed": seed,
                "sample": sample_idx,
                "matched_native": matched_native.stem,
                "original_native": entry,
                "epitope_distance": f"{distance:.2f}",
                "match_quality": quality,
                "matched_to_intended_target": matched_to_intended,
                "num_candidates": len(native_candidates),
                "all_candidates": ", ".join(candidate_names),
            }
        )

    return epitope_match_info
