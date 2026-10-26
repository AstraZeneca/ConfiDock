"""Utilities for DockQ grid evaluation and result formatting.

This module provides utilities for:
- Grid condition directory scanning and resolution
- Boltz input directory resolution
- DockQ result row formatting
- Grid schema parsing
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Grid-mode condition schemas for parsing directory names
BOLTZ_CONDITION_COLS = ["msa_mode", "recycling_steps", "step_scale"]
PROTENIX_CONDITION_COLS = ["msa_mode", "cycle", "step"]
ESMFOLD2_CONDITION_COLS = ["msa_mode", "loops", "steps"]

GRID_SCHEMAS = [
    {
        "name": "boltz",
        "pattern": re.compile(r"msa=(?P<msa_mode>.+?)__recycling=(?P<recycling>\d+)__scale=(?P<scale>[0-9.]+)$"),
        "condition_cols": BOLTZ_CONDITION_COLS,
        "meta_parser": lambda m: {
            "msa_mode": m.group("msa_mode"),
            "recycling_steps": int(m.group("recycling")),
            "step_scale": float(m.group("scale")),
        },
    },
    {
        "name": "protenix",
        "pattern": re.compile(r"msa=(?P<msa_mode>.+?)__cycle=(?P<cycle>\d+)__step=(?P<step>\d+)$"),
        "condition_cols": PROTENIX_CONDITION_COLS,
        "meta_parser": lambda m: {
            "msa_mode": m.group("msa_mode"),
            "cycle": int(m.group("cycle")),
            "step": int(m.group("step")),
        },
    },
    {
        "name": "esmfold2",
        "pattern": re.compile(r"msa=(?P<msa_mode>.+?)__loops=(?P<loops>\d+)__steps=(?P<steps>\d+)$"),
        "condition_cols": ESMFOLD2_CONDITION_COLS,
        "meta_parser": lambda m: {
            "msa_mode": m.group("msa_mode"),
            "loops": int(m.group("loops")),
            "steps": int(m.group("steps")),
        },
    },
]


def format_value(v) -> str | float:
    """Format a value for CSV output.

    Parameters
    ----------
    v : any
        Value to format

    Returns
    -------
    str or float
        Rounded float (6 decimal places) or empty string for None
    """
    if isinstance(v, float):
        return round(v, 6)
    return "" if v is None else v


def resolve_grid_predictions_dir(condition_dir: Path, model: str) -> Path | None:
    """Find the predictions directory within a grid condition folder.

    Parameters
    ----------
    condition_dir : Path
        Grid condition directory (e.g., msa=full_msa__recycling=3__scale=1.0)
    model : str
        Model type ("boltz" or "protenix")

    Returns
    -------
    Path or None
        Predictions directory path, or None if not found
    """

    def _looks_like_predictions_root(path: Path) -> bool:
        if not path.is_dir():
            return False
        for child in path.iterdir():
            if child.is_dir() and not child.name.startswith(".") and not child.name.startswith("_"):
                return True
        return False

    candidates: list[Path] = []
    if model == "boltz":
        candidates.extend(
            [
                condition_dir / "boltz_results__staged_inputs" / "predictions",
                condition_dir / "predictions",
                condition_dir,
            ]
        )
    else:
        candidates.extend(
            [
                condition_dir / "predictions",
                condition_dir,
                condition_dir / "boltz_results__staged_inputs" / "predictions",
            ]
        )

    for candidate in candidates:
        if _looks_like_predictions_root(candidate):
            return candidate
    return None


def _contains_entry_yaml_files(path: Path) -> bool:
    """Check if directory contains entry-specific YAML files.

    Entry YAML files follow the pattern: <pdb_id>_<chains>.yaml
    This excludes config files like grid_config.yaml or hparams.yaml.
    """
    if not path.is_dir():
        return False
    for yaml_file in path.glob("*.yaml"):
        if yaml_file.stem and yaml_file.stem[0].isdigit():
            return True
    for yml_file in path.glob("*.yml"):
        if yml_file.stem and yml_file.stem[0].isdigit():
            return True
    return False


def _build_candidates_from_grid_dir(grid_dir: Path | None) -> list[Path]:
    """Build candidate input directories from grid_dir naming patterns."""
    candidates: list[Path] = []
    if grid_dir is None:
        return candidates

    grid_name = grid_dir.name
    if grid_name.endswith("_boltz2_grid"):
        inputs_name = grid_name.replace("_boltz2_grid", "_boltz2_inputs")
        candidates.append(grid_dir.parent / inputs_name)
    elif grid_name.endswith("_grid"):
        inputs_name = grid_name.replace("_grid", "_inputs")
        candidates.append(grid_dir.parent / inputs_name)

    return candidates


def _build_candidates_from_condition_dir(condition_dir: Path | None) -> list[Path]:
    """Build candidate input directories from condition directory."""
    if condition_dir is None:
        return []
    return [
        condition_dir / "_staged_inputs",
        condition_dir / "staged_inputs",
    ]


def _build_candidates_from_predictions_dir(predictions_dir: Path) -> list[Path]:
    """Build candidate input directories from predictions directory layout."""
    if predictions_dir.name != "predictions":
        return []

    parent = predictions_dir.parent
    return [
        parent / "_staged_inputs",
        parent.parent / "_staged_inputs",
        parent.parent / "staged_inputs",
    ]


def resolve_boltz_input_dir(
    predictions_dir: Path,
    explicit_input_dir: Path | None = None,
    condition_dir: Path | None = None,
    grid_dir: Path | None = None,
) -> Path | None:
    """Resolve the most likely Boltz input-YAML directory.

    Priority:
    1. User-provided explicit_input_dir (if it contains entry YAML files)
    2. Inferred from grid_dir pattern (e.g., *_grid → *_inputs)
    3. Condition-local staged inputs (grid mode)
    4. Paths inferred from predictions_dir layout

    Parameters
    ----------
    predictions_dir : Path
        Predictions directory
    explicit_input_dir : Path, optional
        User-specified input directory
    condition_dir : Path, optional
        Grid condition directory
    grid_dir : Path, optional
        Grid root directory

    Returns
    -------
    Path or None
        Resolved input directory, or None if not found
    """
    # Validate explicit_input_dir contains entry YAML files before using it
    if explicit_input_dir is not None and _contains_entry_yaml_files(explicit_input_dir):
        return explicit_input_dir

    # Build candidate list from all sources
    candidates: list[Path] = []
    candidates.extend(_build_candidates_from_grid_dir(grid_dir))
    candidates.extend(_build_candidates_from_condition_dir(condition_dir))
    candidates.extend(_build_candidates_from_predictions_dir(predictions_dir))

    # Find first valid candidate
    for candidate in candidates:
        if candidate.is_dir() and _contains_entry_yaml_files(candidate):
            return candidate
    return None


def _match_directory_to_schema(dir_name: str) -> tuple[dict[str, Any] | None, dict | None]:
    """Match a directory name against grid schemas.

    Returns
    -------
    tuple
        (matched_schema, matched_meta) or (None, None) if no match
    """
    for schema in GRID_SCHEMAS:
        m = schema["pattern"].match(dir_name)
        if m is not None:
            return schema, schema["meta_parser"](m)
    return None, None


def _should_skip_condition(
    matched_meta: dict,
    msa_modes: list[str] | None,
) -> bool:
    """Check if condition should be skipped based on MSA mode filter."""
    return msa_modes is not None and matched_meta["msa_mode"] not in msa_modes


def _is_schema_consistent(
    matched_cols: list[str],
    condition_cols: list[str],
    schema_name: str | None,
    matched_schema: dict,
    dir_name: str,
) -> bool:
    """Check if matched schema is consistent with previously seen schemas."""
    if not condition_cols:
        return True

    if matched_cols != condition_cols:
        LOGGER.warning(
            "Mixed grid schemas detected (%s vs %s); skipping %s",
            schema_name,
            matched_schema["name"],
            dir_name,
        )
        return False

    return True


def _process_grid_condition_dir(
    d: Path,
    model: str,
    msa_modes: list[str] | None,
    condition_cols: list[str],
    schema_name: str | None,
) -> tuple[dict | None, Path | None, list[str], str | None]:
    """Process a single grid condition directory.

    Returns
    -------
    tuple
        (matched_meta, preds_dir, updated_condition_cols, updated_schema_name)
        Returns (None, None, condition_cols, schema_name) if directory should be skipped
    """
    # Match directory name to schema
    matched_schema, matched_meta = _match_directory_to_schema(d.name)
    if matched_schema is None or matched_meta is None:
        return None, None, condition_cols, schema_name

    # Check MSA mode filter
    if _should_skip_condition(matched_meta, msa_modes):
        return None, None, condition_cols, schema_name

    # Check schema consistency
    matched_cols = list(matched_schema["condition_cols"])
    if not _is_schema_consistent(matched_cols, condition_cols, schema_name, matched_schema, d.name):
        return None, None, condition_cols, schema_name

    # Initialize condition_cols and schema_name on first match
    updated_condition_cols = condition_cols
    updated_schema_name = schema_name
    if not condition_cols:
        updated_condition_cols = matched_cols
        updated_schema_name = str(matched_schema["name"])

    # Resolve predictions directory
    preds_dir = resolve_grid_predictions_dir(d, model)
    if preds_dir is None:
        LOGGER.warning("Predictions dir not found in %s – skipping", d.name)
        return None, None, updated_condition_cols, updated_schema_name

    return matched_meta, preds_dir, updated_condition_cols, updated_schema_name


def scan_grid_conditions(
    grid_dir: Path,
    model: str,
    msa_modes: list[str] | None = None,
) -> tuple[list[tuple[dict, Path]], list[str], str | None]:
    """Scan grid directory for condition subdirectories.

    Supports both naming conventions:
    - ``msa=<mode>__recycling=<int>__scale=<float>`` (Boltz)
    - ``msa=<mode>__cycle=<int>__step=<int>`` (Protenix)

    Parameters
    ----------
    grid_dir : Path
        Grid root directory containing condition subdirectories
    model : str
        Model type ("boltz" or "protenix")
    msa_modes : list of str, optional
        Filter to specific MSA modes

    Returns
    -------
    tuple
        (conditions, condition_cols, schema_name) where:
        - conditions: list of (metadata_dict, predictions_path) tuples
        - condition_cols: list of condition column names
        - schema_name: name of the matched schema ("boltz" or "protenix")
    """
    results: list[tuple[dict, Path]] = []
    condition_cols: list[str] = []
    schema_name: str | None = None

    for d in sorted(grid_dir.iterdir()):
        if not d.is_dir():
            continue

        matched_meta, preds_dir, condition_cols, schema_name = _process_grid_condition_dir(
            d, model, msa_modes, condition_cols, schema_name
        )

        if matched_meta is not None and preds_dir is not None:
            results.append((matched_meta, preds_dir))

    return results, condition_cols, schema_name


def make_dockq_row(
    entry: str,
    seed: int | None,
    sample_idx: int,
    model_path: Path,
    native_path: Path,
    iface_key: str,
    metrics: dict,
    global_dockq: float,
    confidence: dict,
    confidence_cols: list[str],
    structure_metrics: dict | None = None,
    complex_metrics: dict | None = None,
    complex_cols: list[str] | None = None,
) -> dict:
    """Create a CSV row dict from DockQ results.

    Parameters
    ----------
    entry : str
        Entry identifier
    seed : int or None
        Random seed used for prediction
    sample_idx : int
        Sample index
    model_path : Path
        Path to model prediction file
    native_path : Path
        Path to native reference structure
    iface_key : str
        Interface key (e.g., "AB" for chains A-B)
    metrics : dict
        Per-interface DockQ metrics
    global_dockq : float
        Average DockQ across all interfaces
    confidence : dict
        Model confidence metrics
    confidence_cols : list of str
        Confidence column names
    structure_metrics : dict, optional
        Additional structure metrics
    complex_metrics : dict, optional
        Complex-level DockQ metrics
    complex_cols : list of str, optional
        Complex metric column names

    Returns
    -------
    dict
        CSV row with all metrics
    """
    row: dict = {
        "entry": entry,
        "seed": seed if seed is not None else "",
        "sample": sample_idx,
        "model_file": model_path.name,
        "native_file": native_path.name,
        "native_interface": iface_key,
        "model_chains": metrics.get("model_chains", ""),
        "global_dockq": round(global_dockq, 6),
        "DockQ": format_value(metrics.get("DockQ")),
        "iRMSD": format_value(metrics.get("iRMSD")),
        "LRMSD": format_value(metrics.get("LRMSD")),
        "fnat": format_value(metrics.get("fnat")),
        "fnonnat": format_value(metrics.get("fnonnat")),
        "F1": format_value(metrics.get("F1")),
        "clashes": metrics.get("clashes", ""),
    }

    if structure_metrics:
        row.update(structure_metrics)

    # Add complex-level metrics (will be same for all interfaces of this sample)
    if complex_metrics and complex_cols:
        for col in complex_cols:
            row[col] = format_value(complex_metrics.get(col)) if col in complex_metrics else ""
    elif complex_cols:
        # Fill with empty values if not computed
        for col in complex_cols:
            row[col] = ""

    for col in confidence_cols:
        row[col] = format_value(confidence.get(col))

    return row


def validate_interface_coverage(
    interfaces: dict[str, dict],
    native_chain_roles: dict[str, str],
    entry: str,
) -> list[str]:
    """Validate that all expected native interfaces are covered by DockQ results.

    Parameters
    ----------
    interfaces : dict
        DockQ interface results keyed by interface string (e.g., "AB")
    native_chain_roles : dict
        Mapping of native chain IDs to roles
    entry : str
        Entry identifier for logging

    Returns
    -------
    list of str
        List of missing interface keys (empty if all interfaces covered)

    Notes
    -----
    This addresses the issue reported in FoldBench GitHub issue #6, where some
    interfaces may not be scored by DockQ due to chain mapping failures or
    structural issues. Missing interfaces can lead to incorrect evaluation if
    code iterates over expected interfaces rather than actual scored interfaces.
    """
    native_chains = list(native_chain_roles.keys())
    expected_interfaces = {f"{c1}{c2}" for i, c1 in enumerate(native_chains) for c2 in native_chains[i + 1 :]}

    scored_interfaces = set(interfaces.keys())
    missing_interfaces = expected_interfaces - scored_interfaces

    if missing_interfaces:
        LOGGER.warning(
            "Entry %s: %d expected interfaces missing from DockQ results: %s. "
            "This may indicate chain mapping issues or filtered chains.",
            entry,
            len(missing_interfaces),
            sorted(missing_interfaces),
        )

    return sorted(missing_interfaces)
