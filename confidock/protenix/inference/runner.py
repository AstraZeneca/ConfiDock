"""Protenix inference utilities for running predictions and managing MSAs.

This module provides utilities for:
- Finding and invoking the Protenix CLI
- Managing MSA directories and staging
- Checking completion status
- Configuration management
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import yaml

LOGGER = logging.getLogger(__name__)

# Protenix version constants
PROTENIX_VERSION_PYPI = "pypi_v2.0.0"
PROTENIX_VERSION_SHARED = "shared_v0.1.0"


def find_protenix_cli() -> tuple[str, str, str]:
    """Return the protenix runner command, PYTHONPATH, and version identifier.

    Priority:
    1. If we're in the protenix-inference Poetry venv, use local v2.0.0
       (protenix CLI should be available in PATH)
    2. Otherwise, use shared Protenix v0.1.0 (no atom confidence)

    Returns
    -------
    tuple of (cli_command, pythonpath_value, version)
        - cli_command: the command to run
        - pythonpath_value: empty if already in PATH, otherwise path to add to PYTHONPATH
        - version: PROTENIX_VERSION_PYPI or PROTENIX_VERSION_SHARED
    """
    # Check if we're in protenix-inference venv by checking sys.prefix
    # Poetry sets sys.prefix to the venv directory
    is_protenix_venv = "protenix-inference" in sys.prefix

    if is_protenix_venv:
        # runner.batch_inference is installed, use the `protenix pred` subcommand
        LOGGER.info("Using locally-installed Protenix v2.0.0 CLI (via protenix pred command)")
        return "protenix pred", "", PROTENIX_VERSION_PYPI

    # Fall back to a separately-installed Protenix v0.1.0 source checkout (which has no
    # atom confidence), located via PROTENIX_SRC_DIR. Unset means there is no fallback.
    shared_src = os.environ.get("PROTENIX_SRC_DIR", "")
    shared_protenix = Path(shared_src).expanduser() if shared_src else None
    if shared_protenix is not None and shared_protenix.exists():
        LOGGER.info("Using shared Protenix v0.1.0 (atom confidence not available)")
        return f"{sys.executable} -m protenix.runner.batch_inference", str(shared_protenix), PROTENIX_VERSION_SHARED

    # Fallback (shouldn't reach here)
    return "protenix pred", "", PROTENIX_VERSION_PYPI


def _entry_is_complete(
    output_dir: Path,
    name: str,
    seeds: list[int],
    samples: int,
) -> bool:
    """Check if all seed×sample .cif files exist for an entry.

    Parameters
    ----------
    output_dir : Path
        Base output directory
    name : str
        Entry name
    seeds : list of int
        Seeds to check
    samples : int
        Number of samples per seed

    Returns
    -------
    bool
        True if all outputs exist
    """
    for seed in seeds:
        sample_dir = output_dir / name
        if not _all_seed_samples_exist(sample_dir, name, seed, samples):
            return False
    return True


def check_completion(
    output_dir: Path,
    entry_names: list[str],
    seeds: list[int],
    samples: int,
) -> int:
    """Count entries for which all seed × sample .cif files already exist.

    Parameters
    ----------
    output_dir : Path
        Base output directory
    entry_names : list of str
        Entry names to check
    seeds : list of int
        Seeds to check
    samples : int
        Number of samples per seed

    Returns
    -------
    int
        Number of completed entries
    """
    n = 0
    for name in entry_names:
        if _entry_is_complete(output_dir, name, seeds, samples):
            n += 1
    return n


def _all_seed_samples_exist(
    sample_dir: Path,
    name: str,
    seed: int,
    samples: int,
) -> bool:
    """Check if all sample .cif files exist for a given seed.

    Parameters
    ----------
    sample_dir : Path
        Sample directory
    name : str
        Sample name
    seed : int
        Seed to check
    samples : int
        Number of samples per seed

    Returns
    -------
    bool
        True if all .cif files exist for this seed
    """
    pred_dir = sample_dir / f"seed_{seed}" / "predictions"
    for rank in range(samples):
        cif = pred_dir / f"{name}_sample_{rank}.cif"
        if not cif.exists():
            return False
    return True


def _sample_is_complete(
    sample_dir: Path,
    name: str,
    seeds: list[int],
    samples: int,
) -> bool:
    """Check if all seed×sample .cif files exist for a sample.

    Parameters
    ----------
    sample_dir : Path
        Sample directory
    name : str
        Sample name
    seeds : list of int
        Seeds to check
    samples : int
        Number of samples per seed

    Returns
    -------
    bool
        True if all outputs exist
    """
    for seed in seeds:
        if not _all_seed_samples_exist(sample_dir, name, seed, samples):
            return False
    return True


def check_completed_samples(output_dir: Path, seeds: list[int], samples: int) -> set[str]:
    """Return the set of sample names for which every seed×sample .cif exists.

    Parameters
    ----------
    output_dir : Path
        Base output directory containing per-sample subdirectories
    seeds : list of int
        Seeds to check
    samples : int
        Number of samples per seed

    Returns
    -------
    set of str
        Sample names that have all outputs
    """
    completed: set[str] = set()
    for sample_dir in output_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        name = sample_dir.name
        if _sample_is_complete(sample_dir, name, seeds, samples):
            completed.add(name)
    return completed


def write_prediction_config(output_dir: Path, format: str = "yaml", **kwargs) -> None:
    """Persist prediction settings to a config file in output_dir.

    Parameters
    ----------
    output_dir : Path
        Directory to write config file
    format : str
        "yaml" or "json"
    **kwargs
        Configuration key-value pairs to save
    """
    if format == "yaml":
        cfg_path = output_dir / "prediction_config.yaml"
        with cfg_path.open("w") as fh:
            yaml.safe_dump(
                {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(kwargs.items())},
                fh,
                default_flow_style=False,
            )
    else:  # json
        cfg_path = output_dir / "prediction_config.json"
        with cfg_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(kwargs.items())},
                fh,
                indent=2,
            )
    LOGGER.info("Prediction config written to %s", cfg_path)


def print_timings(timings: dict[str, float]) -> None:
    """Print timing statistics in a formatted table.

    Parameters
    ----------
    timings : dict
        Mapping of step names to elapsed times in seconds
    """
    LOGGER.info("=== Timings ===")
    for step, elapsed in timings.items():
        LOGGER.info("  %-25s %.1f s", step, elapsed)
    if timings:
        LOGGER.info("  %-25s %.1f s", "Total", sum(timings.values()))
