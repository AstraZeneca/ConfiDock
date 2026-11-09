"""ESMFold2 specific evaluation utilities.

This module provides ESMFold2-specific utilities for file naming, chain mapping,
and confidence score loading. ESMFold2 outputs CIF files directly in the entry
directory without seed subdirectories.

ESMFold2 layout::

    {predictions_dir}/{entry}/{entry}_sample_{n}.cif
    {predictions_dir}/{entry}/summary_{entry}_sample_{n}.json
    {predictions_dir}/{entry}/pae_{entry}_sample_{n}.npz
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def chain_map_from_stem(stem: str, native_path: Path | None = None) -> dict[str, str]:
    """Return a ``{native_chain: model_chain}`` mapping from a filename stem.

    ESMFold2 preserves the original chain IDs from the filename in its predictions
    (unlike Protenix which uses sequential A, B, C). For example, entry 1a14_HLN
    will have model chains H, L, N.

    If the native_path is provided and the actual native chains differ from the
    filename (e.g., 1a14_HLN.pdb has H, L, T instead of H, L, N), we map them
    positionally: native[0]->model[0], native[1]->model[1], etc.

    Parameters
    ----------
    stem : str
        Filename stem, e.g. ``"8qf5_H_A"``, ``"8qrg_H_L_A"``, or ``"8jlj_HcLcA"``.
    native_path : Path, optional
        Path to native PDB file. If provided, actual native chain IDs are read
        from the file to handle mismatches with the filename.

    Returns
    -------
    dict
        Maps native chain IDs to model chain IDs (e.g., {'H': 'H', 'L': 'L', 'T': 'N'})
    """
    from confidock.boltz.evaluate import _parse_chain_token  # noqa: PLC0415

    # Parse model chain IDs from filename
    tokens = stem.split("_")[1:]
    model_chains = [ch for token in tokens for ch in _parse_chain_token(token)]

    # If native_path provided, read actual native chain IDs from the file
    if native_path is not None and native_path.is_file():
        native_chains: list[str] = []
        seen: set[str] = set()
        with native_path.open() as fh:
            for line in fh:
                rec = line[:6].strip()
                if rec in {"ATOM", "HETATM"} and len(line) >= 22:
                    ch = line[21]
                    if ch not in seen:
                        native_chains.append(ch)
                        seen.add(ch)
        if native_chains:
            # Map positionally: native chains in file order -> model chains from filename
            return {nat: mod for nat, mod in zip(native_chains, model_chains)}

    # Fallback: assume native chains match model chains (identity mapping)
    return {ch: ch for ch in model_chains}


def prediction_samples(
    predictions_dir: Path,
    entry: str,
) -> list[tuple[int, int, Path]]:
    """Return ``(seed, sample_idx, cif_path)`` for all ESMFold2 CIF files of *entry*.

    ESMFold2 layout (flat structure, no seed subdirectories)::

        {predictions_dir}/{entry}/{entry}_sample_{n}.cif  (full_msa mode)
        {predictions_dir}/{entry}/{entry}.cif              (no_msa mode)

    Returns seed=0 for all samples since ESMFold2 doesn't use explicit seeds.
    """
    entry_dir = predictions_dir / entry
    if not entry_dir.is_dir():
        LOGGER.warning("No prediction directory found for entry %s at %s", entry, entry_dir)
        return []

    results: list[tuple[int, int, Path]] = []

    # Pattern 1: {entry}_sample_{n}.cif (e.g., full_msa mode)
    for cif in sorted(entry_dir.glob(f"{entry}_sample_*.cif")):
        m = re.search(r"_sample_(\d+)\.cif$", cif.name)
        if m:
            sample_idx = int(m.group(1))
            results.append((0, sample_idx, cif))  # seed=0 for all ESMFold2 samples

    # Pattern 2: {entry}.cif (e.g., no_msa mode, typically single sample)
    direct_cif = entry_dir / f"{entry}.cif"
    if direct_cif.is_file():
        # Use a sample_idx that won't collide with _sample_* files
        # Check if we already have sample indices, use max+1 or start at 0
        if results:
            max_idx = max(idx for _, idx, _ in results)
            sample_idx = max_idx + 1
        else:
            sample_idx = 0
        results.append((0, sample_idx, direct_cif))

    return results


def find_confidence_file(cif_path: Path, entry: str, sample: int) -> Path:
    """Return the confidence JSON path for an ESMFold2 prediction.

    ESMFold2 names files:
    - ``summary_{entry}_sample_{sample}.json`` (full_msa mode)
    - ``summary_{entry}.json`` (no_msa mode)
    """
    # Check if the CIF follows the _sample_ pattern
    if "_sample_" in cif_path.name:
        return cif_path.parent / f"summary_{entry}_sample_{sample}.json"
    else:
        # Direct naming pattern (e.g., no_msa mode)
        return cif_path.parent / f"summary_{entry}.json"


def load_confidence(json_path: Path) -> dict[str, Any]:
    """Load ESMFold2 confidence metrics from a summary JSON file.

    Parameters
    ----------
    json_path : Path
        Path to the summary JSON file

    Returns
    -------
    dict
        Confidence metrics with keys matching the Protenix schema:
        - plddt: float
        - gpde: float
        - ptm: float
        - iptm: float
    """
    if not json_path.is_file():
        LOGGER.warning("Confidence JSON not found: %s", json_path)
        return {"plddt": None, "gpde": None, "ptm": None, "iptm": None}

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate that data is a dict
        if not isinstance(data, dict):
            LOGGER.warning("Expected dict but got %s in %s", type(data).__name__, json_path)
            return {"plddt": None, "gpde": None, "ptm": None, "iptm": None}

        # ESMFold2 uses the same metric names as Protenix
        return {
            "plddt": data.get("plddt"),
            "gpde": data.get("gpde"),
            "ptm": data.get("ptm"),
            "iptm": data.get("iptm"),
        }
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        LOGGER.warning("Failed to parse confidence JSON %s: %s", json_path, exc)
        return {"plddt": None, "gpde": None, "ptm": None, "iptm": None}
