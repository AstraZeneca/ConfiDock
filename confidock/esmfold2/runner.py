"""ESMFold2 inference utilities for running predictions.

This module provides utilities for:
- Saving PAE matrices from predictions
- Resolving MSA files by sequence hash
- Configuration management
- Building local structure prediction inputs
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def save_pae_matrix(result: Any, output_path: Path) -> tuple[bool, tuple[int, ...] | None]:
    """Save a PAE matrix from result to output_path when available.

    Parameters
    ----------
    result
        Prediction result object (may have a .pae attribute)
    output_path : Path
        Path to save the PAE matrix as .npz

    Returns
    -------
    tuple of (saved, shape)
        - saved: True if PAE was successfully saved
        - shape: Shape of the saved matrix, or None if not saved
    """
    pae = getattr(result, "pae", None)
    if pae is None:
        return False, None

    try:
        if hasattr(pae, "detach"):
            pae_array = pae.detach().cpu().numpy()
        elif hasattr(pae, "cpu") and hasattr(pae, "numpy"):
            pae_array = pae.cpu().numpy()
        else:
            import numpy as np  # noqa: PLC0415

            pae_array = np.asarray(pae)

        import numpy as np  # noqa: PLC0415

        np.savez_compressed(output_path, pae=pae_array)
        return True, tuple(int(x) for x in pae_array.shape)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not save PAE matrix to %s: %s", output_path, exc)
        return False, None


def resolve_a3m(seq: str, msa_dir: Path) -> Path | None:
    """Return the .a3m path for seq from a hash-named MSA directory, or None.

    Parameters
    ----------
    seq : str
        Protein sequence
    msa_dir : Path
        Directory containing .a3m files named by SHA256 hash prefix

    Returns
    -------
    Path or None
        Path to .a3m file if it exists and has content, None otherwise
    """
    digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
    p = msa_dir / f"{digest}.a3m"
    return p if p.exists() and p.stat().st_size >= 10 else None  # noqa: PLR2004


def write_prediction_config(config_dir: Path, **kwargs) -> None:
    """Persist prediction settings to a JSON config file.

    Parameters
    ----------
    config_dir : Path
        Directory to write config file
    **kwargs
        Configuration key-value pairs to save
    """
    cfg_path = config_dir / "prediction_config.json"
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(kwargs.items())},
            fh,
            indent=2,
        )
    LOGGER.info("Prediction config written to %s", cfg_path)


def load_entries(input_json: Path, requested_entries: list[str] | None) -> list[dict[str, Any]]:
    """Load and filter entries from an ESMFold2 input JSON file.

    Parameters
    ----------
    input_json : Path
        Path to JSON file containing entries
    requested_entries : list of str or None
        If provided, only return entries with these names

    Returns
    -------
    list of dict
        Filtered list of entry dictionaries
    """
    with input_json.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list of entries in {input_json}, got {type(payload)!r}")

    entries = [entry for entry in payload if isinstance(entry, dict)]
    if requested_entries:
        wanted = set(requested_entries)
        entries = [entry for entry in entries if entry.get("name") in wanted]
    return entries


def coerce_modifications(raw_modifications: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Coerce modification dictionaries to have correct types.

    Parameters
    ----------
    raw_modifications : list of dict or None
        Raw modification data from JSON

    Returns
    -------
    list of dict
        Modifications with position as int and ccd as str
    """
    if not raw_modifications:
        return []
    return [
        {
            "position": int(mod["position"]),
            "ccd": str(mod["ccd"]),
        }
        for mod in raw_modifications
    ]
