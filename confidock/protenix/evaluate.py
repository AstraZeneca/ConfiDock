"""DockQ evaluation helpers for Protenix predictions.

Protenix-specific file discovery and confidence loading.  The shared
:func:`~confidock.evaluation.dockq.run_dockq` function is re-exported
here for convenience.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from confidock.evaluation.dockq import run_dockq  # noqa: F401

LOGGER = logging.getLogger(__name__)


def chain_map_from_stem(stem: str, native_path: Path | None = None) -> dict[str, str]:
    """Return ``{native_chain: model_chain}`` for a Protenix entry.

    Protenix CIF files use sequential model chain IDs (A, B, C, ...).  For
    SAbDab/processed native PDBs the true native chain IDs should be read from
    the native file itself (order of first ATOM/HETATM occurrence), because the
    filename chain token may not match the chain IDs actually present in the PDB.

    If *native_path* is not provided, this falls back to parsing chain IDs from
    the filename stem and mapping them to sequential model chains.
    """
    from confidock.boltz.evaluate import _parse_chain_token  # noqa: PLC0415

    tokens = stem.split("_")[1:]
    n_chains = sum(len(_parse_chain_token(t)) for t in tokens)
    model_chains = [chr(65 + i) for i in range(n_chains)]

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
            return {nat: mod for nat, mod in zip(native_chains, model_chains)}

    native_chains = [ch for token in tokens for ch in _parse_chain_token(token)]
    return {native_ch: mod for native_ch, mod in zip(native_chains, model_chains)}


def prediction_samples(
    predictions_dir: Path,
    entry: str,
) -> list[tuple[int, int, Path]]:
    """Return ``(seed, sample_idx, cif_path)`` for all Protenix CIF files of *entry*.

    Protenix layout::

        {predictions_dir}/{entry}/seed_{s}/predictions/{entry}_sample_{n}.cif
    """
    entry_dir = predictions_dir / entry
    if not entry_dir.is_dir():
        LOGGER.warning("No prediction directory found for entry %s at %s", entry, entry_dir)
        return []

    results: list[tuple[int, int, Path]] = []
    for seed_dir in sorted(entry_dir.glob("seed_*")):
        m = re.search(r"seed_(\d+)$", seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        pred_dir = seed_dir / "predictions"
        if not pred_dir.is_dir():
            continue
        for cif in sorted(pred_dir.glob(f"{entry}_sample_*.cif")):
            m2 = re.search(r"_sample_(\d+)\.cif$", cif.name)
            if m2:
                results.append((seed, int(m2.group(1)), cif))
    return results


def find_confidence_file(cif_path: Path, entry: str, sample: int) -> Path:
    """Return the confidence JSON path for a Protenix prediction.

    Protenix names files
    ``{entry}_summary_confidence_sample_{sample}.json``.
    """
    return cif_path.parent / f"{entry}_summary_confidence_sample_{sample}.json"


def load_confidence(path: Path) -> dict[str, Any]:
    """Load scalar fields from a Protenix ``summary_confidence`` JSON.

    Returns an empty dict if the file is missing or invalid.
    """
    if not path.is_file():
        LOGGER.debug("Confidence file not found: %s", path)
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return {}


def find_pae_file(cif_path: Path, entry: str, sample: int) -> Path | None:
    """Find confidence data file for a Protenix prediction.

    Checks for .npz file first (preferred), falls back to full_data JSON.

    Parameters
    ----------
    cif_path : Path
        Path to the prediction CIF file
    entry : str
        Entry name
    sample : int
        Sample index

    Returns
    -------
    Path | None
        Path to confidence file (.npz or .json), or None if not found
    """
    # Check for .npz file first (compressed, preferred)
    # Try new naming convention first
    npz_path = cif_path.parent / f"{entry}_confidence_sample_{sample}.npz"
    if npz_path.exists():
        return npz_path

    # Fall back to old naming (for backward compatibility)
    npz_path_old = cif_path.parent / f"{entry}_pae_sample_{sample}.npz"
    if npz_path_old.exists():
        return npz_path_old

    # Fall back to full_data JSON
    json_path = cif_path.parent / f"{entry}_full_data_sample_{sample}.json"
    if json_path.exists():
        return json_path

    return None


def _load_from_npz(pae_path: Path) -> dict[str, Any]:
    """Load PAE matrices from .npz file."""
    import numpy as np

    result = {}
    with np.load(pae_path, allow_pickle=False) as data:
        if "pae" in data:
            result["pae"] = data["pae"]
        if "pde" in data:
            result["pde"] = data["pde"]
        if "contact_probs" in data:
            result["contact_probs"] = data["contact_probs"]
        if "atom_plddt" in data:
            result["atom_plddt"] = data["atom_plddt"]
    return result


def _load_from_json(pae_path: Path) -> dict[str, Any]:
    """Load PAE matrices from full_data JSON."""
    import numpy as np

    result = {}
    with pae_path.open() as fh:
        data = json.load(fh)

    if "token_pair_pae" in data:
        result["pae"] = np.array(data["token_pair_pae"], dtype=np.float32)
    if "token_pair_pde" in data:
        result["pde"] = np.array(data["token_pair_pde"], dtype=np.float32)
    if "contact_probs" in data:
        result["contact_probs"] = np.array(data["contact_probs"], dtype=np.float32)
    if "atom_plddt" in data:
        result["atom_plddt"] = np.array(data["atom_plddt"], dtype=np.float32)

    return result


def load_pae_matrix(pae_path: Path) -> dict[str, Any]:
    """Load PAE and related matrices from .npz or full_data JSON.

    Parameters
    ----------
    pae_path : Path
        Path to .npz or full_data JSON file

    Returns
    -------
    dict
        Dictionary with keys: 'pae', 'pde', 'contact_probs', 'atom_plddt'
        (only present keys are included)
    """
    result = {}

    try:
        if pae_path.suffix == ".npz":
            result = _load_from_npz(pae_path)
        elif pae_path.suffix == ".json":
            result = _load_from_json(pae_path)
        else:
            LOGGER.warning("Unsupported file type: %s", pae_path.suffix)

    except (OSError, KeyError, ValueError) as exc:
        LOGGER.warning("Could not load PAE from %s: %s", pae_path, exc)

    return result
