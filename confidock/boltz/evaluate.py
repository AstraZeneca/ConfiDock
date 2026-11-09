"""Boltz-2 specific evaluation utilities.

This module provides Boltz-specific utilities for file naming, chain mapping,
and confidence score loading.

Core DockQ evaluation functions are in ``confidock.evaluation.dockq``
since they are model-agnostic.

Chain mapping convention
~~~~~~~~~~~~~~~~~~~~~~~~
Both Boltz-2 and Protenix predictions preserve the original PDB chain IDs from
the input (e.g. ``H``, ``L``, ``A``).  Reference (native) PDB files use
sequential IDs ``A``, ``B``, ``C`` …, where ``filename_chain[i]`` maps to
sequential chain ``chr(65 + i)``.  :func:`chain_map_from_stem` derives the
``{native: model}`` mapping from the filename stem automatically.

Example
~~~~~~~
For ``8qrg_H_L_A``:

- Model chains: H, L, A
- Native chains: A, B, C  (sequential in reference PDB)
- ``chain_map``: ``{'A': 'H', 'B': 'L', 'C': 'A'}``
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chain mapping
# ---------------------------------------------------------------------------


def _parse_chain_token(token: str) -> list[str]:
    """Split a concatenated chain-ID token into individual chain IDs.

    Each chain ID is either one character (uppercase or lowercase) or two
    characters (uppercase followed by lowercase, e.g. ``Hc``, ``Lc``, ``Bx``).
    Tokens that are already single-chain (e.g. ``"H"``, ``"A"``) are returned
    as-is so that old-style underscore-separated names continue to work.

    Examples
    --------
    >>> _parse_chain_token("HcLcA")
    ['Hc', 'Lc', 'A']
    >>> _parse_chain_token("NMJ")
    ['N', 'M', 'J']
    >>> _parse_chain_token("hlB")
    ['h', 'l', 'B']
    >>> _parse_chain_token("BbA")
    ['Bb', 'A']
    """
    chains: list[str] = []
    i = 0
    while i < len(token):
        c = token[i]
        if c.isupper() and i + 1 < len(token) and token[i + 1].islower():
            chains.append(token[i : i + 2])
            i += 2
        else:
            chains.append(c)
            i += 1
    return chains


def chain_map_from_stem(stem: str) -> dict[str, str]:
    """Return a ``{native_chain: model_chain}`` mapping from a filename stem.

    Handles both old-style underscore-separated chain IDs
    (e.g. ``"8qrg_H_L_A"``) and the newer concatenated encoding where multiple
    chain IDs are joined without separators
    (e.g. ``"8jlj_HcLcA"`` → chains ``Hc``, ``Lc``, ``A``;
    ``"8z6q_NMJ"`` → chains ``N``, ``M``, ``J``).

    Parameters
    ----------
    stem:
        Filename stem, e.g. ``"8qf5_H_A"``, ``"8qrg_H_L_A"``, or
        ``"8jlj_HcLcA"``.

    Returns
    -------
    dict
        Maps sequential native chain IDs (A, B, C, …) to the original model
        chain IDs (H, L, A, …) as encoded in *stem*.

    Examples
    --------
    >>> chain_map_from_stem("8qf5_H_A")
    {'A': 'H', 'B': 'A'}
    >>> chain_map_from_stem("8qrg_H_L_A")
    {'A': 'H', 'B': 'L', 'C': 'A'}
    >>> chain_map_from_stem("8jlj_HcLcA")
    {'A': 'Hc', 'B': 'Lc', 'C': 'A'}
    >>> chain_map_from_stem("8z6q_NMJ")
    {'A': 'N', 'B': 'M', 'C': 'J'}
    """
    tokens = stem.split("_")[1:]
    model_chains = [ch for token in tokens for ch in _parse_chain_token(token)]
    return {chr(65 + i): ch for i, ch in enumerate(model_chains)}


def chain_map_from_yaml(
    yaml_path: Path,
    native_path: Path | None = None,
) -> dict[str, str]:
    """Return a ``{native_chain: model_chain}`` mapping from a Boltz-2 input YAML.

    When *native_path* is provided the model chain IDs from the YAML are mapped
    to the native chain IDs in the order they appear in the PDB file (positional
    mapping).  This is the correct approach for datasets where the native PDB
    uses non-sequential chain IDs (e.g. ``H``, ``L``, ``T`` for SAbDab).

    When *native_path* is omitted the mapping falls back to sequential native
    chain IDs (``A``, ``B``, ``C``, …), which is correct for datasets that were
    pre-processed to use sequential chain IDs.

    Parameters
    ----------
    yaml_path:
        Path to a Boltz-2 input YAML file (e.g. ``8jln_HcLcA.yaml``).
    native_path:
        Optional path to the native reference PDB.  Chain IDs are read in
        order of first appearance in ATOM/HETATM records.
    """
    import yaml  # noqa: PLC0415

    data = yaml.safe_load(yaml_path.read_text())
    model_chains = [s["protein"]["id"] for s in data.get("sequences", []) if "protein" in s]

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
        return {nat: mod for nat, mod in zip(native_chains, model_chains)}

    return {chr(65 + i): ch for i, ch in enumerate(model_chains)}


# ---------------------------------------------------------------------------
# Prediction file discovery
# ---------------------------------------------------------------------------


def prediction_samples(
    predictions_dir: Path,
    entry: str,
) -> list[tuple[int | None, int, Path]]:
    """Return ``(seed, sample_idx, path)`` tuples for all prediction PDBs of *entry*.

    Handles both ``{entry}_seed{s}_sample{n}.pdb`` (with explicit seed) and
    ``{entry}_sample{n}.pdb`` (no seed) naming conventions.

    Parameters
    ----------
    predictions_dir:
        Root predictions directory (parent of per-entry subdirectories).
    entry:
        Entry name, e.g. ``"8qf5_H_A"``.
    """
    entry_dir = predictions_dir / entry
    if not entry_dir.is_dir():
        LOGGER.warning("No prediction directory found for entry %s at %s", entry, entry_dir)
        return []

    results: list[tuple[int | None, int, Path]] = []
    for pdb in sorted(entry_dir.glob("*.pdb")):
        m_seed = re.search(r"_seed(\d+)_sample(\d+)", pdb.stem)
        if m_seed:
            results.append((int(m_seed.group(1)), int(m_seed.group(2)), pdb))
            continue
        m_noseed = re.search(r"_sample(\d+)", pdb.stem)
        if m_noseed:
            results.append((None, int(m_noseed.group(1)), pdb))

    return results


# ---------------------------------------------------------------------------
# Confidence scores
# ---------------------------------------------------------------------------


def load_confidence(confidence_path: Path) -> dict[str, Any]:
    """Load Boltz-2 confidence metrics from a ``confidence_*.json`` file.

    Returns a flat dict with the top-level scalar fields from the JSON
    (e.g. ``confidence_score``, ``ptm``, ``iptm``, ``complex_plddt``,
    ``complex_iplddt``).  Non-scalar fields (e.g. nested dicts) are skipped.

    Returns an empty dict if the file is missing or invalid.
    """
    if not confidence_path.is_file():
        LOGGER.debug("Confidence file not found: %s", confidence_path)
        return {}
    try:
        with confidence_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Could not read confidence file %s: %s", confidence_path, exc)
        return {}


def find_confidence_file(entry_dir: Path, model_stem: str) -> Path:
    """Return the confidence JSON path for a given model PDB stem.

    Boltz names confidence files ``confidence_{model_stem}.json``.
    """
    return entry_dir / f"confidence_{model_stem}.json"
