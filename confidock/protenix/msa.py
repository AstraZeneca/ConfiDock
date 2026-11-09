"""MSA management utilities for Protenix predictions.

This module provides utilities for:
- Staging MSA directories for Protenix inference
- Injecting MSA paths into Protenix input entries
- Resolving MSA files by sequence hash
"""

from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def resolve_a3m(seq: str, msa_dir: Path) -> Path | None:
    """Return the .a3m path for a sequence from a hash-named MSA directory.

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


def _create_non_pairing_msa(
    non_pairing_dst: Path,
    a3m_src: Path,
    seq: str,
) -> bool:
    """Create non_pairing.a3m file, return True if found, False if missing.

    Parameters
    ----------
    non_pairing_dst : Path
        Destination path for non_pairing.a3m
    a3m_src : Path
        Source .a3m file path
    seq : str
        Protein sequence (for query-only fallback)

    Returns
    -------
    bool
        True if MSA was found and linked, False if query-only was written
    """
    if a3m_src.exists():
        non_pairing_dst.symlink_to(a3m_src.resolve())
        return True
    LOGGER.warning("No .a3m found for seq %s... writing query-only MSA", seq[:20])
    non_pairing_dst.write_text(f">query\n{seq}\n", encoding="utf-8")
    return False


def _create_pairing_msa(pairing_dst: Path, seq: str) -> None:
    """Create query-only pairing.a3m file for heteromer mode.

    Parameters
    ----------
    pairing_dst : Path
        Destination path for pairing.a3m
    seq : str
        Protein sequence
    """
    pairing_dst.write_text(f">query\n{seq}\n", encoding="utf-8")


def _process_protein_chain(
    seq: str,
    sha: str,
    staging_dir: Path,
    msa_dir: Path,
) -> tuple[Path, bool]:
    """Process a single protein chain and create its MSA directory.

    Parameters
    ----------
    seq : str
        Protein sequence
    sha : str
        SHA256 hash prefix for the sequence
    staging_dir : Path
        Root staging directory
    msa_dir : Path
        Source MSA directory

    Returns
    -------
    tuple[Path, bool]
        (per_seq_dir, msa_found) where msa_found is True if MSA existed
    """
    per_seq_dir = staging_dir / sha
    per_seq_dir.mkdir(exist_ok=True)

    non_pairing_dst = per_seq_dir / "non_pairing.a3m"
    pairing_dst = per_seq_dir / "pairing.a3m"

    msa_found = True
    if not non_pairing_dst.exists():
        a3m_src = msa_dir / f"{sha}.a3m"
        msa_found = _create_non_pairing_msa(non_pairing_dst, a3m_src, seq)

    if not pairing_dst.exists():
        _create_pairing_msa(pairing_dst, seq)

    return per_seq_dir, msa_found


def stage_msa_dirs(
    entries: list[dict],
    msa_dir: Path,
    staging_dir: Path,
) -> dict[str, Path]:
    """Create per-sequence MSA directories for Protenix under staging_dir.

    Each directory contains:
    * ``non_pairing.a3m`` – symlink to the pre-fetched ``.a3m`` from *msa_dir*
    * ``pairing.a3m``     – query-only dummy (enables heteromer mode without
                             crashing; effective pairing depth is zero)

    Parameters
    ----------
    entries : list of dict
        Protenix input entries (each with "sequences")
    msa_dir : Path
        Directory containing pre-fetched .a3m files named by SHA256 hash
    staging_dir : Path
        Directory to create per-sequence MSA subdirectories

    Returns
    -------
    dict
        Mapping {sequence: per_seq_dir}
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    seq_to_dir: dict[str, Path] = {}
    n_found = 0
    n_missing = 0

    for entry in entries:
        for chain in entry["sequences"]:
            pc = chain.get("proteinChain")
            if pc is None:
                continue
            seq: str = pc["sequence"]
            if seq in seq_to_dir:
                continue

            sha = hashlib.sha256(seq.encode()).hexdigest()[:16]
            per_seq_dir, msa_found = _process_protein_chain(seq, sha, staging_dir, msa_dir)

            if msa_found:
                n_found += 1
            else:
                n_missing += 1

            seq_to_dir[seq] = per_seq_dir

    LOGGER.info(
        "MSA staging: %d sequences found, %d missing (wrote query-only dummies)",
        n_found,
        n_missing,
    )
    return seq_to_dir


def inject_msa(entries: list[dict], seq_to_msa_dir: dict[str, Path]) -> list[dict]:
    """Deep-copy entries and add ``proteinChain.msa`` for each chain.

    Parameters
    ----------
    entries : list of dict
        Protenix input entries
    seq_to_msa_dir : dict
        Mapping {sequence: msa_directory_path}

    Returns
    -------
    list of dict
        New entries with MSA paths injected
    """
    result = []
    for entry in entries:
        e = copy.deepcopy(entry)
        for chain in e["sequences"]:
            pc = chain.get("proteinChain")
            if pc is None:
                continue
            msa_path = seq_to_msa_dir.get(pc["sequence"])
            if msa_path is not None:
                pc["msa"] = {
                    "precomputed_msa_dir": str(msa_path),
                    "pairing_db": "uniref100",
                }
        result.append(e)
    return result


def _write_msa_paths(
    pc: dict,
    msa_dir: Path,
    keys: tuple[tuple[str, str], ...],
    *,
    include_paired: bool,
) -> None:
    """Point one protein chain at its staged MSA files.

    Parameters
    ----------
    pc : dict
        A ``proteinChain`` block, mutated in place
    msa_dir : Path
        The chain's staged per-sequence MSA directory
    keys : tuple of (str, str)
        ``(entry_key, filename)`` pairs to write, as narrowed by *include_paired*
    include_paired : bool, keyword-only
        When False, any ``pairedMsaPath`` inherited from the input is dropped rather
        than left pointing at a file this call is not writing
    """
    pc.pop("msa", None)
    if not include_paired:
        pc.pop("pairedMsaPath", None)
    for key, name in keys:
        path = Path(msa_dir) / name
        # Only existing paths: need_msa_search treats a declared-but-missing path as a
        # reason to re-run an MMseqs2 search, so a dangling path is worse than none.
        if path.exists():
            pc[key] = str(path)


def inject_msa_paths(
    entries: list[dict],
    seq_to_msa_dir: dict[str, Path],
    *,
    include_paired: bool = True,
) -> list[dict]:
    """Deep-copy entries and add ``pairedMsaPath`` / ``unpairedMsaPath`` per chain.

    This is Protenix's *current* MSA schema, as read by ``runner/msa_search.py``.
    :func:`inject_msa` writes the legacy ``msa.precomputed_msa_dir`` block instead,
    which Protenix accepts but silently rewrites via ``convert_msa_to_new_format``
    (dumping a ``*-update-msa.json`` beside the input). Writing the new keys directly
    means the JSON Protenix opens is the JSON on disk.

    Only paths that exist are written: ``need_msa_search`` treats a declared-but-missing
    path as a reason to re-run an MMseqs2 search, so a dangling path is worse than none.

    Parameters
    ----------
    entries : list of dict
        Protenix input entries
    seq_to_msa_dir : dict
        Mapping {sequence: msa_directory_path}, as returned by :func:`stage_msa_dirs`
    include_paired : bool, keyword-only
        Write ``pairedMsaPath`` as well as ``unpairedMsaPath``. Set False to reference only
        the unpaired MSA. When the staged ``pairing.a3m`` is the query-only dummy that
        :func:`stage_msa_dirs` writes, the two settings are equivalent: ``RawMsa.from_a3m``
        seeds every alignment with the query sequence, so an absent paired MSA and a
        query-only one both featurize to depth 1. Verified bit-identical downstream.

    Returns
    -------
    list of dict
        New entries with ``unpairedMsaPath`` (and optionally ``pairedMsaPath``) set, and any
        legacy ``msa`` block dropped
    """
    keys = (("pairedMsaPath", "pairing.a3m"), ("unpairedMsaPath", "non_pairing.a3m"))
    if not include_paired:
        keys = keys[1:]

    result = []
    for entry in entries:
        e = copy.deepcopy(entry)
        for chain in e["sequences"]:
            pc = chain.get("proteinChain")
            if pc is None:
                continue
            msa_dir = seq_to_msa_dir.get(pc["sequence"])
            if msa_dir is not None:
                _write_msa_paths(pc, msa_dir, keys, include_paired=include_paired)
        result.append(e)
    return result


def strip_msa(entries: list[dict]) -> list[dict]:
    """Deep-copy entries with any MSA fields removed.

    Drops both the legacy ``msa`` block and the current ``pairedMsaPath`` /
    ``unpairedMsaPath`` keys, so a ``no_msa`` condition stays MSA-free even when the
    input JSON has paths baked in by :func:`inject_msa_paths`.

    Parameters
    ----------
    entries : list of dict
        Protenix input entries

    Returns
    -------
    list of dict
        New entries with MSA fields removed
    """
    result = []
    for entry in entries:
        e = copy.deepcopy(entry)
        for chain in e["sequences"]:
            pc = chain.get("proteinChain")
            if pc is None:
                continue
            for key in ("msa", "pairedMsaPath", "unpairedMsaPath"):
                pc.pop(key, None)
        result.append(e)
    return result
