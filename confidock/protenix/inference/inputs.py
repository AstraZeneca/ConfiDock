"""Protenix JSON input preparation utilities.

Provides helpers for parsing PDB files and generating Protenix-compatible
JSON input entries from them.

Typical use::

    from confidock.protenix.inputs import pdb_to_protenix_entry, pdbs_to_protenix_json

    entry = pdb_to_protenix_entry(Path("data/7bbj_H_L_A.pdb"))
    pdbs_to_protenix_json([Path("data/7bbj_H_L_A.pdb")], Path("out/protenix_inputs.json"))

Output format
~~~~~~~~~~~~~
Each entry follows the Protenix sequence input schema::

    {
        "name": "7bbj_H_L_A",
        "sequences": [
            {"proteinChain": {"sequence": "EVQLVE...", "count": 1, "modifications": []}},
            ...
        ]
    }

Chain naming convention
~~~~~~~~~~~~~~~~~~~~~~~
Filename encodes original PDB chain IDs (e.g. ``7bbj_H_L_A.pdb`` → H, L, A),
while the atoms in the file use sequential IDs (A, B, C …).  The mapping is
positional: filename chain[i] → file chain at position i.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from Bio.PDB import PDBParser

from confidock.boltz.inference.inputs import (
    THREE_TO_ONE,
    _parse_chain_labels,
    sanitise_pdb_text,
)

LOGGER = logging.getLogger(__name__)

# Standard residue names that need no modification entry.
# Non-standard residues (e.g. MSE, SEC) present in THREE_TO_ONE will still
# appear in the sequence via their mapped one-letter code but will also
# generate a modification entry.
_STANDARD_AA = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "UNK",  # unknown → X in sequence, no modification entry needed
    }
)

_THREE_TO_ONE_FULL: dict[str, str] = {**THREE_TO_ONE, "UNK": "X"}


def _chain_residues(chain) -> list[tuple[str, int]]:
    """Return ``(resname, seq_position)`` for all standard-location residues.

    HETATM records (``residue.id[0] != ' '``) and water are skipped.
    ``seq_position`` is 1-indexed within the chain's residue list.
    """
    results: list[tuple[str, int]] = []
    pos = 1
    for residue in chain.get_residues():
        if residue.id[0] != " ":  # HETATM / water
            continue
        results.append((residue.get_resname().strip(), pos))
        pos += 1
    return results


def _parse_pdb_filename(stem: str) -> tuple[str | None, list[str]]:
    """Parse PDB filename to extract chain ID hints.

    Returns
    -------
    tuple
        (concat_chain_str, original_chain_ids) where concat_chain_str is only
        set for concatenated format needing deferred parsing.
    """
    parts = stem.split("_")
    if len(parts) >= 3:  # noqa: PLR2004
        return None, parts[1:]
    if len(parts) == 2 and len(parts[1]) > 1:  # noqa: PLR2004
        return parts[1], []
    return None, []


def _build_sequence_from_residues_protenix(
    residues: list[tuple[str, int]], chain_id: str, stem: str
) -> tuple[list[str], list[dict]]:
    """Build sequence letters and modification list from residues for Protenix format.

    Returns
    -------
    tuple
        (seq_letters, modifications) where modifications uses Protenix's ptmType/ptmPosition format.
    """
    seq_letters: list[str] = []
    modifications: list[dict] = []

    for resname, pos in residues:
        one_letter = _THREE_TO_ONE_FULL.get(resname)
        if one_letter is None:
            LOGGER.warning(
                "%s chain %s pos %d: unrecognised residue %r – skipped",
                stem,
                chain_id,
                pos,
                resname,
            )
            continue

        seq_letters.append(one_letter)

        if resname not in _STANDARD_AA:
            modifications.append({"ptmType": f"CCD_{resname}", "ptmPosition": len(seq_letters)})

    return seq_letters, modifications


def _process_chain_for_protenix(chain, original_chain_id: str, stem: str) -> dict | None:
    """Process a single PDB chain into a Protenix sequence entry.

    Returns None if the chain cannot be processed (empty or invalid).
    """
    residues = _chain_residues(chain)

    if not residues:
        LOGGER.warning(
            "%s chain %s (file chain %s): no residues found – skipping",
            stem,
            original_chain_id,
            chain.id,
        )
        return None

    seq_letters, modifications = _build_sequence_from_residues_protenix(residues, original_chain_id, stem)

    if not seq_letters:
        LOGGER.warning("%s chain %s: empty sequence – skipping", stem, original_chain_id)
        return None

    return {
        "proteinChain": {
            "sequence": "".join(seq_letters),
            "count": 1,
            "modifications": modifications,
        }
    }


def pdb_to_protenix_entry(pdb_path: Path) -> dict | None:
    """Convert a single PDB file to a Protenix-format input dict.

    Parameters
    ----------
    pdb_path:
        Path to a PDB file whose stem follows ``{pdbid}_{chain1}[_{chain2}...}``
        (e.g. ``7bbj_H_L_A.pdb``).

    Returns
    -------
    dict | None
        Protenix entry dict, or ``None`` if the file could not be parsed.
    """
    stem = pdb_path.stem
    concat_chain_str, original_chain_ids = _parse_pdb_filename(stem)

    if concat_chain_str is None and not original_chain_ids:
        LOGGER.warning("Skipping %s – filename doesn't encode chain IDs", pdb_path.name)
        return None

    try:
        clean_text = sanitise_pdb_text(pdb_path.read_text())
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(stem, io.StringIO(clean_text))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to parse %s: %s", pdb_path.name, exc)
        return None

    file_chains = list(structure[0].get_chains())

    if concat_chain_str is not None:
        original_chain_ids = _parse_chain_labels(concat_chain_str, len(file_chains))

    if len(file_chains) != len(original_chain_ids):
        LOGGER.warning(
            "%s: filename encodes %d chains but PDB has %d – using positional mapping up to min",
            pdb_path.name,
            len(original_chain_ids),
            len(file_chains),
        )

    sequences: list[dict] = []
    n_chains = min(len(file_chains), len(original_chain_ids))

    for i in range(n_chains):
        entry = _process_chain_for_protenix(file_chains[i], original_chain_ids[i], stem)
        if entry is not None:
            sequences.append(entry)

    if not sequences:
        LOGGER.error("%s: no valid chains extracted", stem)
        return None

    return {"name": stem, "sequences": sequences}


def pdbs_to_protenix_json(
    pdb_paths: list[Path],
    output_json: Path,
) -> list[dict]:
    """Convert a list of PDB files to a Protenix input JSON file.

    Parameters
    ----------
    pdb_paths:
        PDB files to convert (each following the ``{pdbid}_{chains}.pdb`` naming convention).
    output_json:
        Destination JSON file.  Parent directories are created as needed.

    Returns
    -------
    list[dict]
        Protenix-format entries that were written.
    """
    entries: list[dict] = []
    failures: list[str] = []

    for pdb_path in pdb_paths:
        entry = pdb_to_protenix_entry(pdb_path)
        if entry is not None:
            entries.append(entry)
            LOGGER.info("  Converted: %s", pdb_path.name)
        else:
            failures.append(pdb_path.name)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=4)

    LOGGER.info(
        "Done: %d/%d entries written to %s",
        len(entries),
        len(pdb_paths),
        output_json,
    )
    if failures:
        LOGGER.warning("Skipped (%d): %s", len(failures), ", ".join(failures))

    return entries
