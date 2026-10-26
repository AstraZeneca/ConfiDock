"""Boltz-2 YAML input preparation utilities.

Provides helpers for parsing PDB files and generating Boltz-2 compatible
YAML input files from them, as well as MSA-mode staging utilities for the
prediction grid.

Typical use::

    from confidock.boltz.inference.inputs import pdb_to_boltz_yaml
    from confidock.boltz.inference.inputs import stage_yamls, load_sabdab_binders

    yaml_path = pdb_to_boltz_yaml(Path("data/7bbj_H_L_A.pdb"), Path("out/"))
"""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path

import yaml
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

LOGGER = logging.getLogger(__name__)


def _parse_chain_labels(chain_str: str, n_chains: int) -> list[str]:
    """Split *chain_str* into exactly *n_chains* labels.

    Strategies tried in order:

    1. Regex ``[A-Z][a-z]*`` groups — handles ``HcLcA`` → ``["Hc","Lc","A"]``.
    2. Character-by-character — handles ``GgA`` → ``["G","g","A"]``
       and ``hla`` → ``["h","l","a"]``.

    If neither matches *n_chains* exactly the char-by-char result is returned as
    a best-effort fallback (the caller handles length mismatches).
    """
    strategies: list[list[str]] = [
        re.findall(r"[A-Z][a-z]*", chain_str),  # uppercase-start groups: HcLcA→[Hc,Lc,A]
        list(chain_str),  # individual chars:       GgA→[G,g,A]
        [c for c in chain_str if c != "x"],  # drop 'x' separators:   axG→[a,G]
    ]
    for candidate in strategies:
        if len(candidate) == n_chains:
            return candidate
    return list(chain_str)  # best-effort


# Standard three-letter to one-letter amino acid mapping, including common
# non-standard residues that BioPython PPBuilder may miss.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Non-standard / selenomethionine
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


def chain_sequence(chain) -> str:
    """Return the one-letter amino acid sequence for a BioPython chain object.

    PPBuilder is tried first; if it yields nothing (e.g. non-standard residues),
    residues are iterated directly using :data:`THREE_TO_ONE`.
    """
    ppb = PPBuilder()
    seq = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(chain))

    if not seq:
        for residue in chain:
            if residue.get_id()[0] != " ":
                continue  # skip HETATM / water
            one = THREE_TO_ONE.get(residue.get_resname().strip())
            if one:
                seq += one

    return seq


def sanitise_pdb_text(raw: str) -> str:
    """Return PDB text with blank lines and truncated ATOM/HETATM records removed.

    Some files in the dataset are truncated (last line cut mid-record) or
    contain spurious blank lines that cause BioPython's strict parser to raise
    ``PDBConstructionException``.  A valid ATOM/HETATM record is at least 60
    characters wide; anything shorter is dropped.
    """
    kept: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue  # drop blank lines
        record = line[:6].strip()
        if record in {"ATOM", "HETATM"} and len(line) < 60:  # noqa: PLR2004 – truncated, drop it
            continue
        kept.append(line)
    return "\n".join(kept)


def pdb_to_boltz_yaml(pdb_path: Path, output_dir: Path) -> Path | None:
    """Convert a single PDB file to a Boltz-2 YAML input and return its path.

    The filename is expected to follow the ``{pdbid}_{chain1}[_{chain2}...].pdb``
    convention (e.g. ``7bbj_H_L_A.pdb``).  Chain IDs from the filename are used
    as the Boltz-2 sequence labels; the positional mapping is
    ``filename_chain[i] → file chain [A, B, C, ...][i]``.

    Returns ``None`` if the file cannot be processed (error is logged).
    """
    stem = pdb_path.stem  # e.g. "7bbj_H_L_A" or "1a14_HLN"
    parts = stem.split("_")

    # Determine PDB ID and optional chain label hint from filename
    _concat_chain_str: str | None = None
    if len(parts) >= 3:  # noqa: PLR2004
        # Underscore-separated format: "7bbj_H_L_A" → ["H", "L", "A"]
        pdb_id = parts[0]
        label_chain_ids: list[str] = parts[1:]
    elif len(parts) == 2 and len(parts[1]) > 1:  # noqa: PLR2004
        # Concatenated format: deferred until after PDB is loaded so we know
        # the actual chain count (needed to pick the right split strategy).
        pdb_id = parts[0]
        _concat_chain_str = parts[1]
        label_chain_ids = []  # filled in below
    else:
        LOGGER.warning(
            "Unexpected filename format (expected pdbid_chain1_chain2[...] or pdbid_HLA): %s",
            pdb_path.name,
        )
        return None

    raw_text = pdb_path.read_text(encoding="utf-8", errors="replace")
    clean_text = sanitise_pdb_text(raw_text)

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, io.StringIO(clean_text))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to parse %s: %s", pdb_path.name, exc)
        return None

    # Chains in the file use sequential IDs (A, B, C…); map them positionally
    # to the original chain labels from the filename.
    file_chains = list(structure[0].get_chains())

    # For concatenated format, choose the splitting strategy that matches the
    # actual chain count (regex groups, then char-by-char).
    if _concat_chain_str is not None:
        label_chain_ids = _parse_chain_labels(_concat_chain_str, len(file_chains))
    if len(file_chains) > len(label_chain_ids):
        LOGGER.warning(
            "%s: filename encodes %d chains but file has %d – using first %d labels",
            pdb_path.name,
            len(label_chain_ids),
            len(file_chains),
            len(label_chain_ids),
        )
        file_chains = file_chains[: len(label_chain_ids)]
    elif len(file_chains) < len(label_chain_ids):
        LOGGER.warning(
            "%s: filename encodes %d chains but file only has %d – "
            "labelling available chains with first %d original IDs",
            pdb_path.name,
            len(label_chain_ids),
            len(file_chains),
            len(file_chains),
        )
        label_chain_ids = label_chain_ids[: len(file_chains)]

    sequences: list[dict] = []
    for label_id, ch in zip(label_chain_ids, file_chains):
        seq = chain_sequence(ch)
        if not seq:
            LOGGER.warning(
                "Empty sequence for chain %s in %s – skipping chain",
                label_id,
                pdb_path.name,
            )
            continue

        sequences.append(
            {
                "protein": {
                    "id": label_id,
                    "sequence": seq,
                    "msa": "empty",
                }
            }
        )

    if not sequences:
        LOGGER.warning("No valid sequences extracted from %s – skipping file", pdb_path.name)
        return None

    yaml_content = {"version": 1, "sequences": sequences}
    output_path = output_dir / f"{stem}.yaml"

    with output_path.open("w", encoding="utf-8") as fh:
        yaml.dump(
            yaml_content,
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=10_000,  # prevent YAML wrapping long sequences
        )

    return output_path


# ---------------------------------------------------------------------------
# MSA-mode staging
# ---------------------------------------------------------------------------


def _is_binder_chain(chain_id: str) -> bool:
    """Fallback heuristic: heavy chains start with 'H'/'h', light with 'L'/'l'."""
    return bool(chain_id) and chain_id[0].lower() in {"h", "l"}


def _a3m_is_valid(path: str) -> bool:
    """Return True if the referenced .a3m file exists and is non-trivial."""
    p = Path(path)
    if not p.exists() or p.stat().st_size < 10:  # noqa: PLR2004
        return False
    text = p.read_text(encoding="utf-8", errors="replace")
    # A3M files use standard FASTA-style ">" headers.
    has_header = any(line.startswith((">", "#")) for line in text.splitlines())
    has_seq = any(
        not line.startswith((">", "#")) and line.strip() and not any(c.isdigit() for c in line.strip())
        for line in text.splitlines()
    )
    return has_header and has_seq


def _process_sequence_entry(
    entry: dict,
    src_name: str,
    msa_mode: str,
    binder_set: set[str] | None,
) -> None:
    """Process a single sequence entry for MSA mode modification.

    Modifies the entry dict in-place, setting ``msa: empty`` when appropriate.
    """
    p = entry.get("protein")
    if p is None:
        return

    chain_id: str = str(p.get("id", ""))
    msa_val = p.get("msa", "empty")

    is_binder = chain_id in binder_set if binder_set is not None else _is_binder_chain(chain_id)
    strip = msa_mode == "no_msa" or (msa_mode == "antigen_only" and is_binder)

    if not strip and msa_val != "empty" and not _a3m_is_valid(str(msa_val)):
        LOGGER.warning(
            "Invalid a3m for chain %s in %s – falling back to msa: empty",
            chain_id,
            src_name,
        )
        strip = True

    if strip:
        p["msa"] = "empty"


def load_sabdab_binders(tsv_path: Path) -> dict[str, set[str]]:
    """Load SAbDab summary TSV and return mapping: pdb_id -> set of antibody chain IDs.

    The TSV is the standard SAbDab summary file with at minimum the columns
    ``pdb``, ``Hchain``, and ``Lchain``.

    Parameters
    ----------
    tsv_path:
        Path to the SAbDab summary ``.tsv`` file.

    Returns
    -------
    dict
        Maps lower-case PDB ID to the set of antibody chain IDs (heavy + light)
        across all antibodies in that entry.
    """
    binders: dict[str, set[str]] = {}
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pdb = row["pdb"].lower()
            if pdb not in binders:
                binders[pdb] = set()
            for field in ("Hchain", "Lchain"):
                v = row.get(field, "NA").strip()
                if v and v not in ("NA", ""):
                    binders[pdb].add(v)
    LOGGER.info("Loaded SAbDab binder chains for %d PDB entries from %s", len(binders), tsv_path)
    return binders


def stage_yamls(
    yaml_files: list[Path],
    staging_dir: Path,
    msa_mode: str,
    sabdab_binders: dict[str, set[str]] | None = None,
) -> Path:
    """Write MSA-mode-modified copies of *yaml_files* into *staging_dir*.

    full_msa
        Copy as-is, but fall back to ``msa: empty`` for invalid/missing a3m files.
    antigen_only
        Strip MSA from antibody (binder) chains; antigen chains keep their MSA.
    no_msa
        Strip MSA from all chains (Boltz single-sequence mode).

    Parameters
    ----------
    yaml_files:
        Source YAML input files.
    staging_dir:
        Destination directory for modified copies.
    msa_mode:
        One of ``"full_msa"``, ``"antigen_only"``, ``"no_msa"``.
    sabdab_binders:
        Optional mapping from lower-case PDB ID to the set of antibody chain
        IDs, as returned by :func:`load_sabdab_binders`.  When provided it
        takes priority over the H/L name heuristic for ``antigen_only`` mode.

    Returns
    -------
    Path
        *staging_dir* (created if it did not exist).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for src in yaml_files:
        data = yaml.safe_load(src.read_text(encoding="utf-8"))
        pdb_id = src.stem.split("_")[0].lower()
        binder_set: set[str] | None = sabdab_binders.get(pdb_id) if sabdab_binders is not None else None

        for entry in data.get("sequences", []):
            _process_sequence_entry(entry, src.name, msa_mode, binder_set)

        dst = staging_dir / src.name
        dst.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False, width=10_000),
            encoding="utf-8",
        )
    return staging_dir
