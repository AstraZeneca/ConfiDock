"""MSA utilities for Boltz-2 inputs.

Provides helpers for:
- Naming and validating ``.a3m`` cache files
- Decoding Boltz MSA ``.npz`` files to a3m text
- Resolving sequences from a local npz directory before falling back to
  the online ColabFold API
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

try:
    from boltz.data import const as boltz_const
except ModuleNotFoundError:
    boltz_const = None

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token-id ↔ amino-acid-letter mapping
# ---------------------------------------------------------------------------
if boltz_const is not None:
    _TOKENS: list[str] = boltz_const.tokens
    _PROT_TOKEN_TO_LETTER: dict[str, str] = {v: k for k, v in boltz_const.prot_letter_to_token.items()}
    _PROT_TOKEN_TO_LETTER["UNK"] = "X"
else:
    _TOKENS = []
    _PROT_TOKEN_TO_LETTER = {}


def _ensure_boltz_token_mappings() -> None:
    """Ensure boltz token mappings are available for npz decoding."""
    if not _TOKENS or not _PROT_TOKEN_TO_LETTER:
        raise ModuleNotFoundError(
            "boltz is required to decode MSA npz files. Install project dependencies including 'boltz'."
        )


# ---------------------------------------------------------------------------
# General a3m helpers
# ---------------------------------------------------------------------------


def a3m_path(msa_dir: Path, sequence: str) -> Path:
    """Stable filename for a sequence's ``.a3m`` file (SHA-256 hex prefix)."""
    digest = hashlib.sha256(sequence.encode()).hexdigest()[:16]
    return msa_dir / f"{digest}.a3m"


def is_valid_a3m(text: str) -> bool:
    """Return True if *text* looks like a non-trivial a3m file.

    Checks that the file has at least one header line ('>' or '#') and at
    least one non-empty sequence line containing only valid amino acid /
    gap characters (no digits or other junk).
    """
    if not text or len(text.strip()) < 10:  # noqa: PLR2004
        return False
    has_header = False
    has_sequence = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith((">", "#")):
            has_header = True
        elif has_header and line and not any(c.isdigit() for c in line):
            has_sequence = True
            break
    return has_header and has_sequence


# ---------------------------------------------------------------------------
# Local npz → a3m helpers
# ---------------------------------------------------------------------------


def decode_query_seq(npz_path: Path) -> str:
    """Return the query (first) sequence from a Boltz MSA npz as a plain string."""
    f = np.load(npz_path)
    sequences = f["sequences"]
    residues = f["residues"]
    if len(sequences) == 0:
        return ""
    _ensure_boltz_token_mappings()
    s = sequences[0]
    letters = []
    for r in residues[s["res_start"] : s["res_end"]]:
        tok = _TOKENS[r["res_type"]]
        if tok == "-":
            continue
        letters.append(_PROT_TOKEN_TO_LETTER.get(tok, "X"))
    return "".join(letters)


def npz_to_a3m_text(
    npz_path: Path,
    start: int = 0,
    length: int | None = None,
    max_seqs: int | None = None,
) -> str:
    """Convert a Boltz MSA npz to a3m text and return it as a string.

    Parameters
    ----------
    start:
        Starting query position (0-indexed) for cropped sequences.
    length:
        Number of query columns to keep. ``None`` means keep all.
    max_seqs:
        Maximum number of sequences to write (including the query).
        ``None`` means write all.

    Insertion characters are written as lowercase ``'a'``; Boltz counts and
    discards the actual letter so this round-trips losslessly.
    """
    _ensure_boltz_token_mappings()
    f = np.load(npz_path)
    sequences = f["sequences"]
    residues_arr = f["residues"]
    deletions_arr = f["deletions"]

    query_len = int(sequences[0]["res_end"]) - int(sequences[0]["res_start"])
    if length is None:
        length = query_len - start
    end = start + length

    if max_seqs is not None:
        sequences = sequences[:max_seqs]

    lines: list[str] = []
    for i, seq_rec in enumerate(sequences):
        base = int(seq_rec["res_start"])
        res_start = base + start
        res_end = base + end

        del_start_orig = int(seq_rec["del_start"])
        del_end_orig = int(seq_rec["del_end"])

        dels: dict[int, int] = {
            int(d["res_idx"]) - start: int(d["deletion"])
            for d in deletions_arr[del_start_orig:del_end_orig]
            if start <= int(d["res_idx"]) < end
        }

        letters: list[str] = []
        for r in residues_arr[res_start:res_end]:
            tok = _TOKENS[r["res_type"]]
            letters.append("-" if tok == "-" else _PROT_TOKEN_TO_LETTER.get(tok, "X"))

        seq_chars: list[str] = []
        for j, letter in enumerate(letters):
            if j in dels:
                seq_chars.append("a" * dels[j])
            seq_chars.append(letter)

        lines.append(f">seq_{i}")
        lines.append("".join(seq_chars))

    return "\n".join(lines) + "\n"


def build_pdb_index(
    pdb_ids: set[str],
    npz_msa_dir: Path,
) -> dict[str, list[tuple[Path, str]]]:
    """For each pdb_id, return a list of (npz_path, decoded_query_seq)."""
    index: dict[str, list[tuple[Path, str]]] = {}
    for pdb_id in sorted(pdb_ids):
        npz_files = sorted(npz_msa_dir.glob(f"{pdb_id}_*.npz"))
        if not npz_files:
            continue
        entries: list[tuple[Path, str]] = []
        for npz in npz_files:
            try:
                query_seq = decode_query_seq(npz)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Could not decode %s: %s – skipping", npz.name, exc)
                continue
            if query_seq:
                entries.append((npz, query_seq))
        if entries:
            index[pdb_id] = entries
    return index


def find_npz(entries: list[tuple[Path, str]], seq: str) -> tuple[Path, int] | None:
    """Return (npz_path, start_offset) whose decoded query contains *seq*.

    Tries exact match first, then substring search (seq is a crop of the
    full PDB chain used when building the MSA).
    """
    for npz, msa_seq in entries:
        if seq == msa_seq:
            return npz, 0
    for npz, msa_seq in entries:
        idx = msa_seq.find(seq)
        if idx != -1:
            return npz, idx
    return None


def resolve_from_local_npz(
    sequences: list[str],
    seq_to_pdb_id: dict[str, str],
    npz_msa_dir: Path,
    msa_dir: Path,
    max_seqs: int | None = 4096,
) -> set[str]:
    """Try to resolve *sequences* from local npz files.

    Matches each sequence to an npz file by PDB ID and substring search,
    converts it to a3m, and writes to ``a3m_path(msa_dir, seq)``.
    Returns the set of sequences successfully resolved.
    """
    pdb_ids = {seq_to_pdb_id[s] for s in sequences if s in seq_to_pdb_id}
    if not pdb_ids:
        return set()

    LOGGER.info("Building local npz index for %d PDB IDs…", len(pdb_ids))
    index = build_pdb_index(pdb_ids, npz_msa_dir)
    LOGGER.info("Found npz coverage for %d/%d PDB IDs", len(index), len(pdb_ids))

    resolved: set[str] = set()
    for seq in sequences:
        out = a3m_path(msa_dir, seq)
        if out.exists() and out.stat().st_size >= 10:  # noqa: PLR2004
            resolved.add(seq)
            continue

        pdb_id = seq_to_pdb_id.get(seq)
        if pdb_id is None:
            continue
        entries = index.get(pdb_id, [])
        result = find_npz(entries, seq)
        if result is None:
            continue

        npz_path, start = result
        try:
            a3m_text = npz_to_a3m_text(npz_path, start=start, length=len(seq), max_seqs=max_seqs)
            if is_valid_a3m(a3m_text):
                out.write_text(a3m_text)
                resolved.add(seq)
                LOGGER.debug("Resolved %.20s… from local %s", seq, npz_path.name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to convert %s: %s", npz_path.name, exc)

    return resolved
