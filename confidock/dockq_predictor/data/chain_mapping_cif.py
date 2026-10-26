"""Chain identification for CIF files using JSON metadata."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Directory holding the SAbDab metadata JSONs (h_seq/l_seq + h_cdr/l_cdr) used for
# chain typing and CDR labelling. Set CONFIDOCK_SABDAB_JSON_DIR to point at it. Any
# directory of sidecar JSONs works, so structures with no SAbDab entry (e.g. de-novo
# designs, generated decoy grids) can be labelled from a locally-built one -- build it
# with workflows/scripts/inference/build_cdr_sidecar_json.py. There is no default: the
# path is site-specific, so an unset variable means "no CDR metadata available".
_DEFAULT_SABDAB_JSON_DIR = ""

# 3-letter -> 1-letter residue code, for reconstructing a chain's sequence from
# the structure so it can be aligned to the JSON reference sequence. Unknown /
# non-standard residues map to "X" (BLOSUM62 has an X row, so they still align).
_THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def _align_ref_positions_to_structure(ref_seq: str, struct_seq: str) -> Dict[int, int]:
    """Map 1-based reference-sequence positions to structure residue indices.

    Globally aligns the JSON reference sequence (whose CDR positions are defined)
    to the structure's actual residue sequence, then returns
    ``{ref_pos_1based: struct_index_0based}`` for every position that aligns to a
    real residue (gaps are simply absent from the map). This tolerates terminal
    truncation and internal indels between the stored sequence and the modeled
    structure -- the former all-or-nothing exact-length check discarded a chain's
    CDR labels whenever the two differed by even one residue.
    """
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.mode = "global"

    # PairwiseAligner needs valid substitution-matrix symbols; map anything the
    # matrix doesn't know (e.g. "X" is fine, stray chars are not) is left as-is
    # since BLOSUM62 includes B/Z/X/*, and we already coerce unknowns to "X".
    alignment = aligner.align(ref_seq, struct_seq)[0]

    pos_map: Dict[int, int] = {}
    # alignment.aligned -> (ref_blocks, struct_blocks); each block is a matched
    # [start, end) run with equal length in ref and struct, so positions inside a
    # block correspond one-to-one.
    ref_blocks, struct_blocks = alignment.aligned
    for (r0, r1), (s0, s1) in zip(ref_blocks, struct_blocks):
        for offset in range(r1 - r0):
            pos_map[(r0 + offset) + 1] = s0 + offset  # 1-based ref -> 0-based struct
    return pos_map


def sabdab_json_key(pdb_path: str) -> str:
    """Return the JSON basename (no ``.json``) that ``pdb_path``'s metadata is looked up under.

    One decoy's metadata is shared by every pose of the same complex, so the sample
    suffix is stripped off the filename stem:

    * ``_sample_N`` (protenix, esmfold2): ``1i9r_HLA_sample_0.cif`` -> ``1i9r_HLA``
    * ``_seedNN_sampleN`` (boltz2): ``7jtf_ABE_seed42_sample0.pdb`` -> ``7jtf_ABE``

    Exposed rather than inlined into ``load_sabdab_json`` because
    ``workflows/scripts/inference/build_cdr_sidecar_json.py`` has to name the files it
    writes *exactly* as this will look them up. A private copy of the rule in the
    writer would let the two drift, and the symptom would be silent: no JSON found,
    CDR labels all zero, no error.
    """
    filename = Path(pdb_path).stem
    if "_sample_" in filename:
        return filename.split("_sample_")[0]
    if "_seed" in filename and "_sample" in filename:
        return filename.split("_seed")[0]
    return filename


def sabdab_json_search_dirs() -> List[str]:
    """Directories searched for metadata JSONs, in order.

    ``CONFIDOCK_SABDAB_JSON_DIR`` (``os.pathsep``-separated) is searched first, so a
    locally-built sidecar directory adds coverage for complexes absent from a SAbDab
    snapshot without shadowing the ones in it. ``_DEFAULT_SABDAB_JSON_DIR`` is appended
    only when set; empty means the variable alone decides where to look.
    """
    dirs = [d for d in os.environ.get("CONFIDOCK_SABDAB_JSON_DIR", "").split(os.pathsep) if d]
    if _DEFAULT_SABDAB_JSON_DIR:
        dirs.append(_DEFAULT_SABDAB_JSON_DIR)
    return dirs


def find_sabdab_json(pdb_path: str) -> Optional[Path]:
    """Path of the metadata JSON for ``pdb_path``, or None if no directory holds one."""
    key = sabdab_json_key(pdb_path)
    for directory in sabdab_json_search_dirs():
        json_path = Path(directory) / f"{key}.json"
        if json_path.exists():
            return json_path
    return None


def load_sabdab_json(pdb_path: str) -> Optional[Dict]:
    """
    Load SAbDab JSON metadata for a structure.

    Parameters
    ----------
    pdb_path : str
        Path to PDB/CIF file

    Returns
    -------
    Optional[Dict]
        JSON metadata with h_seq, l_seq, t_seq, or None if no matching JSON
        exists in any search directory.
    """
    json_path = find_sabdab_json(pdb_path)
    if json_path is None:
        return None
    with open(json_path) as f:
        return json.load(f)


def identify_chains_from_json(pdb_path: str, chain_lengths: Dict[str, int]) -> Dict[str, str]:
    """
    Identify chain types using JSON metadata and chain lengths.

    For CIF files with numeric chain IDs, matches chains by length
    to h_seq, l_seq, t_seq from JSON metadata.

    Parameters
    ----------
    pdb_path : str
        Path to PDB/CIF file
    chain_lengths : Dict[str, int]
        Mapping of chain_id to number of residues

    Returns
    -------
    Dict[str, str]
        Mapping of chain_id to chain_type (VH, VL, or antigen)
    """
    # Load JSON metadata
    json_data = load_sabdab_json(pdb_path)

    if json_data is None:
        # No JSON found, can't determine chain types
        return {}

    h_len = len(json_data.get("h_seq", ""))
    l_len = len(json_data.get("l_seq", ""))
    t_len = len(json_data.get("t_seq", ""))

    chain_types = {}

    # Match chains by length
    for chain_id, length in chain_lengths.items():
        # Allow ±2 residue tolerance for matching
        if abs(length - h_len) <= 2:
            chain_types[chain_id] = "VH"
        elif abs(length - l_len) <= 2:
            chain_types[chain_id] = "VL"
        elif abs(length - t_len) <= 2:
            chain_types[chain_id] = "antigen"
        else:
            # If no good match, default to antigen
            chain_types[chain_id] = "antigen"

    return chain_types


# (target chain_type, seq field, cdr field, label offset): heavy H1/H2/H3 -> 1/2/3,
# light L1/L2/L3 -> 4/5/6.
_CDR_CHAIN_SPECS = [
    ("VH", "h_seq", "h_cdr", 0),
    ("VL", "l_seq", "l_cdr", 3),
]


def _cdr_positions_to_labels(cdr: Dict, offset: int) -> Dict[int, int]:
    """Flatten a JSON ``h_cdr``/``l_cdr`` block to {1-based seq position: label}."""
    pos_to_label: Dict[int, int] = {}
    for k, positions in cdr.items():
        label = int(k) + offset
        for p in positions:
            pos_to_label[int(p)] = label
    return pos_to_label


def _iter_chains_of_type(structure, chain_types: Dict[str, str], target_type: str):
    """Yield every chain in the structure whose identified type is ``target_type``."""
    for model in structure:
        for chain in model:
            if chain_types.get(chain.id) == target_type:
                yield chain


def _chain_residue_keys(chain) -> Tuple[List[Tuple[str, int, str]], str]:
    """Return a chain's standard residues in order as (label keys, 1-letter sequence)."""
    res_keys: List[Tuple[str, int, str]] = []
    struct_chars: List[str] = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        res_keys.append((chain.id, residue.id[1], residue.id[2] if len(residue.id) > 2 else ""))
        struct_chars.append(_THREE_TO_ONE.get(residue.resname, "X"))
    return res_keys, "".join(struct_chars)


def _ref_to_struct_positions(
    seq: str, struct_seq: str, n_res: int, chain_id: str, seq_key: str
) -> Optional[Dict[int, int]]:
    """Map 1-based reference positions to structure indices, or None if unmappable."""
    if n_res == len(seq):
        # Exact length: the fast path, a straight 1:1 position mapping
        # (identical result to alignment, no aligner call needed).
        return {pos1: pos1 - 1 for pos1 in range(1, len(seq) + 1)}

    # Length differs (terminal truncation / indel). Align the reference sequence
    # to the structure and map CDR positions through the alignment instead of
    # discarding the whole chain.
    try:
        return _align_ref_positions_to_structure(seq, struct_seq)
    except Exception as e:
        print(
            f"Warning: CDR labeling: chain {chain_id} align failed "
            f"({n_res} residues vs {seq_key} length {len(seq)}): {e}; "
            f"leaving this chain's CDR labels 0"
        )
        return None


def _label_chain_cdrs(chain, seq: str, seq_key: str, pos_to_label: Dict[int, int]) -> Dict:
    """CDR labels for one antibody chain: {(chain_id, res_num, icode): label}."""
    # Structure residues in order, with their 1-letter codes, so we can align to
    # the JSON reference sequence rather than assume 1:1.
    res_keys, struct_seq = _chain_residue_keys(chain)
    ref_to_struct = _ref_to_struct_positions(seq, struct_seq, len(res_keys), chain.id, seq_key)
    if ref_to_struct is None:
        return {}

    labels: Dict = {}
    for pos1, label in pos_to_label.items():
        struct_idx = ref_to_struct.get(pos1)
        if struct_idx is not None and 0 <= struct_idx < len(res_keys):
            labels[res_keys[struct_idx]] = label
    return labels


def compute_cdr_labels_from_json(pdb_path: str, structure, chain_types: Dict[str, str]) -> Dict:
    """Assign CDR labels to antibody residues using SAbDab JSON metadata.

    The JSON ``h_cdr`` / ``l_cdr`` fields are ``{"1": [...], "2": [...], "3": [...]}``
    where the lists are **1-based positions into ``h_seq`` / ``l_seq``**. CDR
    identity is a property of the sequence, so it is identical for the real and
    the predicted structure (the latter is folded from the same sequence). We map
    those positions onto each identified VH/VL chain **by residue order** -- no
    residue-number or ANARCI-numbering assumption, which matters because predicted
    structures are numbered 1..N rather than Chothia-numbered.

    Parameters
    ----------
    pdb_path : str
        Path to the structure (used only to locate the matching JSON).
    structure : Bio.PDB structure
        Parsed structure, iterated to walk each chain's residues in order.
    chain_types : Dict[str, str]
        chain_id -> "VH" / "VL" / "antigen".

    Returns
    -------
    Dict[Tuple[str, int, str], int]
        {(chain_id, res_num, insertion_code): label}, label in
        0=none, 1=H1, 2=H2, 3=H3, 4=L1, 5=L2, 6=L3. Only CDR residues are keyed;
        callers should default missing keys to 0. If a chain's standard-residue
        count does not match its sequence length, that chain is skipped (logged)
        rather than mislabeled.
    """
    labels: Dict = {}
    json_data = load_sabdab_json(pdb_path)
    if json_data is None:
        return labels

    for target_type, seq_key, cdr_key, offset in _CDR_CHAIN_SPECS:
        seq = json_data.get(seq_key, "") or ""
        cdr = json_data.get(cdr_key, {}) or {}
        if not seq or not cdr:
            continue

        # 1-based sequence position -> CDR label.
        pos_to_label = _cdr_positions_to_labels(cdr, offset)
        for chain in _iter_chains_of_type(structure, chain_types, target_type):
            labels.update(_label_chain_cdrs(chain, seq, seq_key, pos_to_label))

    return labels
