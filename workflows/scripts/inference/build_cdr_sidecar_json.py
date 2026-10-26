#!/usr/bin/env python
"""Generate SAbDab-style CDR metadata JSONs for structures with no SAbDab entry.

    poetry run python workflows/scripts/inference/build_cdr_sidecar_json.py \
        --structures '/data/poses/*.cif' \
        --out-dir data/dockq_predictor/cdr_json

Hand the resulting directory to ``predict_dockq.py --cdr-json-dir``, or export it as
``CONFIDOCK_SABDAB_JSON_DIR`` for anything else that featurises.

WHY THIS EXISTS
---------------
CDR labels in the DockQ feature pipeline come from exactly one place:
``chain_mapping_cif.compute_cdr_labels_from_json``, which reads ``h_seq``/``l_seq``
plus 1-based ``h_cdr``/``l_cdr`` position lists out of a metadata JSON located by
complex name. A complex that is not in the SAbDab snapshot -- a de-novo design, an
internal target, a generated decoy grid -- finds no JSON, so ``cdr_label`` comes back
all-zero and every CDR-dependent feature block is silently dead: node ``cdr`` (6 of its
7 columns), ``is_cdr``, ``is_cdr_h3``, and edge ``source_is_cdr`` / ``target_is_cdr`` /
``source_is_cdr_h3`` / ``target_is_cdr_h3``. That is 8 of 18 node columns, all of which
were live during training. Nothing raises; the model just scores a worse-informed graph.

Rather than reimplement CDR assignment, this script produces the JSON that the existing
(already-validated, alignment-based) loader expects, deriving CDRs from ANARCI's Chothia
numbering. The structural pipeline is untouched.

CDR definitions are the Chothia ranges: H1 26-32, H2 52-56, H3 95-102, L1 24-34,
L2 50-56, L3 89-97. ANARCI supplies the Chothia *number* per residue, so these ranges
apply correctly even though predicted structures are numbered 1..N. A naive
residue-number lookup cannot be used on this data for that same reason -- which is why
the training path resolves CDRs by sequence position instead.

OUTPUT NAMING
-------------
Files are named with ``chain_mapping_cif.sabdab_json_key`` -- the same function the
loader looks them up with, imported rather than reimplemented so the two cannot drift.
It strips ``_sample_N`` / ``_seedNN_sampleN``, so ``7y8j_HLA_sample_0.cif`` is written
as ``7y8j_HLA.json``: one JSON per complex, not per pose. Poses of one complex share a
sequence, so ANARCI runs once per complex (and results are memoised per sequence, so
complexes sharing an antibody cost nothing extra).

Requires the ``ANARCI`` executable on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from confidock.dockq_predictor.data.chain_mapping_cif import sabdab_json_key  # noqa: E402

# Chothia CDR ranges. The sole copy in the repo: the training path takes CDRs from
# JSON sequence positions, so nothing else hardcodes these bounds.
CHOTHIA_CDR = {
    "H": {1: (26, 32), 2: (52, 56), 3: (95, 102)},
    "L": {1: (24, 34), 2: (50, 56), 3: (89, 97)},
}

# ANARCI's hit line: #|species|chain_type|e-value|score|seqstart_index|seqend_index|
_HIT_RE = re.compile(r"^#\|([^|]*)\|([^|]*)\|")

# ANARCI domain types collapsed onto the two classes CHOTHIA_CDR is keyed by: kappa (K)
# and lambda (L) light chains are one class here, because their CDR ranges are the same.
# A type that is absent -- ANARCI also reports TCR domains (A/B/G/D) -- is not an Fv the
# loader's schema can express, and the chain is treated as antigen.
_CHAIN_CLASSES = {"H": "H", "L": "L", "K": "L"}

# Structure suffixes worth trying, mapped to the parser that reads them.
_SUFFIX_PARSERS = {".cif": "cif", ".mmcif": "cif", ".pdb": "pdb", ".ent": "pdb"}

# What happened to one complex. Named rather than inline strings because they are written
# by write_sidecar and read by print_summary, and a typo in either would silently
# under-report -- a Counter has no missing keys.
EXISTING, SKIPPED, WRITTEN, MULTI_FV = "existing", "skipped", "written", "written_multi_fv"


def iter_structure_files(patterns: List[str]) -> List[Path]:
    """Expand globs / directories / plain paths into a sorted list of structure files.

    A bare directory is walked recursively, so ``--structures /data/poses`` behaves the
    way anyone would expect of a prediction tree with one subdirectory per complex.
    """
    found: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            for suffix in _SUFFIX_PARSERS:
                found.extend(path.rglob(f"*{suffix}"))
        elif path.exists():
            found.append(path)
        else:
            # Not an existing path: treat it as a glob. Path.glob needs a split root,
            # so go through the module-level glob instead.
            from glob import glob

            found.extend(Path(p) for p in glob(pattern, recursive=True))

    keep = [p for p in found if p.suffix.lower() in _SUFFIX_PARSERS]
    # Deduplicate while keeping determinism -- overlapping globs are easy to pass.
    return sorted(set(keep))


def parse_structure(path: Path):
    """Parse a PDB or CIF file, chosen by suffix. Returns None if it will not parse."""
    kind = _SUFFIX_PARSERS.get(path.suffix.lower())
    parser = MMCIFParser(QUIET=True) if kind == "cif" else PDBParser(QUIET=True)
    try:
        return parser.get_structure(path.stem, str(path))
    except Exception as exc:  # noqa: BLE001 -- one bad file must not stop the batch
        print(f"  ! {path.name}: cannot parse ({type(exc).__name__}: {exc})")
        return None


def residue_sequence(chain) -> str:
    """Return the 1-letter sequence of a chain's standard residues, in order."""
    return "".join(protein_letters_3to1.get(r.resname, "X") for r in chain if r.id[0] == " ")


def _run_anarci(seq: str) -> Optional[str]:
    """Run ANARCI on one sequence and return its stdout, or None if it produced none.

    None covers every "this chain has no numbering" outcome -- a non-zero exit, empty
    output, or a timeout -- which the caller reads as "not an antibody chain". A missing
    executable is different in kind and stops the run.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as fh:
        fh.write(f">q\n{seq}\n")
        fasta_path = fh.name
    try:
        result = subprocess.run(
            ["ANARCI", "-i", fasta_path, "--scheme", "chothia"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        raise SystemExit(
            "The ANARCI executable is not on PATH. It is what supplies Chothia "
            "numbering here; without it there is no way to place CDRs on a sequence "
            "that has no SAbDab entry. Install ANARCI (or activate the environment "
            "that has it) and re-run."
        )
    except subprocess.TimeoutExpired:
        print("  ! ANARCI timed out on a chain; treating it as antigen")
        return None
    finally:
        os.unlink(fasta_path)

    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def _seqstart_from_hit(line: str) -> int:
    """Read ``seqstart_index`` out of ANARCI's hit line, defaulting to 0.

    It is the 5th field after the leading '#', and it offsets the numbered block back
    onto the full input sequence -- ANARCI numbers the domain it found, which need not
    start at the chain's first residue.
    """
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    if len(parts) < 6:
        return 0
    try:
        return int(parts[5])
    except ValueError:
        return 0


def _hit_chain_class(stdout: str) -> Tuple[Optional[str], int]:
    """Return (chain class, sequence offset) from ANARCI's most-significant HMM hit.

    The class is ANARCI's domain type collapsed to "H" (heavy) or "L" (light: L or K).
    None means no immunoglobulin domain was reported -- an antigen chain.
    """
    for line in stdout.splitlines():
        match = _HIT_RE.match(line)
        if match is None or match.group(1) == "species":  # the header row, not a hit
            continue
        ctype = match.group(2).strip().upper()
        return _CHAIN_CLASSES.get(ctype), _seqstart_from_hit(line)
    return None, 0


def _chothia_number(line: str) -> Optional[int]:
    """Return the Chothia number of one numbered row, or None if it holds no residue.

    Numbered rows look like ``H 26       G`` (chain, number, [insertion], residue).
    Headers, the ``//`` terminator and gap rows -- whose residue column is '-', so they
    consume a Chothia number without consuming a sequence position -- all return None.
    """
    if not line or line.startswith("#") or line.startswith("//"):
        return None
    parts = line.split()
    if len(parts) < 3 or parts[-1] == "-":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _numbered_positions(stdout: str, seqstart: int) -> Dict[int, int]:
    """Walk ANARCI's numbered rows into {0-based index into the sequence: Chothia number}."""
    pos_map: Dict[int, int] = {}
    cursor = seqstart
    for line in stdout.splitlines():
        number = _chothia_number(line)
        if number is None:
            continue
        pos_map[cursor] = number
        cursor += 1
    return pos_map


def anarci_number(seq: str) -> Tuple[Optional[str], Dict[int, int]]:
    """Run ANARCI on a sequence; return (chain_class, seq_index -> chothia_num).

    ``chain_class`` is ANARCI's domain type collapsed to "H" (heavy) or "L" (light: L or
    K), or None when no immunoglobulin domain is found -- i.e. an antigen chain. The map
    is 0-based index into ``seq`` -> Chothia number, built by walking ANARCI's numbered
    output and skipping its gap ('-') rows, which consume a number but no sequence
    position.
    """
    stdout = _run_anarci(seq)
    if stdout is None:
        return None, {}

    chain_class, seqstart = _hit_chain_class(stdout)
    if chain_class is None:
        return None, {}

    return chain_class, _numbered_positions(stdout, seqstart)


def cdr_positions(pos_map: Dict[int, int], chain_class: str) -> Dict[str, List[int]]:
    """Map index->Chothia numbers into 1-based CDR position lists per CDR id.

    Returns ``{"1": [...], "2": [...], "3": [...]}`` with positions expressed 1-based
    into the chain's own sequence, which is the convention
    ``compute_cdr_labels_from_json`` expects for ``h_cdr``/``l_cdr``.
    """
    out: Dict[str, List[int]] = {"1": [], "2": [], "3": []}
    for idx, chothia in sorted(pos_map.items()):
        for cdr_id, (lo, hi) in CHOTHIA_CDR[chain_class].items():
            if lo <= chothia <= hi:
                out[str(cdr_id)].append(idx + 1)  # 1-based into the sequence
                break
    return {k: v for k, v in out.items() if v}


class _AnarciCache:
    """Memoise ANARCI by sequence.

    Complexes that share an antibody -- a panel against one target, or a design series --
    would otherwise pay for the same numbering once per complex, and ANARCI is by far
    the slowest step here.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[Optional[str], Dict[int, int]]] = {}
        self.calls = 0

    def number(self, seq: str) -> Tuple[Optional[str], Dict[int, int]]:
        """Return the cached numbering for ``seq``, computing it on first sight."""
        if seq not in self._cache:
            self.calls += 1
            self._cache[seq] = anarci_number(seq)
        return self._cache[seq]


def first_model_sequences(structure) -> List[str]:
    """Return the non-empty chain sequences of the structure's first model, in order.

    First model only: poses are single-model, and an NMR ensemble's extra models are the
    same chains again. Empty chains (waters, ligands, anything with no standard residue)
    drop out here so the caller never asks ANARCI to number nothing.
    """
    for model in structure:
        return [seq for seq in (residue_sequence(chain) for chain in model) if seq]
    return []


def record_chain(payload: Dict, chain_class: Optional[str], seq: str, pos_map: Dict[int, int]) -> bool:
    """Add one typed chain to the payload; return False if the schema cannot hold it.

    False means a second heavy or second light chain: the loader keys CDRs off a single
    ``h_seq``/``l_seq`` pair, so a second Fv has nowhere to go and is counted by the
    caller instead of being silently merged into the first.
    """
    if chain_class is None:
        # Antigen. t_seq is unused for CDR labelling but kept so the JSON is a faithful
        # SAbDab-shaped record (chain typing does read its length).
        payload.setdefault("t_seq", seq)
        return True

    prefix = chain_class.lower()  # "H"/"L" -> the JSON's h_*/l_* key pair
    if f"{prefix}_seq" in payload:
        return False
    payload[f"{prefix}_seq"] = seq
    payload[f"{prefix}_cdr"] = cdr_positions(pos_map, chain_class)
    return True


def build_payload(structure, anarci: _AnarciCache) -> Optional[Dict]:
    """Build one complex's SAbDab-shaped metadata, or None if it has no heavy chain.

    Only the first heavy and first light chain are recorded: the loader keys CDRs off
    ``h_seq``/``l_seq``, so a single Fv is what the schema can express. A complex with
    two distinct Fvs gets the first of each, and the second pair's CDRs stay unlabelled
    -- reported by the caller rather than passed off as complete.
    """
    payload: Dict = {"source": "anarci_chothia_sidecar"}
    extra_fv = 0

    for seq in first_model_sequences(structure):
        chain_class, pos_map = anarci.number(seq)
        if not record_chain(payload, chain_class, seq, pos_map):
            extra_fv += 1

    if "h_seq" not in payload:
        return None
    if extra_fv:
        payload["extra_antibody_chains"] = extra_fv
    return payload


def group_by_key(files: List[Path]) -> "OrderedDict[str, List[Path]]":
    """Group structure files by the JSON key the loader will look them up under."""
    groups: "OrderedDict[str, List[Path]]" = OrderedDict()
    for path in files:
        groups.setdefault(sabdab_json_key(str(path)), []).append(path)
    return groups


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--structures",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Structure files, globs, or directories (walked recursively) to derive CDRs from",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write the JSONs into")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite JSONs that already exist (default: leave them and skip the ANARCI run)",
    )
    return parser.parse_args(argv)


def write_sidecar(key: str, paths: List[Path], out_path: Path, anarci: _AnarciCache, overwrite: bool) -> str:
    """Write one complex's JSON and return the outcome to tally it under.

    One of ``EXISTING`` (left alone), ``SKIPPED`` (no heavy chain, so nothing to label),
    ``WRITTEN``, or ``MULTI_FV`` -- a write whose complex had more Fvs than the loader's
    schema can express, which the summary calls out separately.
    """
    if out_path.exists() and not overwrite:
        return EXISTING

    # Any pose of the complex will do: they are folded from the same sequences, and CDR
    # identity is a property of the sequence, not the coordinates.
    structure = parse_structure(paths[0])
    payload = build_payload(structure, anarci) if structure is not None else None
    if payload is None:
        print(f"  - {key:24s} no antibody heavy chain found -- skipped")
        return SKIPPED

    h_n = sum(len(v) for v in payload.get("h_cdr", {}).values())
    l_n = sum(len(v) for v in payload.get("l_cdr", {}).values())
    out_path.write_text(json.dumps(payload))
    n_extra = payload.get("extra_antibody_chains", 0)
    extra = f"  (+{n_extra} unlabelled Fv chain(s))" if n_extra else ""
    print(f"  + {key:24s} H-CDR={h_n:3d} L-CDR={l_n:3d}  from {len(paths)} pose(s){extra}")
    return MULTI_FV if n_extra else WRITTEN


def print_summary(tally: "Counter[str]", out_dir: Path, n_complexes: int, anarci_calls: int) -> None:
    """Report what was written, what was not, and how to point a scorer at the result.

    The last part matters more than it looks: the JSONs are useless unless whatever
    featurises next is told where they are, and nothing about that wiring is visible on
    the scorer's command line.
    """
    n_written = tally[WRITTEN] + tally[MULTI_FV]
    print("\n" + "=" * 72)
    print(f"Wrote {n_written} JSON(s) to {out_dir}")
    if tally[EXISTING]:
        print(f"  {tally[EXISTING]} already existed (pass --overwrite to rebuild)")
    if tally[SKIPPED]:
        print(f"  {tally[SKIPPED]} skipped: no immunoglobulin heavy chain (nothing to label)")
    if tally[MULTI_FV]:
        print(f"  {tally[MULTI_FV]} complex(es) had >1 Fv; only the first heavy/light pair is labelled")
    # Calls are per distinct chain sequence, not per complex: a complex contributes one
    # per chain (heavy, light, antigen), and complexes sharing an antibody share those.
    print(f"  {anarci_calls} ANARCI call(s) on distinct chain sequences, for {n_complexes} complex/complexes")
    if n_written or tally[EXISTING]:
        print(
            "\nPoint the scorer at them with:"
            f"\n  predict_dockq.py ... --cdr-json-dir {out_dir.resolve()}"
            f"\nor, for anything else that featurises:"
            f"\n  export CONFIDOCK_SABDAB_JSON_DIR={out_dir.resolve()}"
        )
    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> None:
    """Scan the structures, run ANARCI per complex, write one JSON per lookup key."""
    args = _parse_args(argv)

    files = iter_structure_files(args.structures)
    if not files:
        raise SystemExit(f"No .cif/.pdb files matched {args.structures}")
    groups = group_by_key(files)
    print(f"{len(files)} structure file(s) -> {len(groups)} complex/complexes")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    anarci = _AnarciCache()
    tally: "Counter[str]" = Counter()

    for key, paths in groups.items():
        tally[write_sidecar(key, paths, args.out_dir / f"{key}.json", anarci, args.overwrite)] += 1

    print_summary(tally, args.out_dir, len(groups), anarci.calls)


if __name__ == "__main__":
    main()
