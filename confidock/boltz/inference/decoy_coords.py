"""Load decoy coordinates onto Boltz-2's canonical atom ordering.

Why this module exists
----------------------
The confidence head consumes ``x_pred`` positionally: it is projected onto
tokens via the sparse ``feats["token_to_rep_atom"]`` matrix, so row *i* of
``x_pred`` **must** be the same atom as row *i* of Boltz-2's internal atom
table.  Our decoys come from four different predictors with mutually
incompatible conventions:

======================  ==========  ====================================
source                  format      deviation from the Boltz-2 table
======================  ==========  ====================================
``boltz2_r1``           ``.pdb``    none (written by Boltz-2 itself)
``esmfold2_r1``         ``.cif``    minimal CIF that ``gemmi`` refuses to
                                    turn into a model (0 models parsed)
``protenix_r1``         ``.cif``    chains renamed ``A/B/C``; terminal
                                    ``OXT`` atoms Boltz-2 does not have
``perturbed_structures`` ``.pdb``   native chain ids (e.g. ``T``) instead
                                    of the Boltz-2 chain ids (e.g. ``A``)
======================  ==========  ====================================

Copying coordinates by file order would therefore shift every atom after the
first ``OXT`` with no error raised -- a silently wrong cache.  Instead we key
on ``(chain index, residue index, atom name)`` and validate residue names, so
any mismatch is a hard failure rather than a silent offset.

Coordinate *frame* needs no alignment: the confidence head only ever sees
``x_pred`` through ``torch.cdist`` distances and local per-token frames, both
of which are invariant to global rotation and translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import gemmi
import numpy as np
from boltz.data.types import StructureV2

__all__ = [
    "DecoyCoords",
    "load_decoy_coords",
    "read_structure_atoms",
]


@dataclass(frozen=True)
class DecoyCoords:
    """Decoy coordinates expressed in Boltz-2's canonical atom order.

    Attributes
    ----------
    coords
        ``(n_atoms, 3)`` float32 array aligned to ``structure.atoms``.
    present
        ``(n_atoms,)`` bool array; ``False`` where the decoy did not provide
        the atom (these are masked out of the confidence loss downstream).

    Notes
    -----
    ``n_atoms`` is ``len(structure.atoms)``, i.e. *unpadded*.  The featurizer pads
    the atom axis to a larger width, so these coordinates must go through
    :func:`~confidock.boltz.inference.cache_runner.build_x_pred` before
    reaching the confidence head.
    """

    coords: np.ndarray
    present: np.ndarray

    @property
    def coverage(self) -> float:
        """Fraction of canonical atoms the decoy actually provided."""
        return float(self.present.mean()) if self.present.size else 0.0


# Residue name aliases: canonical Boltz-2 name -> names a predictor may emit.
# Kept deliberately small; anything unexpected should fail loudly.
_RES_ALIASES: dict[str, set[str]] = {
    "HIS": {"HIS", "HID", "HIE", "HIP"},
    "CYS": {"CYS", "CYX"},
}


def _res_names_match(canonical: str, observed: str) -> bool:
    if canonical == observed:
        return True
    return observed in _RES_ALIASES.get(canonical, set())


def _read_cif_rows(path: Path) -> list[tuple[str, str, str, str, float, float, float]]:
    """Read ``_atom_site`` rows straight from the CIF table.

    We bypass ``gemmi.read_structure`` because the ESMFold CIFs lack the
    metadata gemmi needs to build a model (it silently returns 0 models).
    Reading the loop directly works for every CIF flavour we have.
    """
    block = gemmi.cif.read(str(path))[0]
    table = block.find(
        "_atom_site.",
        [
            "group_PDB",
            "label_atom_id",
            "label_comp_id",
            "label_asym_id",
            "label_seq_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
        ],
    )
    rows = []
    for row in table:
        if row[0] not in ("ATOM", "HETATM"):
            continue
        rows.append(
            (
                row[1].strip().strip('"').strip("'"),  # atom name
                row[2].strip(),  # residue name
                row[3].strip(),  # chain (label_asym_id)
                row[4].strip(),  # residue seq id
                float(row[5]),
                float(row[6]),
                float(row[7]),
            )
        )
    return rows


def _read_pdb_rows(path: Path) -> list[tuple[str, str, str, str, float, float, float]]:
    """Read ATOM/HETATM records from a PDB file.

    SAbDab chain ids are not always a single character: ~8.6% of our entries
    use two-character ids such as ``Dx`` or ``Hc``.  Boltz-2 writes these into
    columns 22-23, which overflows the one-column chain field and shifts every
    subsequent field right by one::

        ATOM      1  N   VAL Dx   1      -3.130 -19.940  -1.627  1.00 69.75
                                 ^^                            <- 2-char id

    Fixed-column parsing of such a line silently reads ``"0 -19.94"`` as the
    z coordinate; ``gemmi`` mis-parses it too (it reports residue numbers like
    ``-456527``).  We therefore detect the overflow -- residue-name field
    followed by a non-blank column 21 -- and shift the remaining fields.
    """
    rows = []
    with path.open() as handle:
        for line in handle:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            if line[16] not in (" ", "A"):  # skip alternate locations
                continue

            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()

            # Standard layout: chain in col 21, residue sequence number in
            # cols 22-25.  With a two-character chain id the name occupies
            # cols 21-22 and everything after it slides one column right, so
            # col 22 holds a letter where a digit or blank belongs.
            shift = 1 if line[22].isalpha() else 0
            if shift:
                chain_id = line[21:23].strip()
            else:
                chain_id = line[21]

            res_seq = line[22 + shift : 27 + shift].strip()
            rows.append(
                (
                    atom_name,
                    res_name,
                    chain_id,
                    res_seq,
                    float(line[30 + shift : 38 + shift]),
                    float(line[38 + shift : 46 + shift]),
                    float(line[46 + shift : 54 + shift]),
                )
            )
    return rows


def _align_residues(canonical: list[str], decoy: list[str]) -> list[tuple[int, int]]:
    """Pair canonical residues with decoy residues by name.

    Most decoys line up 1:1, but a handful of ESMFold/Protenix models contain
    an extra or substituted residue *in the middle* of a chain (e.g. ``6ssp_LxH``
    inserts an ``ASN``).  Pairing purely by index would shift every residue past
    the insertion onto the wrong canonical slot -- coordinates would still be
    written, so the error would be silent.  We therefore align the two residue
    name sequences and only pair positions that genuinely correspond; anything
    unmatched simply stays absent from the presence mask.

    Uses ``difflib.SequenceMatcher``, which yields the matching blocks of the
    longest-common-subsequence alignment.  Fast enough here (chains are a few
    hundred residues and results are reused across all decoys of an entry).
    """
    if canonical == decoy:
        return [(i, i) for i in range(len(canonical))]

    # Normalise aliases (HID/HIE -> HIS, CYX -> CYS) so they align as equal.
    canonical_to_alias = {alias: canon for canon, aliases in _RES_ALIASES.items() for alias in aliases}
    norm_canonical = [canonical_to_alias.get(n, n) for n in canonical]
    norm_decoy = [canonical_to_alias.get(n, n) for n in decoy]

    matcher = SequenceMatcher(a=norm_canonical, b=norm_decoy, autojunk=False)
    pairs: list[tuple[int, int]] = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            pairs.append((block.a + offset, block.b + offset))
    return pairs


def read_structure_atoms(
    path: Path,
) -> list[tuple[str, list[tuple[str, dict[str, tuple[float, float, float]]]]]]:
    """Read a decoy into ``[(chain_id, [(res_name, {atom_name: xyz}), ...]), ...]``.

    Chains and residues are kept in file order; only the *first* model of a
    multi-model file is read (all our decoys are single-model).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        rows = _read_cif_rows(path)
    elif suffix in (".pdb", ".ent"):
        rows = _read_pdb_rows(path)
    else:
        msg = f"Unsupported structure format {suffix!r}: {path}"
        raise ValueError(msg)

    if not rows:
        msg = f"No ATOM/HETATM records found in {path}"
        raise ValueError(msg)

    chains: list[tuple[str, list[tuple[str, dict]]]] = []
    current_chain_id: str | None = None
    current_residues: list[tuple[str, dict]] | None = None
    current_res_key: str | None = None
    current_atoms: dict[str, tuple[float, float, float]] | None = None

    for atom_name, res_name, chain_id, res_seq, x, y, z in rows:
        if chain_id != current_chain_id:
            current_residues = []
            chains.append((chain_id, current_residues))
            current_chain_id = chain_id
            current_res_key = None
        if res_seq != current_res_key:
            current_atoms = {}
            current_residues.append((res_name, current_atoms))
            current_res_key = res_seq
        # Keep the first occurrence of a duplicated atom name.
        current_atoms.setdefault(atom_name, (x, y, z))

    return chains


def load_decoy_coords(
    path: Path,
    structure: StructureV2,
    *,
    min_coverage: float = 0.5,
    fill_missing: bool = True,
) -> DecoyCoords:
    """Map a decoy structure file onto ``structure``'s canonical atom order.

    Parameters
    ----------
    path
        Decoy ``.pdb`` / ``.cif`` file.
    structure
        The Boltz-2 preprocessed ``StructureV2`` for this entry, i.e. the same
        ``processed/structures/<entry>.npz`` the featurizer will tokenize.
        Its atom table defines the target ordering.
    min_coverage
        Fail if fewer than this fraction of canonical atoms were matched.
        Guards against a decoy that parses but shares almost nothing with the
        canonical table (e.g. a backbone-only file).
    fill_missing
        Collapse unmatched atoms onto their residue centre instead of leaving
        them at the origin.  See :func:`_fill_missing_coords`.

    Returns
    -------
    DecoyCoords
        Coordinates and a presence mask, both in canonical atom order.

    Raises
    ------
    ValueError
        If the chain count, per-chain residue count, residue identity, or atom
        coverage disagrees with the canonical structure.  Failing loudly is the
        entire point of this function.
    """
    decoy_chains = read_structure_atoms(path)
    canonical_chains = structure.chains

    if len(decoy_chains) != len(canonical_chains):
        msg = (
            f"{path.name}: chain count mismatch -- decoy has {len(decoy_chains)} "
            f"({[c[0] for c in decoy_chains]}), canonical has "
            f"{len(canonical_chains)} ({[str(c['name']) for c in canonical_chains]})"
        )
        raise ValueError(msg)

    n_atoms = len(structure.atoms)
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    present = np.zeros(n_atoms, dtype=bool)
    atom_names = structure.atoms["name"]

    for chain_idx, chain in enumerate(canonical_chains):
        _decoy_chain_id, decoy_residues = decoy_chains[chain_idx]
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        canonical_residues = structure.residues[res_start:res_end]

        canonical_names = [str(r["name"]) for r in canonical_residues]
        decoy_names = [r[0] for r in decoy_residues]

        for canon_offset, decoy_offset in _align_residues(canonical_names, decoy_names):
            canonical_res = canonical_residues[canon_offset]
            _decoy_res_name, decoy_atoms = decoy_residues[decoy_offset]

            atom_start = int(canonical_res["atom_idx"])
            for k in range(int(canonical_res["atom_num"])):
                atom_idx = atom_start + k
                xyz = decoy_atoms.get(str(atom_names[atom_idx]))
                if xyz is not None:
                    coords[atom_idx] = xyz
                    present[atom_idx] = True

    result = DecoyCoords(coords=coords, present=present)
    if result.coverage < min_coverage:
        msg = (
            f"{path.name}: only {result.coverage:.1%} of {n_atoms} canonical "
            f"atoms matched (minimum {min_coverage:.1%})"
        )
        raise ValueError(msg)

    if fill_missing:
        _fill_missing_coords(coords, present, structure)
    return result


def _fill_missing_coords(coords: np.ndarray, present: np.ndarray, structure: StructureV2) -> None:
    """Replace absent atoms with their residue's centre atom, in place.

    Decoys derived from experimental natives (the perturbed set) inherit
    unresolved side chains -- median coverage is ~99% but the tail reaches 50%.
    Left at the origin, those atoms would sit ~100 A from the protein and
    poison every ``torch.cdist`` distance the confidence head computes.
    Collapsing them onto the residue centre keeps all pairwise distances
    physically plausible; the presence mask records which atoms were inferred
    so they can be excluded from any downstream loss.
    """
    if present.all():
        return
    for residue in structure.residues:
        atom_start = int(residue["atom_idx"])
        atom_end = atom_start + int(residue["atom_num"])
        if present[atom_start:atom_end].all():
            continue
        centre_idx = int(residue["atom_center"])
        if present[centre_idx]:
            anchor = coords[centre_idx]
        else:
            resolved = np.flatnonzero(present[atom_start:atom_end])
            if resolved.size == 0:
                continue  # whole residue absent; leave for the mask to handle
            anchor = coords[atom_start + resolved[0]]
        missing = ~present[atom_start:atom_end]
        coords[atom_start:atom_end][missing] = anchor
