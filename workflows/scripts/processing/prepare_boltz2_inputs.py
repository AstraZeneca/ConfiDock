r"""Prepare Boltz-2 YAML input files from SAbDab / Unsal2026 PDB structures.

For each PDB file in the source directory this script:
1. Reads the chain IDs encoded in the filename (format: {pdbid}_{chains}.pdb).
2. Extracts the amino acid sequence for each chain using BioPython's PPBuilder.
3. Optionally resolves pre-computed MSA files for each chain.
4. Writes a Boltz-2 compatible YAML input file.

Optional date-based filtering (for train/test splits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pass ``--cutoff_date`` to fetch deposition dates from RCSB and keep only
entries deposited on or before that date.  Pass ``--n_samples`` to randomly
sub-sample the filtered set.  Pass ``--exclude_dir`` to skip PDB IDs already
present in another output directory.

MSA injection
~~~~~~~~~~~~~
Pass ``--msa_dir PATH`` (repeatable) to inject pre-computed MSA paths into the
YAML ``msa:`` fields instead of writing ``msa: empty``.  Each directory is
probed in order and the first match is used.

Two lookup strategies are supported automatically:

* **Hash-named** directories (files named ``{sha256[:16]}.a3m``, as written by
  ``fetch_msas.py``): O(1) lookup — no indexing required.
* **Sequence-indexed** directories (e.g. ``data/sabdab_msas_preJune2023/``
  written by ``patch_preJune2023_msas.py``): the first sequence of every
  ``.a3m`` file is decoded at startup and stored in a hash map.

Usage — full SAbDab with MSAs::

    poetry run python workflows/scripts/processing/prepare_boltz2_inputs.py \\
        --pdb_dir /path/to/sabdab_CROP_SIZE_600 \\
        --output_dir data/sabdab_all_boltz2_inputs \\
        --msa_dir data/sabdab_msas \\
        --msa_dir data/sabdab_msas_preJune2023

Usage — pre-June-2023 split with date filter::

    poetry run python workflows/scripts/processing/prepare_boltz2_inputs.py \\
        --pdb_dir /path/to/sabdab_CROP_SIZE_600 \\
        --output_dir data/sabdab_CROP_SIZE_600_preJune2023_boltz2_inputs \\
        --exclude_dir data/sabdab_CROP_SIZE_600_postJune2023_boltz2_inputs \\
        --cutoff_date 2023-06-01 --n_samples 6000 --seed 42

Usage — Unsal2026 (simple, no filtering or MSAs)::

    poetry run python workflows/scripts/processing/prepare_boltz2_inputs.py \\
        --pdb_dir data/Unsal2026/models/PDBs \\
        --output_dir data/Unsal2026/boltz2_inputs
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import re
from datetime import date
from pathlib import Path

import yaml

from confidock.boltz.inference.inputs import pdb_to_boltz_yaml

LOGGER = logging.getLogger(__name__)

# Matches hash-named a3m files written by fetch_msas.py  (16 hex chars)
_HASH_A3M_RE = re.compile(r"^[0-9a-f]{16}\.a3m$")


# ---------------------------------------------------------------------------
# MSA resolver
# ---------------------------------------------------------------------------


def _read_a3m_query(a3m_path: Path) -> str:
    """Return match-state residues from the first (query) sequence in an A3M file.

    In A3M format: uppercase = match states, lowercase = insertions, '-' = gaps.
    This returns only the uppercase letters (match-state residues), which
    represent the aligned positions for sequence matching.
    """
    try:
        lines = a3m_path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    in_seq = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith((">", "#")):
            in_seq = True
            continue
        if in_seq and stripped:
            # Strip lowercase insertions and gaps, return match-state residues only
            return "".join(c for c in stripped if c.isupper())
    return ""


def _is_hash_named_dir(a3m_files: list[Path]) -> bool:
    """Check if a directory uses hash-based naming (at least 50% hash-named files)."""
    hash_count = sum(1 for f in a3m_files if _HASH_A3M_RE.match(f.name))
    return hash_count >= (len(a3m_files) + 1) // 2


def _index_sequence_dir(a3m_files: list[Path], existing_index: dict[str, Path]) -> int:
    """Index sequences from a3m files. Returns count of newly indexed sequences."""
    added = 0
    for a3m in a3m_files:
        q = _read_a3m_query(a3m)
        if q and q not in existing_index:
            existing_index[q] = a3m
            added += 1
    return added


class MsaResolver:
    """Resolve a protein sequence to an absolute ``.a3m`` path.

    Supports two directory types (detected automatically):

    * **Hash-named** (``{sha256[:16]}.a3m``): O(1) — no startup cost.
    * **Sequence-indexed** (any other naming): a3m files are read once at
      startup; the query sequence → path map is used for exact and
      substring matching.
    """

    def __init__(self, msa_dirs: list[Path]) -> None:
        self._hash_dirs: list[Path] = []
        # seq -> a3m_path for non-hash-named dirs
        self._seq_index: dict[str, Path] = {}

        for d in msa_dirs:
            self._process_msa_dir(d)

    def _process_msa_dir(self, msa_dir: Path) -> None:
        """Process a single MSA directory, detecting type and indexing if needed."""
        if not msa_dir.exists():
            LOGGER.warning("--msa_dir %s not found – skipping", msa_dir)
            return

        a3m_files = list(msa_dir.glob("*.a3m"))
        if not a3m_files:
            LOGGER.warning("--msa_dir %s has no .a3m files – skipping", msa_dir)
            return

        if _is_hash_named_dir(a3m_files):
            self._hash_dirs.append(msa_dir)
            LOGGER.info("MSA dir (hash-named): %s  (%d files)", msa_dir, len(a3m_files))
        else:
            LOGGER.info("MSA dir (sequence-indexed): %s  (%d files, indexing…)", msa_dir, len(a3m_files))
            added = _index_sequence_dir(a3m_files, self._seq_index)
            LOGGER.info("  Indexed %d sequences", added)

    def resolve(self, seq: str) -> str | None:
        """Return absolute path to the best matching ``.a3m`` or ``None``."""
        if not seq:
            return None

        # 1. Hash lookup (O(1))
        digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
        for d in self._hash_dirs:
            p = d / f"{digest}.a3m"
            if p.exists() and p.stat().st_size >= 10:  # noqa: PLR2004
                return str(p.resolve())

        # 2. Exact match in sequence index
        if seq in self._seq_index:
            p = self._seq_index[seq]
            if p.exists():
                return str(p.resolve())

        # 3. Substring match (YAML chain may be a crop of a longer indexed seq)
        for indexed_seq, p in self._seq_index.items():
            if indexed_seq.find(seq) != -1 and p.exists():
                return str(p.resolve())

        return None


def _patch_yaml_msas(yaml_path: Path, resolver: MsaResolver) -> int:
    """Patch ``msa: empty`` entries in *yaml_path* in-place.  Returns # patched."""
    data = yaml.safe_load(yaml_path.read_text()) or {}
    if not isinstance(data, dict):
        return 0
    changed = 0
    for entry in data.get("sequences", []):
        p = entry.get("protein")
        if p is None or p.get("msa", "empty") != "empty":
            continue
        resolved = resolver.resolve(p.get("sequence", ""))
        if resolved:
            p["msa"] = resolved
            changed += 1
    if changed:
        with yaml_path.open("w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True, width=10_000)
    return changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Boltz-2 YAML inputs from SAbDab or Unsal2026 PDB files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdb_dir",
        type=Path,
        required=True,
        help="Directory containing input {pdbid}_{chains}.pdb files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for generated YAML files",
    )

    # ── Optional filtering ──────────────────────────────────────────────────
    parser.add_argument(
        "--cutoff_date",
        default=None,
        help="Keep only structures deposited on/before this date (YYYY-MM-DD). "
        "Requires fetching deposition dates from RCSB.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Randomly sample this many entries after filtering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --n_samples",
    )
    parser.add_argument(
        "--exclude_dir",
        type=Path,
        default=None,
        help="Exclude PDB IDs already present in this directory (e.g. the post-cutoff set)",
    )

    # ── MSA injection ───────────────────────────────────────────────────────
    parser.add_argument(
        "--msa_dir",
        type=Path,
        action="append",
        dest="msa_dirs",
        default=[],
        metavar="PATH",
        help="Directory containing pre-computed .a3m files.  Can be repeated; "
        "directories are probed in the order given.",
    )

    return parser.parse_args()


def _collect_pdb_files(pdb_dir: Path) -> list[Path]:
    """Collect and validate PDB files from directory."""
    if not pdb_dir.exists():
        raise SystemExit(f"PDB directory not found: {pdb_dir}")

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    LOGGER.info("Found %d PDB files in %s", len(pdb_files), pdb_dir)
    if not pdb_files:
        raise SystemExit("No PDB files found.")

    return pdb_files


def _apply_exclusions(pdb_files: list[Path], exclude_dir: Path | None) -> list[Path]:
    """Exclude PDB IDs already present in exclude_dir."""
    if exclude_dir is None:
        return pdb_files

    from confidock.rcsb import pdb_id_from_stem

    if not exclude_dir.exists():
        LOGGER.warning("--exclude_dir %s not found – no exclusions applied", exclude_dir)
        return pdb_files

    exclude_ids = {pdb_id_from_stem(f.stem).upper() for f in exclude_dir.iterdir()}
    before = len(pdb_files)
    filtered = [f for f in pdb_files if pdb_id_from_stem(f.stem).upper() not in exclude_ids]
    LOGGER.info(
        "Excluded %d files (PDB IDs in %s); %d remain",
        before - len(filtered),
        exclude_dir,
        len(filtered),
    )
    return filtered


def _apply_date_filter(pdb_files: list[Path], cutoff_date_str: str | None) -> list[Path]:
    """Filter PDB files by deposition date cutoff."""
    if cutoff_date_str is None:
        return pdb_files

    from confidock.rcsb import fetch_deposit_dates, pdb_id_from_stem

    try:
        cutoff = date.fromisoformat(cutoff_date_str)
    except ValueError as exc:
        raise SystemExit(f"Invalid --cutoff_date: {exc}") from exc

    unique_ids = list({pdb_id_from_stem(f.stem).upper() for f in pdb_files})
    LOGGER.info("Fetching deposit dates for %d unique PDB IDs…", len(unique_ids))
    deposit_dates = fetch_deposit_dates(unique_ids)

    filtered = [
        f for f in pdb_files if (d := deposit_dates.get(pdb_id_from_stem(f.stem).upper())) is not None and d <= cutoff
    ]
    LOGGER.info("%d files on or before %s", len(filtered), cutoff)
    if not filtered:
        raise SystemExit("No entries match the cutoff date.")

    return filtered


def _apply_random_sampling(pdb_files: list[Path], n_samples: int | None, seed: int) -> list[Path]:
    """Randomly sample n_samples from pdb_files."""
    if n_samples is None:
        return pdb_files

    n = min(n_samples, len(pdb_files))
    if n < n_samples:
        LOGGER.warning("Requested %d samples but only %d available – using all", n_samples, n)

    # Use random (not secrets) for reproducible scientific sampling with seed.
    # This is not used for security-sensitive operations.
    # S311: Using pseudorandom number generator `random` is safe here for dataset sampling.
    random.seed(seed)  # noqa: S311
    sampled = random.sample(pdb_files, n)  # noqa: S311
    LOGGER.info("Sampled %d entries (seed=%d)", n, seed)
    return sampled


def _generate_yamls_with_msas(
    pdb_files: list[Path],
    output_dir: Path,
    resolver: MsaResolver | None,
) -> tuple[int, int, int]:
    """Generate YAML files and optionally patch MSAs. Returns (successes, failures, chains_patched)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    successes = failures = chains_patched = 0

    for pdb_path in pdb_files:
        yaml_path = pdb_to_boltz_yaml(pdb_path, output_dir)
        if yaml_path is None:
            failures += 1
            continue
        successes += 1
        if resolver is not None:
            chains_patched += _patch_yaml_msas(yaml_path, resolver)

    return successes, failures, chains_patched


def main() -> None:
    """CLI entrypoint for preparing Boltz2 inputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    # ── Pipeline stages ─────────────────────────────────────────────────────
    pdb_files = _collect_pdb_files(args.pdb_dir)
    pdb_files = _apply_exclusions(pdb_files, args.exclude_dir)
    pdb_files = _apply_date_filter(pdb_files, args.cutoff_date)
    pdb_files = _apply_random_sampling(pdb_files, args.n_samples, args.seed)

    # ── MSA resolver ────────────────────────────────────────────────────────
    resolver: MsaResolver | None = None
    if args.msa_dirs:
        resolver = MsaResolver(args.msa_dirs)

    # ── Generate YAMLs ──────────────────────────────────────────────────────
    successes, failures, chains_patched = _generate_yamls_with_msas(pdb_files, args.output_dir, resolver)

    LOGGER.info(
        "Done: %d/%d YAML files written to %s",
        successes,
        len(pdb_files),
        args.output_dir,
    )
    if resolver is not None:
        LOGGER.info("MSA fields injected: %d chains across %d files", chains_patched, successes)
    if failures:
        LOGGER.warning("%d files failed to convert", failures)


if __name__ == "__main__":
    main()
