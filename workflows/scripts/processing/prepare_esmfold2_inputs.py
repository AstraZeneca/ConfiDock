r"""Prepare ESMFold2 input JSON files from PDB structures.

For each PDB file in the source directory this script:
1. Reads the chain IDs encoded in the filename (format: {pdbid}_{chain1}[_{chain2}...].pdb).
2. Extracts the sequence for each chain using BioPython.
3. Writes an ESMFold2-compatible JSON input file.

Usage::

    poetry run python workflows/scripts/processing/prepare_esmfold2_inputs.py \
        --pdb_dir data/Unsal2026/models/PDBs \
        --output_json data/Unsal2026/esmfold2_inputs.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from confidock.esmfold2.inference.inputs import pdbs_to_esmfold2_json

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ESMFold2 input JSON from PDB structures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdb_dir",
        type=Path,
        default=Path("data/Unsal2026/models/PDBs"),
        help="Directory containing input PDB files",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("data/Unsal2026/esmfold2_inputs.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        default=None,
        metavar="ENTRY",
        help=(
            "Restrict to specific entries by stem name "
            "(e.g. 8pyr_H_A 8qf5_H_A). "
            "Defaults to all PDB files found in --pdb_dir."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for preparing ESMFold2 inputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    pdb_dir: Path = args.pdb_dir
    if not pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if args.entries:
        entry_set = set(args.entries)
        pdb_files = [p for p in pdb_files if p.stem in entry_set]
    if not pdb_files:
        LOGGER.error("No PDB files found in %s", pdb_dir)
        return

    LOGGER.info("Found %d PDB files in %s", len(pdb_files), pdb_dir)
    pdbs_to_esmfold2_json(pdb_files, args.output_json)


if __name__ == "__main__":
    main()
