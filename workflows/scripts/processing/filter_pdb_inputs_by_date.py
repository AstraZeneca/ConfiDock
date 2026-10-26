r"""Filter structure input files to those from PDB structures deposited after a cutoff date.

PDB IDs are extracted from filenames (e.g. ``7anq_B_A.yaml``, ``7anq_B_A.pdb``,
``7anq_B_A.json`` → ``7ANQ``).  Any file whose stem begins with a PDB ID
(first ``_``-delimited segment) is supported.  Deposition dates are fetched in
batched requests from the RCSB GraphQL API
(``rcsb_accession_info.deposit_date``).  Only files whose PDB entry was
deposited **strictly after** ``--cutoff_date`` are copied to ``--output_dir``.

Supported extensions by default: ``.yaml``, ``.pdb``, ``.json``.

Usage::

    # YAML inputs
    python workflows/scripts/processing/filter_pdb_inputs_by_date.py \\
        --input_dir   data/Unsal2026/boltz2_inputs \\
        --output_dir  data/Unsal2026/boltz2_inputs_post2023 \\
        --cutoff_date 2023-06-01

    # PDB files
    python workflows/scripts/processing/filter_pdb_inputs_by_date.py \\
        --input_dir   data/Unsal2026/models/PDBs \\
        --output_dir  data/Unsal2026/models/PDBs_post2023 \\
        --cutoff_date 2023-06-01

    # Entries deposited ON OR BEFORE the cutoff (pre-training set)
    python workflows/scripts/processing/filter_pdb_inputs_by_date.py \\
        --input_dir   data/Unsal2026/models/PDBs \\
        --output_dir  data/Unsal2026/models/PDBs_pre2023 \\
        --cutoff_date 2023-06-01 \\
        --before

    # Restrict to a single extension
    python workflows/scripts/processing/filter_pdb_inputs_by_date.py \\
        --input_dir   data/Unsal2026/models/PDBs \\
        --output_dir  data/Unsal2026/models/PDBs_post2023 \\
        --cutoff_date 2023-06-01 \\
        --extensions  .pdb
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from confidock.rcsb import (
    DEFAULT_EXTENSIONS,
    filter_inputs_by_date,
    pdb_id_from_stem,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter PDB inputs (YAML, JSON, PDB) to PDB structures deposited after a cutoff date.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/Unsal2026/boltz2_inputs"),
        help="Directory containing input files to filter",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/Unsal2026/boltz2_inputs_post2023"),
        help="Destination directory for the filtered files",
    )
    parser.add_argument(
        "--cutoff_date",
        default="2023-06-01",
        help="Exclude structures deposited on or before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        metavar="EXT",
        help="File extensions to include (e.g. .yaml .pdb .json)",
    )
    parser.add_argument(
        "--before",
        action="store_true",
        help="Keep entries deposited ON OR BEFORE --cutoff_date (default: strictly after)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print which files would be kept/excluded without copying anything",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for filtering PDB inputs by date."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    try:
        cutoff = date.fromisoformat(args.cutoff_date)
    except ValueError as exc:
        raise SystemExit(f"Invalid --cutoff_date '{args.cutoff_date}': {exc}") from exc

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    kept, excluded, deposit_dates = filter_inputs_by_date(
        input_dir,
        args.output_dir,
        cutoff,
        before=args.before,
        extensions=tuple(args.extensions),
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"\n=== Would KEEP ({len(kept)}) ===")
        for p in kept:
            dep = deposit_dates.get(pdb_id_from_stem(p.stem))
            print(f"  {p.name}  (deposited: {dep})")
        print(f"\n=== Would EXCLUDE ({len(excluded)}) ===")
        for p in excluded:
            dep = deposit_dates.get(pdb_id_from_stem(p.stem))
            print(f"  {p.name}  (deposited: {dep})")


if __name__ == "__main__":
    main()
