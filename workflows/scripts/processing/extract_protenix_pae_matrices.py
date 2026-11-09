#!/usr/bin/env python3
"""Extract PAE matrices from Protenix full_data JSON files and save as compressed .npz.

Protenix stores large confidence data in ``*_full_data_*.json`` files that can be
16+ MB each. This script extracts just the PAE (Predicted Aligned Error) matrix
and saves it as a compressed .npz file, reducing storage by ~95%.

Input structure::

    {predictions_dir}/
        {entry}/
            seed_{seed}/
                predictions/
                    {entry}_full_data_sample_{n}.json  (16 MB)
                    {entry}_sample_{n}.cif
                    {entry}_summary_confidence_sample_{n}.json

Output structure::

    Same directory, adds:
        {entry}_pae_sample_{n}.npz  (~0.5-1 MB)

Usage::

    # Process a single condition directory
    python extract_protenix_pae_matrices.py \
        --predictions_dir /path/to/protenix_r1/msa=full_msa__cycle=10__step=200

    # Process all condition directories under a root
    python extract_protenix_pae_matrices.py \
        --predictions_dir /path/to/protenix_r1 \
        --recursive

    # Delete full_data JSON files after extraction (saves space)
    python extract_protenix_pae_matrices.py \
        --predictions_dir /path/to/protenix_r1 \
        --recursive \
        --delete_full_data

    # Dry run (show what would be processed)
    python extract_protenix_pae_matrices.py \
        --predictions_dir /path/to/protenix_r1 \
        --recursive \
        --dry_run

Note: Despite the script name, this extracts ALL confidence arrays, not just PAE.
The output files are named *_confidence_*.npz to reflect this.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def extract_pae_from_json(json_path: Path, output_npz_path: Path) -> bool:
    """Extract confidence arrays from Protenix full_data JSON and save as .npz.

    Extracts PAE, PDE, contact_probs, atom_plddt, and metadata arrays.

    Parameters
    ----------
    json_path : Path
        Path to *_full_data_*.json file
    output_npz_path : Path
        Path where .npz file should be saved

    Returns
    -------
    bool
        True if extraction succeeded, False otherwise
    """
    try:
        with json_path.open() as f:
            data = json.load(f)

        # Extract PAE matrix (token_pair_pae)
        if "token_pair_pae" not in data:
            LOGGER.warning("No 'token_pair_pae' found in %s", json_path.name)
            return False

        pae = np.array(data["token_pair_pae"], dtype=np.float32)

        # Also extract other potentially useful matrices and metadata
        arrays_to_save = {"pae": pae}

        # Confidence matrices
        if "token_pair_pde" in data:
            arrays_to_save["pde"] = np.array(data["token_pair_pde"], dtype=np.float32)

        if "contact_probs" in data:
            arrays_to_save["contact_probs"] = np.array(data["contact_probs"], dtype=np.float32)

        # Per-atom data
        if "atom_plddt" in data:
            arrays_to_save["atom_plddt"] = np.array(data["atom_plddt"], dtype=np.float32)

        if "atom_to_token_idx" in data:
            arrays_to_save["atom_to_token_idx"] = np.array(data["atom_to_token_idx"], dtype=np.int32)

        # Per-token metadata (important for multi-chain analysis)
        if "token_asym_id" in data:
            arrays_to_save["token_asym_id"] = np.array(data["token_asym_id"], dtype=np.int32)

        if "token_has_frame" in data:
            arrays_to_save["token_has_frame"] = np.array(data["token_has_frame"], dtype=np.int8)

        # Save with compression
        np.savez_compressed(output_npz_path, **arrays_to_save)

        # Log size reduction
        json_size_mb = json_path.stat().st_size / (1024 * 1024)
        npz_size_mb = output_npz_path.stat().st_size / (1024 * 1024)
        reduction_pct = (1 - npz_size_mb / json_size_mb) * 100

        LOGGER.info(
            "  %s: %.1f MB → %.1f MB (%.1f%% reduction)",
            output_npz_path.name,
            json_size_mb,
            npz_size_mb,
            reduction_pct,
        )

        return True

    except (json.JSONDecodeError, OSError, KeyError) as exc:
        LOGGER.error("Failed to process %s: %s", json_path, exc)
        return False


#: Outcome labels ``process_one_file`` reports and ``process_predictions_dir`` counts. Named rather than
#: spelled out at each site, because the two functions have to agree on them for the summary to be right.
PROCESSED, SKIPPED, FAILED = "processed", "skipped", "failed"


def confidence_npz_path(json_path: Path) -> Path:
    """Return where one full_data JSON's extracted arrays belong.

    Beside the JSON, under the same name with ``_full_data_`` swapped for ``_confidence_``: that is the
    name the featuriser globs for, and writing it in place is what makes ``--delete_full_data`` safe --
    the replacement is already where the reader will look.
    """
    npz_name = json_path.name.replace("_full_data_", "_confidence_").replace(".json", ".npz")
    return json_path.parent / npz_name


def full_data_pattern(predictions_dir: Path, recursive: bool) -> str:
    """Return the glob that finds full_data JSONs under whichever level of the tree was passed.

    Protenix writes ``{condition}/{entry}/seed_*/predictions/*_full_data_*.json`` and this script is
    pointed at any level of that. ``--recursive`` sweeps the lot; without it, the presence of a ``seed_*``
    directory is what distinguishes "you gave me one entry" from "you gave me a directory of entries".
    """
    if recursive:
        return "**/*_full_data_*.json"
    if any(predictions_dir.glob("seed_*")):
        return "seed_*/predictions/*_full_data_*.json"
    return "*/seed_*/predictions/*_full_data_*.json"


def process_one_file(
    json_path: Path,
    predictions_dir: Path,
    delete_full_data: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> str:
    """Extract one full_data JSON and report which of ``PROCESSED``/``SKIPPED``/``FAILED`` happened.

    The JSON is only deleted after a successful extraction, and a failure to delete is a warning rather
    than a failure of the file: the arrays are already written, so the run has lost nothing but disk.

    Parameters
    ----------
    json_path : Path
        The ``*_full_data_*.json`` to extract
    predictions_dir : Path
        Root the log lines are made relative to
    delete_full_data : bool
        If True, delete the JSON after a successful extraction
    dry_run : bool
        If True, only report what would be done
    overwrite : bool
        If True, re-extract even if the .npz already exists
    """
    npz_path = confidence_npz_path(json_path)

    if npz_path.exists() and not overwrite:
        LOGGER.debug("Skipping %s (already exists)", npz_path.name)
        return SKIPPED

    if dry_run:
        LOGGER.info("[DRY RUN] Would process: %s", json_path.relative_to(predictions_dir))
        return PROCESSED

    LOGGER.info("Processing %s", json_path.relative_to(predictions_dir))
    if not extract_pae_from_json(json_path, npz_path):
        return FAILED

    if delete_full_data:
        try:
            json_path.unlink()
            LOGGER.info("  Deleted %s", json_path.name)
        except OSError as exc:
            LOGGER.warning("  Could not delete %s: %s", json_path.name, exc)

    return PROCESSED


def process_predictions_dir(
    predictions_dir: Path,
    recursive: bool = False,
    delete_full_data: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Process all full_data JSON files in a predictions directory.

    Parameters
    ----------
    predictions_dir : Path
        Root directory containing predictions (either condition dir or parent)
    recursive : bool
        If True, process all subdirectories recursively
    delete_full_data : bool
        If True, delete full_data JSON files after successful extraction
    dry_run : bool
        If True, only show what would be processed
    overwrite : bool
        If True, re-extract even if .npz already exists

    Returns
    -------
    tuple[int, int, int]
        (processed_count, skipped_count, failed_count)
    """
    if not predictions_dir.is_dir():
        LOGGER.error("Directory not found: %s", predictions_dir)
        return 0, 0, 0

    full_data_files = list(predictions_dir.glob(full_data_pattern(predictions_dir, recursive)))

    if not full_data_files:
        LOGGER.info("No full_data JSON files found in %s", predictions_dir)
        return 0, 0, 0

    LOGGER.info("Found %d full_data JSON files", len(full_data_files))

    outcomes: collections.Counter = collections.Counter()
    for json_path in sorted(full_data_files):
        outcomes[process_one_file(json_path, predictions_dir, delete_full_data, dry_run, overwrite)] += 1

    return outcomes[PROCESSED], outcomes[SKIPPED], outcomes[FAILED]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract confidence arrays from Protenix full_data JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--predictions_dir",
        type=Path,
        required=True,
        help="Root predictions directory (either condition dir or parent)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process all subdirectories recursively",
    )
    parser.add_argument(
        "--delete_full_data",
        action="store_true",
        help="Delete full_data JSON files after successful extraction (saves space)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if .npz files already exist",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be processed without actually doing it",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    if args.dry_run:
        LOGGER.info("DRY RUN MODE - no files will be modified")

    # Process directory
    processed, skipped, failed = process_predictions_dir(
        args.predictions_dir,
        recursive=args.recursive,
        delete_full_data=args.delete_full_data,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("Summary:")
    LOGGER.info("  Processed: %d", processed)
    LOGGER.info("  Skipped:   %d", skipped)
    LOGGER.info("  Failed:    %d", failed)
    LOGGER.info("=" * 60)

    if not args.dry_run and processed > 0:
        LOGGER.info("")
        LOGGER.info("Confidence arrays extracted to .npz files in the same directories")
        if args.delete_full_data:
            LOGGER.info("Original full_data JSON files have been deleted")
        else:
            LOGGER.info("Original full_data JSON files retained (use --delete_full_data to remove)")


if __name__ == "__main__":
    main()
