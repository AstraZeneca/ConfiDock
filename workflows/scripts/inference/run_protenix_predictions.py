r"""Run Protenix structure predictions from a JSON input file.

Protenix writes each prediction under::

    {out_dir}/{sample_name}/seed_{seed}/predictions/
        {sample_name}_sample_{rank}.cif
        {sample_name}_summary_confidence_sample_{rank}.json
        {sample_name}_full_data_sample_{rank}.json

A ``prediction_config.yaml`` is written to ``--output_dir`` before the run to
document every setting used.

MSA retrieval
-------------
Protenix fetches MSAs from the ColabFold/MMseqs2 API at
``https://protenix-server.com/api/msa`` (override with the
``MMSEQS_SERVICE_HOST_URL`` environment variable).

Two-step workflow (recommended for large datasets):

1. Pre-fetch MSAs once and cache them::

       python workflows/scripts/inference/run_protenix_predictions.py \\
           --input_json data/Unsal2026/protenix_inputs_postJune2023.json \\
           --output_dir data/Unsal2026/protenix_predictions \\
           --prefetch_msa

   This runs ``protenix msa`` and writes an enriched JSON
   ``{input_stem}-add-msa.json`` alongside the original, then exits.
   Re-running with ``--prefetch_msa`` is safe: if the enriched JSON already
   exists the MSA step is skipped.

2. Run predictions using the pre-fetched MSAs::

       python workflows/scripts/inference/run_protenix_predictions.py \\
           --input_json data/Unsal2026/protenix_inputs_postJune2023.json \\
           --output_dir data/Unsal2026/protenix_predictions \\
           --seeds 0,1,2 --samples 5 --use_msa

   The script auto-detects the ``*-add-msa.json`` sibling file and uses it
   instead of the original JSON.

Combined (fetch + predict in one call)::

    python workflows/scripts/inference/run_protenix_predictions.py \\
        --input_json  data/Unsal2026/protenix_inputs_postJune2023.json \\
        --output_dir  data/Unsal2026/protenix_predictions \\
        --seeds 0 --samples 5 --prefetch_msa --use_msa

Skipping already-completed entries::

    python workflows/scripts/inference/run_protenix_predictions.py \\
        --input_json  data/Unsal2026/protenix_inputs_postJune2023.json \\
        --output_dir  data/Unsal2026/protenix_predictions \\
        --seeds       0 \\
        --skip_existing

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Import from library
from confidock.protenix.inference.runner import (
    check_completed_samples,
    find_protenix_cli,
    print_timings,
    write_prediction_config,
)

LOGGER = logging.getLogger(__name__)


# Alias library functions for backward compatibility
_completed_samples = check_completed_samples
_find_protenix_cli = find_protenix_cli


def _write_prediction_config(output_dir, **kwargs):
    return write_prediction_config(output_dir, format="yaml", **kwargs)


_print_timings = print_timings

# Defaults ---------------------------------------------------------------

DEFAULT_SEEDS = "101"
DEFAULT_SAMPLES = 5
DEFAULT_CYCLE = 10
DEFAULT_STEP = 200


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Protenix predictions for entries in a JSON input file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="JSON file with a list of Protenix input entries (each has 'name' + 'sequences')",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/Unsal2026/protenix_predictions"),
        help="Root directory for predictions and config",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="protenix_base_20250630_v1.0.0",
        help=(
            "Protenix model name, e.g. 'protenix_base_20250630_v1.0.0'. "
            "The checkpoint is loaded from the protenix release_data/checkpoint/ directory "
            "as {model_name}.pt. See configs/configs_model_type.py for valid names."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=DEFAULT_SEEDS,
        help="Comma-separated list of random seeds, e.g. '0,1,2'",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of diffusion samples (structures) to generate per input per seed",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=DEFAULT_CYCLE,
        help="Number of pairformer recycling cycles",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help="Number of diffusion sampling steps",
    )
    parser.add_argument(
        "--use_msa",
        action="store_true",
        default=False,
        help=(
            "Pass --use_msa true to protenix predict. "
            "If a pre-fetched *-add-msa.json exists it is used automatically; "
            "otherwise MSAs are fetched on-the-fly during inference."
        ),
    )
    parser.add_argument(
        "--prefetch_msa",
        action="store_true",
        default=False,
        help=(
            "Run 'protenix msa' before inference to pre-fetch MSAs from the "
            "ColabFold/MMseqs2 API and cache them as pairing/non-pairing .a3m files. "
            "The resulting *-add-msa.json is then used for prediction. "
            "Skipped if *-add-msa.json already exists. "
            "Implies --use_msa."
        ),
    )
    parser.add_argument(
        "--msa_dir",
        type=Path,
        default=None,
        help="Directory for pre-fetched MSA files when using --prefetch_msa. Defaults to {output_dir}/msa_cache.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip entries whose predictions already exist in output_dir",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Process entries in batches of this size (useful for large datasets "
            "to avoid memory accumulation).  Omit to run all at once."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper functions for main
# ---------------------------------------------------------------------------


def _prefetch_msas_if_needed(
    args: argparse.Namespace,
    input_json: Path,
    msa_dir: Path,
    cli: str,
    pythonpath: str,
) -> Path:
    """Pre-fetch MSAs if requested, return enriched JSON path.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments
    input_json : Path
        Original input JSON
    msa_dir : Path
        MSA directory
    cli : str
        Protenix CLI command

    Returns
    -------
    Path
        Path to enriched JSON (or original if MSA not fetched)
    """
    enriched_json: Path = input_json.with_name(input_json.stem + "-add-msa" + input_json.suffix)

    if not args.prefetch_msa:
        return enriched_json if enriched_json.exists() else input_json

    if enriched_json.exists():
        LOGGER.info("Pre-fetched MSA JSON already exists, skipping fetch: %s", enriched_json)
        return enriched_json

    LOGGER.info("Pre-fetching MSAs via 'protenix msa' → %s", msa_dir)
    msa_dir.mkdir(parents=True, exist_ok=True)

    # Construct MSA command based on CLI type
    # For protenix-inference: use "protenix msa" (not "protenix pred msa")
    # For shared Protenix: MSA may not be supported
    if "protenix pred" in cli:
        # Replace "pred" with "msa" for MSA fetching
        msa_cli = cli.replace("pred", "msa")
        msa_cmd: list[str] = [
            *msa_cli.split(),
            "--input",
            str(input_json),
            "--out_dir",
            str(msa_dir),
        ]
    else:
        # Shared Protenix or other - may not support MSA
        LOGGER.warning("MSA prefetch may not be supported with this Protenix installation: %s", cli)
        msa_cmd: list[str] = [
            *cli.split(),
            "msa",
            "--input",
            str(input_json),
            "--out_dir",
            str(msa_dir),
        ]
    LOGGER.info("MSA command: %s", " ".join(msa_cmd))

    # Set up environment with PYTHONPATH if needed
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = pythonpath

    t0 = time.perf_counter()
    try:
        subprocess.run(msa_cmd, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        LOGGER.error("protenix msa exited with code %d", exc.returncode)
        sys.exit(exc.returncode)
    LOGGER.info("MSA pre-fetch completed in %.1f s", time.perf_counter() - t0)

    if not enriched_json.exists():
        LOGGER.warning(
            "Expected enriched JSON not found at %s; falling back to original input JSON for inference.",
            enriched_json,
        )
        return input_json
    return enriched_json


def _filter_completed_entries(
    entries: list[dict],
    output_dir: Path,
    seeds_list: list[int],
    samples: int,
) -> list[dict]:
    """Filter out already-completed entries.

    Parameters
    ----------
    entries : list of dict
        All entries
    output_dir : Path
        Output directory
    seeds_list : list of int
        Seeds
    samples : int
        Samples per entry

    Returns
    -------
    list of dict
        Incomplete entries
    """
    t0 = time.perf_counter()
    completed = _completed_samples(output_dir, seeds_list, samples)
    LOGGER.info(
        "Skip-existing: %d/%d entries already complete (%.1f s scan)",
        len(completed),
        len(entries),
        time.perf_counter() - t0,
    )
    filtered = [e for e in entries if e["name"] not in completed]
    if not filtered:
        LOGGER.info("All entries already complete. Nothing to do.")
    return filtered


def _run_prediction_batch(
    batch: list[dict],
    batch_idx: int,
    total_batches: int,
    args: argparse.Namespace,
    cli: str,
    pythonpath: str,
    output_dir: Path,
    model_name: str,
    use_msa: bool,
) -> float:
    """Run predictions for a single batch.

    Parameters
    ----------
    batch : list of dict
        Batch entries
    batch_idx : int
        Batch index
    total_batches : int
        Total number of batches
    args : argparse.Namespace
        CLI arguments
    cli : str
        Protenix CLI command
    output_dir : Path
        Output directory
    model_name : str
        Model name
    use_msa : bool
        Use MSA flag

    Returns
    -------
    float
        Elapsed time
    """
    LOGGER.info("Batch %d/%d: %d entries", batch_idx + 1, total_batches, len(batch))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="protenix_batch_") as tmp:
        json.dump(batch, tmp, indent=2)
        batch_json = tmp.name

    if "-m protenix.runner.batch_inference" in cli:
        cmd: list[str] = [
            *cli.split(),
            "--input",
            batch_json,
            "--out_dir",
            str(output_dir),
            "--seeds",
            args.seeds,
            "--sample",
            str(args.samples),
            "--cycle",
            str(args.cycle),
            "--step",
            str(args.step),
            "--model_name",
            model_name,
            "--use_msa",
            str(use_msa).lower(),
            "--use_default_params",
            "false",
        ]
    else:
        cmd: list[str] = [
            *cli.split(),
            "-i",
            batch_json,
            "-o",
            str(output_dir),
            "-s",
            args.seeds,
            "-e",
            str(args.samples),
            "-c",
            str(args.cycle),
            "-p",
            str(args.step),
            "-n",
            model_name,
            "--use_msa",
            str(use_msa).lower(),
            "--use_default_params",
            "false",
        ]

    LOGGER.info("Command: %s", " ".join(cmd))

    # Set up environment with PYTHONPATH if needed
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = pythonpath

    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Protenix exited with code %d", exc.returncode)
        Path(batch_json).unlink(missing_ok=True)
        sys.exit(exc.returncode)

    elapsed = time.perf_counter() - t0
    LOGGER.info(
        "Batch %d/%d finished in %.1f s (%.1f s/entry)", batch_idx + 1, total_batches, elapsed, elapsed / len(batch)
    )
    Path(batch_json).unlink(missing_ok=True)
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Protenix predictions for multiple antibody-antigen complexes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    input_json: Path = args.input_json
    output_dir: Path = args.output_dir

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with input_json.open() as fh:
        entries: list[dict] = json.load(fh)

    if not entries:
        LOGGER.error("No entries found in %s", input_json)
        sys.exit(1)

    LOGGER.info("Found %d entries in %s", len(entries), input_json)

    seeds_list = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name: str = args.model_name
    use_msa: bool = args.use_msa or args.prefetch_msa
    msa_dir: Path = args.msa_dir or (output_dir / "msa_cache")
    cli, pythonpath, _ = _find_protenix_cli()

    # Pre-fetch MSAs if needed
    effective_input_json = _prefetch_msas_if_needed(args, input_json, msa_dir, cli, pythonpath)
    if effective_input_json != input_json:
        LOGGER.info("Using MSA-enriched JSON: %s", effective_input_json)
        with effective_input_json.open() as fh:
            entries = json.load(fh)

    # Write config
    _write_prediction_config(
        output_dir,
        input_json=str(input_json),
        effective_input_json=str(effective_input_json),
        model_name=model_name,
        seeds=args.seeds,
        samples=args.samples,
        cycle=args.cycle,
        step=args.step,
        use_msa=use_msa,
        prefetch_msa=args.prefetch_msa,
        msa_dir=str(msa_dir) if args.prefetch_msa else None,
        skip_existing=args.skip_existing,
    )

    # Filter completed entries if requested
    if args.skip_existing and output_dir.exists():
        entries = _filter_completed_entries(entries, output_dir, seeds_list, args.samples)
        if not entries:
            return

    LOGGER.info("Using protenix CLI: %s", cli)
    LOGGER.info("Running %d entries with seeds=%s, samples=%d", len(entries), args.seeds, args.samples)

    # Run predictions in batches
    batch_size = args.batch_size or len(entries)
    batches = [entries[i : i + batch_size] for i in range(0, len(entries), batch_size)]

    wall_start = time.perf_counter()
    timings: dict[str, float] = {}

    for batch_idx, batch in enumerate(batches):
        elapsed = _run_prediction_batch(
            batch, batch_idx, len(batches), args, cli, pythonpath, output_dir, model_name, use_msa
        )
        timings[f"Batch {batch_idx + 1}/{len(batches)}"] = elapsed

    timings["Total wall time"] = time.perf_counter() - wall_start
    _print_timings(timings)
    LOGGER.info("All predictions complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
