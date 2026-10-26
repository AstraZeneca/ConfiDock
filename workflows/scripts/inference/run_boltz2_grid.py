r"""Run a grid of Boltz-2 predictions over MSA mode × recycling steps × sampling steps.

Grid axes
---------
msa_mode:
    full_msa       – both antigen and binder chains use pre-computed MSA paths
    antigen_only   – antigen chains keep MSA; H/L binder chains set to ``msa: empty``
    no_msa         – ``--no_msa`` flag; all MSA info ignored

recycling_steps: 1, 3, 5
step_scale:      1.0, 1.5, 2.0  (diffusion temperature; lower = more diverse)

With 5 input structures this gives 18 conditions × 5 inputs = 90 total predictions.

Output layout::

    {output_dir}/
        msa=full_msa__recycling=3__sampling=200/
            boltz_results_{input_dir.stem}/predictions/…
            prediction_config.yaml
        msa=antigen_only__recycling=1__sampling=100/
            …
        grid_config.yaml   ← full grid definition written before any run

Usage::

    poetry run python workflows/scripts/inference/run_boltz2_grid.py \\
        --input_dir  data/sabdab_CROP_SIZE_600_postJune2023_boltz2_inputs \\
        --output_dir data/sabdab_CROP_SIZE_600_postJune2023_boltz2_grid \\
        --cache      /data/boltz-cache \\
        --n_inputs   5 \\
        --seed       42

To run on specific YAML files instead of sampling randomly::

    poetry run python workflows/scripts/inference/run_boltz2_grid.py \\
        --input_yamls data/sabdab_CROP_SIZE_600_postJune2023_boltz2_inputs/8jlj_HcLcA.yaml \\
                      data/sabdab_CROP_SIZE_600_postJune2023_boltz2_inputs/8jlk_HcLcA.yaml \\
        --output_dir  data/sabdab_CROP_SIZE_600_postJune2023_boltz2_grid \\
        --cache       /data/boltz-cache
"""

from __future__ import annotations

import argparse
import datetime
import logging
import random
import shutil
import subprocess
import time
from itertools import product
from pathlib import Path

import yaml

from confidock.boltz.inference.inputs import load_sabdab_binders, stage_yamls
from confidock.boltz.inference.runner import (
    BoltzPredictConfig,
    build_command,
    predictions_dir,
    print_timings,
    rename_model_outputs,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

MSA_MODES: list[str] = ["full_msa", "antigen_only", "no_msa"]
RECYCLING_STEPS: list[int] = [1, 3, 5]
STEP_SCALES: list[float] = [1.0, 1.5, 2.0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Boltz-2 prediction grid (MSA mode × recycling × sampling steps).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=Path,
        help="Directory of YAML inputs; --n_inputs are sampled randomly",
    )
    input_group.add_argument(
        "--input_yamls",
        nargs="+",
        type=Path,
        metavar="YAML",
        help="Explicit list of YAML input files to use",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root output directory; one sub-dir per grid condition",
    )
    parser.add_argument(
        "--n_inputs",
        type=int,
        default=None,
        help="Number of YAML inputs to sample when --input_dir is used; default uses all inputs",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split the input list into this many shards and run only one of them (see --shard_index)",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Which shard to run, 0-based; requires --num_shards > 1",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=4,
        help="GPUs to hand Boltz-2 for each condition",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for input sampling and diffusion",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Boltz cache directory (ccd.pkl + mols/)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a Boltz-2 .ckpt file",
    )
    parser.add_argument(
        "--diffusion_samples",
        type=int,
        default=1,
        help="Number of diffusion samples per input per condition",
    )
    parser.add_argument(
        "--msa_modes",
        nargs="+",
        choices=MSA_MODES,
        default=None,
        metavar="MODE",
        help="Restrict grid to these MSA modes (default: all). Choices: %(choices)s",
    )
    parser.add_argument(
        "--use_kernels",
        action="store_true",
        default=False,
        help="Enable cuequivariance CUDA kernels (requires cuequivariance_ops_torch)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the conditions and commands without running anything",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-run conditions even if their output already exists",
    )
    parser.add_argument(
        "--sabdab_tsv",
        type=Path,
        default=None,
        metavar="TSV",
        help=(
            "Path to SAbDab summary TSV (columns: pdb, Hchain, Lchain, …). "
            "When provided, antibody chain identification in antigen_only mode uses "
            "the TSV instead of the H/L chain-name heuristic."
        ),
    )
    parser.add_argument(
        "--conditions_file",
        type=Path,
        default=None,
        metavar="YAML",
        help=(
            "YAML file defining an explicit list of conditions to run instead of the "
            "full grid product. Each entry must have msa_mode, recycling_steps, and "
            "step_scale; optionally diffusion_samples (overrides --diffusion_samples "
            "for that condition). Example entry:\n"
            "  - msa_mode: no_msa\n    recycling_steps: 3\n    step_scale: 1.5\n    diffusion_samples: 3"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _collect_input_yamls(args: argparse.Namespace) -> list[Path]:
    """Collect and return the list of input YAML files based on CLI args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    list of Path
        Resolved paths to YAML input files
    """
    if args.input_yamls:
        yaml_files = [p.resolve() for p in args.input_yamls]
    else:
        all_yamls = sorted(args.input_dir.glob("*.yaml"))
        if not all_yamls:
            raise SystemExit(f"No YAML files found in {args.input_dir}")
        if args.n_inputs is None:
            yaml_files = all_yamls
            LOGGER.info("Using all %d inputs from %s", len(yaml_files), args.input_dir)
        else:
            random.seed(args.seed)
            n = min(args.n_inputs, len(all_yamls))
            yaml_files = random.sample(all_yamls, n)
            LOGGER.info("Sampled %d / %d inputs (seed=%d)", n, len(all_yamls), args.seed)

    if args.num_shards > 1:
        if not 0 <= args.shard_index < args.num_shards:
            raise SystemExit(f"--shard_index {args.shard_index} out of range for --num_shards {args.num_shards}")
        # Striped, not contiguous: the inputs are sorted by name, so a contiguous slice would
        # hand one shard a block of same-entry inputs and the length distribution would differ
        # between shards.  Striping keeps every shard's runtime comparable.
        total = len(yaml_files)
        yaml_files = yaml_files[args.shard_index :: args.num_shards]
        LOGGER.info("Shard %d/%d: %d of %d inputs", args.shard_index, args.num_shards, len(yaml_files), total)

    LOGGER.info("Input files (%d):", len(yaml_files))
    for f in yaml_files:
        LOGGER.info("  %s", f.name)
    return yaml_files


def _load_sabdab_binders_if_needed(
    args: argparse.Namespace,
) -> dict[str, set[str]] | None:
    """Load SAbDab binders if a TSV is provided, with warnings if needed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    dict or None
        Mapping of PDB codes to binder chain IDs, or None
    """
    if args.sabdab_tsv:
        return load_sabdab_binders(args.sabdab_tsv)
    if any(m in (args.msa_modes or MSA_MODES) for m in ("antigen_only",)):
        LOGGER.warning(
            "Running antigen_only mode without --sabdab_tsv: "
            "falling back to H/L chain-name heuristic (may misidentify non-H/L chains)"
        )
    return None


def _build_conditions(args: argparse.Namespace) -> list[dict]:
    """Build the list of grid conditions from CLI args or explicit file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    list of dict
        Grid conditions to run
    """
    if args.conditions_file is not None:
        if not args.conditions_file.exists():
            raise SystemExit(f"--conditions_file not found: {args.conditions_file}")
        with args.conditions_file.open(encoding="utf-8") as _fh:
            conditions: list[dict] = yaml.safe_load(_fh)
        if not isinstance(conditions, list) or not conditions:
            raise SystemExit(f"--conditions_file must contain a non-empty YAML list: {args.conditions_file}")
        LOGGER.info("Loaded %d explicit conditions from %s", len(conditions), args.conditions_file)
    else:
        active_msa_modes = args.msa_modes if args.msa_modes else MSA_MODES
        conditions = [
            {"msa_mode": msa, "recycling_steps": rec, "step_scale": scale}
            for msa, rec, scale in product(active_msa_modes, RECYCLING_STEPS, STEP_SCALES)
        ]
    return conditions


def _write_grid_config(
    output_dir: Path,
    conditions: list[dict],
    yaml_files: list[Path],
    args: argparse.Namespace,
) -> None:
    """Write the grid configuration file before running.

    Parameters
    ----------
    output_dir : Path
        Root output directory
    conditions : list of dict
        Grid conditions
    yaml_files : list of Path
        Input YAML files
    args : argparse.Namespace
        Parsed command-line arguments
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_config = {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "n_conditions": len(conditions),
        "n_inputs": len(yaml_files),
        "inputs": [str(f) for f in yaml_files],
        "seed": args.seed,
        "diffusion_samples": args.diffusion_samples,
        "grid_axes": {
            "msa_modes": MSA_MODES,
            "recycling_steps": RECYCLING_STEPS,
            "step_scales": STEP_SCALES,
        },
        "conditions": conditions,
    }
    grid_config_path = output_dir / "grid_config.yaml"
    grid_config_path.write_text(
        yaml.dump(grid_config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    LOGGER.info("Grid config written → %s", grid_config_path)


def _should_skip_condition(
    cond_output_dir: Path,
    yaml_files: list[Path],
    diffusion_samples: int,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    """Check if a condition should be skipped because it's already complete.

    Parameters
    ----------
    cond_output_dir : Path
        Output directory for this condition
    yaml_files : list of Path
        Input YAML files
    diffusion_samples : int
        Number of diffusion samples
    overwrite : bool
        Whether to overwrite existing outputs
    dry_run : bool
        Whether this is a dry run

    Returns
    -------
    bool
        True if condition should be skipped
    """
    if overwrite or dry_run:
        return False

    staging_dir_stem = "_staged_inputs"
    expected_preds = cond_output_dir / f"boltz_results_{staging_dir_stem}" / "predictions"
    n_expected = len(yaml_files) * diffusion_samples
    n_found = len(list(expected_preds.rglob("*.pdb"))) if expected_preds.exists() else 0
    if n_found >= n_expected:
        LOGGER.info("  Skipping (found %d/%d PDBs) – use --overwrite to re-run", n_found, n_expected)
        return True
    return False


def _run_condition(
    cond: dict,
    cond_output_dir: Path,
    yaml_files: list[Path],
    args: argparse.Namespace,
    sabdab_binders: dict[str, set[str]] | None,
) -> float:
    """Run a single grid condition and return elapsed time.

    Parameters
    ----------
    cond : dict
        Condition parameters
    cond_output_dir : Path
        Output directory for this condition
    yaml_files : list of Path
        Input YAML files
    args : argparse.Namespace
        Parsed command-line arguments
    sabdab_binders : dict or None
        SAbDab binder chain lookup

    Returns
    -------
    float
        Elapsed time in seconds
    """
    msa_mode: str = cond["msa_mode"]
    recycling: int = cond["recycling_steps"]
    step_scale: float = cond["step_scale"]
    diffusion_samples: int = int(cond.get("diffusion_samples", args.diffusion_samples))

    staging_dir = cond_output_dir / "_staged_inputs"
    stage_yamls(yaml_files, staging_dir, msa_mode, sabdab_binders=sabdab_binders)

    config = BoltzPredictConfig(
        cache=args.cache,
        checkpoint=args.checkpoint,
        recycling_steps=recycling,
        sampling_steps=200,
        diffusion_samples=diffusion_samples,
        output_format="pdb",
        write_full_pae=True,
        use_potentials=False,
        override=False,
        devices=args.devices,
        accelerator="gpu",
        seed=args.seed,
        use_msa_server=False,
        no_kernels=not args.use_kernels,
        step_scale=step_scale,
    )
    cmd = build_command(staging_dir, cond_output_dir, config)

    LOGGER.info("Command: %s", " ".join(cmd))

    if args.dry_run:
        return 0.0

    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)  # noqa: S603
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        LOGGER.error("Condition failed (exit %d) – continuing", result.returncode)
        return elapsed

    preds = predictions_dir(cond_output_dir, staging_dir)
    rename_model_outputs(preds, seed=args.seed)

    # Clean up staging dir to save disk space
    shutil.rmtree(staging_dir, ignore_errors=True)

    LOGGER.info("Condition done in %.1fs", elapsed)
    return elapsed


def _finalize_timings(
    timings: dict[str, float],
    wall_start: float,
    yaml_files: list[Path],
    diffusion_samples: int,
    dry_run: bool,
) -> None:
    """Print final timing summary.

    Parameters
    ----------
    timings : dict
        Timing measurements per condition
    wall_start : float
        Wall clock start time
    yaml_files : list of Path
        Input YAML files
    diffusion_samples : int
        Number of diffusion samples
    dry_run : bool
        Whether this was a dry run
    """
    timings["Total"] = time.perf_counter() - wall_start
    if not dry_run:
        n_conditions_run = sum(1 for k, v in timings.items() if k != "Total" and v > 0)
        timings["Boltz-2 grid"] = sum(v for k, v in timings.items() if k != "Total")
        print_timings(
            timings,
            n_inputs=len(yaml_files) * max(n_conditions_run, 1),
            n_samples_per_input=diffusion_samples,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Boltz-2 grid predictions across MSA modes and sampling parameters."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    yaml_files = _collect_input_yamls(args)
    sabdab_binders = _load_sabdab_binders_if_needed(args)
    conditions = _build_conditions(args)

    LOGGER.info(
        "Grid: %d conditions × %d inputs = %d total runs",
        len(conditions),
        len(yaml_files),
        len(conditions) * len(yaml_files),
    )

    _write_grid_config(args.output_dir, conditions, yaml_files, args)

    wall_start = time.perf_counter()
    timings: dict[str, float] = {}

    for cond_idx, cond in enumerate(conditions):
        msa_mode: str = cond["msa_mode"]
        recycling: int = cond["recycling_steps"]
        step_scale: float = cond["step_scale"]
        diffusion_samples: int = int(cond.get("diffusion_samples", args.diffusion_samples))

        tag = f"msa={msa_mode}__recycling={recycling}__scale={step_scale}"
        cond_output_dir = args.output_dir / tag
        cond_output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("[%d/%d] Condition: %s", cond_idx + 1, len(conditions), tag)

        if _should_skip_condition(cond_output_dir, yaml_files, diffusion_samples, args.overwrite, args.dry_run):
            timings[tag] = 0.0
            continue

        elapsed = _run_condition(cond, cond_output_dir, yaml_files, args, sabdab_binders)
        timings[tag] = elapsed

    _finalize_timings(timings, wall_start, yaml_files, args.diffusion_samples, args.dry_run)


if __name__ == "__main__":
    main()
