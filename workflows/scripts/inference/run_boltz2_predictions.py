r"""Run Boltz-2 structure predictions for all YAML inputs in a directory.

Boltz appends a ``boltz_results_{input_dir.stem}`` subfolder to ``--output_dir``
automatically.  Structure and confidence files land under::

    {output_dir}/boltz_results_{input_dir.stem}/predictions/{yaml_stem}/
        {stem}_seed{seed}_sample{N}.pdb
        confidence_{stem}_seed{seed}_sample{N}.json
        plddt_{stem}_seed{seed}_sample{N}.npz
        pae_{stem}_seed{seed}_sample{N}.npz   (if --write_full_pae)

The ``_model_N`` suffix produced by boltz is renamed to ``_seed{seed}_sample{N}``
(or ``_sample{N}`` when no seed is set) after the run.  A
``prediction_config.yaml`` is written to ``--output_dir`` before the run to
document every setting used.

Typical usage::

    python workflows/scripts/inference/run_boltz2_predictions.py \\
        --input_dir  data/Unsal2026/boltz2_inputs \\
        --output_dir data/Unsal2026/boltz2_predictions \\
        --cache      /data/boltz-cache \\
        --checkpoint models/pretrained/boltz2.ckpt

The script attempts to invoke the ``boltz`` CLI; if it is not on ``PATH`` it
falls back to ``python -m boltz``.  All extra Boltz-2 options are exposed as
CLI flags so that the config file always reflects the exact invocation.

"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from confidock.boltz.inference.runner import (
    DEFAULT_ACCELERATOR,
    DEFAULT_DIFFUSION_SAMPLES,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_RECYCLING_STEPS,
    DEFAULT_SAMPLING_STEPS,
    BoltzPredictConfig,
    build_command,
    predictions_dir,
    preprocess_yamls_for_msa,
    print_timings,
    rename_model_outputs,
    write_prediction_config,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Boltz-2 predictions for all YAML inputs in a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/Unsal2026/boltz2_inputs"),
        help="Directory containing Boltz-2 YAML input files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/Unsal2026/boltz2_predictions"),
        help="Root directory for predictions and config",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Boltz cache directory (ccd.pkl + mols/).  Defaults to ~/.boltz or $BOLTZ_CACHE",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a Boltz-2 .ckpt file.  Omit to use the boltz built-in default",
    )
    parser.add_argument(
        "--recycling_steps",
        type=int,
        default=DEFAULT_RECYCLING_STEPS,
        help="Number of recycling steps",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=DEFAULT_SAMPLING_STEPS,
        help="Number of diffusion sampling steps",
    )
    parser.add_argument(
        "--diffusion_samples",
        type=int,
        default=DEFAULT_DIFFUSION_SAMPLES,
        help="Number of diffusion samples (structures) to generate per input",
    )
    parser.add_argument(
        "--output_format",
        default=DEFAULT_OUTPUT_FORMAT,
        choices=["pdb", "mmcif"],
        help="Output structure format",
    )
    parser.add_argument(
        "--write_full_pae",
        action="store_true",
        help="Write the full PAE matrix as a .npz file for each prediction (one per model)",
    )
    parser.add_argument(
        "--use_potentials",
        action="store_true",
        help="Apply inference-time potentials (improves physical quality)",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-run predictions even if output already exists",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPU/CPU devices to use",
    )
    parser.add_argument(
        "--accelerator",
        default=DEFAULT_ACCELERATOR,
        choices=["gpu", "cpu", "tpu"],
        help="Hardware accelerator",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the diffusion sampler (omit for non-deterministic runs)",
    )
    parser.add_argument(
        "--no_msa",
        action="store_true",
        help="Skip MSA computation and run in single-sequence mode (faster but less accurate)",
    )
    parser.add_argument(
        "--no_kernels",
        action="store_true",
        help=(
            "Disable cuequivariance CUDA kernels and fall back to pure PyTorch "
            "(use when cuequivariance-ops-torch is unavailable)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run Boltz-2 predictions for multiple antibody-antigen complexes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    yaml_files = sorted(input_dir.glob("*.yaml"))
    if not yaml_files:
        LOGGER.error("No YAML files found in %s – run prepare_boltz2_inputs.py first", input_dir)
        sys.exit(1)

    LOGGER.info("Found %d YAML input files in %s", len(yaml_files), input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    timings: dict[str, float] = {}

    # When MSA is enabled (default), strip ``msa: empty`` from the YAML inputs
    # by staging modified copies so the originals are not altered.
    # The staging dir is named after the original input dir so that boltz's
    # auto-generated ``boltz_results_{stem}`` output dir has the right name.
    t0 = time.perf_counter()
    if not args.no_msa:
        effective_input_dir = preprocess_yamls_for_msa(input_dir, output_dir / input_dir.name)
    else:
        effective_input_dir = input_dir
    timings["Input staging"] = time.perf_counter() - t0

    # Create config for both writing and command building
    config = BoltzPredictConfig(
        cache=args.cache,
        checkpoint=args.checkpoint,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
        output_format=args.output_format,
        write_full_pae=args.write_full_pae,
        use_potentials=args.use_potentials,
        override=args.override,
        devices=args.devices,
        accelerator=args.accelerator,
        seed=args.seed,
        use_msa_server=not args.no_msa,
        no_kernels=args.no_kernels,
    )

    # Write config before running so it is preserved even if the run fails.
    t0 = time.perf_counter()
    write_prediction_config(output_dir, input_dir, config, no_msa=args.no_msa)
    timings["Config write"] = time.perf_counter() - t0

    cmd = build_command(effective_input_dir, output_dir, config)

    LOGGER.info("Invoking Boltz-2: %s", " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)  # noqa: S603
    timings["Boltz-2 (MSA fetch + diffusion inference)"] = time.perf_counter() - t0

    if result.returncode != 0:
        LOGGER.error("boltz predict exited with code %d", result.returncode)
        sys.exit(result.returncode)

    t0 = time.perf_counter()
    preds = predictions_dir(output_dir, effective_input_dir)
    rename_model_outputs(preds, seed=args.seed)
    timings["Output renaming"] = time.perf_counter() - t0

    timings["Total"] = time.perf_counter() - wall_start

    LOGGER.info("All predictions saved to %s", preds)

    print_timings(
        timings,
        n_inputs=len(yaml_files),
        n_samples_per_input=args.diffusion_samples,
    )


if __name__ == "__main__":
    main()
