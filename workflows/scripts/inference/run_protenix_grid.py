r"""Run a grid of Protenix predictions over MSA mode × recycling cycles × diffusion steps.

Grid axes
---------
msa_mode:
    full_msa  – all chains (antigen + binder) use pre-fetched .a3m MSAs from --msa_dir
    no_msa    – ``--use_msa false``; no MSA information used

cycles:  3, 5, 10   (pairformer recycling cycles; Protenix default: 10)
steps:   100, 200   (diffusion sampling steps; Protenix default: 200)

With 5 input structures this gives 12 conditions × 5 inputs = 60 total predictions.

MSA directory layout
--------------------
The ``--msa_dir`` should contain flat ``.a3m`` files named by the first 16 hex
characters of SHA-256(sequence), e.g.::

    sabdab_msas/
        000095525b64eb5b.a3m
        00043d1fec9123f8.a3m
        ...

These are the files produced by ``fetch_msas.py``.  For each sequence the
script creates a per-sequence staging directory containing:

* ``non_pairing.a3m`` – symlink to the source ``.a3m`` (all MSA hits)
* ``pairing.a3m``     – query-only dummy (no taxonomy info available for
                         cross-chain pairing)

Output layout
-------------
::

    {output_dir}/
        msa=full_msa__cycle=5__step=200/
            {entry_name}/seed_{seed}/predictions/...
            prediction_config.yaml
        msa=no_msa__cycle=3__step=100/
            ...
        _msa_staging/   ← shared MSA dirs, kept for re-use
        grid_config.yaml

Usage::

    poetry run python workflows/scripts/inference/run_protenix_grid.py \\
        --json_dir  /data/sabdab/interfaces \\
        --msa_dir   /data/sabdab/msas \\
        --output_dir data/sabdab_protenix_grid \\
        --n_inputs   5 \\
        --seed       42

To run on specific JSON files instead of sampling randomly::

    poetry run python workflows/scripts/inference/run_protenix_grid.py \\
        --input_jsons \\
            /data/sabdab/interfaces/8jlj_HcLcA.json \\
            /data/sabdab/interfaces/8jlk_HcLcA.json \\
        --msa_dir   /data/sabdab/msas \\
        --output_dir data/sabdab_protenix_grid
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from itertools import product as iproduct
from pathlib import Path

import yaml

from confidock.protenix.inference.runner import check_completion, find_protenix_cli

# Import reusable functions from library
from confidock.protenix.msa import inject_msa, stage_msa_dirs, strip_msa

LOGGER = logging.getLogger(__name__)

# Alias library functions to match script's naming convention
_find_protenix_cli = find_protenix_cli
_stage_msa_dirs = stage_msa_dirs
_inject_msa = inject_msa
_strip_msa = strip_msa
_n_completed = check_completion

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ConditionContext:
    """Context for processing a single grid condition."""

    cond_idx: int
    cond: dict
    conditions: list[dict]
    entries: list[dict]
    entry_names: list[str]
    seq_to_msa_dir: dict[str, Path]
    args: argparse.Namespace
    cli: str
    protenix_path: str
    protenix_version: str
    wall_start: float
    seeds_completed: int
    predictions_completed: int
    total_seeds_needed: int
    timings: dict[str, float]


@dataclass
class SeedProcessingContext:
    """Context for processing a single seed within a condition."""

    current_seed: int
    seed_idx: int
    cond_seeds_list: list[int]
    seed_start_time: float
    base_entries: list[dict]
    cond_output_dir: Path
    msa_mode: str
    cycle: int
    step: int
    use_msa: bool
    tag: str
    args: argparse.Namespace
    cli: str
    protenix_version: str
    env: dict


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

MSA_MODES: list[str] = ["full_msa", "no_msa"]
CYCLES: list[int] = [3, 5, 10]
STEPS: list[int] = [100, 200]

DEFAULT_MODEL = "protenix-v2"
DEFAULT_SEEDS = "101"
DEFAULT_SAMPLES = 1

# Protenix version constants
PROTENIX_VERSION_PYPI = "pypi_v2.0.0"
PROTENIX_VERSION_SHARED = "shared_v0.1.0"


def _log_timestamp(msg: str) -> None:
    """Log a message with a timestamp for debugging hangs."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _calculate_eta(
    completed: int,
    total: int,
    elapsed_seconds: float,
) -> tuple[float, str]:
    """Calculate ETA based on current progress.

    Returns
    -------
        (hours_remaining, formatted_eta_string)
    """
    if completed == 0 or elapsed_seconds == 0:
        return 0.0, "N/A"

    rate = completed / elapsed_seconds  # items per second
    remaining = total - completed
    seconds_remaining = remaining / rate if rate > 0 else 0
    hours_remaining = seconds_remaining / 3600

    # Format ETA
    if hours_remaining < 1:
        eta_str = f"{seconds_remaining/60:.0f}m"
    elif hours_remaining < 24:
        eta_str = f"{hours_remaining:.1f}h"
    else:
        days = hours_remaining / 24
        eta_str = f"{days:.1f}d"

    return hours_remaining, eta_str


def _print_progress_bar(
    completed: int,
    total: int,
    prefix: str = "",
    bar_length: int = 40,
) -> None:
    """Print a simple ASCII progress bar."""
    if total == 0:
        return

    percent = completed / total
    filled = int(bar_length * percent)
    bar = "█" * filled + "░" * (bar_length - filled)

    print(f"\r  {prefix}[{bar}] {completed}/{total} ({percent*100:.1f}%)", end="", flush=True)

    if completed >= total:
        print()  # New line when complete


# ---------------------------------------------------------------------------
# Helpers (functions now imported from confidock.protenix)
# ---------------------------------------------------------------------------

# Note: The following functions are now imported from the library:
# - _find_protenix_cli → protenix.runner.find_protenix_cli
# - _stage_msa_dirs → protenix.msa.stage_msa_dirs
# - _inject_msa → protenix.msa.inject_msa
# - _strip_msa → protenix.msa.strip_msa
# - _n_completed → protenix.runner.check_completion


def _sabdab_json_to_protenix_entry(json_path: Path) -> list[dict]:
    """Convert a sabdab JSON to Protenix entry dict(s).

    Handles two formats:
    - List of Protenix entries (already formatted)
    - Single SAbDab entry (h_seq/l_seq/t_seq format, converts to Protenix)

    Returns a list of Protenix entries (empty if parsing fails or no sequences found).
    """
    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", json_path.name, exc)
        return []

    # If already a list of Protenix entries, return as-is
    if isinstance(data, list):
        if not data:
            LOGGER.warning("Empty JSON list in %s – skipping", json_path.name)
            return []
        return data

    # Convert from SAbDab format (h_seq/l_seq/t_seq) to Protenix format
    if not isinstance(data, dict):
        LOGGER.error("%s is neither dict nor list – skipping", json_path.name)
        return []

    sequences: list[dict] = []
    for key in ("h_seq", "l_seq", "t_seq"):
        seq = data.get(key, "")
        if seq:
            sequences.append(
                {
                    "proteinChain": {
                        "sequence": seq,
                        "count": 1,
                        "modifications": [],
                    }
                }
            )

    if not sequences:
        LOGGER.warning("No sequences found in %s – skipping", json_path.name)
        return []

    return [{"name": json_path.stem, "sequences": sequences}]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Protenix prediction grid (MSA mode × recycling cycles × diffusion steps).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--json_dir",
        type=Path,
        help="Directory of sabdab JSON inputs (each with h_seq / l_seq / t_seq); --n_inputs are sampled randomly",
    )
    input_group.add_argument(
        "--input_jsons",
        nargs="+",
        type=Path,
        metavar="JSON",
        help="Explicit list of sabdab JSON files to use",
    )

    parser.add_argument(
        "--msa_dir",
        type=Path,
        required=True,
        help="Directory of pre-fetched .a3m files named by sha256 hash, e.g. /data/sabdab/msas",
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
        default=5,
        help="Number of JSON inputs to sample when --json_dir is used",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for input sampling",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="Protenix model name – must match the .pt filename stem in --checkpoint_dir",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing {model_name}.pt. "
            "Overrides the default protenix release_data/checkpoint/ location. "
            "E.g. /data/protenix_checkpoint"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=DEFAULT_SEEDS,
        help="Comma-separated Protenix inference seeds, e.g. '0,1,2'",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of diffusion samples per input per condition",
    )
    parser.add_argument(
        "--need_atom_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request atom-level confidence outputs from Protenix. Use --no-need_atom_confidence to disable.",
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
        "--cycles",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=f"Recycling cycle values (default: {CYCLES})",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=f"Diffusion step values (default: {STEPS})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print conditions and commands without running anything",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-run conditions even if their output already exists",
    )
    parser.add_argument(
        "--conditions_file",
        type=Path,
        default=None,
        metavar="YAML",
        help=(
            "YAML file defining an explicit list of conditions to run instead of the "
            "full grid product. Each entry must have msa_mode, cycle, and step; "
            "optionally seeds (overrides --seeds for that condition). Example entry:\n"
            "  - msa_mode: no_msa\n    cycle: 10\n    step: 200\n    seeds: '101,102,103'"
        ),
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU device indices for CUDA_VISIBLE_DEVICES. "
            "E.g., '0,1,2,3' for all 4 A100s, '0' for GPU 0 only. "
            "Default: None (all GPUs visible)"
        ),
    )
    parser.add_argument(
        "--use_ddp",
        action="store_true",
        default=False,
        help=(
            "Use PyTorch DDP (DistributedDataParallel) via torchrun. "
            "Requires --cuda_devices to specify GPUs (e.g., '0,1,2,3'). "
            "Data is automatically split across GPUs via DistributedSampler. "
            "Recommended for multi-GPU inference to avoid redundant data loading."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help=(
            "Number of DataLoader worker processes per GPU. "
            "With DDP on 4 GPUs, total workers = num_workers × 4. "
            "Reduce if hitting OOM errors during data loading (default: 2)"
        ),
    )
    return parser.parse_args()


def _run_ddp_inference(
    batch_json: str,
    output_dir: Path,
    seeds: str,
    samples: int,
    cycle: int,
    step: int,
    model_name: str,
    use_msa: bool,
    need_atom_confidence: bool,
    n_gpus: int,
    cuda_devices: str | None,
    env: dict,
    num_workers: int = 2,
) -> int:
    """Run DDP inference using torchrun.

    Uses torchrun to spawn multiple GPU processes that share data loading via
    DistributedSampler. This is faster than running separate processes because:
    - Model loads once per GPU (not once per batch)
    - Data is automatically split across GPUs
    - No duplicate work

    Args:
        batch_json: Path to input JSON file
        output_dir: Output directory for predictions
        seeds: Comma-separated seed values
        samples: Number of diffusion samples per structure
        cycle: Number of recycling cycles
        step: Number of diffusion steps
        model_name: Protenix model name
        use_msa: Whether to use MSA
        need_atom_confidence: Whether to output atom-level confidence
        n_gpus: Number of GPUs to use
        cuda_devices: CUDA_VISIBLE_DEVICES string (e.g., "0,1,2,3")
        env: Environment variables dict

    Returns
    -------
        Exit code from torchrun
    """
    # Find torchrun in the current Python environment
    torchrun_path = shutil.which("torchrun")
    if not torchrun_path:
        # Try to find it relative to current Python executable
        python_bin_dir = Path(sys.executable).parent
        torchrun_candidate = python_bin_dir / "torchrun"
        if torchrun_candidate.exists():
            torchrun_path = str(torchrun_candidate)
        else:
            LOGGER.error("torchrun not found in PATH or Python bin directory")
            return 1

    LOGGER.info("Using torchrun at: %s", torchrun_path)

    # Find the Protenix inference.py runner
    # Assumes PYTHONPATH includes protenix-src
    protenix_src = Path(env.get("PYTHONPATH", "").split(":")[0])
    inference_py = protenix_src / "runner" / "inference.py"

    if not inference_py.exists():
        LOGGER.error("Protenix inference.py not found at: %s", inference_py)
        LOGGER.error("Ensure PYTHONPATH is set correctly in environment")
        return 1

    # Build torchrun command
    cmd = [
        torchrun_path,
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={n_gpus}",
        str(inference_py),
        "--model_name",
        model_name,
        "--seeds",
        seeds,
        "--dump_dir",
        str(output_dir),
        "--input_json_path",
        batch_json,
        "--use_msa",
        str(use_msa).lower(),
        "--num_workers",
        str(num_workers),
        "--model.N_cycle",
        str(cycle),
        "--sample_diffusion.N_sample",
        str(samples),
        "--sample_diffusion.N_step",
        str(step),
    ]

    # Add atom confidence flag if requested
    if need_atom_confidence:
        cmd.extend(["--need_atom_confidence", "true"])

    LOGGER.info("DDP command: %s", " ".join(cmd))

    # Set CUDA_VISIBLE_DEVICES in environment if specified
    if cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    # Run torchrun
    try:
        result = subprocess.run(
            cmd,
            check=False,
            env=env,
            timeout=432000,  # 4 days
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        LOGGER.error("torchrun timed out after 4 days")
        return -1
    except Exception as exc:
        LOGGER.error("torchrun failed: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# Helper functions for main
# ---------------------------------------------------------------------------


def _collect_input_files(args: argparse.Namespace) -> list[Path]:
    """Collect input JSON files based on CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments

    Returns
    -------
    list of Path
        List of JSON file paths
    """
    if args.input_jsons:
        return [p.resolve() for p in args.input_jsons]

    all_jsons = sorted(args.json_dir.glob("*.json"))
    if not all_jsons:
        raise SystemExit(f"No JSON files found in {args.json_dir}")
    random.seed(args.seed)
    n = min(args.n_inputs, len(all_jsons))
    json_files = random.sample(all_jsons, n)
    LOGGER.info("Sampled %d / %d inputs (seed=%d)", n, len(all_jsons), args.seed)
    return json_files


def _convert_jsons_to_entries(json_files: list[Path]) -> list[dict]:
    """Convert JSON files to Protenix entries.

    Parameters
    ----------
    json_files : list of Path
        JSON files to convert

    Returns
    -------
    list of dict
        Protenix entries
    """
    entries: list[dict] = []
    for jf in json_files:
        file_entries = _sabdab_json_to_protenix_entry(jf)
        entries.extend(file_entries)

    if not entries:
        raise SystemExit("No valid Protenix entries could be built from the input JSONs")
    LOGGER.info("Built %d Protenix entries", len(entries))
    return entries


def _build_grid_conditions(args: argparse.Namespace) -> tuple[list[dict], list[str], list[int], list[int]]:
    """Build grid conditions from CLI arguments or conditions file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments

    Returns
    -------
    tuple
        (conditions, active_msa_modes, active_cycles, active_steps)
    """
    if args.conditions_file is not None:
        if not args.conditions_file.exists():
            raise SystemExit(f"--conditions_file not found: {args.conditions_file}")
        with args.conditions_file.open(encoding="utf-8") as _fh:
            conditions: list[dict] = yaml.safe_load(_fh)
        if not isinstance(conditions, list) or not conditions:
            raise SystemExit(f"--conditions_file must contain a non-empty YAML list: {args.conditions_file}")
        LOGGER.info("Loaded %d explicit conditions from %s", len(conditions), args.conditions_file)
        active_msa_modes = sorted(set(c.get("msa_mode") for c in conditions if "msa_mode" in c))
        active_cycles = sorted(set(c.get("cycle") for c in conditions if "cycle" in c))
        active_steps = sorted(set(c.get("step") for c in conditions if "step" in c))
    else:
        active_msa_modes = args.msa_modes if args.msa_modes else MSA_MODES
        active_cycles = args.cycles if args.cycles else CYCLES
        active_steps = args.steps if args.steps else STEPS
        conditions = [
            {"msa_mode": msa, "cycle": cyc, "step": stp}
            for msa, cyc, stp in iproduct(active_msa_modes, active_cycles, active_steps)
        ]
    return conditions, active_msa_modes, active_cycles, active_steps


def _write_grid_config_file(
    output_dir: Path,
    conditions: list[dict],
    entries: list[dict],
    json_files: list[Path],
    args: argparse.Namespace,
    active_msa_modes: list[str],
    active_cycles: list[int],
    active_steps: list[int],
) -> None:
    """Write grid configuration to YAML file.

    Parameters
    ----------
    output_dir : Path
        Output directory
    conditions : list of dict
        Grid conditions
    entries : list of dict
        Protenix entries
    json_files : list of Path
        Input JSON files
    args : argparse.Namespace
        CLI arguments
    active_msa_modes : list of str
        Active MSA modes
    active_cycles : list of int
        Active cycle values
    active_steps : list of int
        Active step values
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    [int(s.strip()) for s in args.seeds.split(",")]

    grid_config = {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "n_conditions": len(conditions),
        "n_inputs": len(entries),
        "inputs": [str(f) for f in json_files],
        "seed": args.seed,
        "samples": args.samples,
        "need_atom_confidence": args.need_atom_confidence,
        "protenix_seeds": args.seeds,
        "model_name": args.model_name,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "grid_axes": {
            "msa_modes": active_msa_modes,
            "cycles": active_cycles,
            "steps": active_steps,
        },
        "conditions": conditions,
    }
    grid_config_path = output_dir / "grid_config.yaml"
    grid_config_path.write_text(
        yaml.dump(grid_config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    LOGGER.info("Grid config written → %s", grid_config_path)


def _setup_checkpoint_symlink(
    checkpoint_dir: Path,
    model_name: str,
    protenix_path: str,
    protenix_version: str,
) -> None:
    """Set up checkpoint symlink for Protenix.

    Parameters
    ----------
    checkpoint_dir : Path
        Checkpoint directory
    model_name : str
        Model name
    protenix_path : str
        Protenix path
    protenix_version : str
        Protenix version
    """
    try:
        src = (checkpoint_dir / f"{model_name}.pt").resolve()
        if not src.exists():
            LOGGER.warning("Checkpoint not found at %s; protenix will try to download.", src)
            return

        if protenix_version == PROTENIX_VERSION_PYPI:
            ckpt_dst_dir = Path.home() / "checkpoint"
            ckpt_dst_dir.mkdir(parents=True, exist_ok=True)
            dst = ckpt_dst_dir / f"{model_name}.pt"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            LOGGER.info("Symlinked checkpoint (PyPI): %s → %s", src, dst)
        else:
            protenix_pkg_dir = Path(protenix_path) / "protenix"
            ckpt_dst_dir = protenix_pkg_dir / "release_data" / "checkpoint"
            ckpt_dst_dir.mkdir(parents=True, exist_ok=True)
            dst = ckpt_dst_dir / f"{model_name}.pt"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            LOGGER.info("Symlinked checkpoint (Shared): %s → %s", src, dst)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not auto-symlink checkpoint: %s", exc)


def _calculate_total_work(conditions: list[dict], entries: list[dict], args: argparse.Namespace) -> tuple[int, int]:
    """Calculate total predictions and seeds needed.

    Parameters
    ----------
    conditions : list of dict
        Grid conditions
    entries : list of dict
        Protenix entries
    args : argparse.Namespace
        CLI arguments

    Returns
    -------
    tuple
        (total_predictions_needed, total_seeds_needed)
    """
    total_predictions_needed = 0
    total_seeds_needed = 0
    for cond in conditions:
        cond_seeds = str(cond.get("seeds", args.seeds))
        n_seeds = len([s.strip() for s in cond_seeds.split(",")])
        total_predictions_needed += len(entries) * n_seeds * args.samples
        total_seeds_needed += n_seeds
    return total_predictions_needed, total_seeds_needed


def _setup_environment(protenix_version: str, protenix_path: str) -> dict:
    """Set up environment variables for Protenix inference.

    Parameters
    ----------
    protenix_version : str
        Protenix version
    protenix_path : str
        Protenix path

    Returns
    -------
    dict
        Environment variables
    """
    env = os.environ.copy()

    if protenix_version == PROTENIX_VERSION_PYPI:
        current_pythonpath = env.get("PYTHONPATH", "")
        # A Protenix source checkout deliberately *shadows* the pip-installed protenix 2.0.0: it is a
        # later revision, and it is what the published structures were folded with. Overridable via
        # PROTENIX_SRC_DIR because the checkout often has to be staged somewhere a batch job can
        # actually read, which is not where an interactive session keeps it.
        source_protenix_path = env.get("PROTENIX_SRC_DIR", "")
        if source_protenix_path:
            env["PYTHONPATH"] = (
                f"{source_protenix_path}:{current_pythonpath}" if current_pythonpath else source_protenix_path
            )
            LOGGER.info("Protenix source tree (shadows the installed package): %s", source_protenix_path)
    elif protenix_path:
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{protenix_path}:{current_pythonpath}" if current_pythonpath else protenix_path

    shared_hf_cache = Path("/data/hf_cache")
    if shared_hf_cache.exists():
        env["HF_HOME"] = str(shared_hf_cache)
        LOGGER.info("Using shared HuggingFace cache: %s", shared_hf_cache)

    shared_torch_cache = Path("/data/boltz-cache")
    if shared_torch_cache.exists():
        env["TORCH_HOME"] = str(shared_torch_cache)
        LOGGER.info("Using shared Torch cache: %s", shared_torch_cache)

    shared_protenix_root = Path("/data/protenix_checkpoint")
    if (shared_protenix_root / "common" / "components.cif").exists():
        env["PROTENIX_ROOT_DIR"] = str(shared_protenix_root)
        LOGGER.info("Using shared Protenix data cache: %s", shared_protenix_root)

    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    return env


def _filter_incomplete_seeds(
    cond_seeds_list: list[int],
    cond_output_dir: Path,
    entry_names: list[str],
    samples: int,
) -> list[int]:
    """Filter to only incomplete seeds for resumption.

    Parameters
    ----------
    cond_seeds_list : list of int
        All seeds for this condition
    cond_output_dir : Path
        Condition output directory
    entry_names : list[str]
        Entry names
    samples : int
        Number of samples per entry

    Returns
    -------
    list of int
        Incomplete seeds
    """
    incomplete_seeds = []
    for seed in cond_seeds_list:
        seed_done = 0
        for name in entry_names:
            pred_dir = cond_output_dir / name / f"seed_{seed}" / "predictions"
            all_samples_exist = all((pred_dir / f"{name}_sample_{rank}.cif").exists() for rank in range(samples))
            if all_samples_exist:
                seed_done += 1

        if seed_done < len(entry_names):
            incomplete_seeds.append(seed)
            LOGGER.info(
                "  Seed %d: %d / %d entries complete (%.1f%%) - will process",
                seed,
                seed_done,
                len(entry_names),
                (seed_done / len(entry_names)) * 100,
            )
        else:
            LOGGER.info(
                "  Seed %d: %d / %d entries complete (100%%) - skipping", seed, len(entry_names), len(entry_names)
            )
    return incomplete_seeds


def _log_overall_progress(cond_idx: int, seeds_completed: int, total_seeds_needed: int, wall_start: float) -> None:
    """Log overall progress across conditions.

    Parameters
    ----------
    cond_idx : int
        Current condition index
    seeds_completed : int
        Seeds completed so far
    total_seeds_needed : int
        Total seeds needed
    wall_start : float
        Wall clock start time
    """
    if cond_idx > 0:
        overall_elapsed = time.perf_counter() - wall_start
        _, overall_eta = _calculate_eta(seeds_completed, total_seeds_needed, overall_elapsed)
        LOGGER.info("Overall progress: %d/%d seeds complete, ETA: %s", seeds_completed, total_seeds_needed, overall_eta)
        _print_progress_bar(seeds_completed, total_seeds_needed, prefix="Overall", bar_length=50)


def _check_condition_completion(
    cond_output_dir: Path,
    entry_names: list[str],
    cond_seeds_list: list[int],
    samples: int,
    entries: list[dict],
    args: argparse.Namespace,
) -> tuple[bool, int, int]:
    """Check if condition is complete and filter incomplete seeds.

    Parameters
    ----------
    cond_output_dir : Path
        Condition output directory
    entry_names : list of str
        Entry names
    cond_seeds_list : list of int
        Seeds for this condition
    samples : int
        Number of samples
    entries : list of dict
        All entries
    args : argparse.Namespace
        CLI arguments

    Returns
    -------
    tuple
        (should_skip, seeds_to_add, predictions_to_add)
    """
    if args.overwrite or args.dry_run:
        return False, 0, 0

    n_done = _n_completed(cond_output_dir, entry_names, cond_seeds_list, samples)
    if n_done >= len(entries):
        LOGGER.info("  Skipping (all %d entries done) – use --overwrite to re-run", n_done)
        seeds_to_add = len(cond_seeds_list)
        predictions_to_add = len(entries) * len(cond_seeds_list) * samples
        return True, seeds_to_add, predictions_to_add

    return False, 0, 0


def _prepare_condition_entries(
    msa_mode: str, entries: list[dict], seq_to_msa_dir: dict[str, Path]
) -> tuple[list[dict], bool]:
    """Prepare entries for a specific MSA mode.

    Parameters
    ----------
    msa_mode : str
        MSA mode
    entries : list of dict
        Base entries
    seq_to_msa_dir : dict
        Sequence to MSA directory mapping

    Returns
    -------
    tuple
        (base_entries, use_msa)
    """
    if msa_mode == "full_msa":
        return _inject_msa(entries, seq_to_msa_dir), True
    return _strip_msa(entries), False


def _log_seed_progress(seed_idx: int, cond_seeds_list: list[int], seed_start_time: float) -> None:
    """Log progress for current seed within a condition.

    Parameters
    ----------
    seed_idx : int
        Current seed index (0-based)
    cond_seeds_list : list of int
        All seeds in condition
    seed_start_time : float
        Start time for seed processing
    """
    if seed_idx > 0:
        elapsed = time.perf_counter() - seed_start_time
        _, eta = _calculate_eta(seed_idx, len(cond_seeds_list), elapsed)
        _print_progress_bar(seed_idx, len(cond_seeds_list), prefix="Condition seeds", bar_length=40)
        LOGGER.info("  ETA for remaining seeds in condition: %s", eta)


def _prepare_seed_entries(
    base_entries: list[dict],
    cond_output_dir: Path,
    current_seed: int,
    samples: int,
    overwrite: bool,
    dry_run: bool,
) -> list[dict] | None:
    """Prepare entries for a single seed, filtering out completed ones.

    Parameters
    ----------
    base_entries : list of dict
        All base entries
    cond_output_dir : Path
        Condition output directory
    current_seed : int
        Current seed
    samples : int
        Number of samples
    overwrite : bool
        Whether to overwrite existing results
    dry_run : bool
        Whether this is a dry run

    Returns
    -------
    list of dict or None
        Filtered entries to process, or None if all complete
    """
    if overwrite or dry_run:
        return base_entries

    cond_entries = _filter_incomplete_entries(base_entries, cond_output_dir, current_seed, samples)
    n_filtered = len(base_entries) - len(cond_entries)

    if n_filtered > 0:
        LOGGER.info(
            "    Filtered out %d already-complete entries for seed %d, processing %d",
            n_filtered,
            current_seed,
            len(cond_entries),
        )

    if not cond_entries:
        LOGGER.info("    All entries complete for seed %d, skipping", current_seed)
        return None

    return cond_entries


def _log_gpu_setup(current_seed: int, cuda_devices: str | None, use_ddp: bool) -> None:
    """Log GPU configuration for the current seed.

    Parameters
    ----------
    current_seed : int
        Current seed
    cuda_devices : str or None
        CUDA device string
    use_ddp : bool
        Whether using DDP
    """
    if cuda_devices:
        if use_ddp:
            LOGGER.info("    [Seed %d] Multi-GPU DDP inference: %s GPUs", current_seed, len(cuda_devices.split(",")))
        else:
            LOGGER.info("    [Seed %d] Single-GPU mode", current_seed)


def _log_first_run_warning(cli: str, cond_output_dir: Path, current_seed: int) -> None:
    """Log warning if this is the first run (CUDA compilation expected).

    Parameters
    ----------
    cli : str
        CLI command
    cond_output_dir : Path
        Condition output directory
    current_seed : int
        Current seed
    """
    if cli == "protenix" and not (cond_output_dir / f"seed_{current_seed}").exists():
        LOGGER.info(
            "    [Seed %d] First run detected - compiling CUDA kernels... (this may take 5-10 minutes)", current_seed
        )


def _process_single_seed(seed_ctx: SeedProcessingContext) -> tuple[float, int]:
    """Process a single seed for a condition.

    Parameters
    ----------
    seed_ctx : SeedProcessingContext
        Context containing all parameters for seed processing

    Returns
    -------
    tuple
        (elapsed_time, predictions_completed)
    """
    seed_tag = f"{seed_ctx.tag}__seed={seed_ctx.current_seed}"
    LOGGER.info("")
    LOGGER.info(
        "  [Seed %d/%d] Processing seed %d",
        seed_ctx.seed_idx + 1,
        len(seed_ctx.cond_seeds_list),
        seed_ctx.current_seed,
    )

    _log_seed_progress(seed_ctx.seed_idx, seed_ctx.cond_seeds_list, seed_ctx.seed_start_time)

    cond_entries = _prepare_seed_entries(
        seed_ctx.base_entries,
        seed_ctx.cond_output_dir,
        seed_ctx.current_seed,
        seed_ctx.args.samples,
        seed_ctx.args.overwrite,
        seed_ctx.args.dry_run,
    )

    if cond_entries is None:
        return 0.0, 0

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        prefix=f"protenix_grid_{seed_ctx.msa_mode}_seed{seed_ctx.current_seed}_",
    ) as tmp:
        json.dump(cond_entries, tmp, indent=2)
        batch_json = tmp.name

    cond_seeds = str(seed_ctx.current_seed)
    cmd = _build_protenix_command(
        seed_ctx.cli,
        seed_ctx.protenix_version,
        batch_json,
        seed_ctx.cond_output_dir,
        cond_seeds,
        seed_ctx.args,
        seed_ctx.cycle,
        seed_ctx.step,
        seed_ctx.use_msa,
    )

    LOGGER.info("    Command: %s", " ".join(cmd))

    if seed_ctx.args.dry_run:
        Path(batch_json).unlink(missing_ok=True)
        return 0.0, 0

    _write_seed_config(
        seed_ctx.cond_output_dir,
        seed_tag,
        seed_ctx.msa_mode,
        seed_ctx.cycle,
        seed_ctx.step,
        seed_ctx.current_seed,
        seed_ctx.args,
        cond_entries,
    )

    t0 = time.perf_counter()

    if seed_ctx.args.cuda_devices:
        seed_ctx.env["CUDA_VISIBLE_DEVICES"] = seed_ctx.args.cuda_devices
    _log_gpu_setup(seed_ctx.current_seed, seed_ctx.args.cuda_devices, seed_ctx.args.use_ddp)
    _log_first_run_warning(seed_ctx.cli, seed_ctx.cond_output_dir, seed_ctx.current_seed)

    result = _run_inference_subprocess(
        cmd,
        seed_ctx.args,
        seed_ctx.env,
        seed_tag,
        seed_ctx.protenix_version,
        batch_json,
        seed_ctx.cond_output_dir,
        cond_seeds,
        seed_ctx.cycle,
        seed_ctx.step,
        seed_ctx.use_msa,
    )

    elapsed = time.perf_counter() - t0
    Path(batch_json).unlink(missing_ok=True)

    if result.returncode != 0:
        LOGGER.error("Seed %d failed (exit %d) – continuing to next seed", seed_ctx.current_seed, result.returncode)
        return elapsed, 0

    LOGGER.info("    Seed %d done in %.1f s (%.1f hrs)", seed_ctx.current_seed, elapsed, elapsed / 3600)
    predictions_this_seed = len(cond_entries) * seed_ctx.args.samples
    return elapsed, predictions_this_seed


def _write_seed_config(
    cond_output_dir: Path,
    seed_tag: str,
    msa_mode: str,
    cycle: int,
    step: int,
    current_seed: int,
    args: argparse.Namespace,
    cond_entries: list[dict],
) -> None:
    """Write per-seed configuration file.

    Parameters
    ----------
    cond_output_dir : Path
        Condition output directory
    seed_tag : str
        Seed tag
    msa_mode : str
        MSA mode
    cycle : int
        Cycles
    step : int
        Steps
    current_seed : int
        Current seed
    args : argparse.Namespace
        CLI arguments
    cond_entries : list of dict
        Entries for this seed
    """
    seed_config_file = cond_output_dir / f"prediction_config_seed_{current_seed}.yaml"
    seed_config_file.write_text(
        yaml.dump(
            {
                "tag": seed_tag,
                "msa_mode": msa_mode,
                "cycle": cycle,
                "step": step,
                "seed": current_seed,
                "model_name": args.model_name,
                "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
                "samples": args.samples,
                "need_atom_confidence": args.need_atom_confidence,
                "n_entries": len(cond_entries),
            },
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _process_condition_seeds(
    cond_seeds_list: list[int],
    base_entries: list[dict],
    cond_output_dir: Path,
    args: argparse.Namespace,
    cli: str,
    protenix_version: str,
    msa_mode: str,
    cycle: int,
    step: int,
    use_msa: bool,
    env: dict,
    tag: str,
    timings: dict[str, float],
) -> tuple[int, int]:
    """Process all seeds for a condition.

    Parameters
    ----------
    cond_seeds_list : list of int
        Seeds to process
    base_entries : list of dict
        Base entries
    cond_output_dir : Path
        Condition output directory
    args : argparse.Namespace
        CLI arguments
    cli : str
        CLI command
    protenix_version : str
        Protenix version
    msa_mode : str
        MSA mode
    cycle : int
        Cycles
    step : int
        Steps
    use_msa : bool
        Use MSA
    env : dict
        Environment
    tag : str
        Condition tag
    timings : dict
        Timing dictionary

    Returns
    -------
    tuple
        (seeds_completed, predictions_completed)
    """
    seed_start_time = time.perf_counter()
    seeds_completed = 0
    predictions_completed = 0

    for seed_idx, current_seed in enumerate(cond_seeds_list):
        seed_ctx = SeedProcessingContext(
            current_seed=current_seed,
            seed_idx=seed_idx,
            cond_seeds_list=cond_seeds_list,
            seed_start_time=seed_start_time,
            base_entries=base_entries,
            cond_output_dir=cond_output_dir,
            msa_mode=msa_mode,
            cycle=cycle,
            step=step,
            use_msa=use_msa,
            tag=tag,
            args=args,
            cli=cli,
            protenix_version=protenix_version,
            env=env,
        )
        elapsed, predictions_this_seed = _process_single_seed(seed_ctx)

        seed_timing_key = f"{tag}__seed={current_seed}"
        timings[seed_timing_key] = elapsed

        seeds_completed += 1
        predictions_completed += predictions_this_seed

        if seed_idx + 1 < len(cond_seeds_list):
            cond_elapsed = time.perf_counter() - seed_start_time
            _, eta = _calculate_eta(seed_idx + 1, len(cond_seeds_list), cond_elapsed)
            LOGGER.info(
                "  └─ Condition progress: %d/%d seeds (%.1f%%), ETA: %s",
                seed_idx + 1,
                len(cond_seeds_list),
                ((seed_idx + 1) / len(cond_seeds_list)) * 100,
                eta,
            )

    return seeds_completed, predictions_completed


def _process_single_condition(ctx: ConditionContext) -> tuple[int, int]:
    """Process a single grid condition.

    Parameters
    ----------
    ctx : ConditionContext
        Context containing all parameters for condition processing

    Returns
    -------
    tuple
        (updated_seeds_completed, updated_predictions_completed)
    """
    _log_timestamp(f"Condition [{ctx.cond_idx + 1}/{len(ctx.conditions)}] starting...")
    msa_mode: str = ctx.cond["msa_mode"]
    cycle: int = ctx.cond["cycle"]
    step: int = ctx.cond["step"]
    cond_seeds: str = str(ctx.cond.get("seeds", ctx.args.seeds))

    tag = f"msa={msa_mode}__cycle={cycle}__step={step}"
    cond_output_dir = ctx.args.output_dir / tag
    cond_output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("")
    LOGGER.info("─" * 60)
    LOGGER.info("[%d/%d] Condition: %s", ctx.cond_idx + 1, len(ctx.conditions), tag)

    _log_overall_progress(ctx.cond_idx, ctx.seeds_completed, ctx.total_seeds_needed, ctx.wall_start)

    cond_seeds_list = [int(s.strip()) for s in cond_seeds.split(",")]

    should_skip, seeds_to_add, predictions_to_add = _check_condition_completion(
        cond_output_dir, ctx.entry_names, cond_seeds_list, ctx.args.samples, ctx.entries, ctx.args
    )

    if should_skip:
        ctx.timings[tag] = 0.0
        return ctx.seeds_completed + seeds_to_add, ctx.predictions_completed + predictions_to_add

    if not ctx.args.overwrite and not ctx.args.dry_run:
        incomplete_seeds = _filter_incomplete_seeds(cond_seeds_list, cond_output_dir, ctx.entry_names, ctx.args.samples)
        if not incomplete_seeds:
            LOGGER.info("  All seeds complete for condition %s", tag)
            ctx.timings[tag] = 0.0
            return (
                ctx.seeds_completed + len(cond_seeds_list),
                ctx.predictions_completed + len(ctx.entries) * len(cond_seeds_list) * ctx.args.samples,
            )

        cond_seeds_list = incomplete_seeds
        LOGGER.info(
            "  Will process seeds: %s (iterating per-seed for optimal resumption)",
            ",".join(str(s) for s in cond_seeds_list),
        )

    base_entries, use_msa = _prepare_condition_entries(msa_mode, ctx.entries, ctx.seq_to_msa_dir)
    env = _setup_environment(ctx.protenix_version, ctx.protenix_path)

    seeds_added, predictions_added = _process_condition_seeds(
        cond_seeds_list,
        base_entries,
        cond_output_dir,
        ctx.args,
        ctx.cli,
        ctx.protenix_version,
        msa_mode,
        cycle,
        step,
        use_msa,
        env,
        tag,
        ctx.timings,
    )

    cond_elapsed_total = ctx.timings.get(f"{tag}__seed={cond_seeds_list[-1]}", 0.0) if cond_seeds_list else 0.0
    LOGGER.info("")
    _print_progress_bar(len(cond_seeds_list), len(cond_seeds_list), prefix="Condition seeds", bar_length=40)
    LOGGER.info(
        "✓ Condition %s completed (processed %d seeds in %.1f hrs)",
        tag,
        len(cond_seeds_list),
        cond_elapsed_total / 3600,
    )
    LOGGER.info("─" * 60)

    return ctx.seeds_completed + seeds_added, ctx.predictions_completed + predictions_added


def _print_final_summary(
    args: argparse.Namespace,
    predictions_completed: int,
    total_predictions_needed: int,
    seeds_completed: int,
    total_seeds_needed: int,
    wall_elapsed: float,
    timings: dict[str, float],
) -> None:
    """Print final summary statistics.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments
    predictions_completed : int
        Predictions completed
    total_predictions_needed : int
        Total predictions needed
    seeds_completed : int
        Seeds completed
    total_seeds_needed : int
        Total seeds needed
    wall_elapsed : float
        Total wall time
    timings : dict
        Timing dictionary
    """
    if args.dry_run:
        return

    LOGGER.info("")
    LOGGER.info("═" * 60)
    LOGGER.info("FINAL SUMMARY")
    LOGGER.info("═" * 60)

    _print_progress_bar(seeds_completed, total_seeds_needed, prefix="Overall", bar_length=50)
    LOGGER.info("")

    if wall_elapsed > 0 and predictions_completed > 0:
        pred_per_hour = predictions_completed / (wall_elapsed / 3600)
        LOGGER.info(
            "Predictions completed: %d / %d (%.1f%%)",
            predictions_completed,
            total_predictions_needed,
            (predictions_completed / total_predictions_needed * 100) if total_predictions_needed > 0 else 0,
        )
        LOGGER.info("Seeds completed: %d / %d", seeds_completed, total_seeds_needed)
        LOGGER.info("Throughput: %.1f predictions/hour", pred_per_hour)
    else:
        LOGGER.info("Predictions completed: %d / %d", predictions_completed, total_predictions_needed)
        LOGGER.info("Seeds completed: %d / %d", seeds_completed, total_seeds_needed)

    LOGGER.info("Total wall time: %.1f hours (%.1f days)", wall_elapsed / 3600, wall_elapsed / 86400)
    LOGGER.info("")
    LOGGER.info("Per-seed timings:")
    for k, v in sorted(timings.items()):
        if k != "Total" and v > 0:
            LOGGER.info("  %-50s %.1f hrs", k, v / 3600)
    LOGGER.info("")

    LOGGER.info("Output directory: %s", args.output_dir)
    LOGGER.info("═" * 60)


def _filter_incomplete_entries(
    base_entries: list[dict],
    cond_output_dir: Path,
    current_seed: int,
    samples: int,
) -> list[dict]:
    """Filter entries to only those incomplete for a specific seed.

    Parameters
    ----------
    base_entries : list of dict
        All entries
    cond_output_dir : Path
        Condition output directory
    current_seed : int
        Current seed
    samples : int
        Number of samples

    Returns
    -------
    list of dict
        Incomplete entries
    """
    cond_entries = []
    for entry in base_entries:
        name = entry["name"]
        pred_dir = cond_output_dir / name / f"seed_{current_seed}" / "predictions"
        all_samples_exist = all((pred_dir / f"{name}_sample_{rank}.cif").exists() for rank in range(samples))
        if not all_samples_exist:
            cond_entries.append(entry)
    return cond_entries


def _build_protenix_command(
    cli: str,
    protenix_version: str,
    batch_json: str,
    cond_output_dir: Path,
    cond_seeds: str,
    args: argparse.Namespace,
    cycle: int,
    step: int,
    use_msa: bool,
) -> list[str]:
    """Build Protenix inference command.

    Parameters
    ----------
    cli : str
        CLI command prefix
    protenix_version : str
        Protenix version
    batch_json : str
        Batch JSON path
    cond_output_dir : Path
        Output directory
    cond_seeds : str
        Seeds string
    args : argparse.Namespace
        CLI arguments
    cycle : int
        Number of cycles
    step : int
        Number of steps
    use_msa : bool
        Whether to use MSA

    Returns
    -------
    list of str
        Command list
    """
    if protenix_version == PROTENIX_VERSION_PYPI:
        if args.use_ddp:
            cmd: list[str] = [
                sys.executable,
                "-m",
                "protenix.runner.batch_inference",
                "--input",
                batch_json,
                "--out_dir",
                str(cond_output_dir),
                "--seeds",
                cond_seeds,
                "--sample",
                str(args.samples),
                "--cycle",
                str(cycle),
                "--step",
                str(step),
                "--model_name",
                args.model_name,
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
                str(cond_output_dir),
                "-s",
                cond_seeds,
                "-e",
                str(args.samples),
                "-c",
                str(cycle),
                "-p",
                str(step),
                "-n",
                args.model_name,
                "--use_msa",
                str(use_msa).lower(),
                "--use_default_params",
                "false",
            ]
            if args.need_atom_confidence:
                cmd.extend(["--need_atom_confidence", "true"])
    else:
        cmd: list[str] = [
            *cli.split(),
            "--input",
            batch_json,
            "--out_dir",
            str(cond_output_dir),
            "--seeds",
            cond_seeds,
            "--sample",
            str(args.samples),
            "--cycle",
            str(cycle),
            "--step",
            str(step),
            "--model_name",
            args.model_name,
            "--use_msa",
            str(use_msa).lower(),
            "--use_default_params",
            "false",
        ]
        if args.need_atom_confidence:
            LOGGER.warning(
                "Atom-level confidence not available: Shared Protenix v0.1.0 doesn't support --need_atom_confidence"
            )
    return cmd


def _run_inference_subprocess(
    cmd: list[str],
    args: argparse.Namespace,
    env: dict,
    seed_tag: str,
    protenix_version: str,
    batch_json: str,
    cond_output_dir: Path,
    cond_seeds: str,
    cycle: int,
    step: int,
    use_msa: bool,
) -> subprocess.CompletedProcess:
    """Run inference subprocess (DDP or single GPU).

    Parameters
    ----------
    cmd : list of str
        Command list
    args : argparse.Namespace
        CLI arguments
    env : dict
        Environment variables
    seed_tag : str
        Seed tag for logging
    protenix_version : str
        Protenix version
    batch_json : str
        Batch JSON path
    cond_output_dir : Path
        Output directory
    cond_seeds : str
        Seeds string
    cycle : int
        Cycles
    step : int
        Steps
    use_msa : bool
        Use MSA

    Returns
    -------
    subprocess.CompletedProcess
        Completed process
    """
    cmd_str = " ".join(cmd)
    _log_timestamp(f"[{seed_tag}] Subprocess starting...")

    if args.use_ddp and protenix_version == PROTENIX_VERSION_PYPI:
        n_gpus = len(args.cuda_devices.split(",")) if args.cuda_devices else 4
        LOGGER.info("    Using custom DDP launcher with %d GPUs", n_gpus)
        exit_code = _run_ddp_inference(
            batch_json=batch_json,
            output_dir=cond_output_dir,
            seeds=cond_seeds,
            samples=args.samples,
            cycle=cycle,
            step=step,
            model_name=args.model_name,
            use_msa=use_msa,
            need_atom_confidence=args.need_atom_confidence,
            n_gpus=n_gpus,
            cuda_devices=args.cuda_devices,
            env=env,
            num_workers=args.num_workers,
        )
        result = subprocess.CompletedProcess(cmd_str, returncode=exit_code)
        _log_timestamp(f"[{seed_tag}] DDP inference completed with exit code {exit_code}")
    else:
        try:
            result = subprocess.run(
                cmd_str,
                check=False,
                env=env,
                shell=True,  # noqa: S602
                timeout=172800,  # 2 days per seed
            )
            _log_timestamp(f"[{seed_tag}] Subprocess completed with exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            _log_timestamp(f"[{seed_tag}] Subprocess TIMEOUT (>2 days)")
            LOGGER.error("Seed timed out (>2 days) – subprocess execution exceeded timeout")
            result = subprocess.CompletedProcess(cmd_str, returncode=-1)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _collect_input_json_files(args: argparse.Namespace) -> list[Path]:
    """Collect input JSON files based on arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    list of Path
        List of JSON file paths
    """
    if args.input_jsons:
        json_files = [p.resolve() for p in args.input_jsons]
    else:
        all_jsons = sorted(args.json_dir.glob("*.json"))
        if not all_jsons:
            raise SystemExit(f"No JSON files found in {args.json_dir}")
        random.seed(args.seed)
        n = min(args.n_inputs, len(all_jsons))
        json_files = random.sample(all_jsons, n)
        LOGGER.info("Sampled %d / %d inputs (seed=%d)", n, len(all_jsons), args.seed)

    LOGGER.info("Input files (%d):", len(json_files))
    for f in json_files:
        LOGGER.info("  %s", f.name)
    return json_files


def _load_protenix_entries(json_files: list[Path]) -> list[dict]:
    """Convert SAbDab JSONs to Protenix entries.

    Parameters
    ----------
    json_files : list of Path
        Input JSON files

    Returns
    -------
    list of dict
        Protenix entries
    """
    entries: list[dict] = []
    for jf in json_files:
        file_entries = _sabdab_json_to_protenix_entry(jf)
        entries.extend(file_entries)

    if not entries:
        raise SystemExit("No valid Protenix entries could be built from the input JSONs")
    LOGGER.info("Built %d Protenix entries", len(entries))
    return entries


def _build_protenix_grid_conditions(args: argparse.Namespace) -> tuple[list[dict], list[str], list[int], list[int]]:
    """Build grid conditions from arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    tuple
        (conditions, active_msa_modes, active_cycles, active_steps)
    """
    if args.conditions_file is not None:
        if not args.conditions_file.exists():
            raise SystemExit(f"--conditions_file not found: {args.conditions_file}")
        with args.conditions_file.open(encoding="utf-8") as _fh:
            conditions: list[dict] = yaml.safe_load(_fh)
        if not isinstance(conditions, list) or not conditions:
            raise SystemExit(f"--conditions_file must contain a non-empty YAML list: {args.conditions_file}")
        LOGGER.info("Loaded %d explicit conditions from %s", len(conditions), args.conditions_file)
        # Extract unique values from conditions for grid config reporting
        active_msa_modes = sorted(set(c.get("msa_mode") for c in conditions if "msa_mode" in c))
        active_cycles = sorted(set(c.get("cycle") for c in conditions if "cycle" in c))
        active_steps = sorted(set(c.get("step") for c in conditions if "step" in c))
    else:
        active_msa_modes = args.msa_modes if args.msa_modes else MSA_MODES
        active_cycles = args.cycles if args.cycles else CYCLES
        active_steps = args.steps if args.steps else STEPS

        conditions = [
            {"msa_mode": msa, "cycle": cyc, "step": stp}
            for msa, cyc, stp in iproduct(active_msa_modes, active_cycles, active_steps)
        ]
    return conditions, active_msa_modes, active_cycles, active_steps


def _write_protenix_grid_config(
    args: argparse.Namespace,
    json_files: list[Path],
    entries: list[dict],
    conditions: list[dict],
    active_msa_modes: list[str],
    active_cycles: list[int],
    active_steps: list[int],
) -> None:
    """Write Protenix grid configuration to YAML file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    json_files : list of Path
        Input JSON files
    entries : list of dict
        Protenix entries
    conditions : list of dict
        Grid conditions
    active_msa_modes : list of str
        Active MSA modes
    active_cycles : list of int
        Active cycle values
    active_steps : list of int
        Active step values
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid_config = {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "n_conditions": len(conditions),
        "n_inputs": len(entries),
        "inputs": [str(f) for f in json_files],
        "seed": args.seed,
        "samples": args.samples,
        "need_atom_confidence": args.need_atom_confidence,
        "protenix_seeds": args.seeds,
        "model_name": args.model_name,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "grid_axes": {
            "msa_modes": active_msa_modes,
            "cycles": active_cycles,
            "steps": active_steps,
        },
        "conditions": conditions,
    }
    grid_config_path = args.output_dir / "grid_config.yaml"
    grid_config_path.write_text(
        yaml.dump(grid_config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    LOGGER.info("Grid config written → %s", grid_config_path)


def _calculate_total_work(conditions: list[dict], entries: list[dict], args: argparse.Namespace) -> tuple[int, int]:
    """Calculate total predictions and seeds needed.

    Parameters
    ----------
    conditions : list of dict
        Grid conditions
    entries : list of dict
        Protenix entries
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    tuple
        (total_predictions_needed, total_seeds_needed)
    """
    total_predictions_needed = 0
    total_seeds_needed = 0
    for cond in conditions:
        cond_seeds = str(cond.get("seeds", args.seeds))
        n_seeds = len([s.strip() for s in cond_seeds.split(",")])
        total_predictions_needed += len(entries) * n_seeds * args.samples
        total_seeds_needed += n_seeds
    return total_predictions_needed, total_seeds_needed


def _setup_protenix_checkpoint(args: argparse.Namespace, protenix_path: str, protenix_version: str) -> None:
    """Set up Protenix model checkpoint symlink.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    protenix_path : str
        Path to Protenix installation
    protenix_version : str
        Protenix version identifier
    """
    if args.checkpoint_dir is None:
        return

    try:
        src = (args.checkpoint_dir / f"{args.model_name}.pt").resolve()
        if not src.exists():
            LOGGER.warning("Checkpoint not found at %s; protenix will try to download.", src)
            return

        if protenix_version == PROTENIX_VERSION_PYPI:
            # PyPI v2.0.0 looks in ~/checkpoint/
            ckpt_dst_dir = Path.home() / "checkpoint"
            ckpt_dst_dir.mkdir(parents=True, exist_ok=True)
            dst = ckpt_dst_dir / f"{args.model_name}.pt"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            LOGGER.info("Symlinked checkpoint (PyPI): %s → %s", src, dst)
        else:  # shared_v0.1.0
            protenix_pkg_dir = Path(protenix_path) / "protenix"
            ckpt_dst_dir = protenix_pkg_dir / "release_data" / "checkpoint"
            ckpt_dst_dir.mkdir(parents=True, exist_ok=True)
            dst = ckpt_dst_dir / f"{args.model_name}.pt"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            LOGGER.info("Symlinked checkpoint (Shared): %s → %s", src, dst)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not auto-symlink checkpoint: %s", exc)


def main() -> None:
    """Run Protenix grid predictions across MSA and sampling conditions."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    _log_timestamp("Script started")

    # Collect inputs and load entries
    json_files = _collect_input_json_files(args)
    entries = _load_protenix_entries(json_files)

    # Build grid conditions
    conditions, active_msa_modes, active_cycles, active_steps = _build_protenix_grid_conditions(args)

    LOGGER.info(
        "Grid: %d conditions × %d inputs = %d total runs",
        len(conditions),
        len(entries),
        len(conditions) * len(entries),
    )

    # Write grid config
    _write_protenix_grid_config(args, json_files, entries, conditions, active_msa_modes, active_cycles, active_steps)

    # Stage MSA directories (shared across all full_msa conditions)
    seq_to_msa_dir: dict[str, Path] = {}
    if "full_msa" in active_msa_modes:
        staging_dir = args.output_dir / "_msa_staging"
        LOGGER.info("Staging per-sequence MSA dirs in %s …", staging_dir)
        seq_to_msa_dir = _stage_msa_dirs(entries, args.msa_dir, staging_dir)

    # Find Protenix CLI and setup checkpoint
    cli, protenix_path, protenix_version = _find_protenix_cli()
    _setup_protenix_checkpoint(args, protenix_path, protenix_version)

    entry_names = [e["name"] for e in entries]
    wall_start = time.perf_counter()
    timings: dict[str, float] = {}

    # Calculate total work
    total_predictions_needed, total_seeds_needed = _calculate_total_work(conditions, entries, args)
    predictions_completed = 0
    seeds_completed = 0

    LOGGER.info("═" * 60)
    LOGGER.info(
        "Total work: %d conditions × %d seeds = %d predictions",
        len(conditions),
        total_seeds_needed,
        total_predictions_needed,
    )
    LOGGER.info("═" * 60)

    # Run each condition
    for cond_idx, cond in enumerate(conditions):
        ctx = ConditionContext(
            cond_idx=cond_idx,
            cond=cond,
            conditions=conditions,
            entries=entries,
            entry_names=entry_names,
            seq_to_msa_dir=seq_to_msa_dir,
            args=args,
            cli=cli,
            protenix_path=protenix_path,
            protenix_version=protenix_version,
            wall_start=wall_start,
            seeds_completed=seeds_completed,
            predictions_completed=predictions_completed,
            total_seeds_needed=total_seeds_needed,
            timings=timings,
        )
        seeds_completed, predictions_completed = _process_single_condition(ctx)

    wall_elapsed = time.perf_counter() - wall_start
    timings["Total"] = wall_elapsed

    _print_final_summary(
        args,
        predictions_completed,
        total_predictions_needed,
        seeds_completed,
        total_seeds_needed,
        wall_elapsed,
        timings,
    )

    LOGGER.info("All done. Results in %s", args.output_dir)
    _log_timestamp("Script completed successfully")


if __name__ == "__main__":
    main()
