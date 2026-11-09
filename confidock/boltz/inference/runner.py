"""Boltz-2 prediction runner utilities.

Provides helpers for building ``boltz predict`` commands, staging MSA inputs,
post-processing outputs, writing run configs, and reporting timing.

Typical use::

    from confidock.boltz.inference.runner import (
        BoltzPredictConfig,
        build_command,
        preprocess_yamls_for_msa,
        predictions_dir,
        rename_model_outputs,
        write_prediction_config,
        print_timings,
    )
"""

from __future__ import annotations

import datetime
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

LOGGER = logging.getLogger(__name__)

# Boltz-2 defaults (matches ``boltz predict --help``)
DEFAULT_RECYCLING_STEPS = 3
DEFAULT_SAMPLING_STEPS = 200
DEFAULT_DIFFUSION_SAMPLES = 1
DEFAULT_OUTPUT_FORMAT = "pdb"
DEFAULT_ACCELERATOR = "gpu"


@dataclass
class BoltzPredictConfig:
    """Configuration for a Boltz-2 prediction run."""

    recycling_steps: int = DEFAULT_RECYCLING_STEPS
    sampling_steps: int = DEFAULT_SAMPLING_STEPS
    diffusion_samples: int = DEFAULT_DIFFUSION_SAMPLES
    output_format: str = DEFAULT_OUTPUT_FORMAT
    write_full_pae: bool = False
    use_potentials: bool = False
    override: bool = False
    devices: int = 1
    accelerator: str = DEFAULT_ACCELERATOR
    seed: int | None = None
    use_msa_server: bool = False
    no_kernels: bool = False
    step_scale: float | None = None
    cache: Path | None = None
    checkpoint: Path | None = None


# ---------------------------------------------------------------------------
# Input staging
# ---------------------------------------------------------------------------


def preprocess_yamls_for_msa(input_dir: Path, staging_dir: Path) -> Path:
    """Copy YAMLs from *input_dir* to *staging_dir* with ``msa: empty`` removed.

    Stripping ``msa: empty`` tells Boltz to fetch MSAs automatically via its
    MMseqs2 server instead of running in single-sequence mode.

    The staging directory is named after the original input directory so that
    Boltz's auto-generated ``boltz_results_{stem}`` output directory has the
    correct name.

    Returns the staging directory path.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for yaml_path in sorted(input_dir.glob("*.yaml")):
        with yaml_path.open(encoding="utf-8") as fh:
            content = yaml.safe_load(fh)
        for entry in content.get("sequences", []):
            for chain_data in entry.values():
                if isinstance(chain_data, dict) and chain_data.get("msa") == "empty":
                    del chain_data["msa"]
        out_path = staging_dir / yaml_path.name
        with out_path.open("w", encoding="utf-8") as fh:
            yaml.dump(
                content,
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=10_000,
            )
    n = len(list(staging_dir.glob("*.yaml")))
    LOGGER.info("Staged %d MSA-enabled YAML files in %s", n, staging_dir)
    return staging_dir


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def predictions_dir(output_dir: Path, input_dir: Path) -> Path:
    """Return the path where Boltz writes its ``predictions`` subdirectory.

    Boltz appends ``boltz_results_{input.stem}`` to ``--out_dir``, then places
    per-entry subdirectories under ``predictions/``.
    """
    return output_dir / f"boltz_results_{input_dir.stem}" / "predictions"


def rename_model_outputs(preds_dir: Path, seed: int | None) -> None:
    """Rename ``_model_N`` suffixes in *preds_dir* to ``_seed{seed}_sample{N}``.

    Boltz names all outputs ``_model_{N}`` regardless of diffusion sample count
    or seed.  This post-run rename embeds the seed and sample index explicitly.
    """
    seed_tag = f"_seed{seed}" if seed is not None else ""
    pattern = re.compile(r"_model_(\d+)(?=[._])")

    renamed = 0
    for path in sorted(preds_dir.rglob("*")):
        if not path.is_file():
            continue
        new_name = pattern.sub(lambda m: f"{seed_tag}_sample{m.group(1)}", path.name)
        if new_name != path.name:
            path.rename(path.parent / new_name)
            renamed += 1

    LOGGER.info(
        "Renamed %d output files (model_N → %s_sampleN)",
        renamed,
        seed_tag or "(no seed)",
    )


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def boltz_executable() -> list[str]:
    """Return the prefix of a ``boltz predict`` command.

    Tries the ``boltz`` binary on PATH first, then the console script sitting beside the running
    interpreter -- which is where a venv puts it when the venv's bin is not on PATH, e.g. a job that
    invokes ``/path/to/.venv/bin/python`` directly.

    Raises
    ------
    RuntimeError
        If neither location yields a ``boltz`` executable.  This used to fall back to
        ``python -m boltz``, which can never work (the package has no ``__main__``): the grid staged
        all of its inputs, spent two minutes on it, and only then died with an import error.
    """
    if shutil.which("boltz"):
        return ["boltz"]
    sibling = Path(sys.executable).parent / "boltz"
    if sibling.exists():
        return [str(sibling)]
    raise RuntimeError(f"no 'boltz' executable on PATH or beside {sys.executable}")


def build_command(
    input_dir: Path,
    output_dir: Path,
    config: BoltzPredictConfig,
) -> list[str]:
    """Return the full ``boltz predict`` argument list for a prediction run."""
    cmd = boltz_executable() + ["predict", str(input_dir)]

    cmd += ["--out_dir", str(output_dir)]
    cmd += ["--recycling_steps", str(config.recycling_steps)]
    cmd += ["--sampling_steps", str(config.sampling_steps)]
    cmd += ["--diffusion_samples", str(config.diffusion_samples)]
    cmd += ["--output_format", config.output_format]
    cmd += ["--devices", str(config.devices)]
    cmd += ["--accelerator", config.accelerator]

    if config.cache is not None:
        cmd += ["--cache", str(config.cache)]
    if config.checkpoint is not None:
        cmd += ["--checkpoint", str(config.checkpoint)]
    if config.seed is not None:
        cmd += ["--seed", str(config.seed)]
    if config.write_full_pae:
        cmd.append("--write_full_pae")
    if config.use_potentials:
        cmd.append("--use_potentials")
    if config.override:
        cmd.append("--override")
    if config.use_msa_server:
        cmd.append("--use_msa_server")
    if config.no_kernels:
        cmd.append("--no_kernels")
    if config.step_scale is not None:
        cmd += ["--step_scale", str(config.step_scale)]

    return cmd


# ---------------------------------------------------------------------------
# Config serialisation
# ---------------------------------------------------------------------------


def write_prediction_config(
    output_dir: Path,
    input_dir: Path,
    config: BoltzPredictConfig,
    *,
    no_msa: bool = False,
) -> Path:
    """Serialise all prediction settings to ``output_dir/prediction_config.yaml``.

    Returns the path of the written config file.
    """
    config_dict: dict = {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "dataset": "Unsal2026",
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "boltz2_settings": {
            "recycling_steps": config.recycling_steps,
            "sampling_steps": config.sampling_steps,
            "diffusion_samples": config.diffusion_samples,
            "output_format": config.output_format,
            "write_full_pae": config.write_full_pae,
            "use_potentials": config.use_potentials,
            "devices": config.devices,
            "accelerator": config.accelerator,
            "seed": config.seed,
            "msa_mode": "single-sequence (no MSA)" if no_msa else "auto (boltz MMseqs2)",
            "step_scale": config.step_scale,
            "checkpoint": str(config.checkpoint.resolve()) if config.checkpoint else "boltz2_default",
            "cache": str(config.cache.resolve()) if config.cache else "~/.boltz (default)",
        },
    }

    config_path = output_dir / "prediction_config.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.dump(config_dict, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    LOGGER.info("Prediction config written → %s", config_path)
    return config_path


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def fmt_duration(seconds: float) -> str:
    """Return a human-readable duration string (e.g. ``1h 23m 04s``)."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"


def print_timings(
    timings: dict[str, float],
    *,
    n_inputs: int | None = None,
    n_samples_per_input: int | None = None,
) -> None:
    """Print a timing summary table with optional per-input / per-sample throughput."""
    total = timings.get("Total", sum(timings.values()))
    col = max(len(k) for k in timings)
    print("\n─── Timing summary " + "─" * (col + 14))
    for step, elapsed in timings.items():
        pct = 100 * elapsed / total if total > 0 and step != "Total" else None
        pct_str = f"  ({pct:4.1f}%)" if pct is not None else ""
        print(f"  {step:<{col}}  {fmt_duration(elapsed)}{pct_str}")

    boltz_key = next((k for k in timings if k.startswith("Boltz-2")), None)
    if boltz_key and n_inputs:
        boltz_t = timings[boltz_key]
        n_total_samples = n_inputs * (n_samples_per_input or 1)
        print(f"  {'─' * (col + 29)}")
        per_input = boltz_t / n_inputs
        print(
            f"  {'Avg per input (MSA + inference)':<{col}}  "
            f"{fmt_duration(per_input)}  "
            f"({per_input / 60:.1f} min/input)"
        )
        per_sample = boltz_t / n_total_samples
        print(
            f"  {'Avg per sample (all inputs)':<{col}}  "
            f"{fmt_duration(per_sample)}  "
            f"({per_sample / 60:.1f} min/sample, "
            f"{n_samples_per_input or 1} sample(s) × {n_inputs} input(s))"
        )
    print("─" * (col + 33))
