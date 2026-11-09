r"""Run ESMFold2 structure predictions from a JSON input file.

The input JSON is a list of entries with the schema::

    [
      {
        "name": "2cba_A",
        "sequences": [
          {"type": "protein", "id": "A", "sequence": "MSHHW..."}
        ]
      }
    ]

Local mode supports ``protein``, ``dna``, and ``ligand`` sequence objects.
Forge mode currently supports protein-only entries.

Typical usage::

    poetry run python workflows/scripts/inference/run_esmfold2_predictions.py \
        --input_json configs/esmfold2_single_chain_test.json \
        --output_dir data/esmfold2_test_runs \
        --mode local

    BIOHUB_API_TOKEN=... poetry run python workflows/scripts/inference/run_esmfold2_predictions.py \
        --input_json configs/esmfold2_single_chain_test.json \
        --output_dir data/esmfold2_test_runs_forge \
        --mode forge
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

# Import from library
from confidock.esmfold2.inference.runner import (
    coerce_modifications,
    load_entries,
    resolve_a3m,
    save_pae_matrix,
    write_prediction_config,
)

LOGGER = logging.getLogger(__name__)

# Alias library functions for backward compatibility
_save_pae_matrix = save_pae_matrix
_resolve_a3m = resolve_a3m
_write_prediction_config = write_prediction_config
_load_entries = load_entries
_coerce_modifications = coerce_modifications

DEFAULT_NUM_LOOPS = 3
DEFAULT_LOCAL_SAMPLING_STEPS = 50
DEFAULT_FORGE_SAMPLING_STEPS = 32
DEFAULT_NUM_DIFFUSION_SAMPLES = 1
DEFAULT_SEED = 0
DEFAULT_MSA_MAX_SEQUENCES = 1024

# Note: The following functions are now imported from confidock.esmfold2.inference.runner:
# - _save_pae_matrix → esmfold2.runner.save_pae_matrix
# - _resolve_a3m → esmfold2.runner.resolve_a3m
# - _write_prediction_config → esmfold2.runner.write_prediction_config
# - _load_entries → esmfold2.runner.load_entries
# - _coerce_modifications → esmfold2.runner.coerce_modifications


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ESMFold2 predictions for entries in a JSON input file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="JSON file containing a list of ESMFold2 input entries",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/esmfold2_test_runs"),
        help="Root directory for predictions and config",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "forge"],
        default="local",
        help="Use local Hugging Face weights or the Biohub Forge API",
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        default=None,
        metavar="ENTRY",
        help="Restrict to specific entry names from the input JSON",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip entries whose output CIF and summary JSON already exist",
    )
    parser.add_argument(
        "--local_model",
        default="biohub/ESMFold2",
        help="Hugging Face model id or local path for local mode",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for local mode, e.g. cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--forge_model",
        default="esmfold2-fast-2026-05",
        help="Biohub Forge model name for remote inference",
    )
    parser.add_argument(
        "--forge_url",
        default="https://biohub.ai",
        help="Biohub Forge URL",
    )
    parser.add_argument(
        "--token_env_var",
        default="BIOHUB_API_TOKEN",
        help="Environment variable containing the Biohub API token for forge mode",
    )
    parser.add_argument(
        "--num_loops",
        type=int,
        default=DEFAULT_NUM_LOOPS,
        help="Number of recycling loops",
    )
    parser.add_argument(
        "--num_sampling_steps",
        type=int,
        default=None,
        help="Number of diffusion sampling steps; defaults depend on mode",
    )
    parser.add_argument(
        "--num_diffusion_samples",
        type=int,
        default=DEFAULT_NUM_DIFFUSION_SAMPLES,
        help="Number of diffusion samples per entry in local mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used by the sampler",
    )
    parser.add_argument(
        "--msa_dir",
        type=Path,
        default=None,
        help="Directory of pre-fetched .a3m files named by sha256 hash prefix "
        "(as produced by fetch_msas.py). When provided, each protein chain "
        "is given its own MSA at inference time.",
    )
    parser.add_argument(
        "--msa_max_sequences",
        type=int,
        default=DEFAULT_MSA_MAX_SEQUENCES,
        help="Maximum number of sequences to load from each .a3m file",
    )
    parser.add_argument(
        "--save_pae",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a compressed PAE matrix (.npz) for each prediction when available.",
    )
    return parser.parse_args()


def _build_local_spi(
    entry: dict[str, Any], msa_dir: Path | None = None, msa_max_sequences: int = DEFAULT_MSA_MAX_SEQUENCES
):
    from esm.models.esmfold2 import (  # noqa: PLC0415
        DNAInput,
        ESMFold2InputBuilder,
        LigandInput,
        Modification,
        ProteinInput,
        StructurePredictionInput,
    )
    from esm.utils.msa import MSA  # noqa: PLC0415

    sequences = []
    for seq in entry.get("sequences", []):
        seq_type = str(seq["type"]).lower()
        seq_id = str(seq["id"])
        modifications = [Modification(**mod) for mod in _coerce_modifications(seq.get("modifications"))]

        if seq_type == "protein":
            msa = None
            if msa_dir is not None:
                a3m_path = _resolve_a3m(str(seq["sequence"]), msa_dir)
                if a3m_path is not None:
                    msa = MSA.from_a3m(path=a3m_path, remove_insertions=True, max_sequences=msa_max_sequences)
                    LOGGER.info("  MSA loaded for chain %s: depth=%d", seq_id, msa.depth)
                else:
                    LOGGER.warning("  No .a3m found for chain %s – running without MSA", seq_id)
            sequences.append(
                ProteinInput(id=seq_id, sequence=str(seq["sequence"]), modifications=modifications, msa=msa)
            )
        elif seq_type == "dna":
            sequences.append(DNAInput(id=seq_id, sequence=str(seq["sequence"]), modifications=modifications))
        elif seq_type == "ligand":
            ccd = seq.get("ccd")
            smiles = seq.get("smiles")
            sequences.append(LigandInput(id=seq_id, ccd=ccd, smiles=smiles))
        else:
            raise ValueError(f"Unsupported local ESMFold2 input type: {seq_type!r}")

    if not sequences:
        raise ValueError(f"Entry {entry.get('name')!r} has no valid sequences")

    return ESMFold2InputBuilder(), StructurePredictionInput(sequences=sequences)


def _save_local_results(
    result,
    entry_dir: Path,
    name: str,
    args: argparse.Namespace,
    sampling_steps: int,
) -> None:
    """Save local prediction results to disk.

    Parameters
    ----------
    result
        Local prediction result
    entry_dir : Path
        Entry output directory
    name : str
        Entry name
    args : argparse.Namespace
        CLI arguments
    sampling_steps : int
        Number of sampling steps used
    """
    cif_path = entry_dir / f"{name}.cif"
    summary_path = entry_dir / f"summary_{name}.json"
    pae_path = entry_dir / f"pae_{name}.npz"

    cif_path.write_text(result.complex.to_mmcif(), encoding="utf-8")

    pae_saved = False
    pae_shape: tuple[int, ...] | None = None
    if args.save_pae:
        pae_saved, pae_shape = _save_pae_matrix(result, pae_path)

    summary = {
        "name": name,
        "mode": "local",
        "model": args.local_model,
        "device": args.device,
        "num_loops": args.num_loops,
        "num_sampling_steps": sampling_steps,
        "num_diffusion_samples": args.num_diffusion_samples,
        "seed": args.seed,
        "plddt_mean": float(result.plddt.mean()),
        "ptm": float(result.ptm),
        "iptm": float(result.iptm),
        "output_cif": str(cif_path),
        "pae_saved": pae_saved,
    }
    if pae_saved:
        summary["output_pae_npz"] = str(pae_path)
        if pae_shape is not None:
            summary["pae_shape"] = list(pae_shape)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info(
        "%s: pLDDT mean %.3f, pTM %.3f, ipTM %.3f",
        name,
        summary["plddt_mean"],
        summary["ptm"],
        summary["iptm"],
    )


def _process_local_entry(
    entry: dict[str, Any],
    model,
    args: argparse.Namespace,
    sampling_steps: int,
) -> None:
    """Process a single entry in local mode.

    Parameters
    ----------
    entry : dict
        Entry dictionary
    model
        ESMFold2 model
    args : argparse.Namespace
        CLI arguments
    sampling_steps : int
        Number of sampling steps
    """
    name = str(entry["name"])
    entry_dir = args.output_dir / name
    entry_dir.mkdir(parents=True, exist_ok=True)

    if _should_skip_entry(entry_dir, name, args.skip_existing, args.save_pae):
        return

    LOGGER.info("Running local ESMFold2 for %s", name)
    input_builder, spi = _build_local_spi(entry, msa_dir=args.msa_dir, msa_max_sequences=args.msa_max_sequences)
    result = input_builder.fold(
        model,
        spi,
        num_loops=args.num_loops,
        num_sampling_steps=sampling_steps,
        num_diffusion_samples=args.num_diffusion_samples,
        seed=args.seed,
    )

    _save_local_results(result, entry_dir, name, args, sampling_steps)


def _run_local(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Run local mode predictions for all entries.

    Parameters
    ----------
    entries : list of dict
        Entries to process
    args : argparse.Namespace
        CLI arguments
    """
    try:
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Local ESMFold2 inference requires the optional Poetry group. "
            "Install it from a Python 3.12 environment with 'poetry install --with esm'."
        ) from exc

    sampling_steps = args.num_sampling_steps or DEFAULT_LOCAL_SAMPLING_STEPS
    LOGGER.info("Loading ESMFold2 model from %s", args.local_model)
    model = ESMFold2Model.from_pretrained(args.local_model)
    model = model.to(args.device)
    model.eval()

    for entry in entries:
        _process_local_entry(entry, model, args, sampling_steps)


def _should_skip_entry(
    entry_dir: Path,
    name: str,
    skip_existing: bool,
    save_pae: bool,
) -> bool:
    """Check if an entry should be skipped because outputs already exist.

    Parameters
    ----------
    entry_dir : Path
        Entry output directory
    name : str
        Entry name
    skip_existing : bool
        Whether to skip existing outputs
    save_pae : bool
        Whether PAE file is required

    Returns
    -------
    bool
        True if entry should be skipped
    """
    if not skip_existing:
        return False

    cif_path = entry_dir / f"{name}.cif"
    summary_path = entry_dir / f"summary_{name}.json"
    pae_path = entry_dir / f"pae_{name}.npz"

    if not (cif_path.exists() and summary_path.exists()):
        return False
    if save_pae and not pae_path.exists():
        return False

    LOGGER.info("Skipping %s – outputs already exist", name)
    return True


def _build_forge_protein_inputs(
    entry: dict[str, Any],
    msa_dir: Path | None,
    msa_max_sequences: int,
):
    """Build protein inputs for forge mode with optional MSA loading.

    Parameters
    ----------
    entry : dict
        Entry dictionary
    msa_dir : Path | None
        Directory containing MSA files
    msa_max_sequences : int
        Maximum sequences to load from MSA

    Returns
    -------
    list of ProteinInput
        Protein input objects for forge
    """
    from esm.utils.msa import MSA  # noqa: PLC0415
    from esm.utils.structure.input_builder import ProteinInput  # noqa: PLC0415

    name = entry.get("name", "")
    protein_inputs = []

    for seq in entry.get("sequences", []):
        seq_type = str(seq["type"]).lower()
        if seq_type != "protein":
            raise ValueError(f"Forge mode currently supports protein-only entries, got {seq_type!r} in {name}")

        msa = None
        if msa_dir is not None:
            a3m_path = _resolve_a3m(str(seq["sequence"]), msa_dir)
            if a3m_path is not None:
                msa = MSA.from_a3m(path=a3m_path, remove_insertions=True, max_sequences=msa_max_sequences)
                LOGGER.info("  MSA loaded for chain %s: depth=%d", seq["id"], msa.depth)
            else:
                LOGGER.warning("  No .a3m found for chain %s – running without MSA", seq["id"])

        protein_inputs.append(ProteinInput(id=str(seq["id"]), sequence=str(seq["sequence"]), msa=msa))

    if not protein_inputs:
        raise ValueError(f"Entry {name!r} has no protein sequences for forge mode")

    return protein_inputs


def _save_forge_results(
    result,
    entry_dir: Path,
    name: str,
    args: argparse.Namespace,
    sampling_steps: int,
) -> None:
    """Save forge prediction results to disk.

    Parameters
    ----------
    result
        Forge prediction result
    entry_dir : Path
        Entry output directory
    name : str
        Entry name
    args : argparse.Namespace
        CLI arguments
    sampling_steps : int
        Number of sampling steps used
    """
    cif_path = entry_dir / f"{name}.cif"
    summary_path = entry_dir / f"summary_{name}.json"
    pae_path = entry_dir / f"pae_{name}.npz"

    cif_path.write_text(result.complex.to_mmcif(), encoding="utf-8")

    pae_saved = False
    pae_shape: tuple[int, ...] | None = None
    if args.save_pae:
        pae_saved, pae_shape = _save_pae_matrix(result, pae_path)

    summary = {
        "name": name,
        "mode": "forge",
        "model": args.forge_model,
        "url": args.forge_url,
        "num_loops": args.num_loops,
        "num_sampling_steps": sampling_steps,
        "output_cif": str(cif_path),
        "pae_saved": pae_saved,
    }
    if pae_saved:
        summary["output_pae_npz"] = str(pae_path)
        if pae_shape is not None:
            summary["pae_shape"] = list(pae_shape)

    for attr in ("plddt", "ptm", "iptm"):
        value = getattr(result, attr, None)
        if value is None:
            continue
        if attr == "plddt":
            summary["plddt_mean"] = float(value.mean())
        else:
            summary[attr] = float(value)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info(
        "%s: forge run complete%s",
        name,
        f", pLDDT mean {summary['plddt_mean']:.3f}" if "plddt_mean" in summary else "",
    )


def _process_forge_entry(
    entry: dict[str, Any],
    client,
    config,
    args: argparse.Namespace,
    sampling_steps: int,
) -> None:
    """Process a single entry in forge mode.

    Parameters
    ----------
    entry : dict
        Entry dictionary
    client
        Forge inference client
    config
        Folding configuration
    args : argparse.Namespace
        CLI arguments
    sampling_steps : int
        Number of sampling steps
    """
    from esm.utils.structure.input_builder import StructurePredictionInput  # noqa: PLC0415

    name = str(entry["name"])
    entry_dir = args.output_dir / name
    entry_dir.mkdir(parents=True, exist_ok=True)

    if _should_skip_entry(entry_dir, name, args.skip_existing, args.save_pae):
        return

    protein_inputs = _build_forge_protein_inputs(entry, args.msa_dir, args.msa_max_sequences)

    LOGGER.info("Running forge ESMFold2 for %s", name)
    result = client.fold_all_atom(StructurePredictionInput(sequences=protein_inputs), config=config)

    _save_forge_results(result, entry_dir, name, args, sampling_steps)


def _run_forge(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Run forge mode predictions for all entries.

    Parameters
    ----------
    entries : list of dict
        Entries to process
    args : argparse.Namespace
        CLI arguments
    """
    try:
        from esm.sdk.api import FoldingConfig  # noqa: PLC0415
        from esm.sdk.forge import SequenceStructureForgeInferenceClient  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Forge ESMFold2 inference requires the optional Poetry group. "
            "Install it from a Python 3.12 environment with 'poetry install --with esm'."
        ) from exc

    token = os.environ.get(args.token_env_var)
    if not token:
        raise EnvironmentError(f"Forge mode requires the {args.token_env_var} environment variable to be set")

    sampling_steps = args.num_sampling_steps or DEFAULT_FORGE_SAMPLING_STEPS
    client = SequenceStructureForgeInferenceClient(
        model=args.forge_model,
        url=args.forge_url,
        token=token,
    )
    config = FoldingConfig(num_loops=args.num_loops, num_sampling_steps=sampling_steps)

    for entry in entries:
        _process_forge_entry(entry, client, config, args, sampling_steps)


def main() -> None:
    """Run ESMFold2 predictions for multiple antibody-antigen complexes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    if not args.input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    entries = _load_entries(args.input_json, args.entries)
    if not entries:
        raise ValueError(f"No matching entries found in {args.input_json}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_prediction_config(
        args.output_dir,
        input_json=args.input_json,
        output_dir=args.output_dir,
        mode=args.mode,
        entries=args.entries,
        skip_existing=args.skip_existing,
        local_model=args.local_model,
        device=args.device,
        forge_model=args.forge_model,
        forge_url=args.forge_url,
        token_env_var=args.token_env_var,
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=args.num_diffusion_samples,
        seed=args.seed,
        save_pae=args.save_pae,
    )

    LOGGER.info("Running %d ESMFold2 input entries in %s mode", len(entries), args.mode)
    if args.mode == "local":
        _run_local(entries, args)
    else:
        _run_forge(entries, args)


if __name__ == "__main__":
    main()
