r"""Run a grid of ESMFold2 predictions over MSA mode × recycling loops × diffusion steps.

Grid axes
---------
msa_mode:
    full_msa      – all chains use pre-fetched .a3m MSAs from --msa_dir
    antigen_only  – antigen chains use MSA; H/L binder chains get no MSA
    no_msa        – no MSA information used

num_loops:          3, 5, 10   (recycling loops; ESMFold2 default: 3)
num_sampling_steps: 25, 50, 100 (diffusion sampling steps; ESMFold2 default: 50)

With 5 input structures this gives 27 conditions × 5 inputs = 135 total predictions.

MSA directory layout
--------------------
The ``--msa_dir`` should contain flat ``.a3m`` files named by the first 16 hex
characters of SHA-256(sequence), e.g.::

    sabdab_msas/
        000095525b64eb5b.a3m
        00043d1fec9123f8.a3m
        ...

These are the files produced by ``fetch_msas.py``.

A chain may instead name its alignment directly, which makes ``--msa_dir`` optional::

    {"type": "protein", "id": "H", "sequence": "QVQL...", "msaPath": "/.../<sha16>.a3m"}

``msaPath`` wins over the hash lookup and raises if the file is absent, so an input built
against a staging tree that has since moved fails loudly instead of quietly folding MSA-free.
``msa_mode`` still gates it: a ``no_msa`` condition, or ``antigen_only`` on an antibody chain,
ignores ``msaPath`` for the chains it covers. Use
``workflows/scripts/processing/bake_esmfold2_msa_paths.py`` to add the key to an input set.

Output layout
-------------
::

    {output_dir}/
        msa=full_msa__loops=5__steps=50/
            {entry_name}/
                {entry_name}.cif
                summary_{entry_name}.json
                pae_{entry_name}.npz
            prediction_config.yaml
        msa=no_msa__loops=3__steps=25/
            ...
        grid_config.yaml

Usage::

    poetry run python workflows/scripts/inference/run_esmfold2_grid.py \
        --json_dir  /path/to/esmfold2_jsons \
        --msa_dir   /path/to/sabdab_msas \
        --output_dir data/sabdab_esmfold2_grid \
        --n_inputs   5 \
        --seed       42

To run on specific JSON files instead of sampling randomly::

    poetry run python workflows/scripts/inference/run_esmfold2_grid.py \
        --input_jsons \
            /path/to/input1.json \
            /path/to/input2.json \
        --msa_dir   /path/to/sabdab_msas \
        --output_dir data/sabdab_esmfold2_grid
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import time
from itertools import product as iproduct
from pathlib import Path
from typing import Any

import yaml

# Import reusable functions from library
from confidock.boltz.inference.inputs import load_sabdab_binders
from confidock.esmfold2.inference.runner import (
    MSA_PATH_KEY,
    load_entries,
    resolve_chain_a3m,
    save_pae_matrix,
    write_prediction_config,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

MSA_MODES: list[str] = ["full_msa", "antigen_only", "no_msa"]
NUM_LOOPS: list[int] = [3, 5, 10]
NUM_SAMPLING_STEPS: list[int] = [25, 50, 100]

DEFAULT_NUM_DIFFUSION_SAMPLES = 1
DEFAULT_SEED = 0
DEFAULT_MSA_MAX_SEQUENCES = 1024


def _log_timestamp(msg: str) -> None:
    """Log a message with a timestamp."""
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


def _sample_files_exist(
    entry_dir: Path,
    name: str,
    sample_idx: int,
    num_diffusion_samples: int,
    save_pae: bool,
) -> bool:
    """Check if all required files exist for a single sample.

    Parameters
    ----------
    entry_dir : Path
        Entry output directory
    name : str
        Entry name
    sample_idx : int
        Sample index
    num_diffusion_samples : int
        Total number of samples
    save_pae : bool
        Whether PAE file is required

    Returns
    -------
    bool
        True if all required files exist
    """
    suffix = f"_sample_{sample_idx}" if num_diffusion_samples > 1 else ""
    cif_path = entry_dir / f"{name}{suffix}.cif"
    summary_path = entry_dir / f"summary_{name}{suffix}.json"
    pae_path = entry_dir / f"pae_{name}{suffix}.npz"

    if not (cif_path.exists() and summary_path.exists()):
        return False
    if save_pae and not pae_path.exists():
        return False
    return True


def _entry_is_complete(
    output_dir: Path,
    name: str,
    save_pae: bool,
    num_diffusion_samples: int,
) -> bool:
    """Check if all samples for an entry are complete.

    Parameters
    ----------
    output_dir : Path
        Condition output directory
    name : str
        Entry name
    save_pae : bool
        Whether PAE matrices are required
    num_diffusion_samples : int
        Expected number of samples per entry

    Returns
    -------
    bool
        True if all samples exist
    """
    entry_dir = output_dir / name
    for sample_idx in range(num_diffusion_samples):
        if not _sample_files_exist(entry_dir, name, sample_idx, num_diffusion_samples, save_pae):
            return False
    return True


def _check_completion(
    output_dir: Path,
    entry_names: list[str],
    save_pae: bool,
    num_diffusion_samples: int = 1,
) -> int:
    """Count how many entries have complete outputs in output_dir.

    Parameters
    ----------
    output_dir : Path
        Condition output directory
    entry_names : list of str
        Entry names to check
    save_pae : bool
        Whether PAE matrices are required
    num_diffusion_samples : int
        Expected number of samples per entry (default: 1)

    Returns
    -------
    int
        Number of entries with all expected samples complete
    """
    completed = 0
    for name in entry_names:
        if _entry_is_complete(output_dir, name, save_pae, num_diffusion_samples):
            completed += 1
    return completed


def _is_antibody_chain(
    chain_id: str,
    entry_name: str,
    sabdab_binders: dict[str, set[str]] | None = None,
) -> bool:
    """Check if a chain ID is an antibody chain (heavy or light).

    Uses SAbDab TSV if provided, otherwise falls back to H/L heuristic.
    """
    if sabdab_binders is not None:
        # Extract PDB ID from entry name (e.g., "7bbj_HcLcA" -> "7bbj")
        pdb_id = entry_name.split("_")[0].lower()
        binder_chains = sabdab_binders.get(pdb_id, set())
        return chain_id in binder_chains

    # Fallback: H/L heuristic
    return chain_id.upper().startswith(("H", "L"))


def _should_load_msa_for_chain(
    seq_id: str,
    entry_name: str,
    msa_mode: str,
    msa_dir: Path | None,
    sabdab_binders: dict[str, set[str]] | None,
    has_explicit_path: bool = False,
) -> bool:
    """Determine if MSA should be loaded for a chain based on mode.

    Parameters
    ----------
    seq_id : str
        Chain ID
    entry_name : str
        Entry name
    msa_mode : str
        One of "full_msa", "antigen_only", or "no_msa"
    msa_dir : Path | None
        MSA directory
    sabdab_binders : dict | None
        Antibody chain mapping
    has_explicit_path : bool
        Whether the chain names its own ``msaPath``, which removes the need for ``--msa_dir``

    Returns
    -------
    bool
        True if MSA should be loaded
    """
    # msa_mode still wins: a no_msa / antigen_only condition stays MSA-free for the chains it
    # covers even when the input JSON names an alignment for them.
    if msa_dir is None and not has_explicit_path:
        return False
    if msa_mode == "full_msa":
        return True
    if msa_mode == "antigen_only":
        return not _is_antibody_chain(seq_id, entry_name, sabdab_binders)
    return False


def _load_msa_for_protein(
    chain: dict,
    seq_id: str,
    msa_dir: Path | None,
    msa_max_sequences: int,
):
    """Load MSA for a protein chain, or None if not found.

    Parameters
    ----------
    chain : dict
        Sequence dictionary; an explicit ``msaPath`` takes precedence over the hash lookup
    seq_id : str
        Chain ID (for logging)
    msa_dir : Path or None
        Directory containing .a3m files, or None when relying on ``msaPath``
    msa_max_sequences : int
        Maximum sequences to load

    Returns
    -------
    MSA or None
        Loaded MSA or None if not found
    """
    from esm.utils.msa import MSA  # noqa: PLC0415

    a3m_path = resolve_chain_a3m(chain, msa_dir)
    if a3m_path is not None:
        msa = MSA.from_a3m(path=a3m_path, remove_insertions=True, max_sequences=msa_max_sequences)
        source = MSA_PATH_KEY if chain.get(MSA_PATH_KEY) else "hash lookup"
        LOGGER.info("  MSA loaded for chain %s: depth=%d (via %s)", seq_id, msa.depth, source)
        return msa
    LOGGER.warning("  No .a3m found for chain %s – running without MSA", seq_id)
    return None


def _parse_modifications(raw_mods):
    """Parse modification list into Modification objects.

    Parameters
    ----------
    raw_mods : list
        Raw modification dictionaries

    Returns
    -------
    list of Modification
        Parsed modifications
    """
    from esm.models.esmfold2 import Modification  # noqa: PLC0415

    return [Modification(position=int(mod["position"]), ccd=str(mod["ccd"])) for mod in (raw_mods if raw_mods else [])]


def _process_sequence(
    seq: dict,
    entry_name: str,
    msa_dir: Path | None,
    msa_max_sequences: int,
    msa_mode: str,
    sabdab_binders: dict[str, set[str]] | None,
):
    """Process a single sequence entry and return the appropriate Input object.

    Parameters
    ----------
    seq : dict
        Sequence dictionary
    entry_name : str
        Entry name
    msa_dir : Path | None
        MSA directory
    msa_max_sequences : int
        Max MSA sequences
    msa_mode : str
        MSA mode
    sabdab_binders : dict | None
        Antibody chain mapping

    Returns
    -------
    ProteinInput | DNAInput | LigandInput
        Appropriate input object
    """
    from esm.models.esmfold2 import DNAInput, LigandInput, ProteinInput  # noqa: PLC0415

    seq_type = str(seq["type"]).lower()
    seq_id = str(seq["id"])
    modifications = _parse_modifications(seq.get("modifications", []))

    if seq_type == "protein":
        msa = None
        if _should_load_msa_for_chain(
            seq_id, entry_name, msa_mode, msa_dir, sabdab_binders, bool(seq.get(MSA_PATH_KEY))
        ):
            msa = _load_msa_for_protein(seq, seq_id, msa_dir, msa_max_sequences)
        return ProteinInput(id=seq_id, sequence=str(seq["sequence"]), modifications=modifications, msa=msa)
    if seq_type == "dna":
        return DNAInput(id=seq_id, sequence=str(seq["sequence"]), modifications=modifications)
    if seq_type == "ligand":
        return LigandInput(id=seq_id, ccd=seq.get("ccd"), smiles=seq.get("smiles"))
    raise ValueError(f"Unsupported local ESMFold2 input type: {seq_type!r}")


def _build_local_spi(
    entry: dict[str, Any],
    msa_dir: Path | None = None,
    msa_max_sequences: int = DEFAULT_MSA_MAX_SEQUENCES,
    msa_mode: str = "full_msa",
    sabdab_binders: dict[str, set[str]] | None = None,
):
    """Build ESMFold2 input objects for local mode.

    Parameters
    ----------
    entry : dict
        ESMFold2 entry dict with name and sequences.
    msa_dir : Path | None
        Directory containing .a3m MSA files.
    msa_max_sequences : int
        Maximum sequences to load from each MSA.
    msa_mode : str
        One of "full_msa", "antigen_only", or "no_msa".
    sabdab_binders : dict | None
        Mapping of PDB ID to antibody chain IDs for antigen_only mode.
    """
    from esm.models.esmfold2 import ESMFold2InputBuilder, StructurePredictionInput  # noqa: PLC0415

    entry_name = entry.get("name", "")
    sequences = [
        _process_sequence(seq, entry_name, msa_dir, msa_max_sequences, msa_mode, sabdab_binders)
        for seq in entry.get("sequences", [])
    ]

    if not sequences:
        raise ValueError(f"Entry {entry.get('name')!r} has no valid sequences")

    return ESMFold2InputBuilder(), StructurePredictionInput(sequences=sequences)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_local_inference(
    entry: dict,
    model,
    msa_dir: Path | None,
    msa_mode: str,
    msa_max_sequences: int,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    seed: int,
    sabdab_binders: dict[str, set[str]] | None,
) -> list:
    """Run local ESMFold2 inference.

    Parameters
    ----------
    entry : dict
        Entry to process
    model : ESMFold2Model
        Loaded model
    msa_dir : Path | None
        MSA directory
    msa_mode : str
        MSA mode
    msa_max_sequences : int
        Max MSA sequences
    num_loops : int
        Recycling loops
    num_sampling_steps : int
        Diffusion steps
    num_diffusion_samples : int
        Number of samples
    seed : int
        Random seed
    sabdab_binders : dict | None
        Antibody binder mapping

    Returns
    -------
    list
        List of prediction results
    """
    input_builder, spi = _build_local_spi(
        entry,
        msa_dir=msa_dir,
        msa_max_sequences=msa_max_sequences,
        msa_mode=msa_mode,
        sabdab_binders=sabdab_binders,
    )
    results = input_builder.fold(
        model,
        spi,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
    )

    # Handle single result vs list of results
    if not isinstance(results, list):
        results = [results]
    return results


def _load_msa_for_forge(
    seq_id: str,
    chain: dict,
    entry_name: str,
    msa_dir: Path | None,
    msa_mode: str,
    msa_max_sequences: int,
    sabdab_binders: dict[str, set[str]] | None,
):
    """Load MSA for a sequence in Forge mode.

    Parameters
    ----------
    seq_id : str
        Chain ID
    chain : dict
        Sequence dictionary; an explicit ``msaPath`` takes precedence over the hash lookup
    entry_name : str
        Entry name
    msa_dir : Path or None
        MSA directory, or None when relying on ``msaPath``
    msa_mode : str
        MSA mode
    msa_max_sequences : int
        Max MSA sequences
    sabdab_binders : dict | None
        Antibody binder mapping

    Returns
    -------
    MSA or None
        Loaded MSA or None
    """
    from esm.utils.msa import MSA  # noqa: PLC0415

    should_load_msa = msa_mode == "full_msa" or (
        msa_mode == "antigen_only" and not _is_antibody_chain(seq_id, entry_name, sabdab_binders)
    )

    if not should_load_msa:
        return None

    a3m_path = resolve_chain_a3m(chain, msa_dir)
    if a3m_path is not None:
        msa = MSA.from_a3m(path=a3m_path, remove_insertions=True, max_sequences=msa_max_sequences)
        source = MSA_PATH_KEY if chain.get(MSA_PATH_KEY) else "hash lookup"
        LOGGER.info("    MSA loaded for chain %s: depth=%d (via %s)", seq_id, msa.depth, source)
        return msa
    return None


def _build_forge_protein_inputs(
    entry: dict,
    msa_dir: Path | None,
    msa_mode: str,
    msa_max_sequences: int,
    sabdab_binders: dict[str, set[str]] | None,
) -> list:
    """Build protein inputs for Forge API.

    Parameters
    ----------
    entry : dict
        Entry to process
    msa_dir : Path | None
        MSA directory
    msa_mode : str
        MSA mode
    msa_max_sequences : int
        Max MSA sequences
    sabdab_binders : dict | None
        Antibody binder mapping

    Returns
    -------
    list
        List of ProteinInput objects
    """
    from esm.utils.structure.input_builder import ProteinInput  # noqa: PLC0415

    protein_inputs = []
    for seq in entry.get("sequences", []):
        seq_type = str(seq["type"]).lower()
        if seq_type != "protein":
            raise ValueError(f"Forge mode currently supports protein-only entries, got {seq_type!r} in {entry['name']}")

        seq_id = str(seq["id"])
        sequence = str(seq["sequence"])

        msa = None
        if msa_dir is not None or seq.get(MSA_PATH_KEY):
            msa = _load_msa_for_forge(seq_id, seq, entry["name"], msa_dir, msa_mode, msa_max_sequences, sabdab_binders)

        protein_inputs.append(ProteinInput(id=seq_id, sequence=sequence, msa=msa))

    if not protein_inputs:
        raise ValueError(f"Entry {entry['name']!r} has no protein sequences for forge mode")

    return protein_inputs


def _run_forge_inference(
    entry: dict,
    forge_client,
    msa_dir: Path | None,
    msa_mode: str,
    msa_max_sequences: int,
    num_loops: int,
    num_sampling_steps: int,
    sabdab_binders: dict[str, set[str]] | None,
) -> list:
    """Run Forge API ESMFold2 inference.

    Parameters
    ----------
    entry : dict
        Entry to process
    forge_client : SequenceStructureForgeInferenceClient
        Forge client
    msa_dir : Path | None
        MSA directory
    msa_mode : str
        MSA mode
    msa_max_sequences : int
        Max MSA sequences
    num_loops : int
        Recycling loops
    num_sampling_steps : int
        Diffusion steps
    sabdab_binders : dict | None
        Antibody binder mapping

    Returns
    -------
    list
        List of prediction results (single element for Forge)
    """
    from esm.sdk.api import FoldingConfig  # noqa: PLC0415
    from esm.utils.structure.input_builder import StructurePredictionInput  # noqa: PLC0415

    protein_inputs = _build_forge_protein_inputs(entry, msa_dir, msa_mode, msa_max_sequences, sabdab_binders)

    config = FoldingConfig(num_loops=num_loops, num_sampling_steps=num_sampling_steps)
    result = forge_client.fold_all_atom(StructurePredictionInput(sequences=protein_inputs), config=config)

    # Forge returns single result, wrap in list for consistency
    return [result]


def _get_sample_paths(entry_dir: Path, name: str, sample_idx: int, multi_sample: bool) -> tuple[Path, Path, Path]:
    """Get file paths for a sample.

    Parameters
    ----------
    entry_dir : Path
        Entry directory
    name : str
        Entry name
    sample_idx : int
        Sample index
    multi_sample : bool
        Whether multiple samples exist

    Returns
    -------
    tuple
        (cif_path, pae_path, summary_path)
    """
    if multi_sample:
        cif_path = entry_dir / f"{name}_sample_{sample_idx}.cif"
        pae_path = entry_dir / f"pae_{name}_sample_{sample_idx}.npz"
        summary_path = entry_dir / f"summary_{name}_sample_{sample_idx}.json"
    else:
        cif_path = entry_dir / f"{name}.cif"
        pae_path = entry_dir / f"pae_{name}.npz"
        summary_path = entry_dir / f"summary_{name}.json"
    return cif_path, pae_path, summary_path


def _build_sample_summary(
    name: str,
    sample_idx: int,
    result,
    sample_cif_path: Path,
    pae_saved: bool,
    pae_shape: tuple[int, ...] | None,
    sample_pae_path: Path,
    args: argparse.Namespace,
    msa_mode: str,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
) -> dict:
    """Build summary dict for a single sample.

    Parameters
    ----------
    name : str
        Entry name
    sample_idx : int
        Sample index
    result
        Prediction result
    sample_cif_path : Path
        CIF file path
    pae_saved : bool
        Whether PAE was saved
    pae_shape : tuple | None
        PAE shape
    sample_pae_path : Path
        PAE file path
    args : argparse.Namespace
        Command-line arguments
    msa_mode : str
        MSA mode
    num_loops : int
        Recycling loops
    num_sampling_steps : int
        Diffusion steps
    num_diffusion_samples : int
        Number of samples

    Returns
    -------
    dict
        Summary dictionary
    """
    summary = {
        "name": name,
        "sample_idx": sample_idx,
        "mode": args.mode,
        "msa_mode": msa_mode,
        "num_loops": num_loops,
        "num_sampling_steps": num_sampling_steps,
        "num_diffusion_samples": num_diffusion_samples,
        "seed": args.seed,
        "output_cif": str(sample_cif_path),
        "pae_saved": pae_saved,
    }

    # Add mode-specific fields
    if args.mode == "local":
        summary["model"] = args.local_model
        summary["device"] = args.device
    else:
        summary["forge_model"] = args.forge_model
        summary["forge_url"] = args.forge_url

    # Add metrics if available
    for attr in ("plddt", "ptm", "iptm"):
        value = getattr(result, attr, None)
        if value is not None:
            summary[attr if attr != "plddt" else "plddt_mean"] = float(value.mean() if attr == "plddt" else value)

    if pae_saved:
        summary["output_pae_npz"] = str(sample_pae_path)
        if pae_shape is not None:
            summary["pae_shape"] = list(pae_shape)

    return summary


def _log_sample_metrics(name: str, sample_idx: int, summary: dict) -> None:
    """Log metrics for a sample.

    Parameters
    ----------
    name : str
        Entry name
    sample_idx : int
        Sample index
    summary : dict
        Summary dictionary
    """
    if "plddt_mean" in summary:
        LOGGER.info(
            "    %s sample %d: pLDDT mean %.3f, pTM %.3f, ipTM %.3f",
            name,
            sample_idx,
            summary.get("plddt_mean", 0.0),
            summary.get("ptm", 0.0),
            summary.get("iptm", 0.0),
        )
    else:
        LOGGER.info("    %s sample %d: completed", name, sample_idx)


def _save_single_sample(
    result,
    sample_idx: int,
    entry_dir: Path,
    name: str,
    multi_sample: bool,
    args: argparse.Namespace,
    msa_mode: str,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
) -> dict | None:
    """Save a single prediction sample.

    Parameters
    ----------
    result
        Prediction result
    sample_idx : int
        Sample index
    entry_dir : Path
        Entry directory
    name : str
        Entry name
    multi_sample : bool
        Whether multiple samples exist
    args : argparse.Namespace
        Command-line arguments
    msa_mode : str
        MSA mode
    num_loops : int
        Recycling loops
    num_sampling_steps : int
        Diffusion steps
    num_diffusion_samples : int
        Number of samples

    Returns
    -------
    dict or None
        Summary dict if successful, None otherwise
    """
    if not hasattr(result, "complex"):
        LOGGER.error("  Failed for %s sample %d: %s", name, sample_idx, result)
        return None

    sample_cif_path, sample_pae_path, sample_summary_path = _get_sample_paths(entry_dir, name, sample_idx, multi_sample)

    sample_cif_path.write_text(result.complex.to_mmcif(), encoding="utf-8")

    pae_saved = False
    pae_shape: tuple[int, ...] | None = None
    if args.save_pae:
        pae_saved, pae_shape = save_pae_matrix(result, sample_pae_path)

    summary = _build_sample_summary(
        name,
        sample_idx,
        result,
        sample_cif_path,
        pae_saved,
        pae_shape,
        sample_pae_path,
        args,
        msa_mode,
        num_loops,
        num_sampling_steps,
        num_diffusion_samples,
    )

    sample_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log_sample_metrics(name, sample_idx, summary)

    return summary


def _save_prediction_results(
    results: list,
    entry_dir: Path,
    name: str,
    args: argparse.Namespace,
    msa_mode: str,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
) -> int:
    """Save prediction results to disk.

    Parameters
    ----------
    results : list
        Prediction results
    entry_dir : Path
        Entry output directory
    name : str
        Entry name
    args : argparse.Namespace
        Command-line arguments
    msa_mode : str
        MSA mode
    num_loops : int
        Recycling loops
    num_sampling_steps : int
        Diffusion steps
    num_diffusion_samples : int
        Number of samples

    Returns
    -------
    int
        Number of successfully saved samples
    """
    multi_sample = len(results) > 1
    sample_summaries = []

    for sample_idx, result in enumerate(results):
        summary = _save_single_sample(
            result,
            sample_idx,
            entry_dir,
            name,
            multi_sample,
            args,
            msa_mode,
            num_loops,
            num_sampling_steps,
            num_diffusion_samples,
        )
        if summary is not None:
            sample_summaries.append(summary)

    # Write overall summary with all samples
    if sample_summaries:
        overall_summary = {
            "name": name,
            "mode": args.mode,
            "msa_mode": msa_mode,
            "num_loops": num_loops,
            "num_sampling_steps": num_sampling_steps,
            "num_diffusion_samples": num_diffusion_samples,
            "seed": args.seed,
            "samples": sample_summaries,
        }
        (entry_dir / f"summary_{name}_all_samples.json").write_text(
            json.dumps(overall_summary, indent=2), encoding="utf-8"
        )

    return len(sample_summaries)


def _process_single_entry(
    entry: dict,
    cond_output_dir: Path,
    args: argparse.Namespace,
    model,
    forge_client,
    msa_mode: str,
    msa_dir_for_condition: Path | None,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    sabdab_binders: dict[str, set[str]] | None,
) -> bool:
    """Process a single entry for inference.

    Parameters
    ----------
    entry : dict
        Entry to process
    cond_output_dir : Path
        Condition output directory
    args : argparse.Namespace
        Command-line arguments
    model : ESMFold2Model | None
        Local model (or None if forge mode)
    forge_client : SequenceStructureForgeInferenceClient | None
        Forge client (or None if local mode)
    msa_mode : str
        MSA mode
    msa_dir_for_condition : Path | None
        MSA directory for this condition
    num_loops : int
        Recycling loops
    num_sampling_steps : int
        Diffusion steps
    num_diffusion_samples : int
        Number of samples
    sabdab_binders : dict | None
        Antibody binder mapping

    Returns
    -------
    bool
        True if entry was processed (not skipped)
    """
    name = str(entry["name"])
    entry_dir = cond_output_dir / name
    entry_dir.mkdir(parents=True, exist_ok=True)

    # Check if all samples for this entry are already complete
    if not args.overwrite and _entry_is_complete(cond_output_dir, name, args.save_pae, num_diffusion_samples):
        LOGGER.info("  Skipping %s – all %d samples already exist", name, num_diffusion_samples)
        return False

    LOGGER.info("  Running ESMFold2 for %s (mode: %s, %d samples)", name, args.mode, num_diffusion_samples)

    if args.mode == "local":
        results = _run_local_inference(
            entry,
            model,
            msa_dir_for_condition,
            msa_mode,
            args.msa_max_sequences,
            num_loops,
            num_sampling_steps,
            num_diffusion_samples,
            args.seed,
            sabdab_binders,
        )
    else:  # forge mode
        results = _run_forge_inference(
            entry,
            forge_client,
            msa_dir_for_condition,
            msa_mode,
            args.msa_max_sequences,
            num_loops,
            num_sampling_steps,
            sabdab_binders,
        )

    _save_prediction_results(
        results,
        entry_dir,
        name,
        args,
        msa_mode,
        num_loops,
        num_sampling_steps,
        num_diffusion_samples,
    )

    return True


def _log_condition_progress(
    cond_idx: int,
    conditions: list[dict],
    entries_completed: int,
    total_entries_needed: int,
    wall_start: float,
) -> None:
    """Log progress for current condition.

    Parameters
    ----------
    cond_idx : int
        Current condition index
    conditions : list of dict
        All conditions
    entries_completed : int
        Number of entries completed
    total_entries_needed : int
        Total entries needed
    wall_start : float
        Wall clock start time
    """
    if cond_idx > 0:
        overall_elapsed = time.perf_counter() - wall_start
        _, overall_eta = _calculate_eta(entries_completed, total_entries_needed, overall_elapsed)
        LOGGER.info(
            "Overall progress: %d/%d entries complete, ETA: %s",
            entries_completed,
            total_entries_needed,
            overall_eta,
        )
        _print_progress_bar(entries_completed, total_entries_needed, prefix="Overall", bar_length=50)


def _check_and_report_completion(
    cond_output_dir: Path,
    entry_names: list[str],
    args: argparse.Namespace,
    num_diffusion_samples: int,
    entries_completed: int,
) -> tuple[bool, int]:
    """Check completion status and report.

    Parameters
    ----------
    cond_output_dir : Path
        Condition output directory
    entry_names : list of str
        Entry names
    args : argparse.Namespace
        Command-line arguments
    num_diffusion_samples : int
        Number of samples per entry
    entries_completed : int
        Current completed count

    Returns
    -------
    tuple
        (should_skip, updated_entries_completed)
    """
    if args.overwrite or args.dry_run:
        return False, entries_completed

    n_done = _check_completion(cond_output_dir, entry_names, args.save_pae, num_diffusion_samples)
    if n_done >= len(entry_names):
        LOGGER.info("  Skipping (all %d entries done) – use --overwrite to re-run", n_done)
        return True, entries_completed + len(entry_names)

    if n_done > 0:
        LOGGER.info(
            "  %d / %d entries complete (%.1f%%) - will process remaining",
            n_done,
            len(entry_names),
            (n_done / len(entry_names)) * 100,
        )
    return False, entries_completed


def _run_condition_inference(
    all_entries: list[dict],
    cond_output_dir: Path,
    args: argparse.Namespace,
    model,
    forge_client,
    msa_mode: str,
    msa_dir_for_condition: Path | None,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    sabdab_binders: dict[str, set[str]] | None,
    entries_completed: int,
) -> int:
    """Run inference for all entries in a condition.

    Parameters
    ----------
    all_entries : list of dict
        All entries to process
    cond_output_dir : Path
        Condition output directory
    args : argparse.Namespace
        Command-line arguments
    model
        Local model
    forge_client
        Forge client
    msa_mode : str
        MSA mode
    msa_dir_for_condition : Path | None
        MSA directory
    num_loops : int
        Recycling loops
    num_sampling_steps : int
        Diffusion steps
    num_diffusion_samples : int
        Number of samples
    sabdab_binders : dict | None
        Antibody binder mapping
    entries_completed : int
        Current completed count

    Returns
    -------
    int
        Updated entries completed count
    """
    import torch  # local: keeps --help and dry runs off the 10 s torch import

    oom_entries: list[str] = []
    for entry in all_entries:
        try:
            processed = _process_single_entry(
                entry,
                cond_output_dir,
                args,
                model,
                forge_client,
                msa_mode,
                msa_dir_for_condition,
                num_loops,
                num_sampling_steps,
                num_diffusion_samples,
                sabdab_binders,
            )
        except torch.OutOfMemoryError:
            # A ~600-residue complex sits right at the edge of a 40 GB A100 at 6 diffusion samples, so
            # a single unlucky entry used to abort the whole run and throw away every entry after it.
            # Drop the entry instead: it is recorded here and stays absent from the output, which the
            # completion check already treats as not-done, so a later re-run on a bigger card picks it
            # up.  Only OOM is swallowed -- anything else is a bug and should still stop the run.
            oom_entries.append(entry["name"])
            LOGGER.exception("  CUDA OOM on %s -- skipping this entry and continuing", entry["name"])
            torch.cuda.empty_cache()
            continue
        if processed:
            entries_completed += 1
    if oom_entries:
        LOGGER.warning("%d entries skipped after CUDA OOM: %s", len(oom_entries), ", ".join(oom_entries))
    return entries_completed


def _process_condition(
    cond_idx: int,
    cond: dict,
    conditions: list[dict],
    all_entries: list[dict],
    entry_names: list[str],
    args: argparse.Namespace,
    model,
    forge_client,
    sabdab_binders: dict[str, set[str]] | None,
    wall_start: float,
    entries_completed: int,
    total_entries_needed: int,
) -> tuple[float, int]:
    """Process a single grid condition.

    Parameters
    ----------
    cond_idx : int
        Condition index
    cond : dict
        Condition parameters
    conditions : list of dict
        All conditions
    all_entries : list of dict
        All entries to process
    entry_names : list of str
        Entry names
    args : argparse.Namespace
        Command-line arguments
    model : ESMFold2Model | None
        Local model
    forge_client : SequenceStructureForgeInferenceClient | None
        Forge client
    sabdab_binders : dict | None
        Antibody binder mapping
    wall_start : float
        Wall clock start time
    entries_completed : int
        Number of entries completed so far
    total_entries_needed : int
        Total entries needed

    Returns
    -------
    tuple
        (elapsed_time, new_entries_completed)
    """
    _log_timestamp(f"Condition [{cond_idx + 1}/{len(conditions)}] starting...")
    msa_mode: str = cond["msa_mode"]
    num_loops: int = cond["num_loops"]
    num_sampling_steps: int = cond["num_sampling_steps"]
    num_diffusion_samples: int = int(cond.get("num_diffusion_samples", args.num_diffusion_samples))

    tag = f"msa={msa_mode}__loops={num_loops}__steps={num_sampling_steps}"
    cond_output_dir = args.output_dir / tag
    cond_output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("")
    LOGGER.info("─" * 60)
    LOGGER.info("[%d/%d] Condition: %s", cond_idx + 1, len(conditions), tag)

    _log_condition_progress(cond_idx, conditions, entries_completed, total_entries_needed, wall_start)

    should_skip, entries_completed = _check_and_report_completion(
        cond_output_dir, entry_names, args, num_diffusion_samples, entries_completed
    )
    if should_skip:
        return 0.0, entries_completed

    write_prediction_config(
        cond_output_dir,
        msa_mode=msa_mode,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=args.seed,
        local_model=args.local_model,
        device=args.device,
        msa_dir=args.msa_dir,
        msa_max_sequences=args.msa_max_sequences,
        save_pae=args.save_pae,
    )

    if args.dry_run:
        return 0.0, entries_completed

    msa_dir_for_condition = args.msa_dir if msa_mode in ("full_msa", "antigen_only") else None
    t0 = time.perf_counter()

    entries_completed = _run_condition_inference(
        all_entries,
        cond_output_dir,
        args,
        model,
        forge_client,
        msa_mode,
        msa_dir_for_condition,
        num_loops,
        num_sampling_steps,
        num_diffusion_samples,
        sabdab_binders,
        entries_completed,
    )

    elapsed = time.perf_counter() - t0
    LOGGER.info("")
    _print_progress_bar(len(entry_names), len(entry_names), prefix="Condition entries", bar_length=40)
    LOGGER.info("✓ Condition %s completed (%d entries in %.1f hrs)", tag, len(entry_names), elapsed / 3600)
    LOGGER.info("─" * 60)

    return elapsed, entries_completed


def _print_final_summary(
    timings: dict[str, float],
    entries_completed: int,
    total_entries_needed: int,
    wall_elapsed: float,
    args: argparse.Namespace,
) -> None:
    """Print final summary statistics.

    Parameters
    ----------
    timings : dict
        Per-condition timing data
    entries_completed : int
        Total entries completed
    total_entries_needed : int
        Total entries needed
    wall_elapsed : float
        Total wall clock time
    args : argparse.Namespace
        Command-line arguments
    """
    LOGGER.info("")
    LOGGER.info("═" * 60)
    LOGGER.info("FINAL SUMMARY")
    LOGGER.info("═" * 60)

    # Show final overall progress bar
    _print_progress_bar(entries_completed, total_entries_needed, prefix="Overall", bar_length=50)
    LOGGER.info("")

    # Calculate throughput
    if wall_elapsed > 0 and entries_completed > 0:
        entries_per_hour = entries_completed / (wall_elapsed / 3600)
        LOGGER.info(
            "Entries completed: %d / %d (%.1f%%)",
            entries_completed,
            total_entries_needed,
            (entries_completed / total_entries_needed * 100) if total_entries_needed > 0 else 0,
        )
        LOGGER.info("Throughput: %.1f entries/hour", entries_per_hour)
    else:
        LOGGER.info("Entries completed: %d / %d", entries_completed, total_entries_needed)

    LOGGER.info("Total wall time: %.1f hours (%.1f days)", wall_elapsed / 3600, wall_elapsed / 86400)
    LOGGER.info("")
    LOGGER.info("Per-condition timings:")
    for k, v in sorted(timings.items()):
        if k != "Total" and v > 0:
            LOGGER.info("  %-50s %.1f hrs", k, v / 3600)
    LOGGER.info("")

    # Final status check
    LOGGER.info("Output directory: %s", args.output_dir)
    LOGGER.info("═" * 60)


def _setup_inference_mode(args: argparse.Namespace):
    """Set up inference mode (local or forge).

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    tuple
        (model, forge_client) - one will be None depending on mode
    """
    model = None
    forge_client = None

    if args.mode == "local":
        try:
            import torch  # noqa: PLC0415
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Local ESMFold2 inference requires the optional Poetry group. "
                "Install it from a Python 3.12 environment with 'poetry install --with esm'."
            ) from exc

        LOGGER.info("Loading ESMFold2 model from %s", args.local_model)
        model = ESMFold2Model.from_pretrained(f"{args.local_model}")
        model = model.to(args.device)
        model.eval()

        # Log GPU info
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            LOGGER.info("CUDA available: %d GPUs detected", n_gpus)
            LOGGER.info("Using device: %s", args.device)
            if n_gpus > 1:
                LOGGER.info(
                    "Note: To use multiple GPUs, run multiple grid instances in parallel "
                    "with different --device values and split --input_jsons"
                )
        else:
            LOGGER.info("CUDA not available, using CPU")
    else:  # forge mode
        try:
            from esm.sdk.api import FoldingConfig  # noqa: PLC0415, F401
            from esm.sdk.forge import SequenceStructureForgeInferenceClient  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Forge ESMFold2 inference requires the optional Poetry group. "
                "Install it from a Python 3.12 environment with 'poetry install --with esm'."
            ) from exc

        token = os.environ.get(args.token_env_var)
        if not token:
            raise EnvironmentError(
                f"Forge mode requires the {args.token_env_var} environment variable to be set. "
                f"Set it with: export {args.token_env_var}=<your-token>"
            )

        LOGGER.info("Using Forge API: %s (model: %s)", args.forge_url, args.forge_model)
        forge_client = SequenceStructureForgeInferenceClient(
            model=args.forge_model,
            url=args.forge_url,
            token=token,
        )

    return model, forge_client


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an ESMFold2 prediction grid (MSA mode × recycling loops × diffusion steps).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--json_dir",
        type=Path,
        help="Directory of ESMFold2 JSON inputs; --n_inputs are sampled randomly",
    )
    input_group.add_argument(
        "--input_jsons",
        nargs="+",
        type=Path,
        metavar="JSON",
        help="Explicit list of ESMFold2 JSON files to use",
    )

    parser.add_argument(
        "--msa_dir",
        type=Path,
        default=None,
        help="Directory of pre-fetched .a3m files named by sha256 hash. Required for full_msa mode.",
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
        help="Number of JSON inputs to sample when --json_dir is used (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for input sampling and ESMFold2 diffusion",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "forge"],
        default="local",
        help="Use local model (default) or Biohub Forge API for remote inference",
    )
    parser.add_argument(
        "--local_model",
        default="biohub/ESMFold2",
        help="Hugging Face model id or local path for local mode",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "Torch device for local mode, e.g. cuda, cuda:0, or cpu. "
            "Note: ESMFold2 currently processes one structure at a time per GPU. "
            "For multi-GPU parallelization, run multiple grid script instances with different "
            "--input_jsons subsets and different --device values (cuda:0, cuda:1, etc.)."
        ),
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
        "--num_diffusion_samples",
        type=int,
        default=DEFAULT_NUM_DIFFUSION_SAMPLES,
        help="Number of diffusion samples per entry per condition",
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
        help="Save a compressed PAE matrix (.npz) for each prediction",
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
        "--num_loops_values",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=f"Recycling loop values (default: {NUM_LOOPS})",
    )
    parser.add_argument(
        "--num_sampling_steps_values",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=f"Diffusion step values (default: {NUM_SAMPLING_STEPS})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print conditions without running anything",
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
            "full grid product. Each entry must have msa_mode, num_loops, and "
            "num_sampling_steps; optionally num_diffusion_samples (overrides "
            "--num_diffusion_samples for that condition). Example entry:\n"
            "  - msa_mode: no_msa\n    num_loops: 5\n    num_sampling_steps: 50\n    num_diffusion_samples: 2"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _collect_input_files(args: argparse.Namespace) -> list[Path]:
    """Collect input JSON files based on args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    list of Path
        List of JSON file paths to process
    """
    if args.input_jsons:
        json_files = [p.resolve() for p in args.input_jsons]
        LOGGER.info("Input files (%d):", len(json_files))
        for f in json_files:
            LOGGER.info("  %s", f.name)
        return json_files

    # Check if json_dir is actually a single file
    if args.json_dir.is_file() and args.json_dir.suffix == ".json":
        json_files = [args.json_dir.resolve()]
        LOGGER.info("Loading from single JSON file: %s", args.json_dir)
        return json_files

    # Directory of JSON files
    all_jsons = sorted(args.json_dir.glob("*.json"))
    if not all_jsons:
        raise SystemExit(f"No JSON files found in {args.json_dir}")

    if args.n_inputs is None:
        json_files = all_jsons
        LOGGER.info("Using all %d inputs", len(json_files))
    else:
        random.seed(args.seed)
        n = min(args.n_inputs, len(all_jsons))
        json_files = random.sample(all_jsons, n)
        LOGGER.info("Sampled %d / %d inputs (seed=%d)", n, len(all_jsons), args.seed)

    LOGGER.info("Input files (%d):", len(json_files))
    for f in json_files:
        LOGGER.info("  %s", f.name)
    return json_files


def _load_all_entries(json_files: list[Path], args: argparse.Namespace) -> list[dict]:
    """Load entries from JSON files.

    Parameters
    ----------
    json_files : list of Path
        JSON files to load
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    list of dict
        Loaded entries
    """
    all_entries: list[dict] = []
    for jf in json_files:
        entries = load_entries(jf, None)
        all_entries.extend(entries)

    if not all_entries:
        raise SystemExit("No valid ESMFold2 entries could be loaded from the input JSONs")

    LOGGER.info("Loaded %d total entries from %d file(s)", len(all_entries), len(json_files))

    # If sampling was requested and we have a single file with many entries, sample entries
    if len(json_files) == 1 and args.n_inputs is not None and args.n_inputs < len(all_entries):
        total_before_sampling = len(all_entries)
        random.seed(args.seed)
        all_entries = random.sample(all_entries, args.n_inputs)
        LOGGER.info("Sampled %d / %d entries (seed=%d)", len(all_entries), total_before_sampling, args.seed)

    LOGGER.info("Loaded %d entries total", len(all_entries))
    return all_entries


def _load_sabdab_binders_if_needed(args: argparse.Namespace) -> dict[str, set[str]] | None:
    """Load SAbDab binder chain lookup if needed.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    dict or None
        Binder chain mapping if loaded
    """
    if args.sabdab_tsv:
        return load_sabdab_binders(args.sabdab_tsv)

    if "antigen_only" in (args.msa_modes or MSA_MODES):
        LOGGER.warning(
            "Running antigen_only mode without --sabdab_tsv: "
            "falling back to H/L chain-name heuristic (may misidentify non-H/L chains)"
        )
    return None


def _build_grid_conditions(args: argparse.Namespace) -> tuple[list[dict], list[str], list[int], list[int]]:
    """Build grid conditions from arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    tuple
        (conditions, active_msa_modes, active_loops, active_steps)
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
        active_loops = sorted(set(c.get("num_loops") for c in conditions if "num_loops" in c))
        active_steps = sorted(set(c.get("num_sampling_steps") for c in conditions if "num_sampling_steps" in c))
    else:
        active_msa_modes = args.msa_modes if args.msa_modes else MSA_MODES
        active_loops = args.num_loops_values if args.num_loops_values else NUM_LOOPS
        active_steps = args.num_sampling_steps_values if args.num_sampling_steps_values else NUM_SAMPLING_STEPS

        conditions = [
            {"msa_mode": msa, "num_loops": loops, "num_sampling_steps": steps}
            for msa, loops, steps in iproduct(active_msa_modes, active_loops, active_steps)
        ]

    # Validate MSA requirements
    if any(mode in active_msa_modes for mode in ("full_msa", "antigen_only")) and args.msa_dir is None:
        raise SystemExit("--msa_dir is required when using full_msa or antigen_only mode")

    return conditions, active_msa_modes, active_loops, active_steps


def _write_grid_config(
    args: argparse.Namespace,
    json_files: list[Path],
    all_entries: list[dict],
    conditions: list[dict],
    active_msa_modes: list[str],
    active_loops: list[int],
    active_steps: list[int],
) -> None:
    """Write grid configuration to YAML file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    json_files : list of Path
        Input JSON files
    all_entries : list of dict
        All loaded entries
    conditions : list of dict
        Grid conditions
    active_msa_modes : list of str
        Active MSA modes
    active_loops : list of int
        Active loop values
    active_steps : list of int
        Active step values
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid_config = {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "n_conditions": len(conditions),
        "n_inputs": len(json_files),
        "n_entries": len(all_entries),
        "inputs": [str(f) for f in json_files],
        "seed": args.seed,
        "num_diffusion_samples": args.num_diffusion_samples,
        "msa_max_sequences": args.msa_max_sequences,
        "save_pae": args.save_pae,
        "grid_axes": {
            "msa_modes": active_msa_modes,
            "num_loops": active_loops,
            "num_sampling_steps": active_steps,
        },
        "conditions": conditions,
    }
    grid_config_path = args.output_dir / "grid_config.yaml"
    grid_config_path.write_text(
        yaml.dump(grid_config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    LOGGER.info("Grid config written → %s", grid_config_path)


def main() -> None:
    """Run ESMFold2 grid predictions across MSA and sampling conditions."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    _log_timestamp("Script started")

    # Collect inputs and load data
    json_files = _collect_input_files(args)
    all_entries = _load_all_entries(json_files, args)
    sabdab_binders = _load_sabdab_binders_if_needed(args)

    # Build grid conditions
    conditions, active_msa_modes, active_loops, active_steps = _build_grid_conditions(args)

    LOGGER.info(
        "Grid: %d conditions × %d entries = %d total runs",
        len(conditions),
        len(all_entries),
        len(conditions) * len(all_entries),
    )

    # Write grid config
    _write_grid_config(args, json_files, all_entries, conditions, active_msa_modes, active_loops, active_steps)

    # Set up inference mode
    model, forge_client = _setup_inference_mode(args)

    # Run grid
    wall_start = time.perf_counter()
    timings: dict[str, float] = {}
    entry_names = [e["name"] for e in all_entries]
    total_entries_needed = len(conditions) * len(entry_names)
    entries_completed = 0

    LOGGER.info("═" * 60)
    LOGGER.info(
        "Total work: %d conditions × %d entries = %d predictions",
        len(conditions),
        len(entry_names),
        total_entries_needed,
    )
    LOGGER.info("═" * 60)

    for cond_idx, cond in enumerate(conditions):
        elapsed, entries_completed = _process_condition(
            cond_idx,
            cond,
            conditions,
            all_entries,
            entry_names,
            args,
            model,
            forge_client,
            sabdab_binders,
            wall_start,
            entries_completed,
            total_entries_needed,
        )
        tag = f"msa={cond['msa_mode']}__loops={cond['num_loops']}__steps={cond['num_sampling_steps']}"
        timings[tag] = elapsed

    wall_elapsed = time.perf_counter() - wall_start
    timings["Total"] = wall_elapsed

    if not args.dry_run:
        _print_final_summary(timings, entries_completed, total_entries_needed, wall_elapsed, args)

    LOGGER.info("All done. Results in %s", args.output_dir)
    _log_timestamp("Script completed successfully")


if __name__ == "__main__":
    main()
