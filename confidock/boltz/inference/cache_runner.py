"""Run Boltz-2 in score-given-structure mode and cache confidence-head inputs.

This module splits ``Boltz2.forward`` at the trunk/confidence boundary so the
expensive part runs once per entry rather than once per decoy.  See
:mod:`confidock.boltz.inference.confidence_cache` for why that is valid.

Scoring semantics
-----------------
We set ``model.skip_run_structure = True``.  In ``boltz2.py`` that makes the
confidence head receive ``feats["coords"].repeat_interleave(diffusion_samples, 0)``
instead of a freshly diffused structure, so the head scores *exactly the decoy
we supply*.  This is what makes the cache joinable to the decoy's known DockQ
label -- passing structures as YAML ``templates`` would instead have Boltz
re-predict a new structure, and the cached inputs would describe that new
structure rather than the one the label was computed for.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from boltz.data.module.inferencev2 import PredictionDataset, collate
from boltz.data.types import Manifest, Record, StructureV2
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2

from confidock.boltz.inference.confidence_cache import (
    CONFIDENCE_FEAT_KEYS,
    EntryCache,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CacheRunConfig",
    "build_dataset",
    "build_x_pred",
    "feats_for_confidence",
    "load_cache_model",
    "run_confidence_head",
    "run_trunk",
]

#: Confidence metrics worth recording per decoy (scalars only).
CONFIDENCE_SCALARS: tuple[str, ...] = (
    "ptm",
    "iptm",
    "protein_iptm",
    "ligand_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)


@dataclass
class CacheRunConfig:
    """Configuration for a caching run."""

    #: Boltz-2 ``processed`` directory produced by an earlier ``boltz predict``
    #: run; supplies structures, MSAs, records, constraints and the manifest.
    processed_dir: Path
    #: Boltz cache directory containing ``mols/`` and the checkpoint.
    boltz_cache: Path
    #: Where entry caches and decoy coordinates are written.
    output_dir: Path
    checkpoint: Path | None = None
    recycling_steps: int = 3
    #: Fixed so the MSA subsampling in ``trunkv2.py:632`` is reproducible.
    seed: int = 42
    subsample_msa: bool = True
    num_subsampled_msa: int = 1024
    #: fp16 halves the entry caches (149 GiB vs 298 GiB across the dataset) at a
    #: measured cost of |Δiptm| <= 3e-5 when re-scoring from cache.
    store_dtype: str = "float16"
    device: str = "cuda"
    #: Boltz's Triton kernels; disabled by default for portability.
    use_kernels: bool = False

    @property
    def torch_store_dtype(self) -> torch.dtype:
        """``store_dtype`` as a torch dtype."""
        return getattr(torch, self.store_dtype)

    @property
    def mol_dir(self) -> Path:
        """CCD component directory inside the Boltz cache."""
        return self.boltz_cache / "mols"

    @property
    def checkpoint_path(self) -> Path:
        """Explicit ``checkpoint``, else the default one in the Boltz cache."""
        return self.checkpoint or (self.boltz_cache / "boltz2_conf.ckpt")


def load_cache_model(config: CacheRunConfig) -> Boltz2:
    """Load Boltz-2 configured to score supplied structures.

    ``skip_run_structure=True`` bypasses diffusion entirely, so no structure is
    generated and ``feats["coords"]`` is scored verbatim.
    """
    diffusion = Boltz2DiffusionParams()
    pairformer = PairformerArgsV2()
    msa = MSAModuleArgs(
        subsample_msa=config.subsample_msa,
        num_subsampled_msa=config.num_subsampled_msa,
        use_paired_feature=True,
    )
    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False

    model = Boltz2.load_from_checkpoint(
        config.checkpoint_path,
        strict=True,
        map_location="cpu",
        predict_args={
            "recycling_steps": config.recycling_steps,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        diffusion_process_args=asdict(diffusion),
        ema=False,
        use_kernels=config.use_kernels,
        pairformer_args=asdict(pairformer),
        msa_args=asdict(msa),
        steering_args=asdict(steering),
    )
    model.eval()
    model.to(config.device)
    # Score the structure we hand in; never diffuse a new one.
    model.skip_run_structure = True
    model.confidence_prediction = True
    model.run_trunk_and_structure = True
    return model


def build_dataset(config: CacheRunConfig, records: list[Record]) -> PredictionDataset:
    """Build a ``PredictionDataset`` over ``records`` from the processed dir.

    Reusing the existing ``processed`` directory means we inherit Boltz's own
    tokenisation and atom ordering rather than recreating it -- the ordering that
    ``x_pred`` must match.
    """
    processed = config.processed_dir
    return PredictionDataset(
        manifest=Manifest(records=records),
        target_dir=processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=config.mol_dir,
        constraints_dir=processed / "constraints",
        template_dir=processed / "templates",
        extra_mols_dir=processed / "mols",
    )


@torch.no_grad()
def run_trunk(
    model: Boltz2, feats: dict[str, torch.Tensor], recycling_steps: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run input embedder + recycled template/MSA/Pairformer stack.

    Mirrors the trunk portion of ``Boltz2.forward`` (boltz2.py:402-489).  Kept as
    an explicit reimplementation rather than a hook because we need ``s``/``z``
    *before* the structure module and want to invoke the confidence head many
    times against them.  Verified bit-exact against the full forward.
    """
    s_inputs = model.input_embedder(feats)
    s_init = model.s_init(s_inputs)

    z_init = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
    z_init = z_init + model.rel_pos(feats)
    z_init = z_init + model.token_bonds(feats["token_bonds"].float())
    if model.bond_type_feature:
        z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
    z_init = z_init + model.contact_conditioning(feats)

    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)

    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    for _ in range(recycling_steps + 1):
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z_init + model.z_recycle(model.z_norm(z))

        if model.use_templates:
            template_module = (
                model.template_module._orig_mod if model.is_template_compiled else model.template_module  # noqa: SLF001
            )
            z = z + template_module(z, feats, pair_mask, use_kernels=model.use_kernels)

        msa_module = model.msa_module._orig_mod if model.is_msa_compiled else model.msa_module  # noqa: SLF001
        z = z + msa_module(z, s_inputs, feats, use_kernels=model.use_kernels)

        pairformer = (
            model.pairformer_module._orig_mod  # noqa: SLF001
            if model.is_pairformer_compiled
            else model.pairformer_module
        )
        s, z = pairformer(s, z, mask=mask, pair_mask=pair_mask, use_kernels=model.use_kernels)

    return s_inputs, s, z


def feats_for_confidence(
    cached_feats: dict[str, torch.Tensor],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Make a cached ``feats`` dict usable by the confidence head.

    :data:`~confidock.boltz.inference.confidence_cache.CONFIDENCE_FEAT_KEYS`
    deliberately excludes ``coords`` -- storing it per entry would be wrong, since
    each decoy supplies its own.  But :func:`build_x_pred` reads the padded atom
    width from ``feats["coords"]``, so add a zero placeholder of the right shape,
    taken from ``atom_pad_mask``.
    """
    feats = dict(cached_feats)
    if "coords" not in feats:
        n_padded = feats["atom_pad_mask"].shape[-1]
        feats["coords"] = torch.zeros((1, 1, n_padded, 3), device=device)
    return feats


def build_x_pred(
    coords: torch.Tensor | Any,
    feats: dict[str, torch.Tensor],
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Place per-atom ``coords`` into the featurizer's padded atom dimension.

    The featurizer pads the atom axis (``feats["coords"]`` is ``(1, 1, n_padded,
    3)`` with zeros beyond the real atoms, e.g. 2752 slots for 2725 atoms), while
    :func:`~confidock.boltz.inference.decoy_coords.load_decoy_coords`
    returns exactly one row per atom in the structure.  Handing the unpadded
    array straight to the confidence head raises a shape mismatch, so pad it here
    and keep the padded rows at zero as Boltz does.
    """
    target = feats["coords"]
    n_padded = target.shape[-2]
    if not torch.is_tensor(coords):
        coords = torch.from_numpy(coords)
    coords = coords.reshape(-1, 3).to(device=device, dtype=torch.float32)

    n_atoms = coords.shape[0]
    if n_atoms > n_padded:
        msg = f"decoy has {n_atoms} atoms but the featurizer allocated only {n_padded} slots"
        raise ValueError(msg)

    out = torch.zeros((1, 1, n_padded, 3), dtype=torch.float32, device=device)
    out[0, 0, :n_atoms] = coords
    return out


@torch.no_grad()
def run_confidence_head(
    model: Boltz2,
    *,
    s_inputs: torch.Tensor,
    s: torch.Tensor,
    z: torch.Tensor,
    feats: dict[str, torch.Tensor],
    coords: torch.Tensor,
    pred_distogram_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Score ``coords`` with the confidence head against cached trunk outputs.

    ``coords`` is ``(1, 1, n_atoms, 3)`` or ``(1, n_atoms, 3)``; the head
    reshapes 4-D input itself (confidencev2.py:191).
    """
    if pred_distogram_logits is None:
        pred_distogram_logits = model.distogram_module(z)[:, :, :, 0]
    return model.confidence_module(
        s_inputs=s_inputs,
        s=s,
        z=z,
        x_pred=coords,
        feats=feats,
        pred_distogram_logits=pred_distogram_logits,
        multiplicity=1,
        run_sequentially=True,
        use_kernels=model.use_kernels,
    )


def featurize_entry(config: CacheRunConfig, record: Record) -> dict[str, torch.Tensor]:
    """Featurize one entry and move tensors to the run device."""
    dataset = build_dataset(config, [record])
    feats = collate([dataset[0]])
    return {key: (value.to(config.device) if torch.is_tensor(value) else value) for key, value in feats.items()}


def build_entry_cache(
    config: CacheRunConfig,
    record: Record,
    feats: dict[str, torch.Tensor],
    s_inputs: torch.Tensor,
    s: torch.Tensor,
    z: torch.Tensor,
    structure: StructureV2,
) -> EntryCache:
    """Assemble the on-disk cache object for one entry."""
    metadata: dict[str, Any] = {
        "recycling_steps": config.recycling_steps,
        "seed": config.seed,
        "subsample_msa": config.subsample_msa,
        "num_subsampled_msa": config.num_subsampled_msa,
        "n_tokens": int(s.shape[1]),
        "n_atoms": int(len(structure.atoms)),
        "chains": [str(c["name"]) for c in structure.chains],
        "store_dtype": config.store_dtype,
        "use_kernels": config.use_kernels,
    }
    return EntryCache(
        entry=record.id,
        s_inputs=s_inputs,
        s=s,
        z=z,
        feats={k: feats[k] for k in CONFIDENCE_FEAT_KEYS if k in feats},
        metadata=metadata,
    )
