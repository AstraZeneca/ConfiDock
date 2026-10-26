"""Pooled confidence latents: extract once, train the head many times.

Why this exists
---------------
With the confidence module frozen and in ``eval()`` (no dropout), the pooled
vector :func:`~.dockq_head.pool_confidence_latents` produces for a given
``(entry, decoy)`` is **deterministic**.  Training the head directly against the
module therefore recomputes identical numbers every epoch: measured at ~6 min per
712 decoys on an A100, that is ~13.5 h for one epoch over the 97k training decoys,
and the second epoch costs exactly the same again.

Extracting the pooled vectors once turns the fine-tuning problem into a 640-wide
tabular regression: 108k x 640 floats is 264 MiB, fits in RAM, and an epoch over
it takes seconds on one GPU.  That is what makes a hyperparameter sweep or a
ranking-loss ablation affordable rather than a multi-day commitment.

The cost is that the pooling is frozen into the artifact.  Changing
``pool_confidence_latents`` -- or unfreezing the confidence module, where dropout
and updated weights make the latents non-deterministic -- requires re-extraction,
so :func:`save_latent_shard` records a ``pooling_version`` and the module's own
config alongside the data, and :func:`load_latents` refuses to mix versions.

Layout::

    {latent_dir}/
        latents/<entry>.npz     # pooled (n_decoys, in_dim) + labels + baselines
        manifest.shard<N>.jsonl # one line per completed entry
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from confidock.boltz.inference.cache_runner import CONFIDENCE_SCALARS

LOGGER = logging.getLogger(__name__)

__all__ = [
    "POOLING_VERSION",
    "LatentTable",
    "load_latents",
    "save_latent_shard",
]

#: Bump whenever ``pool_confidence_latents`` changes what it returns.  Stored in
#: every shard so a stale cache is a loud error rather than a silently wrong run.
POOLING_VERSION = 1

#: The DockQ sub-score labels the composite head is supervised on.  These are
#: *labels*, not features -- they do not depend on the pooling -- so shards written
#: before the composite loss existed are upgraded in place by
#: ``workflows/scripts/data/backfill_dockq_components.py`` rather than re-extracted
#: on the GPU.  That is why their absence is a pointed error below and not a
#: ``POOLING_VERSION`` bump.
COMPONENT_KEYS = ("fnat", "irmsd", "lrmsd")


@dataclass
class LatentTable:
    """All pooled latents in memory, plus the labels and Boltz baselines.

    Attributes
    ----------
    pooled
        ``(n_decoys, in_dim)`` float32 pooled confidence latents.
    dockq
        ``(n_decoys,)`` ground-truth DockQ.
    fnat, irmsd, lrmsd
        ``(n_decoys,)`` the DockQ sub-scores as reported -- ``fnat`` a fraction, the
        two RMSDs in Angstrom.  The DockQ distance scaling is applied downstream by
        :func:`~.dockq_head.component_targets`, so the table stores the raw values
        and the scaling constants live in one place.
    entry_index
        ``(n_decoys,)`` int index into :attr:`entries`; the grouping key for both
        the within-entry ranking loss and the per-entry correlation report.
    entries
        Unique entry ids, ``entry_index`` indexes this.
    baseline
        ``{metric: (n_decoys,)}`` Boltz confidence scalars, NaN where absent.
    decoy_ids, sources
        Per-decoy provenance.
    """

    pooled: np.ndarray
    dockq: np.ndarray
    fnat: np.ndarray
    irmsd: np.ndarray
    lrmsd: np.ndarray
    entry_index: np.ndarray
    entries: list[str]
    baseline: dict[str, np.ndarray]
    decoy_ids: list[str]
    sources: list[str]

    def __len__(self) -> int:
        """Return the number of decoys."""
        return int(self.pooled.shape[0])

    @property
    def in_dim(self) -> int:
        """Return the pooled feature width."""
        return int(self.pooled.shape[1])

    def subset(self, keep_entries: set[str]) -> "LatentTable":
        """Restrict to decoys whose entry is in ``keep_entries``."""
        wanted = {i for i, e in enumerate(self.entries) if e in keep_entries}
        mask = np.fromiter((i in wanted for i in self.entry_index), bool, len(self))
        kept_entries = [e for e in self.entries if e in keep_entries]
        remap = {self.entries.index(e): j for j, e in enumerate(kept_entries)}
        return LatentTable(
            pooled=self.pooled[mask],
            dockq=self.dockq[mask],
            fnat=self.fnat[mask],
            irmsd=self.irmsd[mask],
            lrmsd=self.lrmsd[mask],
            entry_index=np.array([remap[i] for i in self.entry_index[mask]], dtype=np.int64),
            entries=kept_entries,
            baseline={k: v[mask] for k, v in self.baseline.items()},
            decoy_ids=[d for d, m in zip(self.decoy_ids, mask, strict=True) if m],
            sources=[s for s, m in zip(self.sources, mask, strict=True) if m],
        )


def save_latent_shard(
    path: Path,
    *,
    entry: str,
    pooled: torch.Tensor,
    dockq: list[float],
    components: dict[str, list[float]],
    decoy_ids: list[str],
    sources: list[str],
    baseline: dict[str, list[float]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write one entry's pooled latents.

    ``components`` holds the three DockQ sub-scores (:data:`COMPONENT_KEYS`) as
    reported -- ``fnat`` a fraction, the RMSDs in Angstrom -- one list per key,
    aligned with ``dockq``.

    Float32 throughout: the pooled vectors are means over up to 600^2 pair
    entries, and the head's first op is a LayerNorm, so fp16 rounding here would
    show up directly in the ranking the head learns.
    """
    missing = [k for k in COMPONENT_KEYS if k not in components]
    if missing:
        msg = f"components missing {missing}; the composite head needs all of {COMPONENT_KEYS}"
        raise ValueError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pooled": pooled.detach().float().cpu().numpy(),
        "dockq": np.asarray(dockq, dtype=np.float32),
        **{k: np.asarray(components[k], dtype=np.float32) for k in COMPONENT_KEYS},
        "decoy_ids": np.asarray(decoy_ids, dtype=object),
        "sources": np.asarray(sources, dtype=object),
        "metadata": json.dumps(
            {
                "entry": entry,
                "pooling_version": POOLING_VERSION,
                **(metadata or {}),
            }
        ),
    }
    for key, values in baseline.items():
        payload[f"baseline__{key}"] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(path, **payload)


def load_latents(latent_dir: Path, entries: list[str] | None = None) -> LatentTable:
    """Load pooled latents into one :class:`LatentTable`.

    Parameters
    ----------
    latent_dir
        Directory holding ``latents/<entry>.npz``.
    entries
        Restrict to these entry ids; ``None`` loads everything present.

    Raises
    ------
    ValueError
        If a shard was written by a different :data:`POOLING_VERSION`, if a shard
        predates the DockQ sub-score labels, or if no shards were found.  A silent
        partial load would look like a training set that merely got smaller.
    """
    root = Path(latent_dir)
    root = root / "latents" if (root / "latents").is_dir() else root
    paths = sorted(root.glob("*.npz"))
    if entries is not None:
        wanted = set(entries)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        msg = f"no latent shards found under {root}"
        raise ValueError(msg)

    pooled_parts: list[np.ndarray] = []
    dockq_parts: list[np.ndarray] = []
    component_parts: dict[str, list[np.ndarray]] = {k: [] for k in COMPONENT_KEYS}
    entry_index_parts: list[np.ndarray] = []
    entry_names: list[str] = []
    decoy_ids: list[str] = []
    sources: list[str] = []
    baseline_parts: dict[str, list[np.ndarray]] = {k: [] for k in CONFIDENCE_SCALARS}

    for path in paths:
        blob = np.load(path, allow_pickle=True)
        meta = json.loads(str(blob["metadata"]))
        version = meta.get("pooling_version")
        if version != POOLING_VERSION:
            msg = (
                f"{path} was written with pooling_version={version}, expected "
                f"{POOLING_VERSION}; re-extract the latents"
            )
            raise ValueError(msg)

        absent = [k for k in COMPONENT_KEYS if k not in blob]
        if absent:
            msg = (
                f"{path} has no {absent} labels, which the composite head is "
                "supervised on; run workflows/scripts/data/backfill_dockq_components.py "
                "--latent_dir to add them (no GPU re-extraction needed)"
            )
            raise ValueError(msg)

        n = int(blob["pooled"].shape[0])
        pooled_parts.append(blob["pooled"].astype(np.float32))
        dockq_parts.append(blob["dockq"].astype(np.float32))
        for key in COMPONENT_KEYS:
            component_parts[key].append(blob[key].astype(np.float32))
        entry_index_parts.append(np.full(n, len(entry_names), dtype=np.int64))
        entry_names.append(path.stem)
        decoy_ids.extend(str(d) for d in blob["decoy_ids"])
        sources.extend(str(s) for s in blob["sources"])

        for key in CONFIDENCE_SCALARS:
            field = f"baseline__{key}"
            baseline_parts[key].append(
                blob[field].astype(np.float32) if field in blob else np.full(n, np.nan, dtype=np.float32)
            )

    table = LatentTable(
        pooled=np.concatenate(pooled_parts, axis=0),
        dockq=np.concatenate(dockq_parts, axis=0),
        **{k: np.concatenate(v, axis=0) for k, v in component_parts.items()},
        entry_index=np.concatenate(entry_index_parts, axis=0),
        entries=entry_names,
        # Drop metrics that are absent everywhere; keeping an all-NaN column would
        # only show up as a NaN row in the correlation report.
        baseline={
            k: np.concatenate(v, axis=0)
            for k, v in baseline_parts.items()
            if not np.isnan(np.concatenate(v, axis=0)).all()
        },
        decoy_ids=decoy_ids,
        sources=sources,
    )
    LOGGER.info(
        "loaded %d decoys / %d entries / in_dim=%d from %s",
        len(table),
        len(table.entries),
        table.in_dim,
        root,
    )
    return table
