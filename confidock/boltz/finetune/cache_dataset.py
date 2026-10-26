"""Entry-grouped dataset over the Boltz-2 confidence cache.

One dataset item is **one entry with a batch of its decoys**, not one decoy.  The
entry cache holds ``z`` at ``(1, L, L, 128)`` -- ~46 MiB for a typical entry and
up to ~180 MiB for the largest -- while a decoy adds only ~36 KiB of coordinates.
Loading ``z`` once and scoring all ~33 decoys of that entry against it is the
difference between reading ~150 GiB per epoch and reading ~5 TiB.  Training is
I/O-bound either way (see the ``dockq-training-is-dataloader-bound`` finding), so
this grouping is the single most important performance decision here.

Grouping by entry also happens to be exactly what the ranking loss needs: a
within-entry ranking term requires several decoys of the *same* native in the
same optimiser step, which a decoy-shuffled loader would only supply by accident.

Label hygiene
-------------
672 of the 108,041 cached decoys (0.6%, concentrated in ``boltz2``) have a NaN
DockQ -- the DockQ computation failed for those structures.  Seven entries are
NaN throughout.  These are dropped at construction time rather than at loss time:
a single NaN reaching the loss would propagate to every head parameter through
the backward pass and silently destroy the run.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from confidock.boltz.inference.cache_runner import (
    CONFIDENCE_SCALARS,
    build_x_pred,
    feats_for_confidence,
)
from confidock.boltz.inference.confidence_cache import load_entry_cache

LOGGER = logging.getLogger(__name__)

#: The DockQ sub-score columns the composite head is supervised on.  Backfilled into
#: the cached score JSONLs from the DockQ manifests by
#: ``workflows/scripts/data/backfill_dockq_components.py`` -- the caches built before
#: that script recorded only ``dockq``.
COMPONENT_KEYS = ("fnat", "irmsd", "lrmsd")

#: Token count assumed when an entry's metadata cannot be read.  Above the
#: 600-token maximum of the cached entries on purpose, so an unreadable entry
#: gets the smallest batch rather than the largest.
_UNKNOWN_TOKENS = 4096

#: Measured peak allocation for one grad-enabled forward+backward of the confidence
#: module on an A100-40GB, at ``forward_chunk=1`` under bf16 autocast::
#:
#:     352 tokens, 16 decoys -> 15.1 GiB
#:     438 tokens, 16 decoys -> 22.9 GiB
#:     544 tokens, 16 decoys -> 35.5 GiB
#:      600 tokens,  4 decoys -> 22.4 GiB
#:      600 tokens,  8 decoys -> 29.4 GiB
#:      600 tokens, 12 decoys -> OOM at 34.8 GiB
#:
#: Fitting ``peak = (L/600)^2 * (FIXED + PER_DECOY * d)`` reproduces all five fitted
#: points to within 0.2 GiB and predicts the 12-decoy OOM at 36.3 GiB.  Both terms
#: are O(L^2) because the pairformer's triangular intermediates and ``z_conf`` are.
#:
#: Two earlier beliefs were wrong and are worth recording, because each cost a run:
#:
#: * ``forward_chunk`` *does* bound memory.  Under activation checkpointing only one
#:   chunk's activations are live at a time, so at 600 tokens chunk=1/4 decoys fits
#:   in 33.6 GiB while chunk=4/4 decoys OOMs at 36.6.  An earlier measurement said
#:   otherwise; it was taken with the *chunk* varied and the decoy count held at a
#:   size that could not fit either way.
#: * bf16 is not a uniform halving.  At chunk=4 it was slightly *worse* than fp32
#:   (34.9 vs 33.6 GiB) -- autocast holds bf16 casts alongside the fp32 masters, and
#:   the stack's ``autocast(enabled=False)`` regions keep their activations in fp32.
#:   It pays only in combination with chunk=1, where 600/8 drops from OOM to 29.4.
_MEM_FIXED_GIB = 15.5
_MEM_PER_DECOY_GIB = 1.73
_MEM_REF_TOKENS = 600.0


def decoy_budget(
    n_tokens: int,
    *,
    limit_gib: float = 34.0,
    max_decoys: int = 16,
    weights_gib: float = 1.9,
) -> int:
    """Return how many decoys fit in one grad-enabled batch at ``n_tokens``.

    A fixed ``max_decoys_per_batch`` cannot work here.  Peak memory is quadratic
    in token count, so a value that fits comfortably at 438 tokens is four times
    over budget at 600, and the first fine-tune run duly OOM'd on 49% of its
    batches -- every one of them between 324 and 600 tokens.  Dropping them was
    silent (the trainer catches ``OutOfMemoryError`` and skips), so the model
    trained almost exclusively on small complexes.

    The constants assume the trainer's measured configuration: ``forward_chunk=1``
    under bf16 autocast.  Both matter, and neither is a free tuning dial -- see the
    ``_MEM_*`` comments for what each is worth and which earlier belief about them
    was wrong.  Raising ``forward_chunk`` or disabling autocast invalidates this
    estimate in the expensive direction.

    Returns at least 2, aiming for the pair a within-entry ranking loss needs.  It
    is a target, not a guarantee: fragmentation in a live loop routinely defeats
    this estimate, so the trainer's OOM retry is what actually keeps the largest
    entries in the run (down to a single decoy, which still carries the regression
    terms even though the ranking term needs a pair).
    """
    scale = (max(n_tokens, 1) / _MEM_REF_TOKENS) ** 2
    available = limit_gib - weights_gib - _MEM_FIXED_GIB * scale
    if available <= 0:
        return 2
    fits = int(available / (_MEM_PER_DECOY_GIB * scale))
    return max(2, min(max_decoys, fits))


__all__ = [
    "EntryBatch",
    "EntryGroupedCacheDataset",
    "collate_entry_batches",
    "decoy_budget",
    "split_entries",
]


def _is_bad(value: Any) -> bool:
    """Whether a label is unusable (missing or NaN)."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


@dataclass
class _FilterStats:
    """Tally of what construction discarded, for the one-line dataset summary."""

    n_dropped_nan: int = 0
    n_dropped_flat: int = 0
    n_missing: int = 0
    n_dropped_components: int = 0


def _chunk_indices(entry: str, n_rows: int, per_batch: int) -> list[tuple[str, tuple[int, ...]]]:
    """Split one entry's rows into items of at most ``per_batch`` decoys each.

    The item count is the same ``ceil(n_rows / per_batch)`` a naive stride would give,
    but the rows are spread evenly over those items rather than left as full batches
    plus a remainder.  Striding 33 rows by 16 yields ``16, 16, 1``, and an item of one
    decoy gets no ranking term at all -- :class:`DockQLoss` needs a pair -- so it
    trains absolute quality only.  Balancing gives ``11, 11, 11`` instead: same number
    of forward passes, same memory ceiling (no item exceeds ``per_batch``), every item
    rankable.  On the SAbDab training cache this recovered 388 of 8336 items (4.7%).
    """
    n_items = max(1, math.ceil(n_rows / max(per_batch, 1)))
    base, remainder = divmod(n_rows, n_items)
    items: list[tuple[str, tuple[int, ...]]] = []
    start = 0
    for i in range(n_items):
        stop = start + base + (1 if i < remainder else 0)
        items.append((entry, tuple(range(start, stop))))
        start = stop
    return items


@dataclass
class EntryBatch:
    """One entry's cached trunk state plus a batch of its decoys.

    Attributes
    ----------
    entry
        SAbDab entry id, e.g. ``1a14_HLN``.
    s_inputs, s, z
        Cached trunk outputs, batch dimension 1 -- shared by every decoy here.
    feats
        The cached ``feats`` subset, already given a ``coords`` placeholder by
        :func:`feats_for_confidence` so :func:`build_x_pred` can size the atom
        axis.
    x_pred
        ``(n_decoys, 1, n_padded_atoms, 3)`` stacked decoy coordinates.
    dockq
        ``(n_decoys,)`` ground-truth DockQ labels.
    fnat, irmsd, lrmsd
        ``(n_decoys,)`` the DockQ sub-score inputs: fnat is already a fraction in
        ``[0, 1]``, iRMSD and LRMSD are in Angstrom and are pushed through the DockQ
        distance scaling by
        :func:`~confidock.boltz.finetune.dockq_head.component_targets` before
        they reach the loss.  Carried per-decoy so the composite head can be
        supervised on each sub-score, not just their mean.
    baseline
        ``{metric: (n_decoys,)}`` the Boltz confidence scalars recorded at cache
        time, carried through so evaluation can compare the new head against them
        on identical rows without re-reading the JSONL.
    decoy_ids, sources
        Per-decoy provenance, needed to report correlations per source.
    """

    entry: str
    s_inputs: torch.Tensor
    s: torch.Tensor
    z: torch.Tensor
    feats: dict[str, torch.Tensor]
    x_pred: torch.Tensor
    dockq: torch.Tensor
    fnat: torch.Tensor
    irmsd: torch.Tensor
    lrmsd: torch.Tensor
    baseline: dict[str, torch.Tensor]
    decoy_ids: list[str]
    sources: list[str]

    def __len__(self) -> int:
        """Return the number of decoys in this batch."""
        return len(self.decoy_ids)

    def to(self, device: str | torch.device) -> "EntryBatch":
        """Move every tensor to ``device`` (in place on the container)."""
        self.s_inputs = self.s_inputs.to(device, non_blocking=True)
        self.s = self.s.to(device, non_blocking=True)
        self.z = self.z.to(device, non_blocking=True)
        self.feats = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in self.feats.items()}
        self.x_pred = self.x_pred.to(device, non_blocking=True)
        self.dockq = self.dockq.to(device, non_blocking=True)
        self.fnat = self.fnat.to(device, non_blocking=True)
        self.irmsd = self.irmsd.to(device, non_blocking=True)
        self.lrmsd = self.lrmsd.to(device, non_blocking=True)
        self.baseline = {k: v.to(device, non_blocking=True) for k, v in self.baseline.items()}
        return self


def split_entries(entries: Sequence[str], *, val_fraction: float = 0.1, seed: int = 0) -> tuple[list[str], list[str]]:
    """Split entry ids into train/val, grouped by PDB id.

    Entries sharing a PDB id (e.g. ``1a2y_BAC`` and ``1a2y_ABC``) are the same
    structure with different chain selections, so they must land on the same side
    of the split or the "holdout" leaks.  Splitting on the ``pdb_id`` prefix
    rather than the entry id prevents that.
    """
    by_pdb: dict[str, list[str]] = {}
    for entry in entries:
        by_pdb.setdefault(entry.split("_")[0], []).append(entry)

    pdbs = sorted(by_pdb)
    rng = np.random.default_rng(seed)
    rng.shuffle(pdbs)

    n_val = max(1, int(round(len(pdbs) * val_fraction)))
    val_pdbs = set(pdbs[:n_val])

    train = sorted(e for p in pdbs if p not in val_pdbs for e in by_pdb[p])
    val = sorted(e for p in val_pdbs for e in by_pdb[p])
    return train, val


class EntryGroupedCacheDataset(Dataset):
    """Serve cached entries, each with up to ``max_decoys_per_batch`` decoys.

    Parameters
    ----------
    cache_dir
        Root of the confidence cache (holds ``entries/``, ``decoys/``, ``scores/``).
    entries
        Entry ids to serve.  Use :func:`split_entries` to build these.
    max_decoys_per_batch
        Cap on decoys per item.  An entry with more is split across several items,
        each re-reading ``z`` -- so keep this at or above the typical decoy count
        (33) to avoid paying that twice.  Its real purpose is bounding activation
        memory, which grows linearly in decoy count.
    token_aware_batching
        Size each entry's items by its token count rather than using
        ``max_decoys_per_batch`` flat.  Peak memory is quadratic in tokens, so one
        flat cap cannot fit both a 121-token and a 600-token entry; without this,
        roughly half the training set OOMs and is silently skipped.  Needs
        ``memory_limit_gib`` to match the card.
    memory_limit_gib
        Per-GPU budget for :func:`decoy_budget`, used only when
        ``token_aware_batching`` is set.  Its calibration assumes the trainer runs
        ``forward_chunk=1`` under bf16 autocast.
    shuffle_decoys
        Repartition which decoys appear together, once per epoch (training).  Every
        decoy is still served exactly once per epoch; only its company changes, and
        only after :meth:`set_epoch`.  ``False`` keeps the on-disk order so evaluation
        is reproducible.
    require_label_spread
        Drop items whose decoys all share one DockQ value: a within-entry ranking
        loss has no signal there, and including them only dilutes the gradient.
    """

    def __init__(
        self,
        cache_dir: Path,
        entries: Sequence[str],
        *,
        max_decoys_per_batch: int = 34,
        token_aware_batching: bool = False,
        memory_limit_gib: float = 34.0,
        shuffle_decoys: bool = True,
        require_label_spread: bool = True,
        store_dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_decoys_per_batch = int(max_decoys_per_batch)
        self.token_aware_batching = bool(token_aware_batching)
        self.memory_limit_gib = float(memory_limit_gib)
        self.shuffle_decoys = bool(shuffle_decoys)
        self.store_dtype = store_dtype
        self.seed = int(seed)
        self.epoch = 0

        self._records: dict[str, list[dict[str, Any]]] = {}
        self._items: list[tuple[str, tuple[int, ...]]] = []

        stats = _FilterStats()
        budgets: list[int] = []

        for entry in sorted(entries):
            scores = self.cache_dir / "scores" / f"{entry}.jsonl"
            entry_pt = self.cache_dir / "entries" / f"{entry}.pt"
            if not scores.is_file() or not entry_pt.is_file():
                stats.n_missing += 1
                continue

            rows = self._load_rows(entry, scores, stats)
            if not rows:
                continue
            if require_label_spread and len({round(r["dockq"], 6) for r in rows}) < 2:
                stats.n_dropped_flat += len(rows)
                continue

            self._records[entry] = rows
            per_batch = self._batch_size_for(entry_pt, budgets)
            self._items.extend(_chunk_indices(entry, len(rows), per_batch))

        # A stable integer per entry, used to seed its per-epoch decoy permutation.
        # It has to be stable *across processes*: the items of one entry are handed to
        # different dataloader workers, and they only add up to a partition of that
        # entry's decoys if every worker draws the same permutation.  ``hash(str)`` is
        # salted per process, so it cannot be used here.
        self._entry_ids = {entry: i for i, entry in enumerate(sorted(self._records))}

        if budgets:
            LOGGER.info(
                "token-aware batching at %.0f GiB: %d-%d decoys/item (mean %.1f)",
                self.memory_limit_gib,
                min(budgets),
                max(budgets),
                sum(budgets) / len(budgets),
            )

        LOGGER.info(
            "dataset: %d entries, %d items, %d decoys "
            "(dropped %d NaN-label, %d no-spread, %d missing, %d no-components)",
            len(self._records),
            len(self._items),
            sum(len(v) for v in self._records.values()),
            stats.n_dropped_nan,
            stats.n_dropped_flat,
            stats.n_missing,
            stats.n_dropped_components,
        )

    def _load_rows(self, entry: str, scores: Path, stats: _FilterStats) -> list[dict[str, Any]]:
        """Read one entry's score rows, dropping NaN labels and absent decoy files.

        See the module docstring for why NaN labels are dropped here rather than at
        loss time.  ``stats`` accumulates across entries so the caller can report
        one summary line.

        Rows missing any DockQ sub-score are dropped for the same reason: the
        composite head is supervised on ``fnat``/``irmsd``/``lrmsd`` per decoy, and a
        row without them cannot contribute to that term.  These are counted and
        reported rather than silently skipped -- a drop that correlates with decoy
        source would otherwise read as full coverage.  On the current caches this
        costs 108 of 108,041 train decoys (0.1%, 6 entries) and 0 val decoys.
        """
        rows: list[dict[str, Any]] = []
        for line in scores.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if _is_bad(row.get("dockq")):
                stats.n_dropped_nan += 1
                continue
            if any(_is_bad(row.get(key)) for key in COMPONENT_KEYS):
                stats.n_dropped_components += 1
                continue
            if not (self.cache_dir / "decoys" / entry / f"{row['decoy_id']}.npz").is_file():
                stats.n_missing += 1
                continue
            rows.append(row)
        return rows

    def _batch_size_for(self, entry_pt: Path, budgets: list[int]) -> int:
        """Return the decoy cap for one entry, recording it in ``budgets`` if computed.

        Chunking oversized entries is what keeps activation memory bounded.  With
        token-aware batching the cap comes from this entry's own token count, read
        from the cache metadata via mmap so the multi-GiB ``z`` is never touched
        (~2.3 ms/entry).
        """
        if not self.token_aware_batching:
            return self.max_decoys_per_batch
        per_batch = decoy_budget(
            self._entry_tokens(entry_pt),
            limit_gib=self.memory_limit_gib,
            max_decoys=self.max_decoys_per_batch,
        )
        budgets.append(per_batch)
        return per_batch

    @staticmethod
    def _entry_tokens(entry_pt: Path) -> int:
        """Read an entry's token count without materialising its tensors.

        ``mmap=True`` keeps this to a metadata read; loading the payload properly
        would pull ``z`` (up to 92 MiB fp16) per entry just to learn its shape.
        Falls back to the ``s`` shape for caches written before ``n_tokens`` was
        recorded.  If neither is readable it returns ``_UNKNOWN_TOKENS``, which is
        deliberately large: an unknown entry then gets the *smallest* batch rather
        than the largest, so a bad read degrades to slow instead of to OOM.
        """
        try:
            # weights_only=True: these caches hold only tensors and plain containers.
            payload = torch.load(entry_pt, map_location="cpu", weights_only=True, mmap=True)
        except (OSError, RuntimeError, ValueError):
            return _UNKNOWN_TOKENS
        n_tokens = payload.get("metadata", {}).get("n_tokens")
        if n_tokens:
            return int(n_tokens)
        s = payload.get("s")
        if s is not None and s.ndim >= 2:
            return int(s.shape[1])
        return _UNKNOWN_TOKENS

    def set_epoch(self, epoch: int) -> None:
        """Reseed the decoy permutation so each epoch groups decoys differently.

        Must be called before the epoch's first ``__getitem__``, and with the same
        value on every rank -- the permutation is per (seed, epoch, entry), so a rank
        that forgot to advance would train on a different partition of the same
        decoys.  Coverage does not depend on it: any epoch value still serves every
        decoy exactly once.
        """
        self.epoch = int(epoch)

    @property
    def entries(self) -> list[str]:
        """Return the entry ids retained after filtering."""
        return sorted(self._records)

    @property
    def records(self) -> dict[str, list[dict[str, Any]]]:
        """Return the retained score rows, keyed by entry.

        Exposed for the training-footprint histogram
        (:func:`~.metric_suite.train_std_bin_histogram`), which needs each native's
        true DockQ spread.  It is the post-filter view -- NaN labels and rows without
        sub-scores are already gone -- so the footprint describes what is actually
        trained on rather than what the cache contains.
        """
        return self._records

    @property
    def n_decoys(self) -> int:
        """Return how many decoys survived filtering, across all retained entries."""
        return sum(len(rows) for rows in self._records.values())

    @property
    def item_sizes(self) -> list[int]:
        """Return the decoy count of each item, in item order.

        This is the *planned* size -- what one optimiser step is offered before the
        OOM retry gets a say -- and it is fixed for the run: ``shuffle_decoys``
        repartitions which decoys share an item, never how many.  Items of one decoy
        are worth counting separately, since :class:`DockQLoss` skips its ranking term
        there, so they train absolute quality only.  Balanced chunking means the only
        way to get one is an entry that has a single decoy.
        """
        return [len(idx) for _, idx in self._items]

    def __len__(self) -> int:
        """Return the number of (entry, decoy-chunk) items."""
        return len(self._items)

    def __getitem__(self, index: int) -> EntryBatch:
        """Load one entry's trunk cache and the coordinates of its decoy chunk."""
        entry, idx = self._items[index]
        rows = self._records[entry]

        if self.shuffle_decoys:
            # Repartition, don't resample: permute the entry's rows once per epoch and
            # let each of its items take its own fixed slice of that permutation.  The
            # slices are disjoint and cover every row, so an epoch shows each decoy
            # exactly once while still varying which decoys are ranked against each
            # other.  Sampling each item independently -- what this used to do -- left
            # ~27% of the decoys unvisited per epoch and spent the freed slots on
            # duplicates, since two items of one entry could draw the same row.
            rng = np.random.default_rng((self.seed, self.epoch, self._entry_ids[entry]))
            permutation = rng.permutation(len(rows))
            picked = [rows[permutation[i]] for i in idx]
        else:
            picked = [rows[i] for i in idx]

        cache = load_entry_cache(self.cache_dir / "entries" / f"{entry}.pt", dtype=self.store_dtype)
        feats = feats_for_confidence(cache.feats)

        coords = []
        for row in picked:
            npz = np.load(self.cache_dir / "decoys" / entry / f"{row['decoy_id']}.npz")
            coords.append(build_x_pred(npz["coords"], feats))
        x_pred = torch.cat(coords, dim=0)  # (n_decoys, 1, n_padded, 3)

        baseline = {
            key: torch.tensor([float(r[key]) for r in picked], dtype=torch.float32)
            for key in CONFIDENCE_SCALARS
            if all(key in r and not _is_bad(r.get(key)) for r in picked)
        }

        return EntryBatch(
            entry=entry,
            s_inputs=cache.s_inputs,
            s=cache.s,
            z=cache.z,
            feats=feats,
            x_pred=x_pred,
            dockq=torch.tensor([float(r["dockq"]) for r in picked], dtype=torch.float32),
            fnat=torch.tensor([float(r["fnat"]) for r in picked], dtype=torch.float32),
            irmsd=torch.tensor([float(r["irmsd"]) for r in picked], dtype=torch.float32),
            lrmsd=torch.tensor([float(r["lrmsd"]) for r in picked], dtype=torch.float32),
            baseline=baseline,
            decoy_ids=[r["decoy_id"] for r in picked],
            sources=[r["source"] for r in picked],
        )


def collate_entry_batches(batch: list[EntryBatch]) -> EntryBatch:
    """Pass a single :class:`EntryBatch` through unchanged.

    Entries have different token counts, so two entries cannot share a padded
    tensor without re-padding ``z`` to the larger of the two -- which would waste
    exactly the memory this design is trying to save.  The dataset item is already
    a batch of decoys, so ``batch_size=1`` at the DataLoader level is the intended
    configuration and this collate just unwraps it.
    """
    if len(batch) != 1:
        msg = f"expected batch_size=1 (each item is already an entry batch of decoys), got {len(batch)}"
        raise ValueError(msg)
    return batch[0]


def iter_entry_batches(
    dataset: EntryGroupedCacheDataset, *, shuffle: bool = False, seed: int = 0
) -> Iterator[EntryBatch]:
    """Iterate a dataset in-process (no worker pool), for quick checks."""
    order = list(range(len(dataset)))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for i in order:
        yield dataset[i]
