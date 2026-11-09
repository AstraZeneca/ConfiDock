"""Cache the inputs of Boltz-2's confidence head for offline fine-tuning.

Design
------
The confidence head is called as::

    confidence_module(
        s_inputs=..., s=..., z=..., x_pred=..., feats=...,
        pred_distogram_logits=..., multiplicity=..., ...
    )

Of those, ``s_inputs``/``s``/``z`` come from the trunk and ``pred_distogram_logits``
is one linear layer on ``z``.  The trunk **never sees coordinates**: in
``boltz/model/modules/trunkv2.py`` the only coordinate references are
``feats["template_cb"]``/``feats["template_ca"]`` inside the template modules,
which we do not use.  So for a given entry, ``s``/``z`` are a function of
sequence + MSA alone and are *identical* for every decoy of that entry.

That gives the storage and compute layout:

* **per entry** (3,260 of them) -- run the trunk once, store ``s_inputs``, ``s``,
  ``z`` plus the 18 ``feats`` tensors the confidence head and loss read.
* **per decoy** (108,041 of them) -- store only ``x_pred``, ~33 KiB.

Total: 149 GiB of entry caches plus 3.9 GiB of coordinates.

Verified on ``1a14_HLN``: running the trunk once and then invoking the
confidence head per decoy reproduces the full ``model.forward`` per decoy
**bit-exactly** (max |Δ| = 0.0 on ptm/iptm/plddt/pde) once ``torch.manual_seed``
is fixed -- the seed matters because ``trunkv2.py:632`` subsamples the MSA with
``torch.randperm``.  Timing on an A100 for a 600-token entry: trunk 38.4 s,
confidence head 1.25 s/decoy, i.e. ~80 s for a 33-decoy entry versus ~1322 s if
the trunk were re-run each time.

Storage notes
-------------
``pred_distogram_logits`` (71 GiB across the dataset) is **not** stored: it is
recomputed at training time as ``distogram_module(z)[:, :, :, 0]``, one matmul.

``atom_to_token``, ``token_to_rep_atom`` and ``r_set_to_rep_atom`` are one-hot
``int64`` matrices of 21.5 MiB *each* (220 GiB total -- more than ``z``).  Each
row sums to 1, except for the padded-atom rows of ``atom_to_token``, which are
all-zero, so they are stored as ``int16``/``int32`` index vectors (with -1 for an
all-zero row) and rebuilt on load.  Measured: 84 MiB of ``feats`` per 600-token
entry compacts to 4.9 MiB, verified lossless.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CONFIDENCE_FEAT_KEYS",
    "EntryCache",
    "compact_feats",
    "expand_feats",
    "load_entry_cache",
    "save_entry_cache",
]


#: ``feats`` keys the confidence head reads.
#:
#: Determined empirically, not by grepping: the head's submodules (``rel_pos``,
#: ``contact_conditioning``, the atom encoder) read keys that never appear as a
#: literal ``feats["..."]`` in ``confidencev2.py``, so a grep-derived list is
#: short by seven keys and fails at load time with ``KeyError: 'residue_index'``.
#: Reproduce with a dict subclass that records ``__getitem__`` during a real
#: ``confidence_module`` call (see ``docs/boltz_confidence_caching.md``).
CONFIDENCE_FEAT_KEYS: tuple[str, ...] = (
    "asym_id",
    "atom_pad_mask",
    "atom_to_token",
    "contact_conditioning",
    "contact_threshold",
    "cyclic_period",
    "entity_id",
    "frames_idx",
    "mol_type",
    "residue_index",
    "sym_id",
    "token_bonds",
    "token_index",
    "token_pad_mask",
    "token_to_rep_atom",
    "type_bonds",
    # Not read by the head itself, but by the confidence loss helpers
    # (``compute_ptms`` -> ``compute_frame_pred``) at fine-tuning time.
    "frame_resolved_mask",
    "r_set_to_rep_atom",
)

#: One-hot ``(…, N, M)`` matrices stored as argmax index vectors.  Row sums are
#: 1 for real rows and 0 for padded ones (asserted on save), so the mapping is
#: lossless if empty rows are marked.
_ONE_HOT_KEYS: tuple[str, ...] = (
    "atom_to_token",
    "token_to_rep_atom",
    "r_set_to_rep_atom",
    "contact_conditioning",
)

#: Index recorded for an all-zero row of a one-hot matrix.  ``atom_to_token`` has
#: one such row per padded atom (verified: the zero-row set is exactly
#: ``atom_pad_mask == 0``), so these must round-trip back to zeros rather than to
#: a spurious 1 in column 0.
_EMPTY_ROW = -1


@dataclass
class EntryCache:
    """Trunk outputs and static features for a single entry.

    Attributes
    ----------
    entry
        SAbDab entry id, e.g. ``1a14_HLN``.
    s_inputs, s
        ``(1, n_tokens, token_s)`` input and trunk token representations.
    z
        ``(1, n_tokens, n_tokens, token_z)`` trunk pair representation.
    feats
        The :data:`CONFIDENCE_FEAT_KEYS` subset of the featurizer output.
    metadata
        Provenance: recycling steps, seed, token/atom counts, boltz version.
    """

    entry: str
    s_inputs: torch.Tensor
    s: torch.Tensor
    z: torch.Tensor
    feats: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


def _compact_one_hot(key: str, value: torch.Tensor, out: dict[str, torch.Tensor]) -> None:
    """Store a one-hot matrix as an index vector plus its width.

    Raises if any row sums to something other than 0 or 1: an argmax of an
    ambiguous row would silently corrupt the atom/token mapping.
    """
    row_sums = value.sum(dim=-1)
    bad_mask = (row_sums != 0) & (row_sums != 1)
    if bool(bad_mask.any()):
        bad = int(bad_mask.sum())
        msg = (
            f"{key} is not one-hot: {bad} of {row_sums.numel()} rows sum "
            f"to neither 0 nor 1, cannot store as index vector"
        )
        raise ValueError(msg)
    width = value.shape[-1]
    # _EMPTY_ROW is -1, so int16 must hold width-1 as well as -1.
    dtype = torch.int16 if width < 2**15 else torch.int32
    indices = value.argmax(dim=-1).to(dtype)
    out[key] = torch.where(
        row_sums == 0,
        torch.tensor(_EMPTY_ROW, dtype=dtype, device=indices.device),
        indices,
    )
    out[f"{key}__width"] = torch.tensor(width, dtype=torch.int32)


def _compact_int64(key: str, value: torch.Tensor, out: dict[str, torch.Tensor]) -> None:
    """Narrow an int64 feature to int32 when its values provably fit.

    Small counters and ids (token/atom indices, asym/entity ids, bond types).  A
    future wider feature is stored as-is rather than silently wrapped, and the
    narrowing is flagged so the load path can restore int64.
    """
    fits = value.numel() == 0 or int(value.abs().max()) < 2**31
    out[key] = value.to(torch.int32) if fits else value
    if fits:
        out[f"{key}__int64"] = torch.tensor(True)


def compact_feats(feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Shrink the cached ``feats`` dict before writing it to disk.

    One-hot matrices become index vectors; boolean masks become ``bool``.
    Rows summing to 0 (padded atoms) are stored as :data:`_EMPTY_ROW` and rebuilt
    as all-zero.  Raises if any row sums to something other than 0 or 1, since
    silently storing an argmax of an ambiguous row would corrupt the atom/token
    mapping.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in feats.items():
        if key in _ONE_HOT_KEYS:
            _compact_one_hot(key, value, out)
        elif value.dtype == torch.int64:
            _compact_int64(key, value, out)
        else:
            out[key] = value
    return out


def expand_feats(compact: dict[str, torch.Tensor], *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Invert :func:`compact_feats`, restoring the dtypes the model expects."""
    out: dict[str, torch.Tensor] = {}
    for key, value in compact.items():
        if key.endswith("__width") or key.endswith("__int64"):
            continue
        value = value.to(device)
        if key in _ONE_HOT_KEYS:
            width = int(compact[f"{key}__width"])
            indices = value.long()
            one_hot = torch.zeros((*value.shape, width), dtype=torch.int64, device=value.device)
            # Empty rows scatter a 0 into column 0, leaving the row all-zero.
            filled = (indices != _EMPTY_ROW).long()
            one_hot.scatter_(-1, indices.clamp_min(0).unsqueeze(-1), filled.unsqueeze(-1))
            out[key] = one_hot
        elif f"{key}__int64" in compact:
            out[key] = value.to(torch.int64)
        else:
            out[key] = value
    return out


def save_entry_cache(cache: EntryCache, path: Path, *, dtype: torch.dtype = torch.float16) -> None:
    """Write one entry's trunk cache to ``path`` (a ``.pt`` file).

    ``dtype`` controls the float precision of ``s_inputs``/``s``/``z``, which
    dominate the cache (149 GiB at fp16, 298 GiB at fp32 across the dataset).
    Measured cost of fp16: re-scoring 56 decoys from cache matched the run-time
    scores to |Δiptm| <= 3e-5 and |Δcomplex_ipde| <= 3.5e-4.  Note that decoy
    *coordinates* are a different story and are always stored fp32 -- see
    :func:`save_decoy_coords`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entry": cache.entry,
        "s_inputs": cache.s_inputs.to(dtype).cpu(),
        "s": cache.s.to(dtype).cpu(),
        "z": cache.z.to(dtype).cpu(),
        "feats": {k: v.cpu() for k, v in compact_feats(cache.feats).items()},
        "metadata": cache.metadata,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a killed job never leaves a half-written cache


def load_entry_cache(
    path: Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EntryCache:
    """Read an entry cache written by :func:`save_entry_cache`."""
    # weights_only=True: the payload is only tensors, strings and plain containers,
    # so it never needs arbitrary unpickling.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return EntryCache(
        entry=payload["entry"],
        s_inputs=payload["s_inputs"].to(device=device, dtype=dtype),
        s=payload["s"].to(device=device, dtype=dtype),
        z=payload["z"].to(device=device, dtype=dtype),
        feats=expand_feats(payload["feats"], device=device),
        metadata=payload.get("metadata", {}),
    )


def save_decoy_coords(
    path: Path,
    coords: np.ndarray,
    present: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write a single decoy's ``x_pred`` (and presence mask) to ``path``.

    Coordinates are stored as **float32**, not fp16.  fp16 has ~11 mantissa bits,
    so at the ~100 Å magnitudes these coordinates reach it quantises to ~0.016 Å,
    which shifts the head's outputs measurably (measured on ``1a2y_BAC``:
    |Δiptm| up to 0.005, |Δcomplex_ipde| up to 0.058) -- comparable to the signal
    being learned.  float32 costs ~36 KiB per decoy, so ~3.8 GiB for all ~108k,
    negligible beside the ~147 GiB of entry caches.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed appends ".npz" unless the name already ends in it, so
    # the temp name must keep that suffix last or the file lands elsewhere.
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        coords=coords.astype(np.float32),
        present=present,
        metadata=json.dumps(metadata),
    )
    tmp.replace(path)
