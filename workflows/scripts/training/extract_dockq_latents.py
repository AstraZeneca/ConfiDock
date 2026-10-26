r"""Extract pooled confidence latents for every cached decoy, sharded over GPUs.

Why extract instead of training end-to-end
------------------------------------------
With the confidence module frozen and in ``eval()`` the pooled latents are a
deterministic function of ``(entry, decoy)``, so training the head against the
live module recomputes the same numbers every epoch -- ~13.5 h per epoch over the
97k training decoys on one GPU.  Extracting once across 6 GPUs takes a fraction of
that, and every subsequent epoch, hyperparameter setting, or loss ablation then
runs in seconds against a 264 MiB table.

This does **not** apply under ``--unfreeze_confidence``: there the latents move
with the weights and must be recomputed, which is why
``finetune_dockq_head.py`` retains its live-module path.

Usage (one shard per GPU, via the wrapper)::

    workflows/scripts/training/extract_dockq_latents_all_gpus.sh \
        /data/confidock/sabdab_train/dockq_latents

Single shard::

    CUDA_VISIBLE_DEVICES=0 poetry run python \
        workflows/scripts/training/extract_dockq_latents.py \
        --latent_dir /path/to/dockq_latents --shard_index 0 --num_shards 6

Resumable: an entry whose ``latents/<entry>.npz`` exists is skipped unless
``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import torch

from confidock.boltz.finetune.cache_dataset import EntryGroupedCacheDataset
from confidock.boltz.finetune.dockq_head import pool_confidence_latents
from confidock.boltz.finetune.latent_cache import (
    COMPONENT_KEYS,
    POOLING_VERSION,
    save_latent_shard,
)
from confidock.boltz.finetune.trainer import prepare_confidence_module
from confidock.boltz.inference.cache_runner import CacheRunConfig, load_cache_model
from confidock.paths import default_boltz_cache

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE = Path("/data/confidock/sabdab_train/confidence_cache")
DEFAULT_BOLTZ_CACHE = default_boltz_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--latent_dir", type=Path, required=True)
    parser.add_argument("--boltz_cache", type=Path, default=DEFAULT_BOLTZ_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=None)

    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--forward_chunk",
        type=int,
        default=4,
        help="decoys per confidence forward; z_conf is (chunk, L, L, 128), so this is what bounds GPU memory",
    )
    parser.add_argument("--max_entries", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args(argv)


def select_shard(entries: list[str], shard_index: int, num_shards: int) -> list[str]:
    """Round-robin the entries so each shard gets a comparable size mix.

    Round-robin rather than contiguous blocks: entries are sorted by id and
    correlate with size in places, so contiguous blocks would leave one GPU with
    the large end of the distribution.
    """
    if num_shards <= 1:
        return entries
    return [e for i, e in enumerate(entries) if i % num_shards == shard_index]


@torch.no_grad()
def extract_entry(
    model: object,
    dataset: EntryGroupedCacheDataset,
    index: int,
    *,
    chunk: int,
    device: str,
) -> tuple[str, torch.Tensor, object]:
    """Pool every decoy of one dataset item; returns ``(entry, pooled, batch)``."""
    batch = dataset[index].to(device)
    n = len(batch)

    pred_distogram_logits = model.distogram_module(batch.z)[:, :, :, 0]

    parts: list[torch.Tensor] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        out = model.confidence_module(
            s_inputs=batch.s_inputs,
            s=batch.s,
            z=batch.z,
            x_pred=batch.x_pred[start:stop],
            feats=batch.feats,
            pred_distogram_logits=pred_distogram_logits,
            multiplicity=stop - start,
            run_sequentially=False,
            use_kernels=False,
        )
        parts.append(pool_confidence_latents(out["s_conf"], out["z_conf"], batch.feats).cpu())
        del out

    return batch.entry, torch.cat(parts, dim=0), batch


def _append_jsonl(path: Path, payload: dict) -> None:
    """Append one JSON line; used for both the manifest and the failure log."""
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _resolve_entries(args: argparse.Namespace, out_dir: Path) -> list[str]:
    """Pick this shard's entries, dropping the ones already extracted."""
    all_entries = sorted(p.stem for p in (args.cache_dir / "entries").glob("*.pt"))
    if args.max_entries:
        all_entries = all_entries[: args.max_entries]
    entries = select_shard(all_entries, args.shard_index, args.num_shards)

    if args.overwrite:
        return entries

    todo = [e for e in entries if not (out_dir / f"{e}.npz").is_file()]
    LOGGER.info(
        "shard has %d of %d entries; %d already done, %d to do",
        len(entries),
        len(all_entries),
        len(entries) - len(todo),
        len(todo),
    )
    return todo


def _extract_with_retry(
    model: object,
    dataset: EntryGroupedCacheDataset,
    index: int,
    args: argparse.Namespace,
    failures: Path,
) -> tuple[str, torch.Tensor, object] | None:
    """Extract one item, retrying an OOM at chunk=1; ``None`` if it failed for good.

    An OOM here is a function of this entry's size rather than a persistent fault,
    so halving throughput for one entry beats losing it.  Anything else is recorded
    and skipped -- a shard is hours of work and must not die on one bad entry.
    """
    try:
        return extract_entry(model, dataset, index, chunk=args.forward_chunk, device=args.device)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        LOGGER.warning("OOM at chunk=%d on item %d; retrying at 1", args.forward_chunk, index)
        try:
            return extract_entry(model, dataset, index, chunk=1, device=args.device)
        except Exception:  # noqa: BLE001 - recorded and skipped
            _append_jsonl(failures, {"item": index, "error": traceback.format_exc(limit=3)})
            torch.cuda.empty_cache()
            return None
    except Exception:  # noqa: BLE001 - recorded and skipped
        _append_jsonl(failures, {"item": index, "error": traceback.format_exc(limit=3)})
        return None


def _write_entry(
    out_dir: Path,
    manifest: Path,
    entry: str,
    pooled: torch.Tensor,
    batch: object,
    *,
    seconds: float,
) -> None:
    """Save one entry's latent shard and record it in the shard manifest."""
    n_tokens = int(batch.s.shape[1])
    save_latent_shard(
        out_dir / f"{entry}.npz",
        entry=entry,
        pooled=pooled,
        dockq=[float(v) for v in batch.dockq.cpu()],
        components={key: [float(v) for v in getattr(batch, key).cpu()] for key in COMPONENT_KEYS},
        decoy_ids=batch.decoy_ids,
        sources=batch.sources,
        baseline={k: [float(x) for x in v.cpu()] for k, v in batch.baseline.items()},
        metadata={
            "n_tokens": n_tokens,
            "n_decoys": len(batch),
            "in_dim": int(pooled.shape[1]),
        },
    )
    _append_jsonl(
        manifest,
        {
            "entry": entry,
            "n_decoys": len(batch),
            "n_tokens": n_tokens,
            "seconds": round(seconds, 2),
            "pooling_version": POOLING_VERSION,
        },
    )


def _log_progress(n_done: int, n_failed: int, *, index: int, n_items: int, elapsed: float) -> None:
    """Log throughput and an ETA over the entries completed so far."""
    rate = n_done / max(elapsed, 1e-9)
    remaining = (n_items - index - 1) / max(rate, 1e-9)
    LOGGER.info(
        "%d/%d entries (%.2f entries/s, ~%.0f min left, %d failed)",
        n_done,
        n_items,
        rate,
        remaining / 60,
        n_failed,
    )


def main(argv: list[str] | None = None) -> int:
    """Extract pooled latents for this shard's entries."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format=f"%(asctime)s shard{args.shard_index} %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    out_dir = args.latent_dir / "latents"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = _resolve_entries(args, out_dir)
    if not entries:
        LOGGER.info("nothing to do")
        return 0

    # ``max_decoys_per_batch`` is deliberately huge and ``require_label_spread``
    # off: extraction should emit one file per entry covering every labelled decoy,
    # and any filtering belongs to the training step, which can do it for free.
    dataset = EntryGroupedCacheDataset(
        args.cache_dir,
        entries,
        max_decoys_per_batch=10_000,
        shuffle_decoys=False,
        require_label_spread=False,
    )
    if len(dataset) == 0:
        LOGGER.warning("no usable entries in this shard")
        return 0

    LOGGER.info("loading Boltz-2")
    t0 = time.time()
    model = load_cache_model(
        CacheRunConfig(
            processed_dir=args.cache_dir,
            boltz_cache=args.boltz_cache,
            output_dir=args.latent_dir,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    )
    prepare_confidence_module(model, unfreeze=False)
    LOGGER.info("model ready in %.1fs", time.time() - t0)

    manifest = args.latent_dir / f"manifest.shard{args.shard_index}.jsonl"
    failures = args.latent_dir / f"failures.shard{args.shard_index}.jsonl"
    n_done = 0
    n_failed = 0
    t_start = time.time()

    for index in range(len(dataset)):
        t_entry = time.time()
        result = _extract_with_retry(model, dataset, index, args, failures)
        if result is None:
            n_failed += 1
            continue

        entry, pooled, batch = result
        _write_entry(out_dir, manifest, entry, pooled, batch, seconds=time.time() - t_entry)
        n_done += 1
        del batch, pooled, result

        if args.log_every and n_done % args.log_every == 0:
            _log_progress(
                n_done,
                n_failed,
                index=index,
                n_items=len(dataset),
                elapsed=time.time() - t_start,
            )

    LOGGER.info(
        "done: %d extracted, %d failed, %.1f min",
        n_done,
        n_failed,
        (time.time() - t_start) / 60,
    )
    if n_failed:
        # Non-zero on *any* failure, not only on a shard that extracted nothing.  A
        # partly extracted shard is the dangerous outcome: the latent table is keyed
        # by entry, so a missing one is not an error downstream, just a quietly
        # smaller training set.  Exiting non-zero is also what re-tries it -- the
        # wrapper re-runs the shard, and every entry already written is skipped, so
        # the next attempt costs only the entries that actually failed.
        LOGGER.error(
            "%d entries failed; re-run this shard to retry just those (see %s)",
            n_failed,
            failures.name,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
