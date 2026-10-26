r"""Fine-tune a DockQ head on the cached Boltz-2 confidence-head inputs.

What this does
--------------
Adds a small MLP head that predicts DockQ from the confidence module's own
``s_conf``/``z_conf`` latents, trains it on the cache built by
``cache_boltz2_confidence.py``, and reports how well it correlates with true DockQ
**against every existing Boltz confidence metric** on a held-out split.

The existing metrics are the bar to clear.  Measured on ``sabdab_val_easy``,
Boltz's ``iptm`` gets a global Spearman of 0.599 versus DockQ but a per-entry mean
of only **0.038** -- it tells easy targets from hard ones yet cannot rank decoys of
the same native.  The per-entry column of the report is therefore the one that
matters; ``top1_mean_dockq`` translates it into "how good is the decoy this metric
would actually pick", against an oracle upper bound.

The trunk stays frozen.  Only the new head trains unless
``--unfreeze_confidence`` is given, in which case the 25M-parameter confidence
module trains too at a much lower LR.

Split
-----
Held out from the cached ``sabdab_train`` entries, grouped by **PDB id** so
different chain selections of one structure (``1a2y_BAC`` / ``1a2y_ABC``) cannot
straddle the split.  ``sabdab_val_easy`` is the eventual external holdout: it has
no cache yet (its ``processed`` dirs were cleaned), and its entries share zero
PDB ids with the training cache.  Point ``--eval_cache_dir`` at it once cached.

Usage::

    poetry run python workflows/scripts/training/finetune_dockq_head.py \
        --cache_dir  /data/confidock/sabdab_train/confidence_cache \
        --output_dir runs/dockq_head_v1 \
        --epochs 3

Quick smoke test (a handful of entries, one epoch)::

    ... --limit_train_batches 20 --limit_val_batches 10 --epochs 1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from confidock.boltz.finetune.cache_dataset import (
    EntryGroupedCacheDataset,
    collate_entry_batches,
    split_entries,
)
from confidock.boltz.finetune.dockq_head import DockQHead, DockQLoss
from confidock.boltz.finetune.trainer import (
    FinetuneConfig,
    evaluate,
    prepare_confidence_module,
    train_one_epoch,
)
from confidock.boltz.inference.cache_runner import CacheRunConfig, load_cache_model
from confidock.paths import default_boltz_cache

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE = Path("/data/confidock/sabdab_train/confidence_cache")
DEFAULT_BOLTZ_CACHE = default_boltz_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--boltz_cache", type=Path, default=DEFAULT_BOLTZ_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--eval_cache_dir",
        type=Path,
        default=None,
        help="score this cache instead of the held-out split (e.g. a val_easy cache)",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--confidence_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Composite-loss weights.  Defaults are the SimpleGNN ``comp_sup`` ablation arm
    # (configs/dockq_predictor/ablations/loss_objective.yaml), which won that sweep;
    # keeping the numbers identical is what makes the two models comparable.
    parser.add_argument(
        "--lambda_component",
        type=float,
        default=1.0,
        help="weight on supervising the three DockQ sub-scores individually",
    )
    parser.add_argument(
        "--lambda_rank",
        type=float,
        default=0.5,
        help="weight on the within-entry soft-Spearman ranking term (0 disables it)",
    )
    parser.add_argument("--huber_delta", type=float, default=1.0, help="delta for every Huber term")
    parser.add_argument(
        "--rank_tau",
        type=float,
        default=0.05,
        help="soft-Spearman temperature; smaller is closer to a true rank, larger is smoother",
    )

    parser.add_argument("--max_decoys_per_batch", type=int, default=34)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument(
        "--forward_chunk",
        type=int,
        default=4,
        help="decoys per confidence-module forward; the decoy axis is what bounds "
        "GPU memory (z_conf is (chunk, L, L, 128))",
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--unfreeze_confidence", action="store_true")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--limit_val_batches", type=int, default=None)
    parser.add_argument("--max_entries", type=int, default=None)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> FinetuneConfig:
    """Translate CLI args into a :class:`FinetuneConfig`."""
    return FinetuneConfig(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        boltz_cache=args.boltz_cache,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        confidence_lr=args.confidence_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        accumulate=args.accumulate,
        hidden=args.hidden,
        dropout=args.dropout,
        lambda_component=args.lambda_component,
        lambda_rank=args.lambda_rank,
        huber_delta=args.huber_delta,
        rank_tau=args.rank_tau,
        max_decoys_per_batch=args.max_decoys_per_batch,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        max_tokens=args.max_tokens,
        forward_chunk=args.forward_chunk,
        unfreeze_confidence=args.unfreeze_confidence,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        log_every=args.log_every,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )


def make_loader(dataset: EntryGroupedCacheDataset, config: FinetuneConfig, *, shuffle: bool) -> DataLoader:
    """Wrap a dataset in a DataLoader.

    ``batch_size=1`` is deliberate: each item is already a batch of decoys sharing
    one entry's ``z``.  Workers matter a lot -- the ~46 MiB ``z`` read per item is
    the bottleneck, not the GPU.
    """
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_entry_batches,
        pin_memory=False,  # batches hold O(L^2) tensors; pinning them is costly
        persistent_workers=config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )


def write_report(path: Path, report: dict, label: str) -> None:
    """Write a correlation report as JSON and log the headline comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))

    per = report.get("per_entry", {})
    glob = report.get("global", {})
    top1 = report.get("top1_mean_dockq", {})
    LOGGER.info("%s: %d decoys / %d entries", label, report.get("n_decoys"), report.get("n_entries"))
    LOGGER.info("  %-16s %8s %10s %10s", "metric", "global", "per_entry", "top1_dockq")
    # Best correlation first, NaN last: a constant metric (e.g. a baseline that does
    # not vary within an entry) correlates to NaN and must not sort above real scores.
    for metric in sorted(per, key=lambda m: 9.0 if math.isnan(per[m]) else -per[m]):
        LOGGER.info(
            "  %-16s %8.4f %10.4f %10.4f",
            metric,
            glob.get(metric, float("nan")),
            per.get(metric, float("nan")),
            top1.get(metric, float("nan")),
        )
    LOGGER.info("  %-16s %8s %10s %10.4f", "ORACLE", "-", "-", report.get("oracle_mean_dockq", float("nan")))


def write_predictions(path: Path, rows: list[dict]) -> None:
    """Write per-decoy predictions so the analysis can be redone without the GPU."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Train the DockQ head and report validation correlations."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    config = build_config(args)
    torch.manual_seed(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(config.to_json())

    # ---- split -----------------------------------------------------------
    all_entries = sorted(p.stem for p in (config.cache_dir / "entries").glob("*.pt"))
    if args.max_entries:
        all_entries = all_entries[: args.max_entries]
    if not all_entries:
        LOGGER.error("no entry caches found under %s", config.cache_dir / "entries")
        return 1

    train_entries, val_entries = split_entries(all_entries, val_fraction=config.val_fraction, seed=config.split_seed)
    LOGGER.info(
        "split: %d train / %d val entries (grouped by pdb id, from %d cached)",
        len(train_entries),
        len(val_entries),
        len(all_entries),
    )
    (config.output_dir / "split.json").write_text(json.dumps({"train": train_entries, "val": val_entries}, indent=2))

    train_ds = EntryGroupedCacheDataset(
        config.cache_dir,
        train_entries,
        max_decoys_per_batch=config.max_decoys_per_batch,
        shuffle_decoys=True,
        seed=config.seed,
    )
    val_cache = args.eval_cache_dir or config.cache_dir
    val_ds = EntryGroupedCacheDataset(
        val_cache,
        val_entries if args.eval_cache_dir is None else sorted(p.stem for p in (val_cache / "entries").glob("*.pt")),
        max_decoys_per_batch=config.max_decoys_per_batch,
        shuffle_decoys=False,
        require_label_spread=False,
        seed=config.seed,
    )

    # ---- model -----------------------------------------------------------
    LOGGER.info("loading Boltz-2 (this takes several minutes on a busy box)")
    t0 = time.time()
    model = load_cache_model(
        CacheRunConfig(
            processed_dir=config.cache_dir,
            boltz_cache=config.boltz_cache,
            output_dir=config.output_dir,
            checkpoint=config.checkpoint,
            device=config.device,
        )
    )
    prepare_confidence_module(model, unfreeze=config.unfreeze_confidence)
    LOGGER.info("model ready in %.1fs", time.time() - t0)

    token_s = model.confidence_module.s_to_z.in_features
    token_z = model.confidence_module.s_to_z.out_features
    dockq_head = DockQHead(token_s, token_z, hidden=config.hidden, dropout=config.dropout).to(config.device)
    n_head = sum(p.numel() for p in dockq_head.parameters())
    LOGGER.info(
        "DockQHead: token_s=%d token_z=%d in_dim=%d params=%.2fM",
        token_s,
        token_z,
        dockq_head.in_dim,
        n_head / 1e6,
    )

    groups: list[dict] = [{"params": list(dockq_head.parameters()), "lr": config.lr}]
    if config.unfreeze_confidence:
        groups.append(
            {
                "params": [p for p in model.confidence_module.parameters() if p.requires_grad],
                "lr": config.confidence_lr,
            }
        )
    # Every group sets its own `lr`, so AdamW's default is never used; passing
    # config.lr as well keeps the effective head LR visible at the call site.
    optimizer = torch.optim.AdamW(groups, lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = DockQLoss(
        lambda_component=config.lambda_component,
        lambda_rank=config.lambda_rank,
        huber_delta=config.huber_delta,
        rank_tau=config.rank_tau,
    )

    # ---- baseline (untrained head) --------------------------------------
    val_loader = make_loader(val_ds, config, shuffle=False)
    report, rows = evaluate(model, dockq_head, val_loader, config)
    write_report(config.output_dir / "report_epoch0.json", report, "epoch 0 (untrained head)")

    # ---- train -----------------------------------------------------------
    history = []
    best = -float("inf")
    for epoch in range(1, config.epochs + 1):
        train_ds.set_epoch(epoch)
        train_loader = make_loader(train_ds, config, shuffle=True)
        stats = train_one_epoch(model, dockq_head, loss_fn, optimizer, train_loader, config, epoch=epoch)
        LOGGER.info(
            "epoch %d train: loss %.4f (dockq %.4f comp %.4f rank %.4f) %d steps, %d skipped, %.1fs",
            epoch,
            stats["loss"],
            stats["huber_dockq"],
            stats["huber_comp"],
            stats["rank"],
            int(stats["n_steps"]),
            int(stats["n_skipped"]),
            stats["seconds"],
        )

        report, rows = evaluate(model, dockq_head, val_loader, config)
        write_report(config.output_dir / f"report_epoch{epoch}.json", report, f"epoch {epoch}")
        write_predictions(config.output_dir / f"predictions_epoch{epoch}.csv", rows)
        history.append({"epoch": epoch, "train": stats, "val": report})

        score = report["per_entry"].get("dockq_pred", float("nan"))
        # A NaN score must never win: it means the report had no rankable entries,
        # not a good epoch.
        if not math.isnan(score) and score > best:
            best = score
            torch.save(
                {
                    "dockq_head": dockq_head.state_dict(),
                    "epoch": epoch,
                    "per_entry_spearman": score,
                    "token_s": token_s,
                    "token_z": token_z,
                    "config": json.loads(config.to_json()),
                },
                config.output_dir / "dockq_head_best.pt",
            )
            LOGGER.info("new best per-entry Spearman %.4f (epoch %d)", score, epoch)

    (config.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    LOGGER.info("done; best per-entry Spearman %.4f", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
