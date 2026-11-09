r"""Train the DockQ head on pre-extracted latents (frozen confidence module).

This is the fast baseline and the hyperparameter-search tool.  An epoch is a pass
over a 640-wide table, so a full 3-epoch run takes seconds, versus ~13.5 h per
epoch against the live confidence module.

It answers the question the full fine-tune cannot answer on its own: how much of
the gain comes from the *new head* reading the existing latents, and how much
comes from actually updating the 25M-parameter confidence module.  Run this first,
then compare against ``finetune_dockq_head.py --unfreeze_confidence``.

Usage::

    poetry run python workflows/scripts/training/train_dockq_probe.py \
        --latent_dir /data/confidock/sabdab_train/dockq_latents \
        --output_dir runs/dockq_probe_v1 --epochs 40

Sweep the composite loss's weights -- an epoch is seconds, which is the point::

    ... --lambda_rank 0 --output_dir runs/probe_rank0
    ... --lambda_component 0 --output_dir runs/probe_nocomp

Evaluate on a second, fully disjoint latent set (``val_easy``)::

    ... --eval_latent_dir /data/confidock/sabdab_val_easy/dockq_latents

Model selection stays on the internal holdout; the external set is scored once, at
the selected epoch.  Selecting *on* the external set would make it a second
validation set rather than the holdout the comparison needs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

from confidock.boltz.finetune.cache_dataset import split_entries
from confidock.boltz.finetune.dockq_head import DockQHead, DockQLoss
from confidock.boltz.finetune.latent_cache import load_latents
from confidock.boltz.finetune.probe import (
    evaluate_probe,
    standardise,
    train_probe,
)
from confidock.boltz.finetune.tracking import (
    finish_run,
    flatten_report,
    init_run,
    log_metrics,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_LATENTS = Path("/data/confidock/sabdab_train/dockq_latents")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latent_dir", type=Path, default=DEFAULT_LATENTS)
    parser.add_argument(
        "--eval_latent_dir",
        type=Path,
        default=None,
        help="second latent dir to score at the selected epoch, e.g. val_easy.  "
        "Standardised with the *training* statistics, and never used for model "
        "selection -- the internal holdout stays the selector",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_component", type=float, default=1.0)
    parser.add_argument("--lambda_rank", type=float, default=0.5)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--rank_tau", type=float, default=0.05)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no_standardise",
        action="store_true",
        help="skip z-scoring the pooled features (they have very unequal scales)",
    )
    parser.add_argument("--log_level", default="INFO")

    # Same project as the fine-tune: the probe is its baseline, so the two belong on
    # one set of axes.  Use --wandb_group to keep a sweep's runs together.
    parser.add_argument("--wandb_project", default="confidock-boltz_ft")
    parser.add_argument("--wandb_run_name", default=None, help="defaults to the output dir's name")
    parser.add_argument("--wandb_group", default=None, help="group related runs, e.g. a loss sweep")
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--no_wandb", action="store_true", help="disable W&B for this run")
    return parser.parse_args(argv)


def write_predictions(path: Path, rows: list[dict]) -> None:
    """Write per-decoy predictions for offline analysis."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def log_report(report: dict, label: str) -> None:
    """Log the metric comparison table."""
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
    LOGGER.info(
        "  %-16s %8s %10s %10.4f",
        "ORACLE",
        "-",
        "-",
        report.get("oracle_mean_dockq", float("nan")),
    )


def _load_eval_table(args: argparse.Namespace, table) -> tuple[object | None, int]:
    """Load the external eval set, if any; returns ``(table_or_None, exit_code)``.

    An exit code of 1 means the run must abort: either the external set overlaps the
    training entries, or it was extracted with a different pooling width.
    """
    if not args.eval_latent_dir:
        return None, 0

    eval_table = load_latents(args.eval_latent_dir)
    shared = set(eval_table.entries) & set(table.entries)
    if shared:
        # A shared entry would be trained on and then reported as external, so
        # fail loudly rather than quietly inflate the headline number.
        LOGGER.error(
            "%d entries appear in both --latent_dir and --eval_latent_dir (e.g. %s)",
            len(shared),
            sorted(shared)[:3],
        )
        return None, 1
    if eval_table.in_dim != table.in_dim:
        LOGGER.error(
            "latent width mismatch: %d vs %d -- the two dirs were extracted with different pooling",
            table.in_dim,
            eval_table.in_dim,
        )
        return None, 1
    LOGGER.info(
        "external eval: %d decoys / %d entries from %s",
        len(eval_table),
        len(eval_table.entries),
        args.eval_latent_dir,
    )
    return eval_table, 0


def _standardised_features(args: argparse.Namespace, train_table, val_table, eval_table):
    """Z-score the three feature blocks with the *training* statistics.

    The external set is standardised by train stats exactly as the internal holdout
    is: it is unseen data, so fitting a scaler on it would leak its distribution into
    the features the head reads.
    """
    eval_pooled = eval_table.pooled if eval_table is not None else None
    if args.no_standardise:
        return train_table.pooled, val_table.pooled, eval_pooled
    if eval_pooled is None:
        train_x, val_x = standardise(train_table.pooled, val_table.pooled)
        return train_x, val_x, None
    return standardise(train_table.pooled, val_table.pooled, eval_pooled)


def _scaler_payload(args: argparse.Namespace, train_table) -> dict | None:
    """Build the mean/std a checkpoint needs in order to score new data.

    Without these the checkpoint cannot score anything: the head was trained on
    z-scored features and nothing else records the scaler.  The first probe run
    omitted them, which is why it had to be retrained rather than re-evaluated.
    """
    if args.no_standardise:
        return None
    std = train_table.pooled.std(axis=0, keepdims=True)
    return {
        "mean": train_table.pooled.mean(axis=0, keepdims=True),
        "std": np.where(std < 1e-6, 1.0, std),
    }


def main(argv: list[str] | None = None) -> int:
    """Train the probe and report validation correlations."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    init_run(
        None if args.no_wandb else args.wandb_project,
        name=args.wandb_run_name or args.output_dir.name,
        group=args.wandb_group,
        config=config,
        tags=[*(args.wandb_tags or []), "probe"],
    )

    table = load_latents(args.latent_dir)
    train_entries, val_entries = split_entries(table.entries, val_fraction=args.val_fraction, seed=args.split_seed)
    train_table = table.subset(set(train_entries))
    val_table = table.subset(set(val_entries))
    LOGGER.info(
        "split: %d/%d entries, %d/%d decoys (train/val)",
        len(train_table.entries),
        len(val_table.entries),
        len(train_table),
        len(val_table),
    )
    (args.output_dir / "split.json").write_text(json.dumps({"train": train_entries, "val": val_entries}, indent=2))

    eval_table, code = _load_eval_table(args, table)
    if code:
        return code

    train_x, val_x, eval_x = _standardised_features(args, train_table, val_table, eval_table)

    in_dim = table.in_dim
    # in_dim = 2 * token_z + token_s; recover the widths the head expects.
    token_z = 128
    token_s = in_dim - 2 * token_z
    head = DockQHead(token_s, token_z, hidden=args.hidden, dropout=args.dropout).to(args.device)
    LOGGER.info(
        "DockQHead in_dim=%d params=%.2fM",
        head.in_dim,
        sum(p.numel() for p in head.parameters()) / 1e6,
    )
    if head.in_dim != in_dim:
        LOGGER.error("head in_dim %d != latent width %d", head.in_dim, in_dim)
        return 1

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DockQLoss(
        lambda_component=args.lambda_component,
        lambda_rank=args.lambda_rank,
        huber_delta=args.huber_delta,
        rank_tau=args.rank_tau,
    )

    report, _ = evaluate_probe(head, val_x, val_table, device=args.device)
    log_report(report, "epoch 0 (untrained head)")
    (args.output_dir / "report_epoch0.json").write_text(json.dumps(report, indent=2))
    # Epoch 0 carries the Boltz baselines, so the charts open with the bar to beat.
    log_metrics(flatten_report(report), step=0)

    history = []
    best = -float("inf")
    best_rows: list[dict] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        stats = train_probe(
            head,
            loss_fn,
            optimizer,
            train_x,
            train_table,
            device=args.device,
            epoch=epoch,
            seed=args.seed,
            accumulate=args.accumulate,
            grad_clip=args.grad_clip,
        )
        report, rows = evaluate_probe(head, val_x, val_table, device=args.device)
        score = report["per_entry"].get("dockq_pred", float("nan"))
        LOGGER.info(
            "epoch %d: loss %.4f (dockq %.4f comp %.4f rank %.4f) | val per_entry %.4f global %.4f",
            epoch,
            stats["loss"],
            stats["huber_dockq"],
            stats["huber_comp"],
            stats["rank"],
            score,
            report["global"].get("dockq_pred", float("nan")),
        )
        history.append({"epoch": epoch, "train": stats, "val": report})
        log_metrics(flatten_report(report) | {f"train/{k}": v for k, v in stats.items()}, step=epoch)

        # A NaN score must never win: it means the report had no rankable entries,
        # not a good epoch.
        if not math.isnan(score) and score > best:
            best = score
            best_rows = rows
            # Keep the weights in memory as well as on disk: the external set is
            # scored once, after the loop, and by then ``head`` holds the *last*
            # epoch's weights rather than the selected one.
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            torch.save(
                {
                    "dockq_head": head.state_dict(),
                    "epoch": epoch,
                    "per_entry_spearman": score,
                    "token_s": token_s,
                    "token_z": token_z,
                    "standardise": _scaler_payload(args, train_table),
                },
                args.output_dir / "dockq_head_best.pt",
            )

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    final = max(
        (h for h in history if h["val"]["per_entry"].get("dockq_pred") == best),
        key=lambda h: h["epoch"],
        default=None,
    )
    if final:
        log_report(final["val"], f"best (epoch {final['epoch']})")
        (args.output_dir / "report_best.json").write_text(json.dumps(final["val"], indent=2))
    write_predictions(args.output_dir / "predictions_best.csv", best_rows)

    if eval_table is not None and best_state is not None:
        # Restore the selected epoch's weights: the loop left ``head`` at the last
        # epoch, which is generally overfit relative to the checkpoint.
        head.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})
        eval_report, eval_rows = evaluate_probe(head, eval_x, eval_table, device=args.device)
        log_report(eval_report, f"external eval (weights from epoch {best_epoch})")
        (args.output_dir / "report_external.json").write_text(json.dumps(eval_report, indent=2))
        write_predictions(args.output_dir / "predictions_external.csv", eval_rows)
        # No step: scored once, at the selected epoch, so it is a summary of the run
        # rather than a point on the epoch axis.
        log_metrics(flatten_report(eval_report, prefix="external"))

    LOGGER.info("done; best per-entry Spearman %.4f", best)
    log_metrics({"best/per_entry_spearman": best, "best/epoch": float(best_epoch)})
    finish_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
