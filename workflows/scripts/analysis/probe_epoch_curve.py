#!/usr/bin/env python3
r"""Score the external ``val_easy`` set at EVERY probe epoch.

``train_dockq_probe.py`` deliberately scores the external set once, at the epoch its
*internal* holdout selects -- selecting on the external set would turn the holdout
into a second validation set.  That is the right default for reporting, but it leaves
two questions unanswerable: **would more training help?**, and **does the internal
selector track the external set at all?**

This script answers both by evaluating both sets every epoch.  It is a diagnostic, not
a training tool: its curves must not be used to pick a checkpoint for the headline
comparison in ``docs/boltz_dockq_head_results.md``, or those numbers stop being
holdout numbers.  Latents are on disk, so 120 epochs with two evaluations each costs
about five minutes on one GPU.

Findings as of 2026-08-04 (see the report, section 4): the probe overfits by epoch
10-20, the internal/external correlation across epochs is weak (+0.40) and can be
negative (-0.68 at dropout 0.4), and no configuration exceeds ~0.185 external
per-native Spearman against the GVP GNN's 0.268.

Usage -- one config per GPU, five in parallel::

    for cfg in "1e-4 0.1 base" "1e-2 0.1 wd1e2" "1e-1 0.1 wd1e1" \
               "1e-4 0.4 drop04" "1e-2 0.4 wd1e2drop04"; do
        set -- $cfg
        CUDA_VISIBLE_DEVICES=$i poetry run python \
            workflows/scripts/analysis/probe_epoch_curve.py \
            --weight_decay "$1" --dropout "$2" --tag "$3" &
    done

Each run writes ``{out_dir}/probe_curve_{tag}.json``: one record per epoch with train
loss, internal per-native Spearman, and external per-native / rankable / global /
top-1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from confidock.boltz.finetune.cache_dataset import split_entries
from confidock.boltz.finetune.dockq_head import DockQHead, DockQLoss
from confidock.boltz.finetune.latent_cache import load_latents
from confidock.boltz.finetune.probe import (
    evaluate_probe,
    standardise,
    train_probe,
)

VOLUME = Path("/data/confidock")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latent_dir", type=Path, default=VOLUME / "sabdab_train/dockq_latents")
    parser.add_argument(
        "--eval_latent_dir",
        type=Path,
        default=VOLUME / "sabdab_val_easy/dockq_latents",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("results/probe_curves"))
    parser.add_argument("--tag", default="base")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lambda_component", type=float, default=1.0)
    parser.add_argument("--lambda_rank", type=float, default=0.5)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--rank_tau", type=float, default=0.05)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train the probe, evaluating both sets every epoch."""
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    table = load_latents(args.latent_dir)
    ext_table = load_latents(args.eval_latent_dir)
    shared = set(ext_table.entries) & set(table.entries)
    if shared:
        print(
            f"error: {len(shared)} entries appear in both latent dirs "
            f"(e.g. {sorted(shared)[:3]}) -- the external curve would be a train curve",
            file=sys.stderr,
        )
        return 1

    train_entries, val_entries = split_entries(table.entries, val_fraction=args.val_fraction, seed=args.split_seed)
    train_table = table.subset(set(train_entries))
    val_table = table.subset(set(val_entries))
    # Both held-out sets are z-scored with the *training* statistics; fitting a scaler
    # on either would leak its distribution into the features the head reads.
    train_x, val_x, ext_x = standardise(train_table.pooled, val_table.pooled, ext_table.pooled)
    print(
        f"[{args.tag}] train {len(train_table)} / int-val {len(val_table)} / "
        f"ext {len(ext_table)} decoys, in_dim={table.in_dim}",
        flush=True,
    )

    token_z = 128
    token_s = table.in_dim - 2 * token_z
    head = DockQHead(token_s, token_z, hidden=args.hidden, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DockQLoss(
        lambda_component=args.lambda_component,
        lambda_rank=args.lambda_rank,
        huber_delta=args.huber_delta,
        rank_tau=args.rank_tau,
    )

    rows: list[dict[str, float]] = []
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
        internal, _ = evaluate_probe(head, val_x, val_table, device=args.device)
        external, _ = evaluate_probe(head, ext_x, ext_table, device=args.device)
        rows.append(
            {
                "epoch": epoch,
                "loss": stats["loss"],
                "int_pn": internal["per_entry"]["dockq_pred"],
                "ext_pn": external["per_entry"]["dockq_pred"],
                "ext_rk": external["per_entry_rankable"]["dockq_pred"],
                "ext_g": external["global"]["dockq_pred"],
                "ext_t1": external["top1_mean_dockq"]["dockq_pred"],
            }
        )
        if epoch == 1 or epoch % args.log_every == 0:
            r = rows[-1]
            print(
                f"[{args.tag}] ep {epoch:>3} loss {r['loss']:.4f} | "
                f"int {r['int_pn']:.4f} | ext per-nat {r['ext_pn']:.4f} "
                f"rankable {r['ext_rk']:.4f} top1 {r['ext_t1']:.4f}",
                flush=True,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"probe_curve_{args.tag}.json"
    out.write_text(json.dumps(rows, indent=1))

    internal_curve = np.array([r["int_pn"] for r in rows])
    external_curve = np.array([r["ext_pn"] for r in rows])
    selected = int(internal_curve.argmax())
    best = int(external_curve.argmax())
    print(
        f"\n[{args.tag}] ext best {external_curve.max():.4f} @ ep {best + 1}; "
        f"at the int-selected ep {selected + 1}: {external_curve[selected]:.4f}"
    )
    # A weak or negative correlation means "train longer and keep the best epoch" is
    # not actionable: the selector cannot see which epoch that is.
    print(
        f"[{args.tag}] corr(internal, external) over epochs = "
        f"{np.corrcoef(internal_curve, external_curve)[0, 1]:+.3f}"
    )
    if len(external_curve) > 20:
        print(f"[{args.tag}] epoch-to-epoch noise floor (std ep20+) = {external_curve[19:].std():.4f}")
    print(f"[{args.tag}] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
