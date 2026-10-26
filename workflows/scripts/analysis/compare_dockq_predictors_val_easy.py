#!/usr/bin/env python3
r"""Apples-to-apples comparison of every DockQ predictor on ``sabdab_val_easy``.

Joins four learned predictors and every Boltz-2 confidence scalar onto **identical
rows** -- 33,879 decoys / 763 natives -- and reports global / per-native / rankable
Spearman plus top-1 mean DockQ, with a paired bootstrap over natives for the
differences.  This is the script behind ``docs/boltz_dockq_head_results.md``.

Why a join is needed at all: each predictor writes its own CSV with its own row
keys.  The GVP ablation keys on ``(native_id, decoy_id)`` derived from manifest
*paths*, while the Boltz caches key on a ``source__condition__[seed__]stem``
``decoy_id`` built during caching.  :func:`key_from_path` reconstructs the latter
from the former so the two families can be compared on the same decoys rather than
on two similar-looking supersets.  The reconstruction is verified, not assumed: the
run aborts unless every key matches and the two independently-recorded true-DockQ
columns agree to 1e-6.

Usage::

    poetry run python workflows/scripts/analysis/compare_dockq_predictors_val_easy.py \
        --output_csv results/dockq_predictor_comparison_val_easy.csv

Add ``--by_source`` for the per-generator breakdown, ``--bootstrap 10000`` for the
paired tests (both are on by default; ``--bootstrap 0`` skips the resampling).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

#: Minimum true-DockQ standard deviation for a native to count as rankable.  Kept
#: numerically identical to ``dockq_predictor.training.metrics.RANKABLE_MIN_STD`` and
#: ``boltz.finetune.trainer.RANKABLE_MIN_STD``; if one moves, all three must.
RANKABLE_MIN_STD = 0.05

#: Boltz-2 re-scores every decoy with its pretrained confidence head; these are the
#: scalars the cache stores.  ``ligand_iptm`` is NaN throughout on protein-only
#: complexes and is dropped by :func:`score_all`.
BOLTZ_SCALARS = (
    "ptm",
    "iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO / "data/dockq_predictor/val_easy_manifest.csv"
DEFAULT_GVP = REPO / "checkpoints/ablation_gvp_feature_loo"
DEFAULT_PROBE = REPO / "runs/dockq_probe_v2_valeasy/predictions_external.csv"
DEFAULT_FT = REPO / "runs/confidence_ft_v3/predictions_best.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--gvp_dir",
        type=Path,
        default=DEFAULT_GVP,
        help="ablation root holding full-s{0,1,2}/full/full_val_easy_predictions.csv",
    )
    parser.add_argument("--gvp_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--probe_csv", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--ft_csv", type=Path, default=DEFAULT_FT)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10_000,
        help="paired-bootstrap resamples over natives; 0 to skip",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--by_source", action="store_true", default=True)
    parser.add_argument("--no_by_source", dest="by_source", action="store_false")
    return parser.parse_args(argv)


def key_from_path(path: str, source: str, condition: str) -> str:
    """Rebuild the cache's ``decoy_id`` from a manifest ``pdb_path``.

    The caching runner named each decoy ``source__condition__[seed__]stem`` with the
    ``=`` stripped out of the condition.  Protenix decoys additionally carry the
    ``seed_*`` directory component, because its sampling writes one directory per
    seed and the stems repeat across them.
    """
    parts = path.split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    cond = condition.replace("=", "")
    if source == "protenix":
        seed = next((p for p in parts if p.startswith("seed_")), None)
        return f"{source}__{cond}__{seed}__{stem}"
    return f"{source}__{cond}__{stem}"


def load_joined(args: argparse.Namespace) -> pd.DataFrame:
    """Join every predictor onto one row per decoy, verifying the join as it goes."""
    manifest = pd.read_csv(args.manifest)
    manifest["decoy_id"] = [
        key_from_path(p, s, c)
        for p, s, c in zip(
            manifest["pdb_path"],
            manifest["model_source"],
            manifest["condition"],
            strict=True,
        )
    ]
    if manifest["decoy_id"].duplicated().any():
        raise ValueError(
            "reconstructed decoy_id is not unique in the manifest -- the key scheme "
            "in key_from_path no longer matches how the cache was written"
        )
    df = manifest[["decoy_id", "pdb_path", "entry", "model_source", "dockq", "ptm", "iptm", "plddt"]].rename(
        columns={"ptm": "own_ptm", "iptm": "own_iptm", "plddt": "own_plddt"}
    )

    # --- GVP: per-seed predictions, then the seed mean ------------------------
    # The ablation's ``decoy_id`` column is the absolute ``pdb_path``, not the
    # cache-style key, so it joins on the path directly.
    seed_cols = []
    for seed in args.gvp_seeds:
        path = args.gvp_dir / f"full-s{seed}" / "full" / "full_val_easy_predictions.csv"
        col = f"gvp_s{seed}"
        gvp = pd.read_csv(path, usecols=["decoy_id", "true_dockq", "pred_dockq"]).rename(
            columns={"pred_dockq": col, "decoy_id": "pdb_path"}
        )
        before = len(df)
        df = df.merge(gvp[["pdb_path", col, "true_dockq"]], on="pdb_path", how="inner")
        if len(df) != before:
            raise ValueError(
                f"{path.name}: joined {len(df)} of {before} rows -- the GVP "
                "predictions and the manifest are not the same decoy set"
            )
        # The GVP CSV records its own copy of the label; disagreement means the two
        # sides are keyed to different decoys despite the ids matching.
        drift = float(np.abs(df["true_dockq"] - df["dockq"]).max())
        if drift > 1e-6:
            raise ValueError(f"{path.name}: true-DockQ mismatch up to {drift:.2e}")
        df = df.drop(columns="true_dockq")
        seed_cols.append(col)
    df["gvp"] = df[seed_cols].mean(axis=1)

    # --- Boltz heads and the cached confidence scalars ------------------------
    # Either prediction CSV carries the same cached scalars, but the fine-tune run
    # skips OOM entries, so the scalars are taken from the probe's (complete) file.
    probe = pd.read_csv(args.probe_csv, usecols=["decoy_id", "dockq_pred", *BOLTZ_SCALARS]).rename(
        columns={"dockq_pred": "boltz_probe"}
    )
    df = df.merge(probe, on="decoy_id", how="left", validate="one_to_one")

    ft = pd.read_csv(args.ft_csv, usecols=["decoy_id", "dockq_pred"]).rename(columns={"dockq_pred": "boltz_ft"})
    df = df.merge(ft, on="decoy_id", how="left", validate="one_to_one")

    # Boltz's own ranking composite (boltz/model/models/boltz2.py:1085).  The cache
    # stores the components, not this, so it has to be recomputed here.
    df["confidence_score"] = (4 * df["complex_plddt"] + df["iptm"]) / 5

    for col in ("boltz_probe", "boltz_ft"):
        missing = int(df[col].isna().sum())
        if missing:
            # Left joins, so a miss means the Boltz run skipped those decoys (OOM) or
            # the key scheme drifted.  Silence here would understate a predictor by
            # scoring it on a smaller, easier subset.
            print(f"warning: {col} missing on {missing} of {len(df)} rows")
    return df.drop(columns="pdb_path")


def per_native(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(entries, within-native Spearman)`` for one predictor.

    Natives with fewer than 3 decoys or an undefined correlation are dropped rather
    than counted as zero -- matching ``boltz.finetune.trainer.correlation_report``.
    On ``val_easy`` neither case occurs (minimum 30 decoys per native, no constant
    true DockQ), so this choice is inert here; it matters on other sets.
    """
    entries, values = [], []
    for entry, grp in df.groupby("entry", sort=True):
        if len(grp) < 3:
            continue
        r = spearmanr(grp[col], grp["dockq"]).statistic
        if r is None or np.isnan(r):
            continue
        entries.append(entry)
        values.append(float(r))
    return np.array(entries), np.array(values)


def score_all(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Global / per-native / rankable Spearman and top-1 mean DockQ per predictor."""
    std = df.groupby("entry")["dockq"].std(ddof=0)
    rankable = set(std.index[std >= RANKABLE_MIN_STD])

    rows = []
    for col in cols:
        sub = df[["entry", "dockq", col]].dropna()
        if sub.empty:
            continue  # e.g. ligand_iptm on protein-only complexes
        entries, values = per_native(sub, col)
        mask = np.array([e in rankable for e in entries])
        top1 = [grp.loc[grp[col].idxmax(), "dockq"] for _, grp in sub.groupby("entry")]
        rows.append(
            {
                "predictor": col,
                "global": float(spearmanr(sub[col], sub["dockq"]).statistic),
                "per_native": float(values.mean()),
                "rankable": float(values[mask].mean()) if mask.any() else np.nan,
                "top1_dockq": float(np.mean(top1)),
                "n_decoys": len(sub),
                "n_natives": len(entries),
            }
        )
    out = pd.DataFrame(rows).sort_values("per_native", ascending=False)
    return out.reset_index(drop=True)


def paired_bootstrap(df: pd.DataFrame, pairs: list[tuple[str, str]], *, n: int, seed: int) -> pd.DataFrame:
    """Bootstrap the per-native Spearman difference for each pair, over natives.

    Resampling natives (not decoys) is what makes the interval answer "would this
    ordering hold on a different set of targets?" -- the question a comparison of
    ranking quality actually poses.  Both subsets are reported because the near-flat
    natives dilute every predictor and can hide or invent a difference.
    """
    std = df.groupby("entry")["dockq"].std(ddof=0)
    cols = sorted({c for pair in pairs for c in pair})
    mat, entries = {}, None
    for col in cols:
        ents, vals = per_native(df, col)
        if entries is None:
            entries = ents
        elif not np.array_equal(entries, ents):
            raise ValueError(f"{col} covers a different native set")
        mat[col] = vals
    rk = np.array([std[e] >= RANKABLE_MIN_STD for e in entries])

    rng = np.random.default_rng(seed)
    rows = []
    for subset, label in [(np.ones_like(rk), f"all {len(rk)}"), (rk, f"rankable {int(rk.sum())}")]:
        take = subset.astype(bool)
        idx = rng.integers(0, int(take.sum()), size=(n, int(take.sum())))
        for a, b in pairs:
            d = (mat[a] - mat[b])[take]
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(
                {
                    "subset": label,
                    "comparison": f"{a} - {b}",
                    "delta": float(d.mean()),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "p_gt_0": float((boot > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Join, score, and print every table in the report."""
    args = parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    df = load_joined(args)
    print(f"joined {len(df)} decoys / {df.entry.nunique()} natives")

    learned = ["gvp", "boltz_probe", "boltz_ft"]
    cols = [*learned, "confidence_score", *BOLTZ_SCALARS, "own_iptm", "own_ptm"]
    table = score_all(df, cols)
    oracle = df.groupby("entry")["dockq"].max().mean()
    print("\n=== headline ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nORACLE (per-native max)  top1_dockq = {oracle:.4f}")
    print(f"pick-at-random (mean)    top1_dockq = {df['dockq'].mean():.4f}")

    if args.by_source:
        print("\n=== per-native Spearman by generator ===")
        rows = []
        for src, sub in df.groupby("model_source"):
            row = {
                "generator": src,
                "n_decoys": len(sub),
                "n_natives": sub.entry.nunique(),
                "mean_dockq": sub.dockq.mean(),
            }
            for col in [*learned, "confidence_score"]:
                row[col] = per_native(sub, col)[1].mean()
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.bootstrap:
        pairs = [
            ("gvp", "confidence_score"),
            ("gvp", "boltz_ft"),
            ("gvp", "boltz_probe"),
            ("boltz_ft", "confidence_score"),
            ("boltz_probe", "confidence_score"),
            ("boltz_ft", "boltz_probe"),
        ]
        print(f"\n=== paired bootstrap over natives ({args.bootstrap} resamples) ===")
        boot = paired_bootstrap(df, pairs, n=args.bootstrap, seed=args.seed)
        print(boot.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nwrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
