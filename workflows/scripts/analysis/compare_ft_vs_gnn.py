"""Compare the Boltz confidence fine-tune against the DockQ GNN and the native metrics.

Prints one table per evaluation set (``sabdab_val_easy``, ``sabdab_val_hard``) using a
single code path, so every number in the output is computed the same way.

Why a script rather than reading the reports
--------------------------------------------
``report_external.json`` already holds the fine-tune's metrics, but the GNN's
predictions live in a separate per-decoy CSV produced by the ablation pipeline, and
that pipeline's ``metrics.spearman_r`` differs from ``trainer.correlation_report`` on
degenerate entries (it returns 0.0 where the trainer excludes them).  Recomputing both
sides here removes that as a source of disagreement.

Scope matters more than any other choice here.  Every number below is **pooled**: one
Spearman per native over all ~44 decoys from all three folding models at once.  A
native confidence metric only ever ranks within its own model, so a per-model native
number (~15 decoys) is a different, harder task and is NOT comparable to these -- see
the pooled-vs-per-model correction in the project's GNN-vs-native verdict.

Usage
-----
    python workflows/scripts/analysis/compare_ft_vs_gnn.py
    python workflows/scripts/analysis/compare_ft_vs_gnn.py --bins 0.02 0.05 0.1 0.2
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

#: Matches ``trainer.RANKABLE_MIN_STD`` and ``dockq_predictor.training.metrics``.
RANKABLE_MIN_STD = 0.05

SHARED = pathlib.Path("/data/confidock/shared")
WORKSPACE = pathlib.Path("/data/confidock")

SWEEPS = pathlib.Path("/data/confidock/shared-sweeps")

#: The GNN arm to compare against.  This must be the **ListNet** GNN, whose ranking
#: objective matches the fine-tune's -- that is the only like-for-like comparison.
#:
#: Two traps here, both of which understate the GNN badly:
#:   * the ``gvp`` column in dockq_predictor_comparison_val_easy.csv is an older
#:     *feature*-ablation variant on a different objective (high-std bin 0.5301);
#:   * ``checkpoints/ablation_loss/listnet-s{0,1,2}`` is the right objective but the
#:     older sweep (0.6089).
#: ``confidock-sweeps/pr_gnn`` is the current ListNet GNN and the only source with
#: val_hard predictions, so it is the default.
GNN_ARM = "listnet"
GNN_RUNS = {
    "sabdab_val_easy": {
        f"GNN_{seed}": SWEEPS / f"pr_gnn/runs/{seed}/val_easy/predictions.csv"
        for seed in ("listnet", "listnet_s0", "listnet_s1")
    },
    "sabdab_val_hard": {
        f"GNN_{seed}": SWEEPS / f"pr_gnn/runs/{seed}/val_hard/predictions.csv"
        for seed in ("listnet", "listnet_s0", "listnet_s1")
    },
}

#: (label, predictions csv, prediction column).
SETS = {
    "sabdab_val_easy": {
        "listnet": (SHARED / "runs/confidence_ft_listnet/predictions_external.csv", "dockq_pred"),
        "composite": (SHARED / "runs/confidence_ft_composite/predictions_external.csv", "dockq_pred"),
        "v3": (WORKSPACE / "runs/confidence_ft_v3/predictions_best.csv", "dockq_pred"),
    },
    "sabdab_val_hard": {
        "listnet": (SHARED / "runs/confidence_ft_listnet_valhard/predictions_external.csv", "dockq_pred"),
    },
}

#: Native confidence metrics, read from whichever frame carries them.
NATIVES = ("confidence_score", "complex_plddt", "complex_iplddt", "ptm", "iptm")

#: Natives that also get a per-folding-model z-scored column.  Only the headline one,
#: to keep the table readable -- see :func:`zscore_within_model` for why this matters.
NATIVES_Z = ("confidence_score",)

#: The bin the GNN-vs-native verdict turns on, tested separately from the rankable set.
#: Also the last interior ``--bins`` edge by default, so the two views line up.
HIGH_VARIANCE_MIN_STD = 0.20


def canonical_decoy_id(path: str, model_source: str) -> str:
    """Map a GNN ablation prediction path to the flat ``decoy_id`` the other CSVs use.

    ``.../<model>/msa=full_msa__cycle=10__step=200/<entry>/seed_101/predictions/x.cif``
    becomes ``protenix__msafull_msa__cycle10__step200__seed_101__x``.  The seed
    component only exists for protenix, and omitting it collides several decoys onto one
    id, so it is not optional.
    """
    parts = pathlib.PurePath(path).parts
    condition = next(c for c in parts if "msa=" in c)
    seed = next((c for c in parts if c.startswith("seed_")), None)
    stem = pathlib.PurePath(path).stem
    tail = f"{seed}__{stem}" if seed else stem
    return f"{model_source}__{condition.replace('=', '')}__{tail}"


def load_gnn(name: str) -> tuple[pd.DataFrame | None, list[str]]:
    """Return a frame of ``decoy_id`` plus one column per GNN seed, and missing paths."""
    frames, missing = [], []
    for label, path in GNN_RUNS[name].items():
        if not path.exists():
            missing.append(f"{label} <- {path}")
            continue
        raw = pd.read_csv(path)
        raw["decoy_id"] = [canonical_decoy_id(p, m) for p, m in zip(raw["pdb_path"], raw["model_source"])]
        if raw["decoy_id"].duplicated().any():
            raise SystemExit(f"{path}: canonical decoy_id is not unique -- the mapping is wrong")
        # The first frame contributes the shared columns; later ones only their scores.
        keep = ["dockq", "entry", "model_source", label] if not frames else [label]
        frames.append(raw.rename(columns={"dockq_pred": label}).set_index("decoy_id")[keep])
    if not frames:
        return None, missing
    joined = pd.concat(frames, axis=1, join="inner").reset_index()
    # The seed mean is the ensemble; the per-seed columns stay so the spread is visible
    # rather than hidden inside one averaged number.
    seeds = [c for c in joined.columns if c.startswith("GNN_")]
    joined[f"GNN_{GNN_ARM}_ens"] = joined[seeds].mean(axis=1)
    return joined, missing


def per_native_spearman(frame: pd.DataFrame, column: str) -> dict[str, float]:
    """Spearman of ``column`` vs true DockQ within each native.

    Entries with fewer than 3 decoys, or where either side is constant, are excluded
    rather than counted as zero -- the same convention as
    ``trainer.correlation_report``.
    """
    out: dict[str, float] = {}
    for entry, group in frame.groupby("entry"):
        if len(group) < 3 or group[column].nunique() < 2 or group["dockq"].nunique() < 2:
            continue
        out[entry] = spearmanr(group[column], group["dockq"]).correlation
    return out


def top1_mean_dockq(frame: pd.DataFrame, column: str) -> float:
    """Mean true DockQ of each native's top-ranked decoy."""
    return float(frame.loc[frame.groupby("entry")[column].idxmax(), "dockq"].mean())


def collect_frames(name: str) -> tuple[dict[str, tuple[pd.DataFrame, str]], list[str]]:
    """Return ``{model: (frame, prediction column)}`` for one set, plus the missing paths.

    The GNN contributes one entry per seed column and one for the ensemble mean, all
    sharing the single frame :func:`load_gnn` joined them into.
    """
    frames: dict[str, tuple[pd.DataFrame, str]] = {}
    missing: list[str] = []
    for model, (path, column) in SETS[name].items():
        if not path.exists():
            missing.append(f"{model} <- {path}")
            continue
        frames[model] = (pd.read_csv(path), column)

    gnn, gnn_missing = load_gnn(name)
    missing += gnn_missing
    if gnn is not None:
        for column in [c for c in gnn.columns if c.startswith("GNN_")]:
            frames[column] = (gnn, column)
    return frames, missing


def shared_decoy_ids(frames: dict[str, tuple[pd.DataFrame, str]]) -> set[str]:
    """Return the decoy ids every frame carries -- the inner join the tables are built on."""
    return set.intersection(*(set(frame["decoy_id"]) for frame, _ in frames.values()))


def aligned_rows(frame: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    """Return ``frame``'s rows for ``keys`` in decoy_id order, so all columns line up."""
    return frame[frame["decoy_id"].isin(keys)].sort_values("decoy_id").reset_index(drop=True)


def prediction_columns(
    name: str,
    frames: dict[str, tuple[pd.DataFrame, str]],
    keys: set[str],
    ref_name: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Return one aligned prediction column per model, and the first model's rows.

    Those rows are what every later column is indexed against, so a model whose DockQ
    labels disagree with them is a mismatched evaluation rather than a worse score, and
    exits rather than being tabulated.
    """
    cols: dict[str, np.ndarray] = {}
    base = None
    for model, (frame, column) in frames.items():
        sub = aligned_rows(frame, keys)
        cols[model] = sub[column].to_numpy()
        if base is None:
            base = sub
        elif not np.allclose(sub["dockq"], base["dockq"], atol=1e-6):
            raise SystemExit(f"{name}: {model} disagrees with {ref_name} on the DockQ labels")
    return cols, base


def native_columns(frames: dict[str, tuple[pd.DataFrame, str]], keys: set[str]) -> dict[str, np.ndarray]:
    """Return each native metric's aligned column, from whichever frame carries it."""
    cols: dict[str, np.ndarray] = {}
    for native in NATIVES:
        frame = next((f for f, _ in frames.values() if native in f.columns), None)
        if frame is not None:
            cols[native] = aligned_rows(frame, keys)[native].to_numpy()
    return cols


def zscore_within_model(cols: dict[str, np.ndarray], base: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return per-folding-model z-scored copies of the :data:`NATIVES_Z` columns.

    A pooled per-native Spearman ranks ~44 decoys from three different folding models
    against each other, and a raw native confidence is not commensurable across models
    -- boltz2's 0.96 and esmfold2's 0.96 do not mean the same thing.  Both learned
    predictors are commensurable by construction (one model scores every decoy), so
    comparing them to a RAW pooled native flatters them.  Z-scoring each metric within
    its own folding model fixes the interleaving and is the honest baseline; per-native
    Spearman is invariant to it within a single model, so this changes nothing about the
    per-model view.  Returns nothing when the frames do not say which model a decoy came
    from, since then the correction cannot be applied at all.
    """
    source_col = next((c for c in ("source", "model_source") if c in base.columns), None)
    if source_col is None:
        return {}
    sources = base[source_col].to_numpy()
    out: dict[str, np.ndarray] = {}
    for native in NATIVES_Z:
        if native in cols:
            grouped = pd.Series(cols[native]).groupby(sources)
            out[f"{native}_z"] = grouped.transform(lambda s: (s - s.mean()) / (s.std() or 1.0)).to_numpy()
    return out


def print_header(name: str, missing: list[str]) -> None:
    """Print one set's banner, then the prediction files that were not there to read."""
    print(
        f"\n{'=' * 78}\n{name}   (GNN arm: {GNN_ARM}, "
        f"{len(GNN_RUNS[name])} seeds from confidock-sweeps/pr_gnn)\n{'=' * 78}"
    )
    for entry in missing:
        print(f"  MISSING: {entry}")


def print_scope(work: pd.DataFrame, rankable: list[str]) -> None:
    """Print the decoy and native counts every number below is computed over."""
    natives = work["entry"].nunique()
    print(
        f"  {len(work)} decoys / {natives} natives "
        f"({len(work) / natives:.1f} decoys per native, pooled over all folding models)"
    )
    print(f"  rankable (true-DockQ std >= {RANKABLE_MIN_STD}): {len(rankable)} natives\n")


def print_headline_table(
    work: pd.DataFrame,
    pe: dict[str, dict[str, float]],
    models: list[str],
    scored: dict[str, float],
    rankable: list[str],
    width: int,
) -> None:
    """Print the global, per-native and top-1 rows, one column per model."""
    head = f"  {'metric':<26}" + "".join(f"{m:>{width}}" for m in models)
    print(head)
    print("  " + "-" * (len(head) - 2))

    def row(label: str, fn) -> None:
        print(f"  {label:<26}" + "".join(f"{fn(m):>{width}.4f}" for m in models))

    row("global Spearman", lambda m: spearmanr(work[m], work["dockq"]).correlation)
    row(f"per-native (all {len(scored)})", lambda m: float(np.mean(list(pe[m].values()))))
    row(f"per-native rankable ({len(rankable)})", lambda m: float(np.mean([pe[m][e] for e in rankable])))
    row("top-1 mean DockQ", lambda m: top1_mean_dockq(work, m))
    oracle = work.groupby("entry")["dockq"].max().mean()
    print(f"  {'top-1 ORACLE':<26}{oracle:>{width}.4f}")


def print_bin_table(
    pe: dict[str, dict[str, float]],
    models: list[str],
    scored: dict[str, float],
    stds: pd.Series,
    bins: list[float],
    width: int,
) -> None:
    """Print per-native Spearman split by true-DockQ std bin, skipping empty bins."""
    print("\n  per-native Spearman by true-DockQ std bin (ddof=0)")
    print(f"  {'bin':<12}{'n':>5}" + "".join(f"{m:>{width}}" for m in models))
    edges = [0.0, *bins, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        members = [e for e in scored if lo <= stds[e] < hi]
        if not members:
            continue
        label = f"{lo:.2f}-{min(hi, 1.0):.2f}"
        print(
            f"  {label:<12}{len(members):>5}"
            + "".join(f"{np.mean([pe[m][e] for e in members]):>{width}.4f}" for m in models)
        )


def print_wilcoxon_block(
    subset: str,
    ref_name: str,
    others: list[str],
    pe: dict[str, dict[str, float]],
    entries: list[str],
) -> None:
    """Print the signed-rank test of ``ref_name`` against each other model over ``entries``."""
    print(f"\n  paired Wilcoxon vs {ref_name}, {subset} (n={len(entries)})")
    for other in others:
        a = np.array([pe[ref_name][e] for e in entries])
        b = np.array([pe[other][e] for e in entries])
        _, p = wilcoxon(a, b)
        print(f"    {ref_name} - {other:<18} {a.mean() - b.mean():+.4f}   win rate {(a > b).mean():5.1%}   p={p:.2e}")


def print_wilcoxon_tables(
    pe: dict[str, dict[str, float]],
    models: list[str],
    ref_name: str,
    scored: dict[str, float],
    stds: pd.Series,
    rankable: list[str],
) -> None:
    """Print the paired tests over the rankable set and over the high-variance bin.

    The two subsets are where the comparison can actually go either way, and a mean
    difference over the rankable set can hide a loss confined to the wide natives.
    """
    others = [m for m in models if m != ref_name]
    if not others:
        return
    print_wilcoxon_block("rankable set", ref_name, others, pe, rankable)
    high = [e for e in scored if stds[e] >= HIGH_VARIANCE_MIN_STD]
    print_wilcoxon_block(f"high-variance bin std>={HIGH_VARIANCE_MIN_STD:.2f}", ref_name, others, pe, high)


def print_spread(work: pd.DataFrame, models: list[str]) -> None:
    """Print each model's prediction std against the true DockQ std."""
    print(f"\n  prediction spread (true DockQ std = {work['dockq'].std():.4f})")
    for m in models:
        print(f"    {m:<20} {work[m].std():.4f}")


def report(name: str, bins: list[float]) -> None:
    """Print the comparison table for one evaluation set."""
    frames, missing = collect_frames(name)
    print_header(name, missing)
    if not frames:
        return

    # The fine-tune's ListNet arm is what everything else is differenced against; with
    # no such arm, whichever model was read first stands in as the reference.
    ref_name = "listnet" if "listnet" in frames else next(iter(frames))
    keys = shared_decoy_ids(frames)
    cols, base = prediction_columns(name, frames, keys, ref_name)
    cols.update(native_columns(frames, keys))
    cols.update(zscore_within_model(cols, base))

    work = base[["entry", "dockq"]].copy()
    for model, values in cols.items():
        work[model] = values
    models = list(cols)

    stds = work.groupby("entry")["dockq"].std(ddof=0)
    pe = {m: per_native_spearman(work, m) for m in models}
    scored = pe[ref_name]
    rankable = [e for e in scored if stds[e] >= RANKABLE_MIN_STD]
    width = max(len(m) for m in models) + 2

    print_scope(work, rankable)
    print_headline_table(work, pe, models, scored, rankable, width)
    print_bin_table(pe, models, scored, stds, bins, width)
    print_wilcoxon_tables(pe, models, ref_name, scored, stds, rankable)
    print_spread(work, models)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bins",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10, 0.20],
        help="interior std-bin edges; 0 and 1.01 are added automatically",
    )
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    args = parser.parse_args()
    for name in args.sets:
        report(name, args.bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
