"""Bias analysis for antibody-antigen predictions.

Characterizes systematic biases in predictions between binder (antibody) and
antigen chains, including structure quality differences and interface prediction biases.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidock.evaluation.analysis import (
    build_condition_label,
    check_bias_columns,
    compute_bias_deltas,
    get_grouping_axes,
    infer_chain_topology,
)

LOGGER = logging.getLogger(__name__)

try:
    from scipy import stats
except ImportError:
    stats = None


def run_bias_analysis(
    enriched_csv: Path,
    title: str,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Characterize antigen/binder and chain-style biases with stats + plots.

    Parameters
    ----------
    enriched_csv : Path
        Path to CSV with diversity and structural metrics
    title : str
        Title prefix for plots
    output_dir : Path, optional
        Output directory for results. If None, uses same directory as input CSV.

    Returns
    -------
    tuple of Path
        (summary_text_path, overview_plot_path)

    Raises
    ------
    ValueError
        If CSV is missing required bias columns
    """
    df = pd.read_csv(enriched_csv)

    has_cols, missing = check_bias_columns(df)
    if not has_cols:
        msg = f"Missing required columns for bias analysis: {', '.join(missing)}"
        raise ValueError(msg)

    # Compute bias metrics
    df = compute_bias_deltas(df)
    df["chain_topology"] = infer_chain_topology(df)
    df = build_condition_label(df)

    # Setup output paths
    if output_dir is None:
        output_dir = enriched_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = enriched_csv.stem.replace("_with_diversity", "")
    summary_path = output_dir / f"{stem}_bias_summary.txt"
    plot_path = output_dir / f"{stem}_bias_overview.png"

    # Generate summary text
    _write_bias_summary(df, summary_path)
    LOGGER.info("Saved bias summary → %s", summary_path)

    # Generate overview plot
    _plot_bias_overview(df, plot_path, title)
    LOGGER.info("Saved bias overview plot → %s", plot_path)

    return summary_path, plot_path


def _write_bias_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Write bias analysis summary to text file."""
    bias_cols = [
        "delta_ca_rmsd_antigen_minus_binder",
        "delta_sc_rmsd_antigen_minus_binder",
        "delta_recall_binder_minus_antigen",
    ]

    lines: list[str] = []
    lines.append("Bias Characterization Summary")
    lines.append(f"Rows analyzed: {len(df):,}")
    lines.append("")

    # Global bias signals
    lines.append("GLOBAL BIAS SIGNALS")
    for c in bias_cols:
        s = df[c].dropna()
        lines.append(
            f"- {c}: mean={s.mean():.4f}, median={s.median():.4f}, "
            f"p25={s.quantile(0.25):.4f}, p75={s.quantile(0.75):.4f}"
        )
    lines.append("")

    # Bias by MSA mode
    if "msa_mode" in df.columns:
        lines.append("BIAS BY msa_mode")
        cols = ["DockQ", "interface_contact_f1_mean", *bias_cols]
        lines.append(df.groupby("msa_mode")[cols].mean(numeric_only=True).round(4).to_string())
        lines.append("")

    # Bias by other grouping axes
    for axis in get_grouping_axes(df):
        lines.append(f"BIAS BY {axis}")
        cols = ["DockQ", "interface_contact_f1_mean", *bias_cols]
        lines.append(df.groupby(axis)[cols].mean(numeric_only=True).round(4).to_string())
        lines.append("")

    # Bias by chain topology
    lines.append("BIAS BY chain_topology")
    cols = ["DockQ", "interface_contact_f1_mean", *bias_cols]
    lines.append(df.groupby("chain_topology")[cols].mean(numeric_only=True).round(4).to_string())
    lines.append("")

    # Correlation with DockQ
    lines.append("CORRELATION WITH DockQ")
    for c in bias_cols:
        sub = df[["DockQ", c]].dropna()
        corr = sub["DockQ"].corr(sub[c]) if len(sub) > 10 else np.nan
        lines.append(f"- corr(DockQ, {c}) = {corr:.4f}")
    lines.append("")

    # Statistical tests
    lines.extend(_run_statistical_tests(df, bias_cols))

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _run_paired_wilcoxon_tests(df: pd.DataFrame) -> list[str]:
    """Run paired Wilcoxon tests for antigen vs binder comparisons."""
    test_lines: list[str] = []
    pair_tests = [
        (
            "CA RMSD antigen vs binder",
            "antigen_ca_rmsd_mean",
            "binder_ca_rmsd_mean",
        ),
        (
            "Interface sidechain RMSD antigen vs binder",
            "antigen_interface_sidechain_rmsd_mean",
            "binder_interface_sidechain_rmsd_mean",
        ),
        (
            "Residue recall binder vs antigen",
            "binder_residue_recall_mean",
            "antigen_residue_recall_mean",
        ),
    ]
    for label, a_col, b_col in pair_tests:
        sub = df[[a_col, b_col]].dropna()
        if len(sub) < 10:
            test_lines.append(f"- {label}: insufficient data")
            continue
        try:
            wil = stats.wilcoxon(sub[a_col], sub[b_col], alternative="two-sided")
            test_lines.append(f"- {label}: Wilcoxon W={wil.statistic:.3g}, p={wil.pvalue:.3e}, n={len(sub)}")
        except Exception as exc:  # noqa: BLE001
            test_lines.append(f"- {label}: Wilcoxon failed ({exc})")
    return test_lines


def _run_kruskal_wallis_tests(df: pd.DataFrame, bias_cols: list[str]) -> list[str]:
    """Run Kruskal-Wallis tests by MSA mode."""
    test_lines: list[str] = []
    if "msa_mode" not in df.columns:
        return test_lines

    for col in bias_cols:
        groups = [grp[col].dropna().values for _, grp in df.groupby("msa_mode") if grp[col].notna().any()]
        if len(groups) >= 2:
            try:
                kw = stats.kruskal(*groups)
                test_lines.append(f"- Kruskal by msa_mode on {col}: H={kw.statistic:.3g}, p={kw.pvalue:.3e}")
            except Exception as exc:  # noqa: BLE001
                test_lines.append(f"- Kruskal by msa_mode on {col}: failed ({exc})")
    return test_lines


def _run_statistical_tests(df: pd.DataFrame, bias_cols: list[str]) -> list[str]:
    """Run statistical significance tests on bias metrics."""
    test_lines: list[str] = []
    test_lines.append("STATISTICAL SIGNIFICANCE TESTS")

    if stats is None:
        test_lines.append("- scipy unavailable: significance tests skipped")
        return test_lines

    test_lines.extend(_run_paired_wilcoxon_tests(df))
    test_lines.extend(_run_kruskal_wallis_tests(df, bias_cols))

    return test_lines


def _plot_bias_overview(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Create 2x2 bias overview figure."""
    bias_cols = [
        "delta_ca_rmsd_antigen_minus_binder",
        "delta_sc_rmsd_antigen_minus_binder",
        "delta_recall_binder_minus_antigen",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: Global delta distributions
    labels_map = {
        "delta_ca_rmsd_antigen_minus_binder": "CA RMSD\nAg-Bd",
        "delta_sc_rmsd_antigen_minus_binder": "SC RMSD\nAg-Bd",
        "delta_recall_binder_minus_antigen": "Recall\nBd-Ag",
    }
    non_empty = [(c, df[c].dropna().values) for c in bias_cols if df[c].notna().any()]
    if non_empty:
        vals = [v for _, v in non_empty]
        labels = [labels_map[c] for c, _ in non_empty]
        vp = ax1.violinplot(vals, showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_alpha(0.5)
        vp["cmedians"].set_linewidth(1.8)
        ax1.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
        ax1.set_xticks(range(1, len(labels) + 1))
        ax1.set_xticklabels(labels)
        ax1.set_title("Global bias deltas")
        ax1.set_ylabel("Delta value")
    else:
        ax1.text(0.5, 0.5, "No bias deltas available", ha="center", va="center")
        ax1.set_axis_off()

    # Panel 2: Means by MSA mode
    if "msa_mode" in df.columns:
        grouped = (
            df.groupby("msa_mode")[bias_cols].mean(numeric_only=True).reindex(sorted(df["msa_mode"].dropna().unique()))
        )
        x = np.arange(len(grouped.index))
        w = 0.24
        for i, c in enumerate(bias_cols):
            ax2.bar(x + (i - 1) * w, grouped[c].values, width=w, label=c)
        ax2.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(grouped.index, rotation=15)
        ax2.set_title("Bias by MSA mode")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "msa_mode not available", ha="center", va="center")
        ax2.set_axis_off()

    # Panel 3: DockQ vs antigen-binder sidechain bias
    x_sc = pd.to_numeric(df["delta_sc_rmsd_antigen_minus_binder"], errors="coerce")
    y_dockq = pd.to_numeric(df["DockQ"], errors="coerce")
    c_f1 = pd.to_numeric(df["interface_contact_f1_mean"], errors="coerce")
    valid = x_sc.notna() & y_dockq.notna() & c_f1.notna()
    if valid.any():
        cmin = float(c_f1[valid].min())
        cmax = float(c_f1[valid].max())
        sc = ax3.scatter(
            x_sc[valid],
            y_dockq[valid],
            alpha=0.35,
            s=8,
            c=c_f1[valid],
            cmap="viridis",
            vmin=cmin,
            vmax=cmax,
            linewidths=0,
            rasterized=True,
        )
        cb = fig.colorbar(sc, ax=ax3, orientation="horizontal", fraction=0.06, pad=0.12)
        cb.set_label("interface_contact_f1_mean", labelpad=2)
    else:
        ax3.text(0.5, 0.5, "No valid points for scatter", ha="center", va="center")
    ax3.set_xlabel("delta_sc_rmsd_antigen_minus_binder")
    ax3.set_ylabel("DockQ")
    ax3.set_title("DockQ vs antigen-sidechain penalty")

    # Panel 4: Chain topology boxplot
    topo = (
        df.groupby("chain_topology")["delta_recall_binder_minus_antigen"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().values)
        .to_dict()
    )
    topo = {k: v for k, v in topo.items() if len(v) > 0}
    topo_labels = sorted(topo)
    if topo_labels:
        topo_vals = [topo[k] for k in topo_labels]
        ax4.boxplot(topo_vals, patch_artist=True)
        ax4.set_xticks(range(1, len(topo_labels) + 1))
        ax4.set_xticklabels(topo_labels, rotation=20, ha="right")
    else:
        ax4.text(
            0.5,
            0.5,
            "No non-null delta_recall values by chain_topology",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_xticks([])
    ax4.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax4.set_title("Binder-vs-antigen recall bias by chain topology")
    ax4.set_ylabel("delta_recall_binder_minus_antigen", labelpad=8)
    ax4.yaxis.set_label_position("right")

    fig.suptitle(f"{title} - Bias characterization", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
