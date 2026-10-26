"""Post-process an existing DockQ scores CSV to add diversity metrics and plot.

This script can work with either:
- Complex-level CSV (one row per sample, has complex_dockq)
- Per-interface CSV (multiple rows per sample, has native_interface and DockQ)

For complex-level CSVs, it will try to load the corresponding *_per_interface.csv
for plots that need per-interface data (diversity, components KDE, interface type).

Plots using complex_dockq (via collect_by_entry):
- Violin plots, condition violins, KDE plots (work with either CSV type)

Plots requiring per-interface data:
- Diversity plot, DockQ components KDE, interface-type plot (skipped if no interface CSV)

Usage:
    # On complex-level CSV (will find _per_interface.csv automatically)
    poetry run python workflows/scripts/analysis/add_diversity_metrics.py \
        data/sabdab_boltz2_grid/dockq_scores.csv

    # On per-interface CSV directly
    poetry run python workflows/scripts/analysis/add_diversity_metrics.py \
        data/sabdab_boltz2_grid/dockq_scores_per_interface.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# Add repo root to path so confidock is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from confidock.evaluation import (  # noqa: E402
    compute_diversity_metrics,
    csv_has_bias_metrics,
    infer_condition_cols,
    run_bias_analysis,
)
from confidock.plot import (  # noqa: E402
    plot_condition_kde,
    plot_condition_kde_grid,
    plot_condition_violins,
    plot_cross_condition_heatmap,
    plot_distribution_shift,
    plot_diversity,
    plot_dockq_by_interface_type,
    plot_dockq_components_kde,
    plot_failure_mode_analysis,
    plot_violin,
    plot_violin_gif,
)

# Diversity columns added by compute_diversity_metrics
_DIVERSITY_COLS = [
    "dockq_mean",
    "dockq_std",
    "irmsd_mean",
    "irmsd_std",
    "lrmsd_std",
    "irmsd_range",
    "n_conf_clusters",
    "complex_ca_rmsd_mean",
    "complex_ca_rmsd_std",
    "binder_ca_rmsd_mean",
    "binder_ca_rmsd_std",
    "antigen_ca_rmsd_mean",
    "antigen_ca_rmsd_std",
    "interface_contact_f1_mean",
    "interface_contact_f1_std",
    "binder_residue_recall_mean",
    "binder_residue_recall_std",
    "antigen_residue_recall_mean",
    "antigen_residue_recall_std",
    "binder_interface_sidechain_rmsd_mean",
    "binder_interface_sidechain_rmsd_std",
    "antigen_interface_sidechain_rmsd_mean",
    "antigen_interface_sidechain_rmsd_std",
    "antigen_aligned_binder_ca_rmsd_mean",
    "antigen_aligned_binder_ca_rmsd_std",
]


def _load_input_csv(input_csv: Path) -> list[dict]:
    """Load rows from input CSV file."""
    print(f"Reading {input_csv} …")
    with input_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  {len(rows):,} rows loaded")
    return rows


def _load_interface_csv(input_csv: Path) -> list[dict] | None:
    """Try to load corresponding per-interface CSV for complex-level data."""
    per_interface_csv = input_csv.with_name(input_csv.stem.replace("_with_diversity", "") + "_per_interface.csv")
    if per_interface_csv.is_file():
        print(f"Loading per-interface data for diversity plot: {per_interface_csv}")
        with per_interface_csv.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    print("Note: Per-interface CSV not found, some plots may be limited")
    return None


def _compute_and_write_diversity(
    rows: list[dict],
    condition_cols: list[str],
    is_complex_csv: bool,
    is_interface_csv: bool,
    input_csv: Path,
    output_csv: Path | None,
) -> Path:
    """Compute diversity metrics if needed and write enriched CSV."""
    has_diversity = rows and "dockq_mean" in rows[0] and "dockq_std" in rows[0]
    if not has_diversity and (is_interface_csv or not is_complex_csv):
        print("Computing diversity metrics...")
        compute_diversity_metrics(rows, condition_cols)
    elif has_diversity:
        print("Diversity metrics already present, skipping computation")

    # Write enriched CSV
    output_csv = output_csv or input_csv.with_name(input_csv.stem + "_with_diversity.csv")
    existing_cols = list(rows[0].keys()) if rows else []
    fieldnames = existing_cols + [c for c in _DIVERSITY_COLS if c not in existing_cols]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote enriched CSV → {output_csv}")
    return output_csv


def _create_hybrid_rows_with_irmsd(
    rows: list[dict],
    interface_rows: list[dict],
    condition_cols: list[str],
) -> list[dict]:
    """Merge iRMSD from per-interface rows into complex rows."""
    from collections import defaultdict

    sample_irmsd = defaultdict(list)
    for irow in interface_rows:
        key = (irow["entry"], irow["seed"], irow["sample"]) + tuple(irow.get(c, "") for c in condition_cols)
        if irow.get("iRMSD") not in ("", None):
            sample_irmsd[key].append(float(irow["iRMSD"]))

    hybrid_rows = []
    for row in rows:
        row_copy = dict(row)
        key = (row["entry"], row["seed"], row["sample"]) + tuple(row.get(c, "") for c in condition_cols)
        if key in sample_irmsd and sample_irmsd[key]:
            row_copy["complex_irmsd"] = sum(sample_irmsd[key]) / len(sample_irmsd[key])
        hybrid_rows.append(row_copy)
    return hybrid_rows


def _generate_basic_plots(
    rows: list[dict],
    plot_rows_for_diversity: list[dict],
    output_csv: Path,
    stem: str,
    title: str,
    has_conditions: bool,
    is_complex_csv: bool,
    plot_log: list[str],
) -> None:
    """Generate basic violin and diversity plots."""
    # Diversity plot
    plot_path = output_csv.with_name(stem + "_diversity.png")
    if plot_rows_for_diversity and "native_interface" in plot_rows_for_diversity[0]:
        plot_diversity(plot_rows_for_diversity, plot_path, title=title)
        print(f"Saved diversity plot → {plot_path}")
        plot_log.append(f"✓ {plot_path.name}: per-interface data (DockQ, iRMSD per interface)")
    else:
        print("Skipping diversity plot (requires per-interface data with native_interface column)")
        plot_log.append(f"✗ {plot_path.name}: SKIPPED (requires per-interface data)")

    # Violin plot
    violin_path = output_csv.with_name(stem + "_dockq_violin.png")
    plot_violin(rows, violin_path, title=title)
    print(f"Saved DockQ violin plot → {violin_path}")
    metric_used = (
        "complex_dockq via collect_by_entry" if is_complex_csv else "DockQ (per-interface) via collect_by_entry"
    )
    plot_log.append(f"✓ {violin_path.name}: {metric_used}")

    # Condition violin plots
    if has_conditions:
        cond_violin_path = output_csv.with_name(stem + "_condition_violins.png")
        plot_condition_violins(rows, cond_violin_path, title=title)
        print(f"Saved condition violin plot → {cond_violin_path}")
        plot_log.append(f"✓ {cond_violin_path.name}: {metric_used}")
        try:
            violin_gif_path = output_csv.with_name(stem + "_per_target_conditions.gif")
            plot_violin_gif(rows, violin_gif_path, title=title)
            print(f"Saved condition violin GIF → {violin_gif_path}")
            plot_log.append(f"✓ {violin_gif_path.name}: {metric_used}")
        except Exception as exc:  # noqa: BLE001
            print(f"Condition violin GIF skipped: {exc}")
            plot_log.append(f"✗ {violin_gif_path.name}: SKIPPED ({exc})")


def _generate_kde_plot(
    rows: list[dict],
    plot_rows_for_diversity: list[dict],
    interface_rows: list[dict] | None,
    output_csv: Path,
    stem: str,
    title: str,
    is_complex_csv: bool,
    plot_log: list[str],
) -> None:
    """Generate KDE plot with hybrid approach for complex+interface data."""
    kde_path = output_csv.with_name(stem + "_kde.png")

    if is_complex_csv and "complex_dockq" in rows[0]:
        has_complex_irmsd = any(r.get("complex_irmsd") not in ("", None) for r in rows[:100])

        if has_complex_irmsd:
            plot_condition_kde(rows, kde_path, title=title)
            print(f"Saved KDE plot → {kde_path}")
            plot_log.append(f"✓ {kde_path.name}: complex_dockq, complex_irmsd (complex-level)")
        elif interface_rows:
            condition_cols = infer_condition_cols(rows)
            hybrid_rows = _create_hybrid_rows_with_irmsd(rows, interface_rows, condition_cols)
            plot_condition_kde(hybrid_rows, kde_path, title=title)
            print(f"Saved KDE plot → {kde_path}")
            plot_log.append(f"✓ {kde_path.name}: complex_dockq, mean iRMSD from per-interface (hybrid)")
        else:
            plot_condition_kde(rows, kde_path, title=title)
            print(f"Saved KDE plot → {kde_path}")
            plot_log.append(f"✓ {kde_path.name}: complex_dockq only, no iRMSD data (complex-level)")
    elif plot_rows_for_diversity and "DockQ" in plot_rows_for_diversity[0]:
        plot_condition_kde(plot_rows_for_diversity, kde_path, title=title)
        print(f"Saved KDE plot → {kde_path}")
        plot_log.append(f"✓ {kde_path.name}: DockQ, iRMSD (per-interface)")
    else:
        print("Skipping KDE plot (requires DockQ/iRMSD or complex_dockq data)")
        plot_log.append(f"✗ {kde_path.name}: SKIPPED (no suitable data)")


def _generate_interface_plots(
    plot_rows_for_diversity: list[dict],
    output_csv: Path,
    stem: str,
    title: str,
    plot_log: list[str],
) -> None:
    """Generate plots that require per-interface data."""
    # Components KDE
    components_kde_path = output_csv.with_name(stem + "_components_kde.png")
    if plot_rows_for_diversity and "iRMSD" in plot_rows_for_diversity[0]:
        plot_dockq_components_kde(plot_rows_for_diversity, components_kde_path, title=title)
        print(f"Saved DockQ components KDE plot → {components_kde_path}")
        plot_log.append(f"✓ {components_kde_path.name}: DockQ, iRMSD, LRMSD, fnat (per-interface)")
    else:
        print("Skipping components KDE plot (requires per-interface data)")
        plot_log.append(f"✗ {components_kde_path.name}: SKIPPED (requires per-interface data)")

    # Interface type plot
    interface_type_path = output_csv.with_name(stem + "_by_interface_type.png")
    if plot_rows_for_diversity and "native_interface" in plot_rows_for_diversity[0]:
        plot_dockq_by_interface_type(plot_rows_for_diversity, interface_type_path, title=title)
        print(f"Saved interface-type DockQ plot → {interface_type_path}")
        plot_log.append(f"✓ {interface_type_path.name}: DockQ grouped by native_interface (per-interface)")
    else:
        print("Skipping interface-type plot (requires per-interface data)")
        plot_log.append(f"✗ {interface_type_path.name}: SKIPPED (requires per-interface data)")


def _generate_grid_and_heatmap_plots(
    plot_rows_for_diversity: list[dict],
    output_csv: Path,
    stem: str,
    title: str,
    plot_log: list[str],
) -> None:
    """Generate grid and heatmap plots."""
    # KDE grid
    kde_grid_path = output_csv.with_name(stem + "_kde_grid.png")
    irmsd_grid_path = output_csv.with_name(stem + "_kde_grid_irmsd.png")
    if plot_rows_for_diversity and "DockQ" in plot_rows_for_diversity[0]:
        plot_condition_kde_grid(plot_rows_for_diversity, kde_grid_path, title=title)
        print(f"Saved DockQ KDE grid → {kde_grid_path}")
        print(f"Saved iRMSD KDE grid → {irmsd_grid_path}")
        plot_log.append(f"✓ {kde_grid_path.name}: DockQ (per-interface)")
        plot_log.append(f"✓ {irmsd_grid_path.name}: iRMSD (per-interface)")
    else:
        print("Skipping KDE grid (requires per-interface data with DockQ and iRMSD)")
        plot_log.append(f"✗ {kde_grid_path.name}: SKIPPED (requires per-interface data)")
        plot_log.append(f"✗ {irmsd_grid_path.name}: SKIPPED (requires per-interface data)")

    # Cross-condition heatmap
    heatmap_path = output_csv.with_name(stem + "_cross_condition.png")
    if plot_rows_for_diversity and "DockQ" in plot_rows_for_diversity[0]:
        plot_cross_condition_heatmap(plot_rows_for_diversity, heatmap_path, title=title)
        print(f"Saved cross-condition heatmap → {heatmap_path}")
        plot_log.append(f"✓ {heatmap_path.name}: DockQ aggregated per (entry, condition) (per-interface)")
    else:
        print("Skipping cross-condition heatmap (requires per-interface data with DockQ)")
        plot_log.append(f"✗ {heatmap_path.name}: SKIPPED (requires per-interface data)")


def _generate_analysis_plots(
    plot_rows_for_diversity: list[dict],
    output_csv: Path,
    stem: str,
    title: str,
    plot_log: list[str],
) -> None:
    """Generate distribution shift and failure mode analysis plots."""
    # Distribution shift
    shift_path = output_csv.with_name(stem + "_distribution_shift.png")
    if plot_rows_for_diversity and "DockQ" in plot_rows_for_diversity[0]:
        plot_distribution_shift(plot_rows_for_diversity, shift_path, title=title)
        print(f"Saved distribution shift plot → {shift_path}")
        plot_log.append(f"✓ {shift_path.name}: DockQ per condition, Spearman correlations (per-interface)")
    else:
        print("Skipping distribution shift plot (requires per-interface data with DockQ)")
        plot_log.append(f"✗ {shift_path.name}: SKIPPED (requires per-interface data)")

    # Failure mode analysis
    failure_mode_path = output_csv.with_name(stem + "_failure_mode_analysis.png")
    if plot_rows_for_diversity and "DockQ" in plot_rows_for_diversity[0]:
        plot_failure_mode_analysis(plot_rows_for_diversity, failure_mode_path, title=title)
        print(f"Saved failure mode analysis → {failure_mode_path}")
        plot_log.append(f"✓ {failure_mode_path.name}: DockQ, RMSD, fnat, interface_contact_f1 (per-interface)")
    else:
        print("Skipping failure mode analysis (requires per-interface data with DockQ)")
        plot_log.append(f"✗ {failure_mode_path.name}: SKIPPED (requires per-interface data)")


def _run_bias_analysis_if_needed(
    skip_bias_analysis: bool,
    is_complex_csv: bool,
    interface_rows: list[dict] | None,
    input_csv: Path,
    output_csv: Path,
    stem: str,
    title: str,
    plot_log: list[str],
) -> None:
    """Run bias analysis if not skipped and data is available."""
    if skip_bias_analysis:
        return

    # Try per-interface CSV first (if we loaded it), then output CSV, then input CSV
    bias_source = None
    if is_complex_csv and interface_rows:
        per_interface_csv = input_csv.with_name(input_csv.stem.replace("_with_diversity", "") + "_per_interface.csv")
        if per_interface_csv.is_file() and csv_has_bias_metrics(per_interface_csv):
            bias_source = per_interface_csv

    if bias_source is None:
        bias_source = output_csv if csv_has_bias_metrics(output_csv) else input_csv

    if bias_source != output_csv:
        print(f"Bias analysis using: {bias_source.name}")

    try:
        run_bias_analysis(bias_source, title=title)
        bias_summary_path = output_csv.parent / f"{stem}_bias_summary.txt"
        bias_plot_path = output_csv.parent / f"{stem}_bias_overview.png"
        plot_log.append(f"✓ {bias_summary_path.name}: bias metrics (from {bias_source.name})")
        plot_log.append(f"✓ {bias_plot_path.name}: bias metrics (from {bias_source.name})")
    except ValueError as exc:
        print(f"Skipping bias analysis: {exc}")
        plot_log.append(f"✗ bias analysis: SKIPPED ({exc})")


def _write_plot_log(
    output_csv: Path,
    stem: str,
    input_csv: Path,
    is_complex_csv: bool,
    interface_rows: list[dict] | None,
    plot_log: list[str],
) -> None:
    """Write plot generation log file."""
    log_header = [
        f"Plot Generation Log for {input_csv.name}",
        f"Generated: {output_csv}",
        f"Input type: {'complex-level' if is_complex_csv else 'per-interface'} CSV",
        f"Per-interface data available: {'Yes' if interface_rows else 'No'}",
        "",
    ]

    log_footer = [
        "",
        "Metric descriptions:",
        "  - complex_dockq: DockQ computed with binder chains (H+L) merged as single unit",
        "  - DockQ: per-interface DockQ (separate for each chain pair, e.g., H-A, L-A)",
        "  - complex_irmsd: interface RMSD for merged binder",
        "  - iRMSD: per-interface RMSD",
        "  - collect_by_entry: aggregation that uses complex_dockq when available, "
        "falls back to mean per-interface DockQ",
    ]

    log_path = output_csv.with_name(stem + "_plot_metrics.log")
    with log_path.open("w") as f:
        f.write("\n".join(log_header + plot_log + log_footer))
    print(f"\nPlot metadata log → {log_path}")


def main() -> None:
    """Add diversity metrics and plots to an existing DockQ CSV.

    Optionally, this also runs structural bias characterization with
    significance tests and summary plots when enriched metric columns are
    present.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Existing DockQ scores CSV")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <stem>_diversity.csv next to input)",
    )
    parser.add_argument("--title", default="DockQ", help="Plot title prefix")
    parser.add_argument(
        "--skip_bias_analysis",
        action="store_true",
        help="Skip structural bias characterization, stats tests, and bias plots.",
    )
    args = parser.parse_args()

    if not args.input_csv.is_file():
        sys.exit(f"Input CSV not found: {args.input_csv}")

    # Load input data
    rows = _load_input_csv(args.input_csv)

    # Detect CSV type
    is_complex_csv = rows and "complex_dockq" in rows[0] and "native_interface" not in rows[0]
    is_interface_csv = rows and "native_interface" in rows[0]

    # Load interface data if available
    interface_rows = _load_interface_csv(args.input_csv) if is_complex_csv else None
    plot_rows_for_diversity = interface_rows if interface_rows else rows

    # Infer grid condition columns
    condition_cols = infer_condition_cols(rows)
    has_conditions = bool(condition_cols)

    # Compute diversity metrics and write enriched CSV
    output_csv = _compute_and_write_diversity(
        rows, condition_cols, is_complex_csv, is_interface_csv, args.input_csv, args.output_csv
    )

    # Generate plots
    stem = output_csv.stem.replace("_with_diversity", "")
    plot_log: list[str] = []

    _generate_basic_plots(
        rows, plot_rows_for_diversity, output_csv, stem, args.title, has_conditions, is_complex_csv, plot_log
    )
    _generate_kde_plot(
        rows, plot_rows_for_diversity, interface_rows, output_csv, stem, args.title, is_complex_csv, plot_log
    )
    _generate_interface_plots(plot_rows_for_diversity, output_csv, stem, args.title, plot_log)
    _generate_grid_and_heatmap_plots(plot_rows_for_diversity, output_csv, stem, args.title, plot_log)
    _generate_analysis_plots(plot_rows_for_diversity, output_csv, stem, args.title, plot_log)
    _run_bias_analysis_if_needed(
        args.skip_bias_analysis, is_complex_csv, interface_rows, args.input_csv, output_csv, stem, args.title, plot_log
    )
    _write_plot_log(output_csv, stem, args.input_csv, is_complex_csv, interface_rows, plot_log)


if __name__ == "__main__":
    main()
