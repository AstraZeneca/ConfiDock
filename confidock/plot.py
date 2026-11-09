"""Plotting utilities for DockQ analysis.

Importing this module applies AZ brand styling globally to matplotlib.
All plot functions write directly to disk and log the output path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AstraZeneca brand palette
# ---------------------------------------------------------------------------

AZ_PURPLE = "#8a0051"  # primary purple
AZ_YELLOW = "#EFAB00"  # primary yellow
AZ_GREEN = "#CBD502"  # accent green
AZ_NAVY = "#013660"  # dark navy (text / axis labels)
AZ_SKY = "#97D5E4"  # light sky blue
AZ_PINK = "#E6007C"  # accent pink
AZ_GREY = "#ACC0BF"  # neutral grey

AZ_PALETTE: list[str] = [AZ_PURPLE, AZ_YELLOW, AZ_GREEN, AZ_NAVY, AZ_SKY, AZ_PINK, AZ_GREY]

# DockQ quality thresholds: (value, colour, label)
THRESHOLDS: list[tuple[float, str, str]] = [
    (0.23, AZ_GREY, "acceptable (0.23)"),
    (0.49, AZ_YELLOW, "medium (0.49)"),
    (0.80, AZ_PURPLE, "high (0.80)"),
]

# Common label strings
LABEL_COMPLEX_DOCKQ = "Complex DockQ"
LABEL_IRMSD_ANGSTROM = "iRMSD (Å)"
LEGEND_LOC_LOWER_RIGHT = "lower right"
LEGEND_LOC_UPPER_RIGHT = "upper right"
LEGEND_LOC_UPPER_LEFT = "upper left"

# Common box styles
BBOX_STYLE_ROUNDED = "round,pad=0.4"

# Failure mode labels
FAILURE_MODE_SUCCESS = "Success"
FAILURE_MODE_DOCKING = "Docking Failure"
FAILURE_MODE_FOLDING = "Folding Failure"
FAILURE_MODE_LUCKY = "Lucky Docking"

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

# Where the Roboto TTFs live. Registering them explicitly rather than trusting the
# system font path is what makes a figure look the same in a k8s job as it does in the
# notebook: a job's container has its own HOME, so ~/.local/share/fonts is empty there
# and matplotlib would silently fall back to DejaVu Sans -- silently being the problem,
# since a fallback changes every text extent in the figure but raises nothing.
# Point CONFIDOCK_FONT_DIR at a copy on a shared volume to get the same faces in a job.
# workflows/scripts/utils/install_roboto.py creates either directory.
FONT_DIR = Path(os.environ.get("CONFIDOCK_FONT_DIR", Path.home() / ".local/share/fonts/roboto"))


def _register_fonts(font_dir: Path = FONT_DIR) -> list[str]:
    """Add every TTF in ``font_dir`` to matplotlib's font manager.

    Returns the family names that are now available. Missing directories are not an
    error -- the rcParams below keep DejaVu Sans as the last resort -- but the
    resulting figures will not match the brand style, so it is logged.
    """
    if not font_dir.is_dir():
        LOGGER.info("Font dir %s not found; falling back to whatever fonts exist", font_dir)
        return []
    added = []
    for ttf in sorted(font_dir.glob("*.ttf")):
        fm.fontManager.addfont(str(ttf))
        added.append(ttf.name)
    LOGGER.debug("Registered %d font files from %s", len(added), font_dir)
    return added


_register_fonts()


# ---------------------------------------------------------------------------
# Global rcParams – applied on import
# ---------------------------------------------------------------------------

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        # Roboto first: it is the only one of these that is actually installed on the
        # Linux boxes (Gill Sans MT is the AZ brand face but ships with Office, so it
        # exists only on managed Windows laptops). Left in the stack so a figure
        # regenerated there still gets the brand face.
        "font.sans-serif": ["Roboto", "Gill Sans MT", "Gill Sans", "Arial", "DejaVu Sans"],
        # Mathtext has its OWN font set, so without this the ``$\rho$`` and
        # ``$\bf{...}$`` labels render in DejaVu while the surrounding text is Roboto --
        # visibly two fonts in one axis label. "custom" is what allows pointing it at a
        # text face; Roboto covers the Greek, superscripts and +-/>=/x we actually use.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Roboto",
        "mathtext.it": "Roboto:italic",
        "mathtext.bf": "Roboto:bold",
        "mathtext.cal": "Roboto:italic",  # else mathtext init probes for a cursive face
        "mathtext.default": "regular",  # match body text, not the italic maths default
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=AZ_PALETTE),
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.framealpha": 0.85,
        "legend.edgecolor": AZ_GREY,
        "grid.color": "#E8EEF4",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "text.color": "black",
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


# ---------------------------------------------------------------------------
# Data helper
# ---------------------------------------------------------------------------


def _condition_key_cols(rows: list[dict]) -> list[str]:
    """Return condition columns to keep condition-level samples distinct.

    For grid CSVs, per-target plots should pool over all conditions instead of
    collapsing to the first encountered condition for each (entry, seed, sample).
    """
    if not rows:
        return []
    candidates = [
        ["msa_mode", "recycling_steps", "step_scale"],
        ["msa_mode", "cycle", "step"],
    ]
    for cols in candidates:
        if all(any(str(r.get(c, "")).strip() for r in rows) for c in cols):
            return cols
    return []


def _build_sample_key(row: dict, condition_cols: list[str]) -> tuple | None:
    """Build unique sample key from row and condition columns.

    Parameters
    ----------
    row : dict
        Row data
    condition_cols : list of str
        Condition column names to include in key

    Returns
    -------
    tuple or None
        (entry, seed, sample, *condition_values) or None if required keys missing
    """
    # Check for required keys
    required = ["entry", "seed", "sample"]
    if not all(k in row for k in required):
        return None

    key_parts: list = [row["entry"], row["seed"], row["sample"]]
    key_parts.extend(row.get(c, "") for c in condition_cols)
    return tuple(key_parts)


def _collect_complex_dockq(rows: list[dict], condition_cols: list[str]) -> dict[str, list[float]]:
    """Collect DockQ values using complex_dockq column.

    Parameters
    ----------
    rows : list of dict
        Row data with complex_dockq values
    condition_cols : list of str
        Condition columns for building unique keys

    Returns
    -------
    dict
        Mapping {entry: [dockq_values]}
    """
    seen: set[tuple] = set()
    by_entry: dict[str, list[float]] = {}

    for row in rows:
        key = _build_sample_key(row, condition_cols)
        if key is None or key in seen:
            continue
        seen.add(key)
        val = row.get("complex_dockq")
        if val and str(val).strip():
            try:
                fval = float(val)
                # Skip NaN values
                if not np.isnan(fval):
                    by_entry.setdefault(row["entry"], []).append(fval)
            except (ValueError, TypeError):
                # Skip invalid DockQ values
                continue

    return by_entry


def _collect_interface_dockq(rows: list[dict], condition_cols: list[str]) -> dict[str, list[float]]:
    """Collect DockQ values by computing mean per-interface DockQ.

    Parameters
    ----------
    rows : list of dict
        Row data with per-interface DockQ values
    condition_cols : list of str
        Condition columns for building unique keys

    Returns
    -------
    dict
        Mapping {entry: [mean_dockq_per_sample]}
    """
    from collections import defaultdict

    sample_interfaces = defaultdict(list)
    for row in rows:
        key = _build_sample_key(row, condition_cols)
        if key is None:
            continue
        val = row.get("DockQ")
        if val and str(val).strip():
            try:
                fval = float(val)
                # Skip NaN values
                if not np.isnan(fval):
                    sample_interfaces[key].append(fval)
            except (ValueError, TypeError):
                # Skip invalid DockQ values
                continue

    # Compute mean for each sample
    by_entry: dict[str, list[float]] = {}
    for key, dockq_vals in sample_interfaces.items():
        if dockq_vals:
            entry = key[0]
            mean_dockq = sum(dockq_vals) / len(dockq_vals)
            by_entry.setdefault(entry, []).append(mean_dockq)

    return by_entry


def collect_by_entry(rows: list[dict]) -> dict[str, list[float]]:
    """Return mapping entry → list of DockQ values (one per sample).

    Expected to receive complex-level rows (one row per sample) with complex_dockq.
    Falls back to computing mean per-interface DockQ if complex_dockq unavailable.

    Complex DockQ treats all binder chains (e.g., H+L) as a single unit binding
    to antigen, using DockQ's recommended chain merging approach. This gives a
    weighted metric that properly accounts for each chain's contribution.

    Values are kept distinct across grid conditions.
    """
    condition_cols = _condition_key_cols(rows)

    # Check if complex_dockq is populated
    has_complex_dockq = any(
        row.get("complex_dockq") and str(row["complex_dockq"]).strip() for row in rows[: min(100, len(rows))]
    )

    if has_complex_dockq:
        return _collect_complex_dockq(rows, condition_cols)
    else:
        return _collect_interface_dockq(rows, condition_cols)


def _infer_grid_axes(rows: list[dict]) -> dict[str, object] | None:
    """Infer grid-axis column names and labels from result rows.

    Supports both Boltz-style grids (``recycling_steps`` / ``step_scale``) and
    Protenix-style grids (``cycle`` / ``step``).
    """
    candidates = [
        {
            "primary_col": "recycling_steps",
            "secondary_col": "step_scale",
            "primary_name": "recycling",
            "secondary_name": "scale",
            "primary_short": "r",
            "secondary_short": "s",
            "secondary_is_float": True,
        },
        {
            "primary_col": "cycle",
            "secondary_col": "step",
            "primary_name": "cycle",
            "secondary_name": "step",
            "primary_short": "c",
            "secondary_short": "t",
            "secondary_is_float": False,
        },
    ]
    for spec in candidates:
        primary_col = str(spec["primary_col"])
        secondary_col = str(spec["secondary_col"])
        has_primary = any(str(r.get(primary_col, "")).strip() for r in rows)
        has_secondary = any(str(r.get(secondary_col, "")).strip() for r in rows)
        if has_primary and has_secondary:
            return spec
    return None


def _grid_primary_value(row: dict, spec: dict[str, object]) -> int:
    return int(float(str(row[str(spec["primary_col"])])))


def _grid_secondary_value(row: dict, spec: dict[str, object]) -> float | int:
    value = float(str(row[str(spec["secondary_col"])]))
    if bool(spec["secondary_is_float"]):
        return value
    return int(value)


def _grid_secondary_text(value: float | int, spec: dict[str, object]) -> str:
    if bool(spec["secondary_is_float"]):
        return f"{float(value):.1f}"
    return str(int(value))


def _grid_condition_matches(
    row: dict,
    msa_mode: str,
    primary: int,
    secondary: float | int,
    spec: dict[str, object],
) -> bool:
    try:
        row_primary = _grid_primary_value(row, spec)
        row_secondary = _grid_secondary_value(row, spec)
    except (KeyError, TypeError, ValueError):
        return False
    return row.get("msa_mode") == msa_mode and row_primary == primary and row_secondary == secondary


def _sorted_grid_conditions(rows: list[dict], spec: dict[str, object]) -> tuple[list[int], list[float | int]]:
    primary_vals = sorted(
        {_grid_primary_value(r, spec) for r in rows if str(r.get(str(spec["primary_col"]), "")).strip()}
    )
    secondary_vals = sorted(
        {_grid_secondary_value(r, spec) for r in rows if str(r.get(str(spec["secondary_col"]), "")).strip()}
    )
    return primary_vals, secondary_vals


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_dockq(rows: list[dict], output_path: Path, title: str = "DockQ") -> None:
    """Boxplot of global DockQ distributions, one box per entry."""
    by_entry = collect_by_entry(rows)
    entries = sorted(by_entry)
    data = [by_entry[e] for e in entries]

    fig, ax = plt.subplots(figsize=(max(6, len(entries) * 1.4), 5))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops={"color": AZ_NAVY, "linewidth": 2},
        whiskerprops={"color": AZ_GREY, "linewidth": 1},
        capprops={"color": AZ_GREY, "linewidth": 1},
        flierprops={"marker": "o", "color": AZ_GREY, "markersize": 4, "alpha": 0.5},
    )
    for patch, colour in zip(bp["boxes"], AZ_PALETTE):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    for i, vals in enumerate(data, start=1):
        ax.scatter([i] * len(vals), vals, color=AZ_NAVY, s=18, zorder=3, alpha=0.5)

    # Determine data range for appropriate axis limits
    all_vals = [v for vals in data for v in vals]
    data_min = min(all_vals) if all_vals else 0.0
    data_max = max(all_vals) if all_vals else 1.0
    y_margin = (data_max - data_min) * 0.02 if data_max > data_min else 0.02

    ax.set_xticks(range(1, len(entries) + 1))
    ax.set_xticklabels(entries, rotation=30, ha="right")
    ax.set_ylabel(LABEL_COMPLEX_DOCKQ)
    ax.set_ylim(data_min - y_margin, data_max + y_margin)
    ax.set_title(f"{title} – Complex DockQ distributions per target")
    # Only show thresholds if data range is roughly 0-1
    if data_max <= 1.5:
        for val, colour, label in THRESHOLDS:
            ax.axhline(val, color=colour, linewidth=0.9, linestyle="--", label=label)
        ax.legend(loc=LEGEND_LOC_LOWER_RIGHT)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved boxplot → %s", output_path)


def _add_threshold_lines(ax, data_max: float, orientation: str) -> None:
    """Add DockQ threshold reference lines to the plot.

    Parameters
    ----------
    ax
        Matplotlib axes object
    data_max : float
        Maximum data value to determine if thresholds should be shown
    orientation : str
        "vertical" or "horizontal" - determines line direction
    """
    if data_max <= 1.5:
        for val, colour, label in THRESHOLDS:
            if orientation == "vertical":
                ax.axhline(val, color=colour, linewidth=0.9, linestyle="--", label=label)
                ax.legend(loc=LEGEND_LOC_LOWER_RIGHT)
            else:
                ax.axvline(val, color=colour, linewidth=0.9, linestyle="--", label=label)
                ax.legend(loc=LEGEND_LOC_UPPER_RIGHT)


def _configure_violin_axes(
    ax,
    entries: list[str],
    ticks: range,
    label_fs: int,
    data_min: float,
    data_max: float,
    y_margin: float,
    vert: bool,
) -> None:
    """Configure axes for violin plot based on orientation.

    Parameters
    ----------
    ax
        Matplotlib axes object
    entries : list of str
        Entry labels
    ticks : range
        Tick positions
    label_fs : int
        Label font size
    data_min, data_max : float
        Data range for axis limits
    y_margin : float
        Margin for axis limits
    vert : bool
        True for vertical orientation, False for horizontal
    """
    if vert:
        ax.set_xticks(ticks)
        ax.set_xticklabels(entries, rotation=45, ha="right", fontsize=label_fs)
        ax.set_ylabel(LABEL_COMPLEX_DOCKQ)
        ax.set_ylim(data_min - y_margin, data_max + y_margin)
        _add_threshold_lines(ax, data_max, "vertical")
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(entries, fontsize=label_fs)
        ax.set_xlabel(LABEL_COMPLEX_DOCKQ)
        ax.set_xlim(data_min - y_margin, data_max + y_margin)
        ax.invert_yaxis()
        _add_threshold_lines(ax, data_max, "horizontal")


def plot_violin(rows: list[dict], output_path: Path, title: str = "DockQ") -> None:
    """Violin plot, one violin per target; switches to horizontal layout for >40 targets.

    Black line = median, orange dashed = Q1/Q3.
    """
    by_entry = collect_by_entry(rows)
    entries = sorted(by_entry)
    data = [by_entry[e] for e in entries]
    n = len(entries)

    if n <= 40:
        fig_w, fig_h, vert = max(8, n * 1.1), 6, True
    else:
        fig_w, fig_h, vert = 12, max(10, min(n * 0.3, 50)), False

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    parts = ax.violinplot(
        data,
        positions=range(1, n + 1),
        vert=vert,
        showmedians=True,
        showextrema=False,
        quantiles=[[0.25, 0.75]] * n,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(AZ_PURPLE)
        pc.set_alpha(0.55)
    parts["cmedians"].set_color(AZ_YELLOW)
    parts["cmedians"].set_linewidth(2)
    parts["cquantiles"].set_color(AZ_NAVY)
    parts["cquantiles"].set_linewidth(1.5)
    parts["cquantiles"].set_linestyle("--")

    label_fs = max(5, min(9, 300 // n))
    ticks = range(1, n + 1)

    # Determine data range for appropriate axis limits
    all_vals = [v for vals in data for v in vals]
    data_min = min(all_vals) if all_vals else 0.0
    data_max = max(all_vals) if all_vals else 1.0
    y_margin = (data_max - data_min) * 0.02 if data_max > data_min else 0.02

    _configure_violin_axes(ax, entries, ticks, label_fs, data_min, data_max, y_margin, vert)

    ax.set_title(f"{title} – per-target DockQ distribution (n={n} targets)\nyellow = median  |  navy dashed = Q1/Q3")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved violin plot → %s", output_path)


def plot_heatmap(rows: list[dict], output_path: Path, title: str = "DockQ") -> None:
    """Heatmap: targets (rows) × DockQ bins (columns, width 0.1), colour = fraction of decoys."""
    from matplotlib.colors import LinearSegmentedColormap

    by_entry = collect_by_entry(rows)
    entries = sorted(by_entry)
    n = len(entries)

    # Handle empty data
    if n == 0:
        LOGGER.warning("No entries to plot in heatmap")
        # Create minimal empty plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        LOGGER.info("Saved empty heatmap → %s", output_path)
        return

    bin_edges = np.arange(0.0, 1.01, 0.1)
    bin_labels = [f"{e:.1f}–{e + 0.1:.1f}" for e in bin_edges[:-1]]
    n_bins = len(bin_labels)

    matrix = np.zeros((n, n_bins))
    for i, entry in enumerate(entries):
        vals = by_entry[entry]
        counts, _ = np.histogram(vals, bins=bin_edges)
        total = len(vals)
        matrix[i] = counts / total if total > 0 else counts

    az_cmap = LinearSegmentedColormap.from_list("az_purple", ["#FFFFFF", AZ_YELLOW, AZ_PURPLE])

    fig_h = max(6, min(n * 0.28, 50))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap=az_cmap, vmin=0, vmax=1, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Fraction of decoys")

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    label_fs = max(4, min(9, 300 // n))
    ax.set_yticklabels(entries, fontsize=label_fs)
    ax.set_xlabel("DockQ bin")
    ax.set_ylabel("Target")
    ax.set_title(f"{title} – DockQ coverage heatmap (fraction of decoys per bin)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved heatmap → %s", output_path)


def plot_global_summary(rows: list[dict], output_path: Path, title: str = "DockQ") -> None:
    """Three-panel global summary.

    Top: pooled histogram + KDE across all decoys.
    Bottom-left: per-target mean DockQ vs std DockQ scatter, coloured by fraction ≥ 0.49.
    Bottom-right: fraction of decoys with DockQ ≥ 0.49 per target, sorted descending.
    """
    from matplotlib.colors import LinearSegmentedColormap

    by_entry = collect_by_entry(rows)
    entries = sorted(by_entry)
    all_vals = np.array([v for vals in by_entry.values() for v in vals])

    means = np.array([np.mean(by_entry[e]) for e in entries])
    stds = np.array([np.std(by_entry[e]) for e in entries])
    frac_medium = np.array([sum(v >= 0.49 for v in by_entry[e]) / len(by_entry[e]) for e in entries])

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax_kde = fig.add_subplot(gs[0, :])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    # Panel 1: pooled histogram + KDE
    bins = np.linspace(0, 1, 41)
    ax_kde.hist(all_vals, bins=bins, density=True, alpha=0.40, color=AZ_YELLOW, label="histogram")
    try:
        from scipy.stats import gaussian_kde

        xs = np.linspace(0, 1, 300)
        ax_kde.plot(xs, gaussian_kde(all_vals)(xs), color=AZ_PURPLE, linewidth=2, label="KDE")
    except Exception:  # noqa: BLE001
        pass
    for val, colour, label in THRESHOLDS:
        ax_kde.axvline(val, color=colour, linewidth=1, linestyle="--", label=label)
    ax_kde.set_xlabel("DockQ")
    ax_kde.set_ylabel("Density")
    ax_kde.set_title(f"{title} – pooled DockQ  (n={len(all_vals):,} decoys across {len(entries)} targets)")
    ax_kde.legend()

    # Panel 2: mean vs std scatter coloured by fraction ≥ 0.49
    rg_cmap = LinearSegmentedColormap.from_list("az_rg", [AZ_GREY, AZ_YELLOW, AZ_PURPLE])
    sc = ax_scatter.scatter(
        means,
        stds,
        c=frac_medium,
        cmap=rg_cmap,
        vmin=0,
        vmax=1,
        s=60,
        edgecolors=AZ_GREY,
        linewidths=0.5,
        zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax_scatter)
    cb.set_label("Fraction DockQ ≥ 0.49")
    ax_scatter.set_xlabel("Mean DockQ per target")
    ax_scatter.set_ylabel("Std DockQ per target")
    ax_scatter.set_title("Per-target mean vs std DockQ")
    ax_scatter.set_xlim(-0.02, 1.02)

    # Panel 3: fraction ≥ 0.49 per target (sorted descending)
    sort_idx = np.argsort(frac_medium)[::-1]
    ax_bar.bar(range(len(entries)), frac_medium[sort_idx], color=AZ_PURPLE, alpha=0.8)
    ax_bar.set_xticks(range(len(entries)))
    label_fs = max(4, min(8, 250 // max(len(entries), 1)))
    ax_bar.set_xticklabels([entries[i] for i in sort_idx], rotation=90, fontsize=label_fs)
    ax_bar.set_ylabel("Fraction DockQ ≥ 0.49")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("Fraction medium/high-quality decoys per target (sorted)")
    ax_bar.axhline(0.5, color=AZ_GREY, linewidth=0.8, linestyle="--")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved global summary → %s", output_path)


def _style_violin_parts(parts: dict) -> None:
    """Apply consistent styling to violin plot parts.

    Parameters
    ----------
    parts : dict
        Dictionary returned by ax.violinplot()
    """
    for pc in parts["bodies"]:
        pc.set_facecolor(AZ_PURPLE)
        pc.set_alpha(0.55)

    # Handle optional keys that may not be present depending on violin plot parameters
    if "cmedians" in parts:
        parts["cmedians"].set_color(AZ_YELLOW)
        parts["cmedians"].set_linewidth(2)
    if "cquantiles" in parts:
        parts["cquantiles"].set_color(AZ_NAVY)
        parts["cquantiles"].set_linewidth(1.5)
        parts["cquantiles"].set_linestyle("--")


def _configure_gif_frame_axes(
    ax,
    valid_entries: list[str],
    ticks: range,
    label_fs: int,
    frame_min: float,
    frame_max: float,
    y_margin: float,
    vert: bool,
) -> None:
    """Configure axes for a single GIF frame.

    Parameters
    ----------
    ax
        Matplotlib axes object
    valid_entries : list of str
        Entry labels for this frame
    ticks : range
        Tick positions
    label_fs : int
        Label font size
    frame_min, frame_max : float
        Data range for axis limits
    y_margin : float
        Margin for axis limits
    vert : bool
        True for vertical orientation, False for horizontal
    """
    if vert:
        ax.set_xticks(ticks)
        ax.set_xticklabels(valid_entries, rotation=45, ha="right", fontsize=label_fs)
        ax.set_ylabel("DockQ")
        ax.set_ylim(frame_min - y_margin, frame_max + y_margin)
        _add_threshold_lines(ax, frame_max, "vertical")
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(valid_entries, fontsize=label_fs)
        ax.set_xlabel("DockQ")
        ax.set_xlim(frame_min - y_margin, frame_max + y_margin)
        ax.invert_yaxis()
        _add_threshold_lines(ax, frame_max, "horizontal")


def _create_gif_frame(
    rows: list[dict],
    all_entries: list[str],
    msa_mode: str,
    primary: int,
    secondary: float | int,
    spec: dict,
    fig_w: float,
    fig_h: float,
    vert: bool,
    title: str,
) -> object | None:
    """Create a single GIF frame for a condition.

    Parameters
    ----------
    rows : list of dict
        All data rows
    all_entries : list of str
        All entry names (for consistent ordering)
    msa_mode : str
        MSA mode for this condition
    primary, secondary : int or float
        Grid condition values
    spec : dict
        Grid axis specification
    fig_w, fig_h : float
        Figure dimensions
    vert : bool
        Vertical orientation flag
    title : str
        Plot title prefix

    Returns
    -------
    PIL.Image or None
        Frame image, or None if no data for this condition
    """
    import io

    from PIL import Image

    cond_label = (
        f"msa={msa_mode}  {spec['primary_name']}={primary}  "
        f"{spec['secondary_name']}={_grid_secondary_text(secondary, spec)}"
    )
    cond_rows = [r for r in rows if _grid_condition_matches(r, msa_mode, primary, secondary, spec)]
    by_entry = collect_by_entry(cond_rows)
    valid_entries = [e for e in all_entries if by_entry.get(e)]
    valid_data = [by_entry[e] for e in valid_entries]
    nv = len(valid_entries)
    if nv == 0:
        return None

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    parts = ax.violinplot(
        valid_data,
        positions=range(1, nv + 1),
        vert=vert,
        showmedians=True,
        showextrema=False,
        quantiles=[[0.25, 0.75]] * nv,
    )
    _style_violin_parts(parts)

    all_frame_vals = [v for vals in valid_data for v in vals]
    frame_min = min(all_frame_vals) if all_frame_vals else 0.0
    frame_max = max(all_frame_vals) if all_frame_vals else 1.0
    y_margin = (frame_max - frame_min) * 0.02 if frame_max > frame_min else 0.02

    label_fs = max(5, min(9, 300 // nv))
    ticks = range(1, nv + 1)
    _configure_gif_frame_axes(ax, valid_entries, ticks, label_fs, frame_min, frame_max, y_margin, vert)

    ax.set_title(
        f"{title} – per-target DockQ distribution (n={nv} targets)\n"
        f"condition: {cond_label}\n"
        "yellow = median  |  navy dashed = Q1/Q3"
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    frame = Image.open(buf).copy()
    buf.close()
    return frame


def plot_violin_gif(
    rows: list[dict],
    output_path: Path,
    title: str = "DockQ",
    fps: float = 1.5,
) -> None:
    """Animated GIF cycling through per-condition per-target violin plots.

    Each frame corresponds to one unique (msa_mode, recycling_steps, step_scale)
    condition and uses the **same target ordering** as :func:`plot_violin`, so
    differences between conditions are easy to spot visually.  The condition is
    displayed as the figure title.

    Requires *Pillow* (``pip install pillow``).
    """
    if not rows or "msa_mode" not in rows[0]:
        LOGGER.warning("plot_violin_gif: no condition columns found – skipping")
        return

    spec = _infer_grid_axes(rows)
    if spec is None:
        LOGGER.warning("plot_violin_gif: no grid-axis columns found – skipping")
        return

    try:
        from PIL import Image
    except ImportError:
        LOGGER.warning("plot_violin_gif: pillow not installed – skipping GIF")
        return

    all_entries = sorted({r["entry"] for r in rows})
    n = len(all_entries)

    conditions = sorted({(r["msa_mode"], _grid_primary_value(r, spec), _grid_secondary_value(r, spec)) for r in rows})

    if n <= 40:  # noqa: PLR2004
        fig_w, fig_h, vert = max(8, n * 1.1), 6, True
    else:
        fig_w, fig_h, vert = 12, max(10, min(n * 0.3, 50)), False

    frames: list = []
    for msa_mode, primary, secondary in conditions:
        frame = _create_gif_frame(rows, all_entries, msa_mode, primary, secondary, spec, fig_w, fig_h, vert, title)
        if frame is not None:
            frames.append(frame)

    if not frames:
        LOGGER.warning("plot_violin_gif: no frames generated")
        return

    # Ensure all frames share the same canvas size
    target_size = frames[0].size
    frames = [f.resize(target_size, Image.LANCZOS) if f.size != target_size else f for f in frames]

    duration_ms = int(1000 / fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    LOGGER.info(
        "Saved violin GIF (%d frames @ %.1f fps) → %s",
        len(frames),
        fps,
        output_path,
    )


def _collect_condition_data(
    mode_rows: list[dict],
    mode: str,
    conditions: list[tuple],
    cond_labels: list[str],
    spec: dict,
) -> tuple[list[list[float]], list[str]]:
    """Collect DockQ values for each condition.

    Parameters
    ----------
    mode_rows : list of dict
        Rows filtered for a specific MSA mode
    mode : str
        MSA mode identifier
    conditions : list of tuples
        Grid conditions (primary, secondary)
    cond_labels : list of str
        Condition labels for display
    spec : dict
        Grid axis specification

    Returns
    -------
    tuple
        (data, valid_labels) - lists of values and labels for conditions with data
    """
    data: list[list[float]] = []
    valid_labels: list[str] = []
    for (primary, secondary), label in zip(conditions, cond_labels):
        seen: set[tuple] = set()
        vals: list[float] = []
        for r in mode_rows:
            if not _grid_condition_matches(r, mode, primary, secondary, spec):
                continue
            key = (r["entry"], r["seed"], r["sample"])
            if key in seen:
                continue
            seen.add(key)
            v = r.get("complex_dockq") or r.get("global_dockq")
            if v not in ("", None):
                vals.append(float(v))
        if vals:
            data.append(vals)
            valid_labels.append(label)
    return data, valid_labels


def _add_jittered_scatter(ax, data: list[list[float]], rng) -> None:
    """Add jittered scatter points over violin plots.

    Parameters
    ----------
    ax
        Matplotlib axes object
    data : list of lists
        Data for each violin
    rng
        Numpy random generator
    """
    for i, vals in enumerate(data, start=1):
        jitter = rng.uniform(-0.15, 0.15, min(len(vals), 2000))
        sample = vals if len(vals) <= 2000 else list(rng.choice(vals, 2000, replace=False))  # type: ignore[arg-type]
        ax.scatter(i + jitter, sample, color=AZ_NAVY, s=5, alpha=0.20, zorder=2)


def _configure_condition_violin_axes(
    ax,
    positions: list[int],
    valid_labels: list[str],
    spec: dict,
    col_idx: int,
) -> None:
    """Configure axes for condition violin subplot.

    Parameters
    ----------
    ax
        Matplotlib axes object
    positions : list of int
        X positions for violins
    valid_labels : list of str
        Condition labels
    spec : dict
        Grid axis specification
    col_idx : int
        Column index (0-based)
    """
    ax.set_xticks(positions)
    ax.set_xticklabels(valid_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(-0.02, 1.06)
    for val, colour, lbl in THRESHOLDS:
        ax.axhline(val, color=colour, linewidth=0.9, linestyle="--", label=lbl if col_idx == 0 else None)
    if col_idx == 0:
        ax.set_ylabel("DockQ")
        ax.legend(loc=LEGEND_LOC_LOWER_RIGHT, fontsize=7)
    ax.set_xlabel(
        f"condition  ({spec['primary_short']} = {spec['primary_name']}, "
        f"{spec['secondary_short']} = {spec['secondary_name']})"
    )


def plot_condition_violins(
    rows: list[dict],
    output_path: Path,
    title: str = "DockQ",
) -> None:
    """Compare DockQ distributions across grid conditions (recycling × step_scale).

    One violin per unique (recycling_steps, step_scale) combination, pooling
    global_dockq across all entries × samples for that condition.  When multiple
    msa_modes are present the plot is faceted into one column per mode.

    Requires rows to contain the grid-mode columns written by
    ``compute_dockq_scores.py`` (msa_mode, recycling_steps, step_scale).
    """
    spec = _infer_grid_axes(rows)
    if not rows or spec is None:
        LOGGER.warning("plot_condition_violins: no condition columns found – skipping")
        return

    msa_modes = sorted({r["msa_mode"] for r in rows})
    primary_vals, secondary_vals = _sorted_grid_conditions(rows, spec)

    conditions = [(primary, secondary) for primary in primary_vals for secondary in secondary_vals]
    cond_labels = [
        f"{spec['primary_short']}{primary} {spec['secondary_short']}{_grid_secondary_text(secondary, spec)}"
        for primary, secondary in conditions
    ]

    n_modes = len(msa_modes)
    fig, axes = plt.subplots(
        1,
        n_modes,
        figsize=(max(8, len(conditions) * 1.1 * n_modes + 2), 6),
        sharey=True,
        squeeze=False,
    )

    rng = np.random.default_rng(42)

    for col_idx, mode in enumerate(msa_modes):
        ax = axes[0][col_idx]
        mode_rows = [r for r in rows if r["msa_mode"] == mode]

        data, valid_labels = _collect_condition_data(mode_rows, mode, conditions, cond_labels, spec)

        if not data:
            ax.set_title(f"msa={mode}\n(no data)")
            continue

        n = len(data)
        positions = list(range(1, n + 1))
        parts = ax.violinplot(
            data,
            positions=positions,
            vert=True,
            showmedians=True,
            showextrema=False,
            quantiles=[[0.25, 0.75]] * n,
        )
        _style_violin_parts(parts)
        _add_jittered_scatter(ax, data, rng)

        _configure_condition_violin_axes(ax, positions, valid_labels, spec, col_idx)
        ax.set_title(f"msa={mode}", fontsize=10)

        # sample-count annotation above each violin
        for i, vals in enumerate(data, start=1):
            ax.text(i, 1.03, f"n={len(vals)}", ha="center", va="bottom", fontsize=6, color=AZ_GREY)

    fig.suptitle(f"{title} – DockQ distribution by grid condition", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved condition violin plot → %s", output_path)


def _deduplicate_diversity_rows(rows: list[dict], spec: dict | None, has_conditions: bool) -> list[dict]:
    """Deduplicate to one row per (condition × entry × interface).

    Parameters
    ----------
    rows : list of dict
        Input rows
    spec : dict or None
        Grid axis specification
    has_conditions : bool
        Whether condition columns are present

    Returns
    -------
    list of dict
        Deduplicated rows
    """
    seen: set = set()
    unique: list[dict] = []
    for r in rows:
        # Check for required keys
        if "entry" not in r or "native_interface" not in r:
            continue

        if has_conditions and spec is not None:
            primary_col = str(spec["primary_col"])
            secondary_col = str(spec["secondary_col"])
            primary_val = r.get(primary_col, "")
            secondary_val = r.get(secondary_col, "")
            condition_key = (r.get("msa_mode", ""), primary_val, secondary_val)
        else:
            condition_key = ("",)

        key = condition_key + (r["entry"], r["native_interface"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _setup_diversity_conditions(
    unique: list[dict], spec: dict | None, has_conditions: bool
) -> tuple[list[str], list[tuple], list[str], int, int]:
    """Set up MSA modes, conditions, and labels for diversity plot.

    Parameters
    ----------
    unique : list of dict
        Deduplicated rows
    spec : dict or None
        Grid axis specification
    has_conditions : bool
        Whether condition columns are present

    Returns
    -------
    tuple
        (msa_modes, conditions, cond_labels, n_modes, n_conds)
    """
    if has_conditions:
        msa_modes = sorted({r["msa_mode"] for r in unique})
        primary_vals, secondary_vals = _sorted_grid_conditions(unique, spec)
        conditions = [(primary, secondary) for primary in primary_vals for secondary in secondary_vals]
        cond_labels = [
            f"{spec['primary_short']}{primary} {spec['secondary_short']}{_grid_secondary_text(secondary, spec)}"
            for primary, secondary in conditions
        ]
        n_modes = len(msa_modes)
        n_conds = len(conditions)
    else:
        msa_modes = ["all"]
        conditions = [("", "")]
        cond_labels = ["all"]
        n_modes = 1
        n_conds = 1
    return msa_modes, conditions, cond_labels, n_modes, n_conds


def _style_diversity_violin_parts(parts: dict, mode_palette: dict, mode: str) -> None:
    """Style violin plot parts for diversity plots.

    Parameters
    ----------
    parts : dict
        Violin plot parts from ax.violinplot()
    mode_palette : dict
        Color palette for MSA modes
    mode : str
        Current MSA mode
    """
    colour = mode_palette.get(mode, AZ_SKY)
    for pc in parts["bodies"]:
        pc.set_facecolor(colour)
        pc.set_alpha(0.55)
    parts["cmedians"].set_color(AZ_PURPLE)
    parts["cmedians"].set_linewidth(2)
    parts["cquantiles"].set_color(AZ_NAVY)
    parts["cquantiles"].set_linewidth(1.5)
    parts["cquantiles"].set_linestyle("--")


def _add_diversity_scatter(ax, data: list[list[float]], rng) -> None:
    """Add scatter points to diversity violin plots.

    Parameters
    ----------
    ax
        Matplotlib axes
    data : list of lists
        Data for each violin
    rng
        Numpy random generator
    """
    for i, vals in enumerate(data, 1):
        n_pts = min(len(vals), 400)
        samp = vals[:n_pts] if len(vals) <= 400 else list(rng.choice(vals, n_pts, replace=False))  # type: ignore[arg-type]
        ax.scatter(
            i + rng.uniform(-0.15, 0.15, n_pts),
            samp,
            color=AZ_NAVY,
            s=5,
            alpha=0.30,
            zorder=2,
        )


def _collect_diversity_metric_data(
    mode_rows: list[dict],
    mode: str,
    conditions: list[tuple],
    cond_labels: list[str],
    spec: dict | None,
    has_conditions: bool,
    metric_col: str,
) -> tuple[list[list[float]], list[str]]:
    """Collect data for a specific diversity metric across conditions.

    Parameters
    ----------
    mode_rows : list of dict
        Rows for current MSA mode
    mode : str
        MSA mode identifier
    conditions : list of tuples
        Condition specifications
    cond_labels : list of str
        Condition labels
    spec : dict or None
        Grid axis specification
    has_conditions : bool
        Whether conditions are present
    metric_col : str
        Metric column name to collect

    Returns
    -------
    tuple
        (data, valid_labels) - metric values and labels for conditions with data
    """

    def _fv(row_list: list[dict], col: str) -> list[float]:
        return [float(r[col]) for r in row_list if r.get(col) not in ("", None)]

    data: list[list[float]] = []
    valid_labels: list[str] = []
    for (primary, secondary), label in zip(conditions, cond_labels):
        grp = (
            [r for r in mode_rows if _grid_condition_matches(r, mode, primary, secondary, spec)]
            if has_conditions
            else mode_rows
        )
        vals = _fv(grp, metric_col)
        if vals:
            data.append(vals)
            valid_labels.append(label)
    return data, valid_labels


def _plot_diversity_violin_panel(
    ax,
    data: list[list[float]],
    valid_labels: list[str],
    mode_palette: dict,
    mode: str,
    rng,
    ylabel: str,
    col_idx: int,
    has_conditions: bool,
    title: str,
) -> None:
    """Plot one diversity violin panel.

    Parameters
    ----------
    ax
        Matplotlib axes
    data : list of lists
        Metric data
    valid_labels : list of str
        Condition labels
    mode_palette : dict
        Color palette
    mode : str
        MSA mode
    rng
        Random generator
    ylabel : str
        Y-axis label
    col_idx : int
        Column index
    has_conditions : bool
        Whether conditions are present
    title : str
        Plot title
    """
    if data:
        n = len(data)
        parts = ax.violinplot(
            data,
            positions=range(1, n + 1),
            showmedians=True,
            showextrema=False,
            quantiles=[[0.25, 0.75]] * n,
        )
        _style_diversity_violin_parts(parts, mode_palette, mode)
        _add_diversity_scatter(ax, data, rng)

    ax.set_xticks(range(1, len(valid_labels) + 1))
    ax.set_xticklabels(valid_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel if col_idx == 0 else "")
    ax.set_title(f"msa={mode}" if has_conditions else title, fontsize=10)


def _plot_diversity_cluster_panel(
    ax,
    cluster_data: list[list[float]],
    valid_labels: list[str],
    col_idx: int,
    has_conditions: bool,
    title: str,
    mode: str | None = None,
) -> None:
    """Plot conformational cluster stacked bar panel.

    Parameters
    ----------
    ax
        Matplotlib axes
    cluster_data : list of lists
        Cluster count data
    valid_labels : list of str
        Condition labels
    col_idx : int
        Column index
    has_conditions : bool
        Whether conditions are present
    title : str
        Plot title
    mode : str, optional
        MSA mode for title
    """
    if cluster_data:
        max_k = int(max(v for vals in cluster_data for v in vals))
        bottoms = np.zeros(len(cluster_data))
        k_colors = [AZ_GREY, AZ_SKY, AZ_YELLOW, AZ_PINK, AZ_PURPLE]
        for k in range(1, max_k + 1):
            fracs = np.array([sum(1 for v in vals if round(v) == k) / len(vals) for vals in cluster_data])
            ax.bar(
                range(len(cluster_data)),
                fracs,
                bottom=bottoms,
                color=k_colors[(k - 1) % len(k_colors)],
                alpha=0.85,
                label=f"{k} cluster{'s' if k > 1 else ''}",
            )
            bottoms += fracs
        ax.set_xticks(range(len(valid_labels)))
        ax.set_xticklabels(valid_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Fraction of groups" if col_idx == 0 else "")
        ax.set_title(f"msa={mode}" if has_conditions and mode else title, fontsize=10)
        if col_idx == 0:
            ax.legend(fontsize=7, loc=LEGEND_LOC_UPPER_RIGHT)
        for i, vals in enumerate(cluster_data):
            ax.text(i, 1.03, f"n={len(vals)}", ha="center", va="bottom", fontsize=6, color=AZ_GREY)


def _plot_diversity_scatter_panel(
    ax,
    unique: list[dict],
    msa_modes: list[str],
    mode_palette: dict,
    has_conditions: bool,
    title: str,
) -> None:
    """Plot mean iRMSD vs std scatter panel.

    Parameters
    ----------
    ax
        Matplotlib axes
    unique : list of dict
        Deduplicated rows
    msa_modes : list of str
        MSA mode identifiers
    mode_palette : dict
        Color palette
    has_conditions : bool
        Whether conditions are present
    title : str
        Plot title
    """
    for mode in msa_modes:
        mode_rows = [r for r in unique if r.get("msa_mode", "all") == mode] if has_conditions else unique
        paired = [
            (float(r["irmsd_mean"]), float(r["irmsd_std"]))
            for r in mode_rows
            if r.get("irmsd_mean") not in ("", None) and r.get("irmsd_std") not in ("", None)
        ]
        if paired:
            xs, ys = zip(*paired)
            ax.scatter(
                xs,
                ys,
                s=10,
                alpha=0.30,
                color=mode_palette.get(mode, AZ_PURPLE),
                label=f"msa={mode}" if has_conditions else title,
                zorder=2,
            )
    ax.set_xlabel(f"Mean {LABEL_IRMSD_ANGSTROM}")
    ax.set_ylabel("iRMSD std (Å)")
    ax.set_title(
        "Mean iRMSD vs iRMSD std per (entry × interface × condition)\nupper-right = high error + high variability"
    )
    if has_conditions:
        ax.legend(markerscale=3, fontsize=8)


def plot_diversity(
    rows: list[dict],
    output_path: Path,
    title: str = "DockQ",
) -> None:
    """Three-row diversity summary of predicted ensembles across conditions.

    Requires rows to have been annotated by ``_compute_diversity_metrics``
    (columns ``irmsd_std``, ``irmsd_mean``, ``n_conf_clusters``).
    Silently skips if those columns are absent.

    Row 1 (per msa_mode facet): violin of **iRMSD std** per (recycling × scale)
        condition — how consistently each condition places the interface relative
        to native.  A narrow distribution means all 5 samples land at similar
        iRMSD values; a wide one indicates the ensemble spans different poses.

    Row 2 (per msa_mode facet): stacked bar of **conformational cluster counts**
        — the fraction of (entry × interface) groups whose 5 samples fall into
        1, 2, 3 … distinct iRMSD clusters (2 Å gap threshold).  More multi-cluster
        groups = more bimodal / multi-modal sampling.

    Row 3 (spanning full width): scatter of **mean iRMSD vs iRMSD std** per
        (condition × entry × interface) group, coloured by msa_mode.  Points in
        the upper-right quadrant have both high average error *and* high
        variability — the model is uncertain and exploring diverse poses.
    """
    if not rows or "irmsd_std" not in rows[0]:
        LOGGER.warning("plot_diversity: diversity columns not found – skipping")
        return

    spec = _infer_grid_axes(rows)
    has_conditions = "msa_mode" in rows[0] and spec is not None

    unique = _deduplicate_diversity_rows(rows, spec, has_conditions)
    msa_modes, conditions, cond_labels, n_modes, n_conds = _setup_diversity_conditions(unique, spec, has_conditions)

    from matplotlib.gridspec import GridSpec

    fig_w = max(10, n_conds * 1.3 * n_modes + 4)
    fig = plt.figure(figsize=(fig_w, 18))
    gs = GridSpec(4, n_modes, figure=fig, hspace=0.60, wspace=0.30, height_ratios=[2, 2, 2, 1.8])
    rng = np.random.default_rng(42)

    mode_palette = {m: c for m, c in zip(msa_modes, AZ_PALETTE)}

    for col_idx, mode in enumerate(msa_modes):
        mode_rows = [r for r in unique if r.get("msa_mode", "all") == mode] if has_conditions else unique

        # Row 0: DockQ std violin
        ax_dq = fig.add_subplot(gs[0, col_idx])
        dockq_data, valid_dq = _collect_diversity_metric_data(
            mode_rows, mode, conditions, cond_labels, spec, has_conditions, "dockq_std"
        )
        _plot_diversity_violin_panel(
            ax_dq, dockq_data, valid_dq, mode_palette, mode, rng, "DockQ std", col_idx, has_conditions, title
        )
        ax_dq.set_ylim(bottom=0)

        # Row 1: iRMSD std violin
        ax_a = fig.add_subplot(gs[1, col_idx])
        irmsd_data, valid_a = _collect_diversity_metric_data(
            mode_rows, mode, conditions, cond_labels, spec, has_conditions, "irmsd_std"
        )
        _plot_diversity_violin_panel(
            ax_a, irmsd_data, valid_a, mode_palette, mode, rng, "iRMSD std (Å)", col_idx, has_conditions, title
        )
        ax_a.axhline(0, color=AZ_GREY, linewidth=0.5, linestyle=":")

        # Row 2: Conformational clusters stacked bar
        ax_b = fig.add_subplot(gs[2, col_idx])
        cluster_data, valid_b = _collect_diversity_metric_data(
            mode_rows, mode, conditions, cond_labels, spec, has_conditions, "n_conf_clusters"
        )
        _plot_diversity_cluster_panel(ax_b, cluster_data, valid_b, col_idx, has_conditions, title, mode)

    # Row 3: mean iRMSD vs iRMSD std scatter (full width)
    ax_c = fig.add_subplot(gs[3, :])
    _plot_diversity_scatter_panel(ax_c, unique, msa_modes, mode_palette, has_conditions, title)

    # Row headers
    fig.text(0.01, 0.97, "A  DockQ spread", fontsize=10, fontweight="bold", color=AZ_PURPLE, va="top")
    fig.text(0.01, 0.73, "B  iRMSD spread", fontsize=10, fontweight="bold", color=AZ_NAVY, va="top")
    fig.text(0.01, 0.49, "C  Conformational clusters", fontsize=10, fontweight="bold", color=AZ_NAVY, va="top")
    fig.text(0.01, 0.25, "D  Mean iRMSD vs spread", fontsize=10, fontweight="bold", color=AZ_NAVY, va="top")

    fig.suptitle(f"{title} – Ensemble Diversity", fontsize=14, fontweight="bold", y=1.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved diversity plot → %s", output_path)


def _detect_metric_columns(rows: list[dict]) -> tuple[str, str, str, str]:
    """Detect whether to use complex-level or per-interface metrics.

    Parameters
    ----------
    rows : list of dict
        Data rows

    Returns
    -------
    tuple
        (dockq_col, irmsd_col, metric_type_dockq, metric_type_irmsd)
    """
    has_complex = rows and "complex_dockq" in rows[0]
    dockq_col = "complex_dockq" if has_complex else "DockQ"

    has_complex_irmsd = has_complex and any(
        r.get("complex_irmsd") not in ("", None) for r in rows[: min(100, len(rows))]
    )
    irmsd_col = "complex_irmsd" if has_complex_irmsd else "iRMSD"

    metric_type_dockq = "complex-level" if has_complex else "per-interface"
    metric_type_irmsd = "complex-level" if has_complex_irmsd else "per-interface"

    return dockq_col, irmsd_col, metric_type_dockq, metric_type_irmsd


def _plot_kde_curve(ax, xs: np.ndarray, vals: list[float], colour: str, label: str, gaussian_kde) -> None:
    """Plot a single KDE curve with fill.

    Parameters
    ----------
    ax
        Matplotlib axes
    xs : ndarray
        X values for KDE
    vals : list of float
        Data values
    colour : str
        Line/fill color
    label : str
        Legend label
    gaussian_kde
        KDE function from scipy
    """
    if len(vals) > 10:
        ys = gaussian_kde(vals, bw_method="scott")(xs)
        ax.plot(xs, ys, color=colour, linewidth=2.5, label=label)
        ax.fill_between(xs, ys, alpha=0.12, color=colour)


def plot_condition_kde(
    rows: list[dict],
    output_path: Path,
    title: str = "DockQ",
) -> None:
    """Overlaid KDE distributions of DockQ and iRMSD, one curve per msa_mode.

    Left panel: DockQ pooled across all samples under each msa_mode, with
        vertical reference lines at the three quality thresholds.
    Right panel: iRMSD pooled across all samples; 99th-percentile clipped to
        keep the x-axis readable.

    Both panels pool *all* recycling-step and step-scale conditions within each
    msa_mode, since the diversity analysis showed those hyperparameters have
    negligible effect on the distributions.

    Works with both complex-level (complex_dockq, complex_irmsd) and per-interface
    (DockQ, iRMSD) data. Prefers complex-level metrics when available.
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        LOGGER.warning("plot_condition_kde: scipy not available – skipping")
        return

    has_conditions = "msa_mode" in (rows[0] if rows else {})
    msa_modes = sorted({r["msa_mode"] for r in rows}) if has_conditions else ["all"]

    dockq_col, irmsd_col, metric_type_dockq, metric_type_irmsd = _detect_metric_columns(rows)

    fig, (ax_dq, ax_ir) = plt.subplots(1, 2, figsize=(14, 5))

    all_ir = [float(r[irmsd_col]) for r in rows if r.get(irmsd_col) not in ("", None)]
    ir_max = float(min(np.percentile(all_ir, 99), 50.0)) if all_ir else 30.0
    xs_dq = np.linspace(0.0, 1.0, 500)
    xs_ir = np.linspace(0.0, ir_max, 500)

    for i, mode in enumerate(msa_modes):
        colour = AZ_PALETTE[i % len(AZ_PALETTE)]
        mode_rows = [r for r in rows if r.get("msa_mode") == mode] if has_conditions else rows
        dq_vals = [float(r[dockq_col]) for r in mode_rows if r.get(dockq_col) not in ("", None)]
        ir_vals = [float(r[irmsd_col]) for r in mode_rows if r.get(irmsd_col) not in ("", None)]
        base_lbl = f"msa={mode}" if has_conditions else "all"

        _plot_kde_curve(ax_dq, xs_dq, dq_vals, colour, f"{base_lbl}  (n={len(dq_vals):,})", gaussian_kde)

        ir_clip = [v for v in ir_vals if v <= ir_max]
        _plot_kde_curve(ax_ir, xs_ir, ir_clip, colour, f"{base_lbl}  (n={len(ir_vals):,})", gaussian_kde)

    for val, colour, lbl in THRESHOLDS:
        ax_dq.axvline(val, color=colour, linewidth=1.2, linestyle="--", label=lbl)

    ax_dq.set_xlabel("DockQ")
    ax_dq.set_ylabel("Density")
    ax_dq.set_xlim(0.0, 1.0)
    dockq_subtitle = f" ({metric_type_dockq})" if metric_type_dockq == "complex-level" else ""
    ax_dq.set_title(f"{title} – DockQ distribution{dockq_subtitle} (all samples)")
    ax_dq.legend(fontsize=8)

    ax_ir.set_xlabel(LABEL_IRMSD_ANGSTROM)
    ax_ir.set_ylabel("Density")
    ax_ir.set_xlim(0.0, ir_max)
    irmsd_subtitle = f" ({metric_type_irmsd})" if metric_type_irmsd != metric_type_dockq else ""
    ax_ir.set_title(f"{title} – iRMSD distribution{irmsd_subtitle}  (clipped at 99th pct = {ir_max:.1f} Å)")
    ax_ir.legend(fontsize=8)

    fig.suptitle(f"{title} – Quality distributions by MSA mode", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved KDE plot → %s", output_path)


def _collect_component_metric_values(mode_rows: list[dict], metric_col: str) -> list[float]:
    """Collect float values for a metric column.

    Parameters
    ----------
    mode_rows : list of dict
        Rows for a specific MSA mode
    metric_col : str
        Metric column name

    Returns
    -------
    list of float
        Metric values
    """
    return [float(r[metric_col]) for r in mode_rows if r.get(metric_col) not in ("", None)]


def plot_dockq_components_kde(
    rows: list[dict],
    output_path: Path,
    title: str = "DockQ",
) -> None:
    """Overlaid KDE distributions of DockQ component metrics by MSA mode.

    Three-panel figure showing separate distributions for:
    - iRMSD (interface RMSD)
    - LRMSD (ligand RMSD)
    - fnat (fraction of native contacts)

    Each panel shows one curve per msa_mode, pooling across all
    recycling-step and step-scale conditions.
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        LOGGER.warning("plot_dockq_components_kde: scipy not available – skipping")
        return

    has_conditions = "msa_mode" in (rows[0] if rows else {})
    msa_modes = sorted({r["msa_mode"] for r in rows}) if has_conditions else ["all"]

    fig, (ax_ir, ax_lr, ax_fn) = plt.subplots(1, 3, figsize=(18, 5))

    # Determine plot ranges
    all_ir = _collect_component_metric_values(rows, "iRMSD")
    all_lr = _collect_component_metric_values(rows, "LRMSD")
    ir_max = float(min(np.percentile(all_ir, 99), 50.0)) if all_ir else 30.0
    lr_max = float(min(np.percentile(all_lr, 99), 50.0)) if all_lr else 30.0

    xs_ir = np.linspace(0.0, ir_max, 500)
    xs_lr = np.linspace(0.0, lr_max, 500)
    xs_fn = np.linspace(0.0, 1.0, 500)

    for i, mode in enumerate(msa_modes):
        colour = AZ_PALETTE[i % len(AZ_PALETTE)]
        mode_rows = [r for r in rows if r.get("msa_mode") == mode] if has_conditions else rows

        ir_vals = _collect_component_metric_values(mode_rows, "iRMSD")
        lr_vals = _collect_component_metric_values(mode_rows, "LRMSD")
        fn_vals = _collect_component_metric_values(mode_rows, "fnat")

        base_lbl = f"msa={mode}" if has_conditions else "all"

        # iRMSD panel
        ir_clip = [v for v in ir_vals if v <= ir_max]
        _plot_kde_curve(ax_ir, xs_ir, ir_clip, colour, f"{base_lbl}  (n={len(ir_vals):,})", gaussian_kde)

        # LRMSD panel
        lr_clip = [v for v in lr_vals if v <= lr_max]
        _plot_kde_curve(ax_lr, xs_lr, lr_clip, colour, f"{base_lbl}  (n={len(lr_vals):,})", gaussian_kde)

        # fnat panel
        _plot_kde_curve(ax_fn, xs_fn, fn_vals, colour, f"{base_lbl}  (n={len(fn_vals):,})", gaussian_kde)

    # Style iRMSD panel
    ax_ir.set_xlabel(LABEL_IRMSD_ANGSTROM)
    ax_ir.set_ylabel("Density")
    ax_ir.set_xlim(0.0, ir_max)
    ax_ir.set_title(f"Interface RMSD\n(clipped at 99th pct = {ir_max:.1f} Å)")
    ax_ir.legend(fontsize=8)

    # Style LRMSD panel
    ax_lr.set_xlabel("LRMSD (Å)")
    ax_lr.set_ylabel("Density")
    ax_lr.set_xlim(0.0, lr_max)
    ax_lr.set_title(f"Ligand RMSD\n(clipped at 99th pct = {lr_max:.1f} Å)")
    ax_lr.legend(fontsize=8)

    # Style fnat panel
    ax_fn.set_xlabel("fnat")
    ax_fn.set_ylabel("Density")
    ax_fn.set_xlim(0.0, 1.0)
    ax_fn.set_title("Fraction of Native Contacts")
    ax_fn.legend(fontsize=8)

    fig.suptitle(f"{title} – DockQ Component Distributions by MSA Mode", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved DockQ components KDE plot → %s", output_path)


def _setup_interface_type_axes(n_interfaces: int) -> tuple:
    """Set up subplot grid for interface type plots.

    Parameters
    ----------
    n_interfaces : int
        Number of interface types

    Returns
    -------
    tuple
        (fig, axes_list, n_rows, n_cols)
    """
    n_cols = min(3, n_interfaces)
    n_rows = (n_interfaces + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_interfaces == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten() if n_interfaces > 1 else [axes]

    return fig, axes_list, n_rows, n_cols


def _plot_interface_type_panel(
    ax,
    rows: list[dict],
    iface_type: str,
    msa_modes: list[str],
    has_conditions: bool,
    xs_dq: np.ndarray,
    gaussian_kde,
    ax_idx: int,
) -> None:
    """Plot KDE curves for one interface type panel.

    Parameters
    ----------
    ax
        Matplotlib axes
    rows : list of dict
        All data rows
    iface_type : str
        Interface type identifier
    msa_modes : list of str
        MSA mode identifiers
    has_conditions : bool
        Whether conditions are present
    xs_dq : ndarray
        X values for KDE
    gaussian_kde
        KDE function from scipy
    ax_idx : int
        Axes index (for legend control)
    """
    for i, mode in enumerate(msa_modes):
        colour = AZ_PALETTE[i % len(AZ_PALETTE)]

        iface_rows = [
            r
            for r in rows
            if r.get("native_interface") == iface_type and (not has_conditions or r.get("msa_mode") == mode)
        ]

        dq_vals = [float(r["DockQ"]) for r in iface_rows if r.get("DockQ") not in ("", None)]
        base_lbl = f"msa={mode}" if has_conditions else "all"

        _plot_kde_curve(ax, xs_dq, dq_vals, colour, f"{base_lbl}  (n={len(dq_vals):,})", gaussian_kde)

    # Add DockQ thresholds
    for val, colour, lbl in THRESHOLDS:
        ax.axvline(val, color=colour, linewidth=1.2, linestyle="--", label=lbl if ax_idx == 0 else None)

    ax.set_xlabel("DockQ")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"Interface: {iface_type}")
    ax.legend(fontsize=8)


def plot_dockq_by_interface_type(
    rows: list[dict],
    output_path: Path,
    title: str = "DockQ",
) -> None:
    """KDE distributions of DockQ by interface type and MSA mode.

    Creates separate panels for different interface types (e.g., H-L, H-A, L-A
    for antibody-antigen complexes), with one curve per MSA mode in each panel.
    Automatically detects interface types from the data.
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        LOGGER.warning("plot_dockq_by_interface_type: scipy not available – skipping")
        return

    if not rows or "native_interface" not in rows[0]:
        LOGGER.warning("plot_dockq_by_interface_type: no native_interface column – skipping")
        return

    has_conditions = "msa_mode" in (rows[0] if rows else {})
    msa_modes = sorted({r["msa_mode"] for r in rows}) if has_conditions else ["all"]

    interface_types = sorted({r["native_interface"] for r in rows if r.get("native_interface")})
    n_interfaces = len(interface_types)

    if n_interfaces == 0:
        LOGGER.warning("plot_dockq_by_interface_type: no interface types found – skipping")
        return

    fig, axes, _, _ = _setup_interface_type_axes(n_interfaces)
    xs_dq = np.linspace(0.0, 1.0, 500)

    for ax_idx, iface_type in enumerate(interface_types):
        _plot_interface_type_panel(
            axes[ax_idx], rows, iface_type, msa_modes, has_conditions, xs_dq, gaussian_kde, ax_idx
        )

    # Hide unused subplots
    for ax_idx in range(n_interfaces, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(f"{title} – DockQ Distribution by Interface Type", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved interface-type DockQ plot → %s", output_path)


def plot_temporal_split_kde(
    rows: list[dict],
    output_path: Path,
    title: str = "DockQ",
    temporal_col: str = "temporal_split",
) -> None:
    """KDE distributions comparing performance on different temporal splits.

    Creates overlay KDE plots showing DockQ and component metrics for
    structures deposited before vs after a training cutoff date.

    Requires a 'temporal_split' column in the rows data.
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        LOGGER.warning("plot_temporal_split_kde: scipy not available – skipping")
        return

    import pandas as pd

    df = pd.DataFrame(rows)

    # Check if temporal split column exists
    if temporal_col not in df.columns:
        LOGGER.warning(f"plot_temporal_split_kde: column '{temporal_col}' not found – skipping")
        return

    # Filter out Unknown
    df_clean = df[df[temporal_col].isin(["Pre-Oct2021", "Post-Oct2021"])]

    if len(df_clean) == 0:
        LOGGER.warning("plot_temporal_split_kde: no valid temporal split data – skipping")
        return

    # Create 2x2 grid: DockQ, iRMSD, LRMSD, fnat
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_dq, ax_ir, ax_lr, ax_fn = axes.flatten()

    # Prepare data ranges
    all_ir = [float(r["iRMSD"]) for r in rows if r.get("iRMSD") not in ("", None)]
    all_lr = [float(r["LRMSD"]) for r in rows if r.get("LRMSD") not in ("", None)]
    ir_max = float(min(np.percentile(all_ir, 99), 50.0)) if all_ir else 30.0
    lr_max = float(min(np.percentile(all_lr, 99), 50.0)) if all_lr else 30.0

    xs_dq = np.linspace(0.0, 1.0, 500)
    xs_ir = np.linspace(0.0, ir_max, 500)
    xs_lr = np.linspace(0.0, lr_max, 500)
    xs_fn = np.linspace(0.0, 1.0, 500)

    # Color scheme
    colors = {"Pre-Oct2021": AZ_NAVY, "Post-Oct2021": AZ_PINK}

    for split in ["Pre-Oct2021", "Post-Oct2021"]:
        split_df = df_clean[df_clean[temporal_col] == split]
        color = colors[split]

        # DockQ
        dq_vals = split_df["DockQ"].dropna().astype(float).values
        if len(dq_vals) > 10:
            ys = gaussian_kde(dq_vals, bw_method="scott")(xs_dq)
            ax_dq.plot(xs_dq, ys, color=color, linewidth=2.5, label=f"{split} (n={len(dq_vals):,})")
            ax_dq.fill_between(xs_dq, ys, alpha=0.15, color=color)

        # iRMSD
        ir_vals = split_df["iRMSD"].dropna().astype(float).values
        if len(ir_vals) > 10:
            ir_clip = ir_vals[ir_vals <= ir_max]
            ys = gaussian_kde(ir_clip, bw_method="scott")(xs_ir)
            ax_ir.plot(xs_ir, ys, color=color, linewidth=2.5, label=f"{split} (n={len(ir_vals):,})")
            ax_ir.fill_between(xs_ir, ys, alpha=0.15, color=color)

        # LRMSD
        lr_vals = split_df["LRMSD"].dropna().astype(float).values
        if len(lr_vals) > 10:
            lr_clip = lr_vals[lr_vals <= lr_max]
            ys = gaussian_kde(lr_clip, bw_method="scott")(xs_lr)
            ax_lr.plot(xs_lr, ys, color=color, linewidth=2.5, label=f"{split} (n={len(lr_vals):,})")
            ax_lr.fill_between(xs_lr, ys, alpha=0.15, color=color)

        # fnat
        fn_vals = split_df["fnat"].dropna().astype(float).values
        if len(fn_vals) > 10:
            ys = gaussian_kde(fn_vals, bw_method="scott")(xs_fn)
            ax_fn.plot(xs_fn, ys, color=color, linewidth=2.5, label=f"{split} (n={len(fn_vals):,})")
            ax_fn.fill_between(xs_fn, ys, alpha=0.15, color=color)

    # Style DockQ panel
    for val, colour, lbl in THRESHOLDS:
        ax_dq.axvline(val, color=colour, linewidth=1.2, linestyle="--", alpha=0.6)
    ax_dq.set_xlabel("DockQ")
    ax_dq.set_ylabel("Density")
    ax_dq.set_xlim(0.0, 1.0)
    ax_dq.set_title("DockQ Distribution")
    ax_dq.legend(fontsize=9)

    # Style iRMSD panel
    ax_ir.set_xlabel(LABEL_IRMSD_ANGSTROM)
    ax_ir.set_ylabel("Density")
    ax_ir.set_xlim(0.0, ir_max)
    ax_ir.set_title(f"Interface RMSD (clipped at 99th pct = {ir_max:.1f} Å)")
    ax_ir.legend(fontsize=9)

    # Style LRMSD panel
    ax_lr.set_xlabel("LRMSD (Å)")
    ax_lr.set_ylabel("Density")
    ax_lr.set_xlim(0.0, lr_max)
    ax_lr.set_title(f"Ligand RMSD (clipped at 99th pct = {lr_max:.1f} Å)")
    ax_lr.legend(fontsize=9)

    # Style fnat panel
    ax_fn.set_xlabel("fnat")
    ax_fn.set_ylabel("Density")
    ax_fn.set_xlim(0.0, 1.0)
    ax_fn.set_title("Fraction of Native Contacts")
    ax_fn.legend(fontsize=9)

    # Calculate summary statistics
    pre_dq = df_clean[df_clean[temporal_col] == "Pre-Oct2021"]["DockQ"].dropna().astype(float)
    post_dq = df_clean[df_clean[temporal_col] == "Post-Oct2021"]["DockQ"].dropna().astype(float)

    stats_text = f"Pre-Oct2021: mean={pre_dq.mean():.3f}, median={pre_dq.median():.3f}\n"
    stats_text += f"Post-Oct2021: mean={post_dq.mean():.3f}, median={post_dq.median():.3f}"

    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3),
    )

    fig.suptitle(
        f"{title} – Performance by Training Cutoff\n"
        f"Pre-Oct2021: Training-era structures | Post-Oct2021: Novel structures",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved temporal split KDE plot → %s", output_path)


def _validate_failure_mode_data(rows: list[dict]) -> tuple[bool, list[str]]:
    """Validate required columns for failure mode analysis."""
    required_cols = [
        "binder_ca_rmsd_mean",
        "antigen_ca_rmsd_mean",
        "interface_contact_f1_mean",
        "fnat",
        "iRMSD",
        "DockQ",
    ]
    has_cols = all(col in (rows[0] if rows else {}) for col in required_cols)
    return has_cols, required_cols


def _prepare_failure_mode_dataframe(rows: list[dict], required_cols: list[str]):
    """Prepare and clean dataframe for failure mode analysis."""
    import pandas as pd

    df = pd.DataFrame(rows)
    has_conditions = "msa_mode" in df.columns
    msa_modes = sorted(df["msa_mode"].unique()) if has_conditions else ["all"]

    # Convert to numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter out rows with missing data
    df_clean = df.dropna(subset=required_cols).copy()

    return df_clean, has_conditions, msa_modes


def _add_scatter_for_modes(ax, df_clean, msa_modes, has_conditions, x_col, y_col):
    """Add scatter plots for each MSA mode."""
    for i, mode in enumerate(msa_modes):
        colour = AZ_PALETTE[i % len(AZ_PALETTE)]
        mode_df = df_clean[df_clean["msa_mode"] == mode] if has_conditions else df_clean

        ax.scatter(
            mode_df[x_col],
            mode_df[y_col],
            alpha=0.3,
            s=10,
            color=colour,
            label=f"msa={mode}" if has_conditions else "all",
        )


def _plot_panel_a(ax, df_clean, msa_modes, has_conditions):
    """Plot Panel A: Binder CA RMSD vs fnat."""
    _add_scatter_for_modes(ax, df_clean, msa_modes, has_conditions, "binder_ca_rmsd_mean", "fnat")

    ax.axvline(2.0, color=AZ_GREY, linestyle="--", linewidth=1, alpha=0.6, label="2Å threshold")
    ax.axhline(0.5, color=AZ_GREY, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Binder CA RMSD (Å, binder-aligned)")
    ax.set_ylabel("fnat (Fraction Native Contacts)")
    ax.set_title(
        "Binder Structure vs Contact Recovery\n"
        + "Bottom-left: Good structure, wrong paratope\n"
        + "Top-left: Good structure, correct contacts"
    )
    ax.legend(fontsize=7)
    ax.set_xlim(0, min(df_clean["binder_ca_rmsd_mean"].quantile(0.99), 20))


def _plot_panel_b(ax, df_clean, msa_modes, has_conditions):
    """Plot Panel B: Antigen CA RMSD vs interface contact F1."""
    _add_scatter_for_modes(ax, df_clean, msa_modes, has_conditions, "antigen_ca_rmsd_mean", "interface_contact_f1_mean")

    ax.axvline(2.0, color=AZ_GREY, linestyle="--", linewidth=1, alpha=0.6, label="2Å threshold")
    ax.axhline(0.5, color=AZ_GREY, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Antigen CA RMSD (Å, antigen-aligned)")
    ax.set_ylabel("Interface Contact F1")
    ax.set_title(
        "Antigen Structure vs Interface Contacts\n"
        + "Bottom-left: Good antigen structure, wrong epitope\n"
        + "Top-left: Good antigen structure, correct interface"
    )
    ax.legend(fontsize=7)
    ax.set_xlim(0, min(df_clean["antigen_ca_rmsd_mean"].quantile(0.99), 20))


def _compute_monomer_quality(df_clean) -> str:
    """Compute monomer quality metric and return xlabel."""
    import pandas as pd

    if "complex_ca_rmsd_mean" in df_clean.columns:
        df_clean["monomer_quality"] = pd.to_numeric(df_clean["complex_ca_rmsd_mean"], errors="coerce")
        xlabel = "Monomer CA RMSD (Å)\n(mean of per-chain local RMSD, each chain self-aligned)"
    else:
        df_clean["monomer_quality"] = (df_clean["binder_ca_rmsd_mean"] + df_clean["antigen_ca_rmsd_mean"]) / 2
        xlabel = "Combined CA RMSD (Å)\n(average of binder-aligned + antigen-aligned)"

    return xlabel


def _add_panel_c_quadrant_annotations(ax, xlim, ylim):
    """Add quadrant annotations to Panel C."""
    # Bottom-left: Success
    ax.text(
        0.5,
        0.5,
        "SUCCESS\nGood monomers\nCorrect docking",
        fontsize=9,
        color="darkgreen",
        ha="center",
        va="bottom",
        fontweight="bold",
        bbox=dict(boxstyle=BBOX_STYLE_ROUNDED, facecolor="lightgreen", alpha=0.4),
    )
    # Top-left: Docking failure
    ax.text(
        0.5,
        ylim[1] * 0.95,
        "DOCKING FAILURE\nGood monomers\nWrong epitope/paratope",
        fontsize=9,
        color=AZ_NAVY,
        ha="center",
        va="top",
        fontweight="bold",
        bbox=dict(boxstyle=BBOX_STYLE_ROUNDED, facecolor="yellow", alpha=0.4),
    )
    # Top-right: Folding failure
    ax.text(
        xlim[1] * 0.95,
        ylim[1] * 0.95,
        "FOLDING FAILURE\nBad monomers\nBad interface",
        fontsize=9,
        color="darkred",
        ha="right",
        va="top",
        fontweight="bold",
        bbox=dict(boxstyle=BBOX_STYLE_ROUNDED, facecolor="lightcoral", alpha=0.4),
    )
    # Bottom-right: Lucky docking
    ax.text(
        xlim[1] * 0.95,
        0.5,
        "LUCKY DOCKING\nBad monomers\nCorrect interface",
        fontsize=8,
        color=AZ_NAVY,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightskyblue", alpha=0.4),
    )


def _plot_panel_c(ax, df_clean, msa_modes, has_conditions) -> bool:
    """Plot Panel C: Monomer quality vs interface RMSD. Returns False if no data."""
    xlabel = _compute_monomer_quality(df_clean)

    df_clean = df_clean.dropna(subset=["monomer_quality", "iRMSD"])
    if len(df_clean) == 0:
        return False

    _add_scatter_for_modes(ax, df_clean, msa_modes, has_conditions, "monomer_quality", "iRMSD")

    # Add quadrant lines
    ax.axvline(2.0, color=AZ_GREY, linestyle="--", linewidth=1, alpha=0.6, label="2Å thresholds")
    ax.axhline(2.0, color=AZ_GREY, linestyle="--", linewidth=1, alpha=0.6)

    xlim = ax.get_xlim()
    ylim = (0, min(df_clean["iRMSD"].quantile(0.99), 20))
    ax.set_ylim(ylim)

    _add_panel_c_quadrant_annotations(ax, xlim, ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Interface RMSD (Å)\n(antigen-aligned, measures docking accuracy)")
    ax.set_title("Monomer Quality vs Interface Placement\n" + "Disentangles folding accuracy from docking accuracy")
    ax.legend(fontsize=7, loc=LEGEND_LOC_UPPER_LEFT)

    return True


def _classify_failure_mode(row) -> str:
    """Classify a prediction into a failure mode."""
    good_monomers = row["monomer_quality"] < 2.0
    correct_interface = row["iRMSD"] < 2.0

    if good_monomers:
        return FAILURE_MODE_SUCCESS if correct_interface else FAILURE_MODE_DOCKING
    else:
        return FAILURE_MODE_LUCKY if correct_interface else FAILURE_MODE_FOLDING


def _get_failure_mode_colors() -> dict:
    """Get color mapping for failure modes."""
    return {
        FAILURE_MODE_SUCCESS: AZ_GREEN,
        FAILURE_MODE_DOCKING: AZ_YELLOW,
        FAILURE_MODE_FOLDING: AZ_PINK,
        FAILURE_MODE_LUCKY: AZ_SKY,
    }


def _plot_panel_d_with_conditions(ax, df_clean, msa_modes):
    """Plot Panel D with MSA mode conditions."""
    mode_counts = df_clean.groupby(["msa_mode", "failure_mode"]).size().unstack(fill_value=0)
    mode_fracs = mode_counts.div(mode_counts.sum(axis=1), axis=0)

    x = np.arange(len(msa_modes))
    width = 0.2
    colors = _get_failure_mode_colors()
    failure_modes = [FAILURE_MODE_SUCCESS, FAILURE_MODE_DOCKING, FAILURE_MODE_FOLDING, FAILURE_MODE_LUCKY]

    for i, failure_mode in enumerate(failure_modes):
        if failure_mode in mode_fracs.columns:
            vals = [mode_fracs.loc[mode, failure_mode] if mode in mode_fracs.index else 0 for mode in msa_modes]
            ax.bar(x + i * width, vals, width, label=failure_mode, color=colors.get(failure_mode, AZ_GREY), alpha=0.8)

    ax.set_xlabel("MSA Mode")
    ax.set_ylabel("Fraction of Predictions")
    ax.set_title("Failure Mode Distribution by MSA Mode\n(2Å thresholds for monomer CA RMSD and interface RMSD)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(msa_modes, rotation=15, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)


def _plot_panel_d_without_conditions(ax, df_clean):
    """Plot Panel D without MSA mode conditions."""
    counts = df_clean["failure_mode"].value_counts()
    colors = _get_failure_mode_colors()
    colors_list = [colors.get(mode, AZ_GREY) for mode in counts.index]

    ax.bar(range(len(counts)), counts.values, color=colors_list, alpha=0.8)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Failure Mode Distribution\n(2Å thresholds)")


def _plot_panel_d(ax, df_clean, msa_modes, has_conditions):
    """Plot Panel D: Failure mode distribution."""
    df_clean["failure_mode"] = df_clean.apply(_classify_failure_mode, axis=1)

    if has_conditions:
        _plot_panel_d_with_conditions(ax, df_clean, msa_modes)
    else:
        _plot_panel_d_without_conditions(ax, df_clean)


def plot_failure_mode_analysis(
    rows: list[dict],
    output_path: Path,
    title: str = "DockQ",
) -> None:
    """Diagnose failure modes: monomer structure quality vs interface placement.

    Creates scatter plots to distinguish:
    - Good monomers + Good interface → Success
    - Good monomers + Bad interface → Wrong epitope/paratope (docking failure)
    - Bad monomers + Bad interface → Structure prediction failure
    - Bad monomers + Good interface → Lucky docking

    Alignment strategy for clarity:
    - **Monomer accuracy**: Each chain aligned to itself independently
      (complex_ca_rmsd_mean = mean of per-chain local RMSD)
    - **Interface accuracy**: Antigen-aligned iRMSD - wrong epitope shows up
      as high iRMSD even when antigen structure is good

    Panel A: Binder monomer quality (binder-aligned CA RMSD) vs fnat
    Panel B: Antigen monomer quality (antigen-aligned CA RMSD) vs interface contact F1
    Panel C: Combined monomer quality vs interface RMSD (antigen-aligned for iRMSD)
    Panel D: Failure mode distribution by MSA mode

    Requires columns: binder_ca_rmsd_mean, antigen_ca_rmsd_mean,
    complex_ca_rmsd_mean (preferred), interface_contact_f1_mean, fnat, iRMSD, DockQ
    """
    has_cols, required_cols = _validate_failure_mode_data(rows)
    if not has_cols:
        LOGGER.warning("plot_failure_mode_analysis: missing required columns – skipping")
        return

    df_clean, has_conditions, msa_modes = _prepare_failure_mode_dataframe(rows, required_cols)

    if len(df_clean) == 0:
        LOGGER.warning("plot_failure_mode_analysis: no valid data – skipping")
        return

    # Create 2x2 grid of diagnostic plots
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_panel_a(ax_a, df_clean, msa_modes, has_conditions)

    ax_b = fig.add_subplot(gs[0, 1])
    _plot_panel_b(ax_b, df_clean, msa_modes, has_conditions)

    ax_c = fig.add_subplot(gs[1, 0])
    if not _plot_panel_c(ax_c, df_clean, msa_modes, has_conditions):
        LOGGER.warning("plot_failure_mode_analysis: no valid data for panel C – skipping")
        plt.close(fig)
        return

    ax_d = fig.add_subplot(gs[1, 1])
    _plot_panel_d(ax_d, df_clean, msa_modes, has_conditions)

    fig.suptitle(
        f"{title} – Failure Mode Analysis: Monomer Quality vs Interface Placement", fontsize=14, fontweight="bold"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved failure mode analysis → %s", output_path)


def _get_msa_label(mode: str) -> str:
    """Convert MSA mode to display label."""
    return {
        "antigen_only": "Antigen-only\nMSA",
        "full_msa": "Full\nMSA",
        "no_msa": "No\nMSA",
    }.get(mode, mode.replace("_", " ").title())


def _prepare_msa_modes(rows: list[dict]) -> tuple[list[str], dict[str, str]]:
    """Extract and order MSA modes from rows.

    Returns
    -------
    tuple
        (msa_modes, msa_labels)
    """
    preferred_msa_order = ["antigen_only", "full_msa", "no_msa"]
    msa_found = sorted({str(r.get("msa_mode", "")) for r in rows if str(r.get("msa_mode", "")).strip()})
    msa_modes = [m for m in preferred_msa_order if m in msa_found] + [
        m for m in msa_found if m not in preferred_msa_order
    ]
    if not msa_modes:
        msa_modes = ["all"]
    msa_labels = {m: _get_msa_label(m) for m in msa_modes}
    return msa_modes, msa_labels


def _calculate_irmsd_clip(rows: list[dict]) -> float:
    """Calculate 99th percentile iRMSD clip value."""
    ir_all = []
    for r in rows:
        try:
            ir_all.append(float(r["iRMSD"]))
        except (ValueError, KeyError, TypeError):
            pass
    return float(np.percentile(ir_all, 99)) if ir_all else 30.0


def _compute_bracket_positions(
    primary_values: list[int],
    n_secondary: int,
    gs_left: float,
    stride: float,
    axes_w: float,
) -> list[tuple[int, float, float]]:
    """Compute x-positions for recycling group brackets."""
    positions = []
    for gi, primary in enumerate(primary_values):
        cs = gi * n_secondary
        x_l = gs_left + cs * stride
        x_r = gs_left + (cs + n_secondary - 1) * stride + axes_w
        positions.append((primary, x_l, x_r))
    return positions


def _normalize_axes_array(axes, n_rows: int, n_cols: int):
    """Normalize axes array to 2D regardless of subplot dimensions."""
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    elif n_rows == 1:
        return axes.reshape(1, -1)
    elif n_cols == 1:
        return axes.reshape(-1, 1)
    return axes


def _plot_kde_cell(
    ax,
    vals: list[float],
    x_min: float,
    x_max: float,
    color: str,
    metric: str,
    gaussian_kde,
) -> None:
    """Plot KDE curve in a single grid cell."""
    xs = np.linspace(x_min, x_max, 300)
    if len(vals) > 10:
        ys = gaussian_kde(vals, bw_method="scott")(xs)
        ax.fill_between(xs, ys, alpha=0.35, color=color)
        ax.plot(xs, ys, color=color, lw=2.0)

    if metric == "dockq":
        for thresh, col, _ in THRESHOLDS:
            ax.axvline(thresh, color=col, ls="--", lw=1.2, alpha=0.9)


def _configure_grid_cell_axes(
    ax,
    ri: int,
    ci: int,
    n_rows: int,
    msa: str,
    msa_labels: dict,
    xlabel: str,
    metric: str,
    x_min: float,
    x_max: float,
) -> None:
    """Configure axes styling for a single grid cell."""
    ax.set_xlim(x_min, x_max)
    ax.tick_params(axis="both", labelsize=8, length=3, pad=2)
    ax.set_xticks([0, 0.5, 1.0] if metric == "dockq" else np.linspace(0, x_max, 4).round(0))

    if ri < n_rows - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(xlabel, fontsize=9, labelpad=3)

    if ci > 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Density", fontsize=9, labelpad=3)
        ax.annotate(
            msa_labels[msa],
            xy=(-0.38, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            rotation=90,
        )


def _add_recycling_group_labels(
    fig,
    bracket_positions: list[tuple[int, float, float]],
    y_text: float,
    y_line: float,
    spec: dict,
) -> None:
    """Add recycling group labels and bracket lines."""
    from matplotlib.lines import Line2D

    for primary, x_l, x_r in bracket_positions:
        x_mid = (x_l + x_r) / 2
        fig.text(
            x_mid,
            y_text,
            f"{str(spec['primary_name']).title()} = {primary}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            transform=fig.transFigure,
        )
        fig.add_artist(
            Line2D(
                [x_l, x_r],
                [y_line, y_line],
                transform=fig.transFigure,
                color=AZ_GREY,
                lw=2.0,
                clip_on=False,
            )
        )


def _add_scale_labels(
    fig,
    columns: list[tuple],
    gs_left: float,
    stride: float,
    axes_w: float,
    y_line: float,
    spec: dict,
) -> None:
    """Add scale labels on the bracket line."""
    for ci, (_, secondary) in enumerate(columns):
        x_col = gs_left + ci * stride + axes_w / 2
        fig.text(
            x_col,
            y_line + 0.003,
            f"{str(spec['secondary_name']).title()} = {_grid_secondary_text(secondary, spec)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            transform=fig.transFigure,
        )


def _add_threshold_legend(fig, metric: str) -> None:
    """Add threshold legend for DockQ plots."""
    if metric == "dockq":
        handles = [plt.Line2D([0], [0], color=col, ls="--", lw=1.5, label=lbl) for _, col, lbl in THRESHOLDS]
        fig.legend(
            handles=handles,
            loc=LEGEND_LOC_UPPER_RIGHT,
            bbox_to_anchor=(0.995, 0.995),
            fontsize=9,
            framealpha=0.9,
            title="Thresholds",
            title_fontsize=9,
        )


def _extract_dockq_values(cond_rows: list[dict], _clip: float) -> tuple[list[float], float, float]:
    """Extract DockQ values from condition rows."""
    vals = []
    for r in cond_rows:
        try:
            v = float(r["DockQ"])
            if 0.0 <= v <= 1.0:
                vals.append(v)
        except (ValueError, KeyError, TypeError):
            pass
    return vals, 0.0, 1.0


def _extract_irmsd_values(cond_rows: list[dict], clip: float) -> tuple[list[float], float, float]:
    """Extract iRMSD values from condition rows."""
    vals = []
    for r in cond_rows:
        try:
            v = float(r["iRMSD"])
            if 0.0 <= v <= clip:
                vals.append(v)
        except (ValueError, KeyError, TypeError):
            pass
    return vals, 0.0, clip


def _draw_kde_grid(
    rows: list[dict],
    msa_modes: list[str],
    msa_labels: dict,
    columns: list[tuple],
    spec: dict,
    metric: str,
    color: str,
    x_vals_fn,
    xlabel: str,
    suptitle: str,
    irmsd_clip: float,
    gaussian_kde,
) -> plt.Figure:
    """Draw a single KDE grid (either DockQ or iRMSD)."""
    n_rows, n_cols = len(msa_modes), len(columns)

    # Layout constants
    gs_left, gs_right = 0.07, 0.99
    gs_top, gs_bottom = 0.76, 0.08
    gs_wspace, gs_hspace = 0.25, 0.55

    # Compute layout metrics
    total_w = gs_right - gs_left
    axes_w = total_w / (n_cols + (n_cols - 1) * gs_wspace)
    stride = axes_w * (1 + gs_wspace)
    y_line = gs_top + 0.075
    y_text = gs_top + 0.125

    primary_values = sorted({c[0] for c in columns})
    n_secondary = len({c[1] for c in columns})
    bracket_positions = _compute_bracket_positions(primary_values, n_secondary, gs_left, stride, axes_w)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(20, 8.5),
        gridspec_kw=dict(
            hspace=gs_hspace,
            wspace=gs_wspace,
            top=gs_top,
            bottom=gs_bottom,
            left=gs_left,
            right=gs_right,
        ),
    )

    axes = _normalize_axes_array(axes, n_rows, n_cols)

    for ri, msa in enumerate(msa_modes):
        for ci, (primary, secondary) in enumerate(columns):
            ax = axes[ri, ci]
            cond_rows = [r for r in rows if _grid_condition_matches(r, msa, primary, secondary, spec)]
            vals, x_min, x_max = x_vals_fn(cond_rows, irmsd_clip)

            _plot_kde_cell(ax, vals, x_min, x_max, color, metric, gaussian_kde)
            _configure_grid_cell_axes(ax, ri, ci, n_rows, msa, msa_labels, xlabel, metric, x_min, x_max)

    _add_recycling_group_labels(fig, bracket_positions, y_text, y_line, spec)
    _add_scale_labels(fig, columns, gs_left, stride, axes_w, y_line, spec)
    _add_threshold_legend(fig, metric)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.995)
    return fig


def plot_condition_kde_grid(
    rows: list[dict],
    output_path: Path | str,
    title: str = "Boltz-2",
) -> None:
    """Two slide-ready 3×9 KDE grids saved as separate files.

    Rows    = MSA mode (derived from input rows)
    Columns = recycling × scale (derived from input rows)

    Generates:
      <output_path>          – DockQ KDE grid (purple)
      <stem>_irmsd<ext>      – iRMSD KDE grid (navy)

    Each figure is single-metric (one KDE per cell) with slide-optimised
    dimensions (~20"×7.5"), large fonts, and recycling group headers.
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        LOGGER.warning("plot_condition_kde_grid: scipy not available – skipping")
        return

    if not rows:
        LOGGER.warning("plot_condition_kde_grid: no rows provided – skipping")
        return

    msa_modes, msa_labels = _prepare_msa_modes(rows)

    spec = _infer_grid_axes(rows)
    if spec is None:
        LOGGER.warning("plot_condition_kde_grid: missing grid-axis values – skipping")
        return

    primary_values, secondary_values = _sorted_grid_conditions(rows, spec)
    if not primary_values or not secondary_values:
        LOGGER.warning("plot_condition_kde_grid: missing recycling/scale condition values – skipping")
        return

    columns = [(primary, secondary) for primary in primary_values for secondary in secondary_values]
    irmsd_clip = _calculate_irmsd_clip(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_conditions = len(msa_modes) * len(columns)

    # Generate DockQ grid
    fig_dq = _draw_kde_grid(
        rows,
        msa_modes,
        msa_labels,
        columns,
        spec,
        "dockq",
        AZ_PURPLE,
        _extract_dockq_values,
        "DockQ",
        f"{title} – DockQ KDE: all {n_conditions} conditions  "
        f"(rows = MSA mode · cols = {spec['secondary_name']} · groups = {spec['primary_name']})",
        irmsd_clip,
        gaussian_kde,
    )
    fig_dq.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig_dq)
    LOGGER.info("Saved DockQ KDE grid → %s", output_path)

    # Generate iRMSD grid
    irmsd_path = output_path.with_name(
        output_path.stem.replace("_kde_grid", "") + "_kde_grid_irmsd" + output_path.suffix
    )
    fig_ir = _draw_kde_grid(
        rows,
        msa_modes,
        msa_labels,
        columns,
        spec,
        "irmsd",
        AZ_NAVY,
        _extract_irmsd_values,
        LABEL_IRMSD_ANGSTROM,
        f"{title} – iRMSD KDE: all {n_conditions} conditions  "
        f"(rows = MSA mode · cols = {spec['secondary_name']} · groups = "
        f"{spec['primary_name']} · clipped at 99th pct = {irmsd_clip:.1f} Å)",
        irmsd_clip,
        gaussian_kde,
    )
    fig_ir.savefig(irmsd_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ir)
    LOGGER.info("Saved iRMSD KDE grid → %s", irmsd_path)


def _build_condition_matrix(rows: list[dict], spec: dict) -> tuple:
    """Build the condition × entry matrix for heatmap."""
    from collections import defaultdict as _dd

    cond_tuples: list[tuple] = sorted(
        {(r["msa_mode"], _grid_primary_value(r, spec), _grid_secondary_value(r, spec)) for r in rows}
    )
    cond_to_idx = {c: i for i, c in enumerate(cond_tuples)}
    n_conds = len(cond_tuples)

    entries = sorted({r["entry"] for r in rows})
    entry_to_idx = {e: i for i, e in enumerate(entries)}
    n_entries = len(entries)

    accum: dict = _dd(list)
    for r in rows:
        if r.get("DockQ") in ("", None):
            continue
        cond = (r["msa_mode"], _grid_primary_value(r, spec), _grid_secondary_value(r, spec))
        accum[(entry_to_idx[r["entry"]], cond_to_idx[cond])].append(float(r["DockQ"]))

    matrix = np.full((n_entries, n_conds), np.nan)
    for (ei, ci), vals in accum.items():
        matrix[ei, ci] = float(np.mean(vals))

    return matrix, entries, cond_tuples, n_entries, n_conds


def _sort_matrix_by_row_means(matrix, entries):
    """Sort matrix rows by descending mean DockQ."""
    row_means = np.nanmean(matrix, axis=1)
    sort_idx = np.argsort(row_means)[::-1]
    matrix_sorted = matrix[sort_idx]
    entries_sorted = [entries[i] for i in sort_idx]
    row_stds = np.nanstd(matrix, axis=1)[sort_idx]
    return matrix_sorted, entries_sorted, row_stds


def _create_condition_labels(cond_tuples, spec) -> list[str]:
    """Create short condition labels for heatmap x-axis."""
    _short = {"antigen_only": "ag", "full_msa": "fm", "no_msa": "nm"}
    return [
        f"{_short.get(m, m[:3])}\n{spec['primary_short']}{primary}"
        f"{spec['secondary_short']}{_grid_secondary_text(secondary, spec)}"
        for m, primary, secondary in cond_tuples
    ]


def _plot_heatmap_panel(ax_heat, matrix_s, entries_s, cond_labels, cond_tuples, n_entries, n_conds, title, fig):
    """Plot the main heatmap panel."""
    from matplotlib.colors import LinearSegmentedColormap

    az_cmap = LinearSegmentedColormap.from_list("az", ["#FFFFFF", AZ_YELLOW, AZ_PURPLE])
    im = ax_heat.imshow(matrix_s, aspect="auto", cmap=az_cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02, label="Mean DockQ")

    # White separators between msa_mode groups
    for i in range(1, n_conds):
        if cond_tuples[i][0] != cond_tuples[i - 1][0]:
            ax_heat.axvline(i - 0.5, color="white", linewidth=2.5)

    ax_heat.set_xticks(range(n_conds))
    ax_heat.set_xticklabels(cond_labels, rotation=90, fontsize=6)
    label_fs = max(4, min(8, 300 // n_entries))
    if n_entries <= 80:
        ax_heat.set_yticks(range(n_entries))
        ax_heat.set_yticklabels(entries_s, fontsize=label_fs)
    else:
        ax_heat.set_yticks([])
        ax_heat.set_ylabel(f"Entries  (n={n_entries}), sorted by mean DockQ ↓", fontsize=9)
    ax_heat.set_title(
        f"{title} – Mean DockQ per condition  (averaged over samples + interfaces)",
        fontsize=9,
        pad=18,
    )


def _add_msa_group_labels(ax_heat, cond_tuples, n_conds):
    """Add MSA mode group labels above heatmap."""
    _short = {"antigen_only": "ag", "full_msa": "fm", "no_msa": "nm"}
    mode_span: dict[str, list[int]] = {}
    for i, (m, _, _) in enumerate(cond_tuples):
        mode_span.setdefault(m, []).append(i)
    for m_idx, (mode, idxs) in enumerate(sorted(mode_span.items())):
        x_frac = (np.mean(idxs) + 0.5) / n_conds
        ax_heat.text(
            x_frac,
            1.01,
            _short.get(mode, mode),
            transform=ax_heat.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=AZ_PALETTE[m_idx % len(AZ_PALETTE)],
            clip_on=False,
        )


def _plot_std_bar_panel(ax_bar, row_stds, n_entries):
    """Plot cross-condition std bar panel."""
    ax_bar.barh(range(n_entries), row_stds, color=AZ_SKY, alpha=0.85)
    ax_bar.set_xlabel("Cross-cond.\nDockQ std", fontsize=8)
    ax_bar.set_yticks([])
    ax_bar.invert_yaxis()
    std_max = float(np.nanmax(row_stds)) if row_stds.size else 0.1
    ax_bar.set_xlim(0, std_max * 1.1)
    med_std = float(np.nanmedian(row_stds))
    ax_bar.axvline(med_std, color=AZ_PURPLE, linewidth=1.2, linestyle="--", label=f"med={med_std:.3f}")
    ax_bar.legend(fontsize=7, loc=LEGEND_LOC_LOWER_RIGHT)
    ax_bar.set_title("Cond.\nvariability", fontsize=8)


def _plot_std_kde_panel(ax_kde, row_stds, gaussian_kde):
    """Plot KDE of cross-condition std distribution."""
    std_clean = row_stds[~np.isnan(row_stds)]
    if len(std_clean) > 10:
        xs = np.linspace(0.0, float(std_clean.max()), 400)
        ys = gaussian_kde(std_clean)(xs)
        ax_kde.plot(xs, ys, color=AZ_PURPLE, linewidth=2.5)
        ax_kde.fill_between(xs, ys, alpha=0.15, color=AZ_PURPLE)
        med = float(np.median(std_clean))
        mn = float(np.mean(std_clean))
        ax_kde.axvline(med, color=AZ_YELLOW, linewidth=2.0, linestyle="--", label=f"median = {med:.3f}")
        ax_kde.axvline(mn, color=AZ_NAVY, linewidth=2.0, linestyle=":", label=f"mean   = {mn:.3f}")
        ax_kde.legend(fontsize=9)
    ax_kde.set_xlabel("Cross-condition DockQ std  (per entry, averaged over interfaces + samples)")
    ax_kde.set_ylabel("Density")
    ax_kde.set_title(
        "Distribution of cross-condition variability\nLow → robustly good/bad;   High → condition-sensitive",
        fontsize=10,
    )


def plot_cross_condition_heatmap(
    rows: list[dict],
    output_path: Path,
    title: str = "DockQ",
) -> None:
    """Heatmap of per-entry mean DockQ across all grid conditions.

    Each cell shows the mean DockQ for one (entry, condition) pair, averaged
    over all samples and interfaces.  Entries are sorted descending by their
    mean across all 27 conditions so the best-performing targets appear at the
    top.

    Layout:
    - Left (wide): heatmap entries × conditions, grouped by msa_mode with white
      separators.  Condition labels use short codes: ag/fm/nm + r{recycling}
      s{scale}.
    - Middle (narrow): horizontal bar showing per-entry cross-condition DockQ
      std — high = sensitive to hyperparameter choice; low = robustly good/bad.
    - Right (medium): KDE of the cross-condition std distribution across all
      entries, with median and mean marked.

    Skips silently when no condition columns are present.
    """
    spec = _infer_grid_axes(rows)
    if not rows or "msa_mode" not in (rows[0] if rows else {}) or spec is None:
        LOGGER.warning("plot_cross_condition_heatmap: no condition columns – skipping")
        return

    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        LOGGER.warning("plot_cross_condition_heatmap: scipy not available – skipping")
        return

    # Build matrix
    matrix, entries, cond_tuples, n_entries, n_conds = _build_condition_matrix(rows, spec)
    matrix_s, entries_s, row_stds = _sort_matrix_by_row_means(matrix, entries)
    cond_labels = _create_condition_labels(cond_tuples, spec)

    # Figure layout
    fig_h = max(8, min(n_entries * 0.13 + 2, 55))
    fig_w = max(18, n_conds * 0.50 + 10)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[n_conds, 2, 6],
        wspace=0.12,
        left=0.06,
        right=0.97,
        bottom=0.12,
        top=0.93,
    )
    ax_heat = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    ax_kde = fig.add_subplot(gs[2])

    # Plot panels
    _plot_heatmap_panel(ax_heat, matrix_s, entries_s, cond_labels, cond_tuples, n_entries, n_conds, title, fig)
    _add_msa_group_labels(ax_heat, cond_tuples, n_conds)
    _plot_std_bar_panel(ax_bar, row_stds, n_entries)
    _plot_std_kde_panel(ax_kde, row_stds, gaussian_kde)

    fig.suptitle(f"{title} – Cross-condition diversity", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved cross-condition heatmap → %s", output_path)


# ---------------------------------------------------------------------------
# Distribution shift analysis
# ---------------------------------------------------------------------------


def _get_msa_color_palette() -> dict[str, str]:
    """Return color mapping for MSA modes and models."""
    return {
        "full_msa": AZ_GREEN,
        "antigen_only": AZ_YELLOW,
        "no_msa": AZ_NAVY,
        "Boltz2": AZ_PURPLE,
        "Protenix": AZ_PINK,
    }


def _format_column_label(col: str) -> str:
    """Format column name for display (preserve DockQ capitalization)."""
    if col.lower() == "dockq":
        return "DockQ"
    return col.replace("_", " ").title()


def _format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def _prepare_correlation_data(df, x_col: str, y_col: str, hue_col: str | None):
    """Filter and validate data for correlation plot."""
    if x_col not in df.columns or y_col not in df.columns:
        return None

    mask = df[x_col].notna() & df[y_col].notna()
    if hue_col:
        mask &= df[hue_col].notna()
    data = df[mask]

    return data if len(data) > 0 else None


def _create_hue_label(hue_val: str, hue_data, x_col: str, y_col: str, show_stats: bool, pearsonr) -> str:
    """Create label for hue-colored scatter points."""
    if show_stats and len(hue_data) >= 10:
        hue_x = hue_data[x_col].values
        hue_y = hue_data[y_col].values
        hue_r, _ = pearsonr(hue_x, hue_y)
        return f"{hue_val} (n={len(hue_data)}, r={hue_r:.2f})"
    else:
        return f"{hue_val} (n={len(hue_data)})"


def _plot_scatter_by_hue(ax, data, x_col: str, y_col: str, hue_col: str, show_per_hue_stats: bool, pearsonr) -> None:
    """Plot scatter points colored by hue column."""
    msa_colors = _get_msa_color_palette()
    hue_vals = sorted(data[hue_col].unique())

    for hue_val in hue_vals:
        hue_data = data[data[hue_col] == hue_val]
        color = msa_colors.get(hue_val, None)
        label = _create_hue_label(hue_val, hue_data, x_col, y_col, show_per_hue_stats, pearsonr)
        ax.scatter(hue_data[x_col], hue_data[y_col], alpha=0.4, s=15, label=label, color=color, edgecolors="none")

    ax.legend(fontsize=9, loc="best", framealpha=0.9)


def _add_regression_line(ax, x: np.ndarray, y: np.ndarray) -> None:
    """Add regression line to scatter plot."""
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color=AZ_PINK, linestyle="--", alpha=0.8, linewidth=2, label="Overall fit")


def _build_correlation_title(
    title: str,
    data,
    pearson_r: float,
    pearson_p: float,
    spearman_r: float,
    spearman_p: float,
    hue_col: str | None,
    show_per_hue_stats: bool,
) -> str:
    """Build title string with correlation statistics."""
    title_str = f"{title}\n"
    if hue_col and show_per_hue_stats:
        title_str += f"Overall: n={len(data)}, Pearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}"
    else:
        p_str = _format_pvalue(pearson_p)
        s_str = _format_pvalue(spearman_p)
        title_str += f"n={len(data)}, Pearson r={pearson_r:.3f} (p {p_str})\nSpearman ρ={spearman_r:.3f} (p {s_str})"
    return title_str


def plot_correlation_scatter(
    rows: list[dict],
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    hue_col: str | None = None,
    show_per_hue_stats: bool = False,
) -> None:
    """Scatter plot with correlation statistics.

    Parameters
    ----------
    rows:
        List of row dicts from a DataFrame-like structure.
    x_col:
        Column name for x-axis metric.
    y_col:
        Column name for y-axis metric.
    output_path:
        Destination PNG file.
    title:
        Plot title.
    hue_col:
        Optional column to color points by (e.g., 'msa_mode', 'model').
    show_per_hue_stats:
        If True and hue_col is set, show per-category correlation in legend.

    Notes
    -----
    - Uses AZ brand colors for MSA modes (full_msa=green, antigen_only=orange, no_msa=blue)
    - Shows overall Pearson and Spearman correlations in title
    - Adds regression line for overall data
    - Per-hue statistics shown in legend if show_per_hue_stats=True
    """
    import pandas as pd  # noqa: PLC0415
    from scipy.stats import pearsonr, spearmanr  # noqa: PLC0415

    df = pd.DataFrame(rows)
    data = _prepare_correlation_data(df, x_col, y_col, hue_col)

    if data is None:
        LOGGER.warning(f"Columns {x_col} or {y_col} not found or no valid data, skipping {output_path.name}")
        return

    x = data[x_col].values
    y = data[y_col].values
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(10, 7))

    if hue_col:
        _plot_scatter_by_hue(ax, data, x_col, y_col, hue_col, show_per_hue_stats, pearsonr)
    else:
        ax.scatter(x, y, alpha=0.3, s=10, color=AZ_PURPLE, edgecolors="none")

    _add_regression_line(ax, x, y)

    ax.set_xlabel(_format_column_label(x_col), fontsize=11, color="black")
    ax.set_ylabel(_format_column_label(y_col), fontsize=11, color="black")

    title_str = _build_correlation_title(
        title, data, pearson_r, pearson_p, spearman_r, spearman_p, hue_col, show_per_hue_stats
    )
    ax.set_title(title_str, fontsize=12, fontweight="bold", color="black")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved correlation scatter plot: {output_path}")


def _detect_available_chain_columns(df) -> tuple[list[str], list[str]]:
    """Detect which chain pTM columns are available in dataframe."""
    chain_mapping = [
        ("ptm_H", "Heavy Chain"),
        ("ptm_L", "Light Chain"),
        ("ptm_Ag", "Antigen"),
    ]
    chain_cols = []
    chain_labels = []

    for col, label in chain_mapping:
        if col in df.columns and df[col].notna().sum() > 0:
            chain_cols.append(col)
            chain_labels.append(label)

    return chain_cols, chain_labels


def _prepare_chain_violin_data(chain_data, chain_col: str, msa_modes: list[str], msa_colors: dict) -> tuple:
    """Prepare data structures for violin plot."""
    positions = []
    data_by_mode = []
    colors = []

    for pos, mode in enumerate(msa_modes):
        mode_data = chain_data[chain_data["msa_mode"] == mode][chain_col].values
        if len(mode_data) > 0:
            positions.append(pos)
            data_by_mode.append(mode_data)
            colors.append(msa_colors.get(mode, AZ_GREY))

    return positions, data_by_mode, colors


def _style_violin_parts_ptm(parts: dict, colors: list[str]) -> None:
    """Apply color and style to violin plot parts."""
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")
        pc.set_linewidth(1)

    for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
        if partname in parts:
            parts[partname].set_edgecolor("black")
            parts[partname].set_linewidth(1.5)


def _add_jittered_scatter_ptm(ax, positions: list, data_by_mode: list, colors: list[str]) -> None:
    """Add jittered scatter points to violin plot."""
    rng = np.random.default_rng(42)
    for pos, mode_data, color in zip(positions, data_by_mode, colors):
        if len(mode_data) > 500:
            mode_data = rng.choice(mode_data, 500, replace=False)
        jitter = rng.normal(0, 0.05, size=len(mode_data))
        ax.scatter(pos + jitter, mode_data, alpha=0.2, s=5, color=color, edgecolors="none")


def _configure_chain_panel_axes(
    ax, positions: list, msa_modes: list[str], data_by_mode: list, idx: int, chain_label: str
) -> None:
    """Configure axes for a single chain panel."""
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{mode}\n(n={len(d)})" for mode, d in zip(msa_modes, data_by_mode)])
    ax.set_ylabel("pTM Score" if idx == 0 else "", fontsize=11, color="black")
    ax.set_title(chain_label, fontsize=12, fontweight="bold", color="black")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")


def _plot_single_chain_panel(ax, df, chain_col: str, chain_label: str, idx: int, msa_colors: dict) -> None:
    """Plot violin distribution for a single chain."""
    chain_data = df[df[chain_col].notna()].copy()
    if len(chain_data) == 0:
        return

    msa_modes = sorted(chain_data["msa_mode"].unique())
    positions, data_by_mode, colors = _prepare_chain_violin_data(chain_data, chain_col, msa_modes, msa_colors)

    parts = ax.violinplot(data_by_mode, positions=positions, widths=0.7, showmeans=True, showmedians=True)
    _style_violin_parts_ptm(parts, colors)
    _add_jittered_scatter_ptm(ax, positions, data_by_mode, colors)
    _configure_chain_panel_axes(ax, positions, msa_modes, data_by_mode, idx, chain_label)


def plot_chain_ptm_distributions(
    rows: list[dict],
    output_path: Path,
    title: str = "Chain-specific pTM by MSA Mode",
) -> None:
    """Plot distributions of chain-specific pTM values by MSA mode.

    Creates a multi-panel violin/box plot showing the distribution of pTM
    values for Heavy, Light, and Antigen chains across different MSA modes.

    Parameters
    ----------
    rows:
        List of row dicts containing 'ptm_H', 'ptm_L', 'ptm_Ag', and 'msa_mode'.
    output_path:
        Destination PNG file.
    title:
        Figure title.
    """
    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame(rows)
    chain_cols, chain_labels = _detect_available_chain_columns(df)

    if not chain_cols or "msa_mode" not in df.columns:
        LOGGER.warning("Missing required columns for chain pTM distribution plot")
        return

    n_chains = len(chain_cols)
    fig, axes = plt.subplots(1, n_chains, figsize=(6 * n_chains, 6), sharey=True)
    if n_chains == 1:
        axes = [axes]

    msa_colors = _get_msa_color_palette()

    for idx, (chain_col, chain_label) in enumerate(zip(chain_cols, chain_labels)):
        _plot_single_chain_panel(axes[idx], df, chain_col, chain_label, idx, msa_colors)

    fig.suptitle(title, fontsize=14, fontweight="bold", color="black", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved chain pTM distribution plot: {output_path}")


def _prepare_distribution_shift_data(rows: list[dict], spec: dict):
    """Prepare dataframe for distribution shift analysis."""
    import pandas as pd

    _REC = str(spec["primary_col"])
    _SCALE = str(spec["secondary_col"])
    _MSA = "msa_mode"

    df = pd.DataFrame(rows)
    for col in [_MSA, _REC, _SCALE, "entry", "DockQ"]:
        if col not in df.columns:
            LOGGER.warning("plot_distribution_shift: missing column '%s' – skipping", col)
            return None

    df["DockQ"] = pd.to_numeric(df["DockQ"], errors="coerce")
    df[_REC] = pd.to_numeric(df[_REC], errors="coerce")
    df[_SCALE] = pd.to_numeric(df[_SCALE], errors="coerce")
    df["_cond"] = df[_MSA] + "  R=" + df[_REC].astype(int).astype(str) + "  S=" + df[_SCALE].astype(str)

    return df


def _compute_spearman_matrix(pivot, n_conds):
    """Compute Spearman correlation matrix between conditions."""
    from scipy.stats import spearmanr

    rho_mat = np.full((n_conds, n_conds), np.nan)
    for i in range(n_conds):
        rho_mat[i, i] = 1.0
        for j in range(i + 1, n_conds):
            mask = pivot.iloc[i].notna() & pivot.iloc[j].notna()
            if mask.sum() < 5:
                continue
            rho, _ = spearmanr(pivot.iloc[i][mask], pivot.iloc[j][mask])
            rho_mat[i, j] = rho_mat[j, i] = float(rho)
    return rho_mat


def _create_short_labels(cond_order):
    """Create short condition labels."""
    _MSA_SHORT = {"antigen_only": "Ag", "full_msa": "Fl", "no_msa": "No"}
    short_labels = []
    for c in cond_order:
        parts = c.split()
        msa = _MSA_SHORT.get(parts[0], parts[0][:2])
        rec = parts[1].split("=")[1] if len(parts) > 1 else ""
        sc = parts[2].split("=")[1] if len(parts) > 2 else ""
        short_labels.append(f"{msa} R{rec} S{sc}")
    return short_labels


def _compute_condition_stats(df, cond_order, dockq_threshold):
    """Compute mean, std, and success rate per condition."""
    cond_mean = df.groupby("_cond")["DockQ"].mean().reindex(cond_order)
    cond_std = df.groupby("_cond")["DockQ"].std().reindex(cond_order)
    df["_hit"] = df["DockQ"] > dockq_threshold
    cond_succ = df.groupby("_cond")["_hit"].mean().reindex(cond_order)
    return cond_mean, cond_std, cond_succ


def _plot_spearman_heatmap(ax_corr, rho_mat, short_labels, n_conds, msa_tags, msa_unique, fig):
    """Plot Spearman correlation heatmap."""
    _MSA_ORDER = {"antigen_only": 0, "full_msa": 1, "no_msa": 2}

    im = ax_corr.imshow(rho_mat, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    cbar = fig.colorbar(im, ax=ax_corr, shrink=0.40, pad=0.01)
    cbar.set_label("Spearman ρ", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Group boundary lines
    for k in range(1, n_conds):
        if msa_tags[k] != msa_tags[k - 1]:
            bnd = k - 0.5
            ax_corr.axhline(bnd, color="white", linewidth=2.0)
            ax_corr.axvline(bnd, color="white", linewidth=2.0)

    ax_corr.set_xticks(range(n_conds))
    ax_corr.set_xticklabels(short_labels, rotation=90, fontsize=5.5)
    ax_corr.set_yticks(range(n_conds))
    ax_corr.set_yticklabels(short_labels, fontsize=5.5)
    ax_corr.set_title("Spearman ρ between conditions\n(per-entry mean DockQ rank correlation)", fontsize=10)

    # MSA-mode group annotations
    msa_spans: dict[str, list[int]] = {}
    for i, msa in enumerate(msa_tags):
        msa_spans.setdefault(msa, []).append(i)
    for m_idx, msa in enumerate(msa_unique):
        idxs = msa_spans[msa]
        x_frac = (np.mean(idxs) + 0.5) / n_conds
        ax_corr.text(
            x_frac,
            -0.05,
            msa.replace("_", " "),
            transform=ax_corr.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=AZ_PALETTE[m_idx % len(AZ_PALETTE)],
            clip_on=False,
        )


def _plot_mean_dockq_bars(ax_bar, y_pos, cond_mean, cond_std, bar_colors, short_labels, dockq_threshold):
    """Plot mean DockQ bars per condition."""
    ax_bar.barh(
        y_pos,
        cond_mean.values,
        xerr=cond_std.values,
        color=bar_colors,
        alpha=0.85,
        height=0.75,
        error_kw={"elinewidth": 0.7, "capsize": 2.0},
    )
    ax_bar.axvline(
        dockq_threshold,
        color=AZ_NAVY,
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
        label=f"DockQ = {dockq_threshold}",
    )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(short_labels, fontsize=5.5)
    ax_bar.set_xlabel("Mean DockQ ± std", fontsize=8)
    ax_bar.set_title("Mean DockQ per condition", fontsize=10)
    ax_bar.legend(fontsize=7, loc=LEGEND_LOC_LOWER_RIGHT)


def _plot_success_rate_bars(ax_succ, y_pos, cond_succ, bar_colors, dockq_threshold):
    """Plot success rate bars per condition."""
    ax_succ.barh(y_pos, cond_succ.values, color=bar_colors, alpha=0.85, height=0.75)
    ax_succ.set_xlim(0, 1)
    ax_succ.set_xlabel(f"Fraction (DockQ > {dockq_threshold})", fontsize=8)
    ax_succ.set_title("Success rate per condition", fontsize=10)
    ax_succ.tick_params(labelleft=False)


def _plot_parameter_effect_scatter(ax, df, param_col, param_vals, msa_unique, spec, param_name, is_secondary=False):
    """Plot effect scatter for recycling or scale parameter."""
    from scipy.stats import spearmanr

    if len(param_vals) < 2:
        return

    lo_val, hi_val = param_vals[0], param_vals[-1]

    for m_idx, msa in enumerate(msa_unique):
        sub_lo = df[(df["msa_mode"] == msa) & (df[param_col] == lo_val)].groupby("entry")["DockQ"].mean()
        sub_hi = df[(df["msa_mode"] == msa) & (df[param_col] == hi_val)].groupby("entry")["DockQ"].mean()
        common = sub_lo.index.intersection(sub_hi.index)
        if len(common) < 3:
            continue
        color = AZ_PALETTE[m_idx % len(AZ_PALETTE)]
        ax.scatter(
            sub_hi.loc[common].values,
            sub_lo.loc[common].values,
            alpha=0.45,
            s=20,
            color=color,
            label=msa.replace("_", " "),
        )

    # Spearman annotation
    all_lo = df[df[param_col] == lo_val].groupby("entry")["DockQ"].mean()
    all_hi = df[df[param_col] == hi_val].groupby("entry")["DockQ"].mean()
    common_all = all_lo.index.intersection(all_hi.index)
    if len(common_all) >= 5:
        rho, _ = spearmanr(all_hi.loc[common_all], all_lo.loc[common_all])
        ax.text(
            0.04,
            0.93,
            f"ρ = {rho:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            color=AZ_NAVY,
            fontweight="bold",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if is_secondary:
        lo_text = _grid_secondary_text(lo_val, spec)
        hi_text = _grid_secondary_text(hi_val, spec)
        ax.set_xlabel(f"Mean DockQ  ({param_name} = {hi_text})", fontsize=9)
        ax.set_ylabel(f"Mean DockQ  ({param_name} = {lo_text})", fontsize=9)
        ax.set_title(
            f"{param_name.title()} effect: {lo_text} vs {hi_text}\n"
            "Points on diagonal → changing this axis shifts spread more than ranking",
            fontsize=9,
        )
    else:
        ax.set_xlabel(f"Mean DockQ  ({param_name} = {hi_val:.0f})", fontsize=9)
        ax.set_ylabel(f"Mean DockQ  ({param_name} = {lo_val:.0f})", fontsize=9)
        ax.set_title(
            f"{param_name.title()} effect: {lo_val:.0f} vs {hi_val:.0f}\n"
            "Points on diagonal → same entries succeed; shift → bias",
            fontsize=9,
        )

    ax.legend(fontsize=7, markerscale=1.8, loc=LEGEND_LOC_UPPER_LEFT)


def plot_distribution_shift(
    rows: list[dict],
    output_path: Path,
    title: str = "Boltz-2",
    dockq_threshold: float = 0.49,
) -> None:
    """Five-panel figure probing whether grid conditions introduce systematic bias.

    Parameters
    ----------
    rows:
        List of row dicts from the DockQ CSV.  Must contain columns
        ``msa_mode``, ``recycling_steps``, ``step_scale``, ``entry``, ``DockQ``.
    output_path:
        Destination PNG file.
    title:
        Figure super-title prefix.
    dockq_threshold:
        DockQ value used to define a "successful" prediction (default 0.49,
        the accepted medium-quality cut-off).

    Panels
    ------
    A (left, tall):
        27 × 27 Spearman ρ heatmap between conditions based on per-entry
        mean DockQ.  High off-diagonal ρ → conditions are interchangeable
        samplers of the same distribution.  Low ρ → distribution shift.
        Group boundaries between msa_modes are highlighted in white.
    B (top-centre):
        Mean DockQ ± std per condition (horizontal bars, colour = msa_mode).
    C (top-right):
        Success rate (fraction of entries with mean DockQ > threshold) per
        condition.  Same y-ordering as B.
    D (bottom-centre):
        Recycling-effect scatter: DockQ(recycling_max) vs DockQ(recycling_min)
        per entry (mean across scales and samples).  Points on the 45° line →
        same entries succeed/fail regardless of recycling steps (no bias).
        Points below the line → fewer recycling steps hurts that entry more.
    E (bottom-right):
        Scale (temperature) effect scatter: DockQ(scale_max) vs
        DockQ(scale_min) per entry.  Points on the diagonal → scale only
        changes spread, not which entries succeed.
    """
    spec = _infer_grid_axes(rows)
    if spec is None:
        LOGGER.warning("plot_distribution_shift: no condition columns – skipping")
        return

    df = _prepare_distribution_shift_data(rows, spec)
    if df is None:
        return

    _REC = str(spec["primary_col"])
    _SCALE = str(spec["secondary_col"])

    # Per-entry per-condition mean DockQ
    pivot = df.groupby(["_cond", "entry"])["DockQ"].mean().unstack("entry")

    # Canonical sort
    _MSA_ORDER = {"antigen_only": 0, "full_msa": 1, "no_msa": 2}

    def _cond_key(cond: str) -> tuple:
        parts = cond.split()
        msa = parts[0]
        rec = float(parts[1].split("=")[1]) if len(parts) > 1 else 0.0
        sc = float(parts[2].split("=")[1]) if len(parts) > 2 else 0.0
        return (_MSA_ORDER.get(msa, 99), rec, sc)

    cond_order = sorted(pivot.index, key=_cond_key)
    pivot = pivot.loc[cond_order]
    n_conds = len(cond_order)
    msa_tags = [c.split()[0] for c in cond_order]
    msa_unique = sorted(set(msa_tags), key=lambda m: _MSA_ORDER.get(m, 99))

    # Compute matrices and stats
    rho_mat = _compute_spearman_matrix(pivot, n_conds)
    short_labels = _create_short_labels(cond_order)
    cond_mean, cond_std, cond_succ = _compute_condition_stats(df, cond_order, dockq_threshold)

    # Color per condition
    bar_colors = [AZ_PALETTE[msa_unique.index(msa_tags[i]) % len(AZ_PALETTE)] for i in range(n_conds)]

    # Figure layout
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[2.2, 2.0, 2.0],
        height_ratios=[1, 1],
        wspace=0.30,
        hspace=0.38,
        left=0.06,
        right=0.98,
        bottom=0.06,
        top=0.92,
    )
    ax_corr = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_succ = fig.add_subplot(gs[0, 2], sharey=ax_bar)
    ax_rec = fig.add_subplot(gs[1, 1])
    ax_scale = fig.add_subplot(gs[1, 2])

    # Plot panels
    _plot_spearman_heatmap(ax_corr, rho_mat, short_labels, n_conds, msa_tags, msa_unique, fig)

    y_pos = np.arange(n_conds)
    _plot_mean_dockq_bars(ax_bar, y_pos, cond_mean, cond_std, bar_colors, short_labels, dockq_threshold)
    _plot_success_rate_bars(ax_succ, y_pos, cond_succ, bar_colors, dockq_threshold)

    rec_vals = sorted(df[_REC].dropna().unique())
    scale_vals = sorted(df[_SCALE].dropna().unique())

    _plot_parameter_effect_scatter(
        ax_rec, df, _REC, rec_vals, msa_unique, spec, spec["primary_name"], is_secondary=False
    )
    _plot_parameter_effect_scatter(
        ax_scale, df, _SCALE, scale_vals, msa_unique, spec, spec["secondary_name"], is_secondary=True
    )

    fig.suptitle(f"{title} – Distribution shift analysis", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved distribution shift plot → %s", output_path)
