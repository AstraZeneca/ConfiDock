"""Diversity metrics for ensembles of predicted protein structures.

This module provides utilities for quantifying diversity across multiple
predictions for the same target, including clustering and statistical aggregation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def count_irmsd_clusters(values: list[float], gap: float = 2.0) -> int:
    """Count conformational clusters via a gap threshold on sorted iRMSD values.

    With 5 samples and a 2 Å gap this is a lightweight proxy for whether the
    ensemble is unimodal (all samples near the same interface geometry) or
    multi-modal (some samples at clearly different poses).

    Parameters
    ----------
    values : list of float
        iRMSD values (in Ångströms)
    gap : float
        Gap threshold in Ångströms (default: 2.0)

    Returns
    -------
    int
        Number of clusters

    Notes
    -----
    Algorithm: sort values, count the number of gaps > threshold.
    Example: [1.0, 1.2, 4.5, 4.8] with gap=2.0 → 2 clusters (gap of 3.3 Å between 1.2 and 4.5)
    """
    if len(values) < 2:
        return 1 if values else 0

    sorted_vals = sorted(values)
    return 1 + sum(b - a > gap for a, b in zip(sorted_vals, sorted_vals[1:]))


def _extract_numeric_values(group_rows: list[dict], col: str) -> list[float]:
    """Extract numeric values for a column from a group of rows."""
    return [float(r[col]) for r in group_rows if r.get(col) not in ("", None)]


def _compute_mean_std_for_column(group_rows: list[dict], col: str) -> dict[str, float | str]:
    """Compute mean and std for a column."""
    values = _extract_numeric_values(group_rows, col)
    if not values:
        return {f"{col}_mean": "", f"{col}_std": ""}
    return {
        f"{col}_mean": round(float(np.mean(values)), 6),
        f"{col}_std": round(float(np.std(values)), 6) if len(values) > 1 else "",
    }


def _compute_core_dockq_metrics(
    dockq_v: list[float],
    irmsd_v: list[float],
    lrmsd_v: list[float],
) -> dict[str, Any]:
    """Compute core DockQ diversity metrics."""
    return {
        "dockq_mean": round(float(np.mean(dockq_v)), 6) if dockq_v else "",
        "dockq_std": round(float(np.std(dockq_v)), 6) if len(dockq_v) > 1 else "",
        "irmsd_mean": round(float(np.mean(irmsd_v)), 6) if irmsd_v else "",
        "irmsd_std": round(float(np.std(irmsd_v)), 6) if len(irmsd_v) > 1 else "",
        "lrmsd_std": round(float(np.std(lrmsd_v)), 6) if len(lrmsd_v) > 1 else "",
        "irmsd_range": round(max(irmsd_v) - min(irmsd_v), 6) if len(irmsd_v) > 1 else "",
        "n_conf_clusters": count_irmsd_clusters(irmsd_v) if irmsd_v else "",
    }


def _compute_structure_metrics(group_rows: list[dict]) -> dict[str, Any]:
    """Compute structure-level mean/std metrics across samples."""
    metrics: dict[str, Any] = {}
    for col in [
        "complex_ca_rmsd",
        "binder_ca_rmsd",
        "antigen_ca_rmsd",
        "interface_contact_f1",
        "binder_residue_recall",
        "antigen_residue_recall",
        "binder_interface_sidechain_rmsd",
        "antigen_interface_sidechain_rmsd",
        "antigen_aligned_binder_ca_rmsd",
    ]:
        metrics.update(_compute_mean_std_for_column(group_rows, col))
    return metrics


def _group_rows_by_condition(
    rows: list[dict],
    group_key_cols: list[str],
) -> dict[tuple, list[dict]]:
    """Group rows by condition columns."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(c, "") for c in group_key_cols)
        groups[key].append(row)
    return groups


def compute_diversity_metrics(rows: list[dict], condition_cols: list[str]) -> None:
    """Annotate each row in-place with per-(condition × entry × interface) diversity metrics.

    Groups rows by their condition columns + entry + native_interface (i.e. all
    samples for one target under one grid condition), then computes:

    - ``dockq_mean`` / ``dockq_std``: mean and std of DockQ across samples
    - ``irmsd_mean`` / ``irmsd_std``: mean and std of iRMSD across samples
    - ``lrmsd_std``: std of LRMSD across samples
    - ``irmsd_range``: max(iRMSD) − min(iRMSD)
    - ``n_conf_clusters``: number of distinct iRMSD clusters (2 Å gap threshold)
    - ``<metric>_mean`` / ``<metric>_std`` for various structure-level metrics

    Parameters
    ----------
    rows : list of dict
        Rows to annotate (modified in-place)
    condition_cols : list of str
        Column names that define experimental conditions
        (e.g., ["msa_mode", "recycling_steps", "step_scale"] for Boltz-2)

    Notes
    -----
    This function modifies `rows` in-place by adding diversity metric columns.
    """
    group_key_cols = condition_cols + ["entry", "native_interface"]
    groups = _group_rows_by_condition(rows, group_key_cols)

    for group_rows in groups.values():
        # Extract core metric values
        dockq_v = _extract_numeric_values(group_rows, "DockQ")
        irmsd_v = _extract_numeric_values(group_rows, "iRMSD")
        lrmsd_v = _extract_numeric_values(group_rows, "LRMSD")

        # Compute all metrics
        metrics = _compute_core_dockq_metrics(dockq_v, irmsd_v, lrmsd_v)
        metrics.update(_compute_structure_metrics(group_rows))

        # Annotate all rows in this group with the same diversity metrics
        for row in group_rows:
            row.update(metrics)


def _safe_float(value) -> float | None:
    """Return float(value) or None for empty/non-numeric values."""
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | str:
    """Return the population mean for a non-empty list, else empty string."""
    if not values:
        return ""
    return round(float(np.mean(values)), 6)


def _std(values: list[float]) -> float | str:
    """Return the population standard deviation for n>1, else empty string."""
    if len(values) < 2:
        return ""
    return round(float(np.std(values)), 6)


def _extract_numeric(row: dict, key: str) -> float | None:
    """Extract a numeric value from a row dictionary by key."""
    return _safe_float(row.get(key, ""))


def _aggregate_sample_interface_metric(group_rows: list[dict], col: str) -> float | str:
    """Compute per-sample mean of an interface-level metric across interfaces."""
    values = [v for v in (_extract_numeric(r, col) for r in group_rows) if v is not None]
    return _mean(values)


def annotate_complex_diversity_metrics(
    complex_rows: list[dict],
    condition_cols: list[str],
    sample_metric_cols: list[str],
    sample_prefix: str = "__sample_",
) -> None:
    """Annotate complex rows with diversity metrics computed at complex level.

    Groups by (conditions + entry), then computes diversity across samples using:
    - complex-level DockQ metrics (complex_dockq/complex_irmsd/complex_lrmsd)
    - per-sample interface-aggregated structural metrics stored under
      ``{sample_prefix}{metric_name}`` keys.
    """
    if not complex_rows:
        return

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in complex_rows:
        key = tuple([row.get("entry", "")] + [row.get(c, "") for c in condition_cols])
        groups[key].append(row)

    for group_rows in groups.values():
        dockq_v = [v for v in (_extract_numeric(r, "complex_dockq") for r in group_rows) if v is not None]
        irmsd_v = [v for v in (_extract_numeric(r, "complex_irmsd") for r in group_rows) if v is not None]
        lrmsd_v = [v for v in (_extract_numeric(r, "complex_lrmsd") for r in group_rows) if v is not None]

        group_metrics = {
            "dockq_mean": _mean(dockq_v),
            "dockq_std": _std(dockq_v),
            "irmsd_mean": _mean(irmsd_v),
            "irmsd_std": _std(irmsd_v),
            "lrmsd_std": _std(lrmsd_v),
            "irmsd_range": round(max(irmsd_v) - min(irmsd_v), 6) if len(irmsd_v) > 1 else "",
            "n_conf_clusters": count_irmsd_clusters(irmsd_v) if irmsd_v else "",
        }

        for src_col in sample_metric_cols:
            values = [
                v for v in (_extract_numeric(r, f"{sample_prefix}{src_col}") for r in group_rows) if v is not None
            ]
            group_metrics[f"{src_col}_mean"] = _mean(values)
            group_metrics[f"{src_col}_std"] = _std(values)

        for row in group_rows:
            row.update(group_metrics)


def compute_interface_mean(group_rows: list[dict], metric: str) -> float | str:
    """Compute mean value for an interface metric within one sample group."""
    return _aggregate_sample_interface_metric(group_rows, metric)
