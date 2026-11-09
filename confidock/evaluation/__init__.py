"""Evaluation utilities for protein structure predictions.

This module provides metrics and utilities for evaluating predicted protein
structures against native references:

- Structure-level RMSD metrics (CA, backbone, sidechain)
- Interface contact metrics (precision, recall, F1)
- Diversity metrics across prediction samples
- SAbDab-specific chain role inference
- Bias analysis between binder and antigen predictions
- Analysis utilities for grid conditions and chain topology
"""

from __future__ import annotations

from confidock.evaluation.analysis import (
    check_bias_columns,
    compute_bias_deltas,
    csv_has_bias_metrics,
    get_grouping_axes,
    infer_chain_topology,
    infer_condition_cols,
)
from confidock.evaluation.bias import run_bias_analysis
from confidock.evaluation.dockq import (
    run_dockq,
    run_complex_level_dockq,
    run_dockq_with_complex_score,
)
from confidock.evaluation.diversity import (
    annotate_complex_diversity_metrics,
    compute_interface_mean,
    compute_diversity_metrics,
    count_irmsd_clusters,
)
from confidock.evaluation.dockq_utils import (
    format_value,
    make_dockq_row,
    resolve_boltz_input_dir,
    resolve_grid_predictions_dir,
    scan_grid_conditions,
    validate_interface_coverage,
    BOLTZ_CONDITION_COLS,
    PROTENIX_CONDITION_COLS,
    ESMFOLD2_CONDITION_COLS,
    GRID_SCHEMAS,
)
from confidock.evaluation.metrics import compute_structure_metrics
from confidock.evaluation.sabdab import (
    infer_chain_roles,
    interface_role_pair,
    load_sabdab_role_counts,
)
from confidock.evaluation.structures import (
    chain_ids_in_order,
    load_structure_pair,
)
from confidock.evaluation.epitope_matching import (
    align_on_framework,
    compute_interface_ca_rmsd,
    compute_cdr_center_distance,
    match_epitope_by_interface_position,
    detect_multiple_binding_modes,
)
from confidock.evaluation.sequence_matching import (
    extract_sequence_from_structure,
    match_chains_by_sequence,
    match_chains_using_json,
    get_chain_map_with_sequence_matching,
    sequence_identity,
)
from confidock.evaluation.collection import (
    collect_rows_boltz,
    collect_rows_esmfold2,
    collect_rows_protenix,
)

__all__ = [
    # Metrics
    "compute_structure_metrics",
    # Structures
    "load_structure_pair",
    "chain_ids_in_order",
    # Epitope Matching
    "align_on_framework",
    "compute_interface_ca_rmsd",
    "compute_cdr_center_distance",
    "match_epitope_by_interface_position",
    "detect_multiple_binding_modes",
    # Sequence Matching
    "extract_sequence_from_structure",
    "match_chains_by_sequence",
    "match_chains_using_json",
    "get_chain_map_with_sequence_matching",
    "sequence_identity",
    # DockQ
    "run_dockq",
    "run_complex_level_dockq",
    "run_dockq_with_complex_score",
    # Diversity
    "annotate_complex_diversity_metrics",
    "compute_interface_mean",
    "compute_diversity_metrics",
    "count_irmsd_clusters",
    # SAbDab
    "infer_chain_roles",
    "interface_role_pair",
    "load_sabdab_role_counts",
    # Analysis
    "infer_condition_cols",
    "get_grouping_axes",
    "infer_chain_topology",
    "check_bias_columns",
    "compute_bias_deltas",
    "csv_has_bias_metrics",
    # Bias
    "run_bias_analysis",
    # DockQ Utils
    "format_value",
    "make_dockq_row",
    "resolve_boltz_input_dir",
    "resolve_grid_predictions_dir",
    "scan_grid_conditions",
    "validate_interface_coverage",
    "BOLTZ_CONDITION_COLS",
    "PROTENIX_CONDITION_COLS",
    "ESMFOLD2_CONDITION_COLS",
    "GRID_SCHEMAS",
    # Collection
    "collect_rows_boltz",
    "collect_rows_esmfold2",
    "collect_rows_protenix",
]
