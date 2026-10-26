#!/usr/bin/env python
"""
Build a DockQ-predictor training manifest from grid-inference DockQ score CSVs.

This script:
1. Loads the ``dockq_scores.csv`` written by the grid pipeline for each folding model
2. Selects complexes, keeping ALL decoys of each selected complex
3. Resolves every decoy's structure path into manifest rows
4. Validates that those structures exist on disk

Selection is at the COMPLEX level, never per row: per-native ranking metrics (and the
within-complex ranking loss) need several decoys of the same complex, so a row-wise
sample would break training. With ``--stratify-antigen-clusters`` one shared set of
complexes is drawn round-robin across antigen clusters and used for every model, which
both maximises antigen diversity (clusters are very skewed) and makes the models
directly comparable on the same complexes.

Usage:
    poetry run python workflows/scripts/analysis/sample_dockq_for_training.py \
      --boltz2-csv    <GRID_ROOT>/boltz2_r1/dockq/dockq_scores.csv \
      --boltz2-dir    <GRID_ROOT>/boltz2_r1 \
      --esmfold2-csv  <GRID_ROOT>/esmfold2_r1/dockq/dockq_scores.csv \
      --esmfold2-dir  <GRID_ROOT>/esmfold2_r1 \
      --protenix-csv  <GRID_ROOT>/protenix_r1/dockq/dockq_scores.csv \
      --protenix-dir  <GRID_ROOT>/protenix_r1 \
      --stratify-antigen-clusters data/sabdab_trainset_pipeline/mmseqs/antigen_cluster.tsv \
      --n-total 800 \
      --output data/dockq_predictor/dockq_training_manifest_800.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# The manifest-building helpers live in the package so they can be reused by the
# ablation runner and other code, not just this CLI. Re-exported here for
# backward compatibility with anything importing them from this script.
from confidock.dockq_predictor.data.manifest_builder import (
    build_antigen_cluster_map,
    convert_to_training_format,
    load_and_sample_dockq,
    print_statistics,
    select_stratified_entries,
    validate_manifest,
)

__all__ = [
    "build_antigen_cluster_map",
    "select_stratified_entries",
    "load_and_sample_dockq",
    "convert_to_training_format",
    "validate_manifest",
    "print_statistics",
]


def main():
    parser = argparse.ArgumentParser(description="Sample DockQ scores for training manifest")

    # One --<model>-csv / --<model>-dir pair per folding model. A model is included
    # only if both are given and the CSV exists, so a partial grid still builds a
    # manifest from whatever finished.
    parser.add_argument("--boltz2-csv", type=Path, default=None, help="Boltz2 DockQ CSV")
    parser.add_argument("--esmfold2-csv", type=Path, default=None, help="ESMFold2 DockQ CSV")
    parser.add_argument("--protenix-csv", type=Path, default=None, help="Protenix DockQ CSV")

    parser.add_argument("--boltz2-dir", type=Path, default=None, help="Boltz2 grid root (holds the condition dirs)")
    parser.add_argument("--esmfold2-dir", type=Path, default=None, help="ESMFold2 grid root (holds the condition dirs)")
    parser.add_argument("--protenix-dir", type=Path, default=None, help="Protenix grid root (holds the condition dirs)")

    parser.add_argument("--output", type=Path, required=True, help="Output manifest CSV")

    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of complexes per model source (all decoys of each are kept)",
    )

    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")

    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip file existence validation (faster but risky)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["boltz2", "esmfold2", "protenix"],
        choices=["boltz2", "esmfold2", "protenix"],
        help="Which models to include",
    )

    parser.add_argument(
        "--stratify-antigen-clusters",
        type=Path,
        default=None,
        help=(
            "Path to mmseqs antigen_cluster.tsv. When given, selects one SHARED "
            "set of complexes stratified across antigen clusters and pulls all "
            "models' decoys for those complexes (instead of sampling each model "
            "independently)."
        ),
    )

    parser.add_argument(
        "--n-total",
        type=int,
        default=None,
        help=(
            "Total number of complexes to select (shared across all models). "
            "Only used with --stratify-antigen-clusters. Defaults to --n-samples."
        ),
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SAMPLING DOCKQ SCORES FOR TRAINING")
    print("=" * 80)
    print(f"Random seed: {args.random_seed}")
    print(f"Models: {', '.join(args.models)}")

    stratified = args.stratify_antigen_clusters is not None
    csv_by_model = {"boltz2": args.boltz2_csv, "esmfold2": args.esmfold2_csv, "protenix": args.protenix_csv}
    dir_by_model = {"boltz2": args.boltz2_dir, "esmfold2": args.esmfold2_dir, "protenix": args.protenix_dir}
    active_models = [
        m
        for m in args.models
        if csv_by_model[m] is not None and dir_by_model[m] is not None and csv_by_model[m].exists()
    ]
    if not active_models:
        print("\nERROR: no model has both --<model>-csv (existing) and --<model>-dir set.")
        sys.exit(1)

    preselected_entries = None
    if stratified:
        n_total = args.n_total if args.n_total is not None else args.n_samples
        print(f"Mode: STRATIFIED by antigen cluster ({n_total} complexes shared across models)")
        print(f"Cluster file: {args.stratify_antigen_clusters}")

        # Shared complexes = intersection of entries across all active models.
        entry_sets = []
        for m in active_models:
            entries = set(pd.read_csv(csv_by_model[m], usecols=["entry"])["entry"].unique())
            print(f"  {m}: {len(entries)} unique complexes")
            entry_sets.append(entries)
        shared_entries = set.intersection(*entry_sets)
        print(f"  Complexes common to all {len(active_models)} models: {len(shared_entries)}")

        cluster_map = build_antigen_cluster_map(args.stratify_antigen_clusters)
        preselected_entries = set(select_stratified_entries(shared_entries, cluster_map, n_total, args.random_seed))
    else:
        print(f"Mode: RANDOM sampling ({args.n_samples} complexes per model, sampled independently)")

    # Load and sample from each model
    sampled_dfs = []
    for model in active_models:
        df_model = load_and_sample_dockq(
            csv_by_model[model],
            model,
            args.n_samples,
            args.random_seed,
            preselected_entries=preselected_entries,
        )
        manifest_model = convert_to_training_format(df_model, dir_by_model[model], model)
        sampled_dfs.append(manifest_model)

    if not sampled_dfs:
        print("\nERROR: No valid model sources found!")
        sys.exit(1)

    # Combine
    manifest_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\nCombined {len(manifest_df)} entries from {len(sampled_dfs)} model sources")

    # Validate
    if not args.skip_validation:
        manifest_df = validate_manifest(manifest_df)

        if len(manifest_df) == 0:
            print("\nERROR: No valid entries after validation!")
            sys.exit(1)

    # Print statistics
    print_statistics(manifest_df)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(args.output, index=False)

    print(f"\nSaved manifest to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
