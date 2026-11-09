"""Build DockQ training/validation manifests from grid-layout DockQ score CSVs.

These helpers turn the per-model ``dockq_scores.csv`` files produced by the grid
inference pipeline (boltz2 / esmfold2 / protenix) into a manifest of individual
decoys, resolving each decoy's predicted-structure path. A manifest row is one
decoy: where its structure is, which folding model produced it, and its
ground-truth DockQ.

The feature set is computed from coordinates alone (see ``feature_utils``), so a
decoy needs NOTHING but its structure file -- no companion PAE / confidence file.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import islice, zip_longest
from pathlib import Path

import numpy as np
import pandas as pd


def build_antigen_cluster_map(cluster_tsv: Path) -> dict:
    """
    Build a mapping of ``entry`` (interface_id) -> antigen cluster id.

    The mmseqs cluster TSV has two columns: ``representative<TAB>member``,
    where each id looks like ``<interface_id>|<chain>`` (e.g. ``1a14_HLN|A``).
    Every interface maps to exactly one representative, which we use as the
    cluster id.
    """
    df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"])
    df["entry"] = df["member"].str.split("|").str[0]
    df["cluster"] = df["rep"].str.split("|").str[0]
    cluster_map = dict(zip(df["entry"], df["cluster"]))
    n_clusters = df["cluster"].nunique()
    print(f"  Loaded antigen clusters: {len(cluster_map)} interfaces -> {n_clusters} clusters")
    return cluster_map


def _bucket_entries_by_cluster(entries: set, cluster_map: dict) -> dict:
    """Group entries by antigen cluster, in a deterministic order.

    Entries absent from ``cluster_map`` land in a synthetic ``__unclustered__``
    bucket (and are counted in a warning) so they can still be selected.
    """
    buckets = defaultdict(list)
    n_missing = 0
    for entry in sorted(entries):  # sort for determinism before shuffling
        cluster = cluster_map.get(entry)
        if cluster is None:
            cluster = "__unclustered__"
            n_missing += 1
        buckets[cluster].append(entry)

    if n_missing:
        print(f"  WARNING: {n_missing} entries had no antigen cluster (grouped as __unclustered__)")

    return buckets


def _round_robin_select(buckets: dict, cluster_order: list, n_total: int) -> list:
    """Draw up to ``n_total`` entries, one per cluster per pass.

    ``zip_longest`` lays the buckets out in passes -- the i-th element of every
    cluster, in ``cluster_order`` -- and exhausted clusters pad with None, which is
    dropped. So every cluster contributes once before any contributes twice, and a
    short bucket simply stops appearing. Yields fewer than ``n_total`` only when
    every bucket is exhausted.
    """
    passes = zip_longest(*(buckets[cluster] for cluster in cluster_order))
    drawn = (entry for one_pass in passes for entry in one_pass if entry is not None)
    return list(islice(drawn, n_total))


def select_stratified_entries(
    entries: set,
    cluster_map: dict,
    n_total: int,
    random_seed: int = 42,
) -> list:
    """
    Select ``n_total`` complexes stratified across antigen clusters.

    Uses round-robin allocation: entries are shuffled within each cluster, then
    drawn one cluster at a time in repeated passes until ``n_total`` are chosen.
    This guarantees maximum cluster coverage first (one per cluster), only
    over-representing large clusters once every cluster has contributed. This is
    the right behaviour here because antigen clusters are very skewed
    (median size 2, max ~791) and plain random sampling would over-sample the
    few huge clusters.

    Entries absent from ``cluster_map`` are grouped under a synthetic
    ``__unclustered__`` cluster so they can still be selected.
    """
    # Reproducible per seed, but NOT stream-compatible with the legacy RandomState
    # this replaced: re-running an old seed selects a different set of complexes,
    # so a manifest built before this change cannot be regenerated, only reused.
    rng = np.random.default_rng(random_seed)

    # Bucket entries by cluster.
    buckets = _bucket_entries_by_cluster(entries, cluster_map)

    # Shuffle within each cluster so the "first" pick per cluster is random.
    for cluster in buckets:
        rng.shuffle(buckets[cluster])

    # Round-robin across clusters in a fixed order for reproducibility.
    cluster_order = sorted(buckets.keys())
    if n_total >= len(entries):
        print(f"  Requested {n_total} >= available {len(entries)} complexes; using all.")
        return sorted(entries)

    selected = _round_robin_select(buckets, cluster_order, n_total)

    n_clusters_used = len({cluster_map.get(e, "__unclustered__") for e in selected})
    print(f"  Stratified selection: {len(selected)} complexes across {n_clusters_used} antigen clusters")
    return selected


def load_and_sample_dockq(
    csv_path: Path,
    model_source: str,
    n_samples: int,
    random_seed: int = 42,
    preselected_entries: set = None,
) -> pd.DataFrame:
    """
    Load DockQ CSV and sample complexes, keeping ALL decoys per complex.

    Sampling is done at the complex (``entry``) level rather than per row, so
    every selected complex retains its full set of decoys (~10 per complex:
    multiple conditions x samples/seeds). This is required for per-native
    ranking metrics, which need >=2 decoys per complex to be meaningful.

    Parameters
    ----------
    csv_path : Path
        Path to DockQ scores CSV
    model_source : str
        Model source name (boltz2, esmfold2, protenix)
    n_samples : int
        Number of complexes to sample (all decoys of each are kept). Ignored
        when ``preselected_entries`` is provided.
    random_seed : int
        Random seed for reproducibility
    preselected_entries : set, optional
        If given, the CSV is filtered to exactly these entries instead of
        being randomly sampled. Used to share one stratified set of complexes
        across all models.
    """
    print(f"\nProcessing {model_source}:")
    print(f"  Loading: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")

    # Sample at the complex level so we keep every decoy per complex.
    unique_entries = df["entry"].drop_duplicates()
    print(f"  Unique complexes: {len(unique_entries)}")

    if preselected_entries is not None:
        df_sampled = df[df["entry"].isin(preselected_entries)]
        n_present = df_sampled["entry"].nunique()
        print(f"  Filtered to shared selection: {n_present} complexes -> {len(df_sampled)} decoys")
    elif len(unique_entries) > n_samples:
        sampled_entries = unique_entries.sample(n=n_samples, random_state=random_seed)
        df_sampled = df[df["entry"].isin(sampled_entries)]
        print(f"  Sampled: {n_samples} complexes -> {len(df_sampled)} decoys")
    else:
        df_sampled = df
        print(f"  Using all {len(unique_entries)} complexes -> {len(df_sampled)} decoys (< {n_samples})")

    decoys_per_complex = df_sampled.groupby("entry").size()
    print(
        f"  Decoys/complex: min={decoys_per_complex.min()} "
        f"max={decoys_per_complex.max()} mean={decoys_per_complex.mean():.1f}"
    )

    # Add model source
    df_sampled = df_sampled.copy()
    df_sampled["model_source"] = model_source

    return df_sampled


def _boltz2_condition(row, base_dir: Path) -> str:
    """Build the boltz2 grid condition dir name for one row.

    ``step_scale`` is read from the CSV as a float (e.g. 1.0), but the grid dirs
    are inconsistently named: some runs use "scale=1.0", others the bare integer
    "scale=1". Try both spellings and prefer whichever condition dir actually
    exists on disk, falling back to the float spelling when neither does.
    """
    scale = row["step_scale"]
    scale_variants = [f"{scale}"]
    if float(scale).is_integer():
        scale_variants.append(f"{int(scale)}")

    prefix = f"msa={row['msa_mode']}__recycling={int(row['recycling_steps'])}__scale="
    for sv in scale_variants:
        if (base_dir / (prefix + sv)).exists():
            return prefix + sv
    return prefix + scale_variants[0]


def _condition_and_pred_dir(row, base_dir: Path, model_source: str, entry: str) -> tuple[str, Path]:
    """Resolve a row's condition string and predicted-structure directory.

    Each folding model names its grid conditions from different CSV columns and
    nests its outputs differently, so both are derived together per model. An
    unrecognised ``model_source`` falls back to ``("unknown", base_dir / entry)``.
    """
    if model_source == "boltz2":
        condition = _boltz2_condition(row, base_dir)
        return condition, base_dir / condition / "boltz_results__staged_inputs" / "predictions" / entry

    if model_source == "esmfold2":
        condition = f"msa={row['msa_mode']}__loops={int(row['loops'])}__steps={int(row['steps'])}"
        return condition, base_dir / condition / entry

    if model_source == "protenix":
        condition = f"msa={row['msa_mode']}__cycle={int(row['cycle'])}__step={int(row['step'])}"
        # Protenix has seed_XXX/predictions/ structure
        return condition, base_dir / condition / entry / f"seed_{int(row['seed'])}" / "predictions"

    return "unknown", base_dir / entry


def convert_to_training_format(df: pd.DataFrame, base_dir: Path, model_source: str) -> pd.DataFrame:
    """
    Convert DockQ CSV to training manifest format.

    Expected columns in df:
    - entry: e.g., "1a14_HLN"
    - model_file: e.g., "1a14_HLN_seed42_sample0.pdb"
    - native_file: e.g., "1a14_HLN.pdb"
    - complex_dockq, complex_fnat, complex_irmsd, complex_lrmsd
    - model_source: Added by load_and_sample_dockq

    Returns manifest with columns:
    - pdb_id, chains, entry, pdb_path, model_source, condition
    - dockq, fnat, irmsd, lrmsd (fnat/irmsd/lrmsd are carried as provenance --
      only ``dockq`` is read as a training label)
    - ptm, iptm, plddt (the folding model's own confidence, for comparing the
      trained predictor against native confidence; not read during training)
    """
    manifest_entries = []

    for _, row in df.iterrows():
        entry = row["entry"]

        # Parse entry to get pdb_id and chains
        parts = entry.split("_")
        pdb_id = parts[0]
        chains = "_".join(parts[1:])

        # Condition string + output layout are both model-specific.
        condition, pred_dir = _condition_and_pred_dir(row, base_dir, model_source, entry)
        pdb_path = pred_dir / row["model_file"]

        manifest_entry = {
            "pdb_id": pdb_id,
            "chains": chains,
            "entry": entry,
            "pdb_path": str(pdb_path),
            "model_source": model_source,
            "condition": condition,
            "dockq": row["complex_dockq"],
            "fnat": row["complex_fnat"],
            "irmsd": row["complex_irmsd"],
            "lrmsd": row["complex_lrmsd"],
            # Include confidence metrics (column names differ by model)
            "ptm": row.get("ptm", np.nan),
            "iptm": row.get("iptm", np.nan),
            "plddt": row.get("plddt", row.get("complex_plddt", np.nan)),
        }

        manifest_entries.append(manifest_entry)

    return pd.DataFrame(manifest_entries)


def validate_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that every manifest row's predicted structure exists on disk.

    Only the structure file is required: the feature set is computed from
    coordinates alone, so a decoy with no PAE / confidence sidecar is perfectly
    trainable and must not be dropped.

    Returns filtered DataFrame with only valid entries.
    """
    print("\nValidating file existence...")

    valid_mask = []

    for idx, row in manifest_df.iterrows():
        valid = Path(row["pdb_path"]).exists()

        if not valid:
            print(f"  Missing PDB: {row['pdb_path']}")

        valid_mask.append(valid)

        if (idx + 1) % 100 == 0:
            n_valid = sum(valid_mask)
            print(f"  Validated {idx + 1}/{len(manifest_df)} entries ({n_valid} valid)")

    n_valid = sum(valid_mask)
    n_invalid = len(valid_mask) - n_valid

    print("\nValidation complete:")
    print(f"  Valid entries: {n_valid}")
    print(f"  Invalid entries: {n_invalid}")

    return manifest_df[valid_mask].copy()


def print_statistics(manifest_df: pd.DataFrame):
    """Print manifest statistics."""
    print("\n" + "=" * 80)
    print("MANIFEST STATISTICS")
    print("=" * 80)

    print(f"\nTotal entries: {len(manifest_df)}")

    print("\nEntries per model source:")
    print(manifest_df["model_source"].value_counts().sort_index())

    print("\nDockQ distribution:")
    print(manifest_df["dockq"].describe())

    # Quality tiers. Kept local rather than assigned onto manifest_df: callers print the
    # statistics and then write the manifest out (sample_dockq_for_training.py does both),
    # so a column added here would end up in the CSV the trainer reads.
    bins = [0, 0.23, 0.49, 0.8, 1.0]
    labels = ["Unacceptable (<0.23)", "Acceptable (0.23-0.49)", "Medium (0.49-0.8)", "High (>0.8)"]
    quality_tier = pd.cut(manifest_df["dockq"], bins=bins, labels=labels).rename("quality_tier")

    print("\nQuality tier distribution:")
    tier_counts = quality_tier.value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = 100 * count / len(manifest_df)
        print(f"  {tier:30s}: {count:4d} ({pct:5.1f}%)")

    print("\nQuality tier per model source:")
    crosstab = pd.crosstab(manifest_df["model_source"], quality_tier, margins=True)
    print(crosstab)

    print("\nDockQ statistics per model:")
    print(manifest_df.groupby("model_source")["dockq"].describe())
