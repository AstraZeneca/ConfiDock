"""PyTorch Geometric Dataset for DockQ prediction."""

import hashlib
import os
from typing import Any, Dict, List, Optional

import torch
from torch_geometric.data import Data, Dataset

from .feature_utils import compute_edge_features, compute_node_features, edge_feature_dim, node_feature_dim
from .graph_builder import build_graph

# Bump when a feature block's CONTENT changes without changing its width.
#
# This is the only guard against silently reusing stale graphs from the on-disk
# cache. The width changes below are caught automatically (nodedim/edgedim are in
# the key), but editing, say, the RBF gamma or the interaction-type distance
# thresholds keeps the width identical while changing every value -- and a cache
# hit would then feed the model features it was never built with. There is no
# cheap way to hash the feature CODE, so this is a manual counter: change a
# feature computation, bump this.
FEATURE_VERSION = 1


def _native_id_from_path(pdb_path: str) -> str:
    """Complex key = "{pdb_id}_{chains}" (first two '_'-tokens of the filename).

    NOT just the pdb_id: ~60 pdb_ids in this set have multiple binding modes that
    differ only by chain string, so keying on pdb_id alone would merge distinct
    complexes. This matches the manifest 'entry' column.

    The extension is stripped FIRST: decoy files carry a "_sample_N" suffix
    (``7y8j_HLA_sample_0.cif``) so the first two tokens are already clean, but a
    single-output grid cell emits a bare ``7y8j_HLA.cif`` whose second token is
    ``HLA.cif`` -- without stripping, that decoy splits off into its own singleton
    group ("7y8j_HLA.cif") instead of joining "7y8j_HLA".
    """
    stem = os.path.splitext(pdb_path.split("/")[-1])[0]
    return "_".join(stem.split("_")[:2])


def _chain_types_from_entry(entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Chain typing from the manifest's chain string, or None to fall back.

    Returns None when the entry has no chain string, when parsing fails, or
    when the parsed string carries no antibody chain (e.g. "EFC") -- in all
    three cases the graph builder falls back to JSON metadata or ANARCI.

    Module-level, not a ``DockQDataset`` staticmethod: it reads no class state,
    and in the class body ``entry.get(...)`` sits beside the dataset's own
    ``get(idx: int)``, which static analysis resolves to in preference to
    ``dict.get`` -- reporting the string key as a wrongly-typed argument.
    """
    chain_str = entry.get("chain_str")
    if chain_str is None:
        return None

    from .chain_mapping import parse_chain_string

    try:
        chain_types = parse_chain_string(chain_str, entry["pdb_path"])
    except Exception as e:
        print(f"Warning: Failed to parse chain string '{chain_str}': {e}")
        return None

    if not any(ct in ["VH", "VL"] for ct in chain_types.values()):
        return None
    return chain_types


class DockQDataset(Dataset):
    """
    Dataset for DockQ prediction.

    Each sample is a predicted antibody-antigen complex, turned into a
    residue-level interface graph with an 18-dim node / 96-dim edge feature set
    (see ``feature_utils``), labelled with its ground-truth DockQ score.
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        interface_cutoff: float = 8.0,
        knn_k: int = 16,
        cache_graphs: bool = True,
        graph_cache_dir: Optional[str] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        entries : List[Dict[str, Any]]
            List of data entries, each with:
            - pdb_path: str - path to the predicted complex (PDB or CIF)
            - model_source: str - "boltz2", "esmfold2" or "protenix"
            - chain_str: str, optional - chain annotation from the manifest
            - dockq: float - ground truth DockQ score
        interface_cutoff : float
            Distance cutoff for flagging interface residues (Angstroms)
        knn_k : int
            Number of nearest neighbors for the k-NN graph
        cache_graphs : bool
            Keep built graphs in memory. Residue graphs are small enough that this
            is worth it; combined with ``persistent_workers`` it means each worker
            builds its share once and reads them from RAM thereafter.
        graph_cache_dir : str, optional
            Directory for an on-disk graph cache. When set, each built graph is
            saved to / loaded from a ``.pt`` file keyed by a content hash of all
            inputs that determine the graph. This pays graph construction once
            (first epoch) and reads pre-built graphs thereafter, WITHOUT the
            per-worker RAM growth of the in-memory cache. Independent of and
            combinable with ``cache_graphs``.
        transform : callable, optional
            Transform applied to each sample
        pre_transform : callable, optional
            Pre-transform applied once during processing
        pre_filter : callable, optional
            Filter to exclude samples
        """
        self.entries = entries
        self.interface_cutoff = interface_cutoff
        self.knn_k = knn_k
        self.cache_graphs = cache_graphs
        self.graph_cache_dir = graph_cache_dir
        if self.graph_cache_dir is not None:
            os.makedirs(self.graph_cache_dir, exist_ok=True)

        # In-memory cache for graphs
        self._cache = {}

        super().__init__(None, transform, pre_transform, pre_filter)

    def len(self) -> int:
        """Return number of samples."""
        return len(self.entries)

    def _disk_cache_path(self, entry: Dict[str, Any]) -> Optional[str]:
        """Return the on-disk cache file path for an entry, or None if disabled.

        The key hashes every input that affects the built graph's content, so a
        cached file is only reused when it would be byte-identical to a fresh
        build. Changing a cutoff, knn_k, the feature widths or FEATURE_VERSION
        yields a different key, so stale graphs are never silently reused.
        """
        if self.graph_cache_dir is None:
            return None
        key_parts = [
            str(entry.get("pdb_path")),
            str(entry.get("model_source")),
            str(entry.get("chain_str")),
            f"cutoff={self.interface_cutoff}",
            f"knn={self.knn_k}",
            f"nodedim={node_feature_dim()}",
            f"edgedim={edge_feature_dim()}",
            f"featver={FEATURE_VERSION}",
        ]
        digest = hashlib.sha256("||".join(key_parts).encode()).hexdigest()
        return os.path.join(self.graph_cache_dir, f"{digest}.pt")

    def _load_cached_graph(self, disk_path: Optional[str], pdb_path: str) -> Optional[Data]:
        """Load a graph from the on-disk cache, or None on a miss.

        Corrupt/partial files (e.g. from an interrupted write) are treated as a
        miss so the caller rebuilds.
        """
        if disk_path is None or not os.path.exists(disk_path):
            return None
        try:
            data = torch.load(disk_path, weights_only=False)
        except Exception as e:
            print(f"Warning: failed to load cached graph {disk_path}: {e}")
            return None
        # (Re)derive native_id on every load. It's metadata, not part of the
        # cache-key hash, so this keeps a large on-disk cache valid while making
        # it immune to a stale/buggy value: graphs cached before the
        # extension-stripping fix stored the wrong id (e.g. "7y8j_HLA.cif"), and
        # those DO have the attribute set, so an "only if missing" backfill would
        # leave them wrong. Always overwriting is cheap and self-correcting.
        data.native_id = _native_id_from_path(pdb_path)
        return data

    @staticmethod
    def _write_cached_graph(data: Data, disk_path: Optional[str]) -> None:
        """Persist a built graph to the on-disk cache (no-op when disabled).

        Writes to a unique temp file then renames (atomic on the same filesystem)
        so a concurrent worker or an interrupted write can never leave a
        half-written file that a later read would trust.
        """
        if disk_path is None:
            return
        try:
            tmp_path = f"{disk_path}.tmp.{os.getpid()}"
            torch.save(data, tmp_path)
            os.replace(tmp_path, disk_path)
        except Exception as e:
            print(f"Warning: failed to write cached graph {disk_path}: {e}")

    def get(self, idx: int) -> Data:
        """
        Get a single sample.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        Data
            PyG Data object with features and labels, or None if the structure
            could not be parsed / featurised (``FilteredDockQDataset`` resamples).
        """
        # Check in-memory cache first
        if self.cache_graphs and idx in self._cache:
            return self._cache[idx]

        entry = self.entries[idx]

        # Check on-disk cache next. A hit skips graph construction + feature
        # computation entirely.
        disk_path = self._disk_cache_path(entry)
        data = self._load_cached_graph(disk_path, entry["pdb_path"])

        if data is None:
            data = self._build_sample(entry)
            if data is None:
                return None  # Unparseable / unfeaturisable structure
            self._write_cached_graph(data, disk_path)

        # Cache the processed graph in memory
        if self.cache_graphs:
            self._cache[idx] = data

        return data

    def _build_sample(self, entry: Dict[str, Any]) -> Optional[Data]:
        """Build and featurise one entry's graph, or None if any stage fails."""
        pdb_path = entry["pdb_path"]
        model_source = entry["model_source"]
        dockq = entry["dockq"]

        # Get chain types from manifest if available
        chain_types = _chain_types_from_entry(entry)

        # Build graph
        try:
            data = build_graph(
                pdb_path=pdb_path,
                interface_cutoff=self.interface_cutoff,
                knn_k=self.knn_k,
                chain_types=chain_types,
            )
        except Exception as e:
            print(f"Error building graph for {pdb_path}: {e}")
            return None  # Skip this sample instead of crashing

        # Compute node features. Real per-node pLDDT comes from the graph builder
        # (B-factor column of the predicted structure); fall back to a flat 70.0
        # only if it is absent, in which case the plddt block is constant and
        # carries no information.
        try:
            plddt = data.plddt if hasattr(data, "plddt") else torch.ones(data.num_nodes) * 70.0
            data.x = compute_node_features(
                residues=data.residues,
                cdr_labels=data.cdr_label,
                plddt=plddt,
            )
        except Exception as e:
            print(f"Error computing node features for {pdb_path}: {e}")
            return None  # Skip this sample

        # Compute edge features
        try:
            data.edge_attr = compute_edge_features(
                edge_index=data.edge_index,
                pos=data.pos,
                residues=data.residues,
                model_source=model_source,
                chain_ids=data.chain_id,
            )
        except Exception as e:
            print(f"Error computing edge features for {pdb_path}: {e}")
            return None  # Skip this sample

        # Add label
        data.y_dockq = torch.tensor([dockq], dtype=torch.float32)

        # Add metadata
        data.pdb_path = pdb_path
        data.model_source = model_source
        # Complex identifier = "{pdb_id}_{chains}" (first two '_'-tokens of the
        # filename, e.g. "1a3r_HLP"). NOT just the pdb_id: ~60 pdb_ids in this
        # set have multiple binding modes that differ only by chain string, so
        # keying on pdb_id alone would merge distinct complexes. This matches the
        # manifest 'entry' column. Attached here so a batch carries per-graph
        # native ids for the within-complex ranking loss.
        data.native_id = _native_id_from_path(pdb_path)

        # Clean up temporary attributes. `residues` carries Bio.PDB-derived dicts
        # (including per-residue heavy-atom arrays) that the features have already
        # consumed; leaving them on would multiply the cache size and make the
        # graph un-collatable into a batch.
        delattr(data, "residues")
        delattr(data, "chain_types")

        return data

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}({len(self)} samples)"


def create_data_splits(
    entries: List[Dict[str, Any]],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> tuple:
    """
    Create train/val/test splits, grouped by PDB ID.

    Splitting is ALWAYS by pdb_id, never by row. Each complex contributes many
    decoys, so a row-wise split puts decoys of the same complex on both sides of
    the boundary and the val score measures memorisation rather than
    generalisation. Grouping by pdb_id (not by native_id) is the stricter choice:
    it also keeps the alternative binding modes of one pdb_id together.

    Parameters
    ----------
    entries : List[Dict[str, Any]]
        All data entries
    val_fraction : float
        Fraction of PDB IDs for validation
    test_fraction : float
        Fraction of PDB IDs for test
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (train_entries, val_entries, test_entries)
    """
    import random
    from pathlib import Path

    random.seed(random_seed)

    # Group entries by PDB ID
    pdb_groups = {}
    for entry in entries:
        # Extract PDB ID from path (e.g., "7xa3_HcLcA" -> "7xa3")
        pdb_path = entry["pdb_path"]
        filename = Path(pdb_path).name
        pdb_id = filename.split("_")[0]  # Take first part before underscore

        if pdb_id not in pdb_groups:
            pdb_groups[pdb_id] = []
        pdb_groups[pdb_id].append(entry)

    # Shuffle PDB IDs
    pdb_ids = list(pdb_groups.keys())
    random.shuffle(pdb_ids)

    # Split PDB IDs into train/val/test
    n_pdbs = len(pdb_ids)
    n_test_pdbs = int(n_pdbs * test_fraction)
    n_val_pdbs = int(n_pdbs * val_fraction)
    n_train_pdbs = n_pdbs - n_test_pdbs - n_val_pdbs

    train_pdb_ids = pdb_ids[:n_train_pdbs]
    val_pdb_ids = pdb_ids[n_train_pdbs : n_train_pdbs + n_val_pdbs]
    test_pdb_ids = pdb_ids[n_train_pdbs + n_val_pdbs :]

    # Collect all entries for each split
    train_entries = []
    val_entries = []
    test_entries = []

    for pdb_id in train_pdb_ids:
        train_entries.extend(pdb_groups[pdb_id])
    for pdb_id in val_pdb_ids:
        val_entries.extend(pdb_groups[pdb_id])
    for pdb_id in test_pdb_ids:
        test_entries.extend(pdb_groups[pdb_id])

    # Shuffle within each split
    random.shuffle(train_entries)
    random.shuffle(val_entries)
    random.shuffle(test_entries)

    print(f"Split by PDB ID: {len(train_pdb_ids)} train, {len(val_pdb_ids)} val, {len(test_pdb_ids)} test PDBs")
    print(f"  -> {len(train_entries)} train, {len(val_entries)} val, {len(test_entries)} test samples")

    return train_entries, val_entries, test_entries
