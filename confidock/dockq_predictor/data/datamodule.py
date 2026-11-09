"""PyTorch Lightning DataModule for DockQ predictor."""

import random
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

from .dataset import DockQDataset, _native_id_from_path, create_data_splits

# Re-exported for callers that group batches by complex; the canonical
# definition lives in dataset.py so batch grouping and data.native_id stay in
# lockstep (same key derivation, one source of truth).
__all__ = ["_native_id_from_path"]


def _init_loader_worker(_worker_id: int) -> None:
    """Pin each dataloader worker to ONE intra-op torch thread.

    Workers inherit the parent's intra-op pool, which torch sizes from the visible
    HOST cpu count (128 here) rather than the pod's CPU budget -- so N workers
    quietly ask for N*128 threads. Graph loading is per-sample tensor work with
    nothing to parallelise intra-op, so those threads only add contention:
    benchmarked cold-build throughput at 16 workers was ~3x SLOWER than at 8 until
    this cap was applied.
    """
    torch.set_num_threads(1)


class GroupedComplexBatchSampler(Sampler):
    """Yield batches that pack whole chunks of decoys from the same complex.

    A within-complex ranking loss only has gradient when a batch holds >=2 decoys
    of the same complex. Random batching over ~800 complexes almost never does
    that, so this sampler builds each batch from a few complexes x
    ``decoys_per_complex`` decoys each. Every epoch it reshuffles complex order
    and the decoys within each complex, so batch composition still varies.

    ``batch_size`` is treated as decoys-per-batch; ``batch_size //
    decoys_per_complex`` complexes go into each batch (at least 1). The last,
    smaller batch is kept.
    """

    def __init__(
        self,
        entries: List[Dict],
        batch_size: int,
        decoys_per_complex: int,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.decoys_per_complex = max(2, decoys_per_complex)
        self.seed = seed
        self.epoch = 0

        # Map native_id -> list of dataset indices (order matches the dataset's
        # entries list, so these indices address the dataset directly).
        self.complex_to_indices: Dict[str, List[int]] = {}
        for idx, entry in enumerate(entries):
            nid = _native_id_from_path(entry["pdb_path"])
            self.complex_to_indices.setdefault(nid, []).append(idx)

        self._num_batches = None

    def _build_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        complexes_per_batch = max(1, self.batch_size // self.decoys_per_complex)

        # Emit fixed-size chunks of decoys per complex, then shuffle chunks and
        # pack them into batches. Reshuffled each epoch for varied composition.
        chunks: List[List[int]] = []
        for indices in self.complex_to_indices.values():
            idxs = indices[:]
            rng.shuffle(idxs)
            for start in range(0, len(idxs), self.decoys_per_complex):
                chunk = idxs[start : start + self.decoys_per_complex]
                if len(chunk) >= 2:  # a size-1 tail chunk carries no ranking signal
                    chunks.append(chunk)
        rng.shuffle(chunks)

        batches: List[List[int]] = []
        for start in range(0, len(chunks), complexes_per_batch):
            batch = [i for chunk in chunks[start : start + complexes_per_batch] for i in chunk]
            batches.append(batch)
        return batches

    def __iter__(self):
        """Yield batches for one epoch, advancing the shuffle seed."""
        batches = self._build_batches()
        self._num_batches = len(batches)
        # Advance so the next epoch reshuffles (DataLoader calls __iter__ once per
        # epoch). This self-advance is the ONLY thing that varies the shuffle: there
        # used to be a set_epoch() here for Lightning to drive, but Lightning's
        # _set_sampler_epoch looks at dataloader.sampler and dataloader.batch_sampler
        # .sampler -- and this class IS the batch_sampler, with no .sampler attribute
        # -- so it was never called. Verified against lightning_fabric 2.5.0.
        self.epoch += 1
        yield from batches

    def __len__(self):
        """Return the number of batches in one epoch."""
        if self._num_batches is None:
            self._num_batches = len(self._build_batches())
        return self._num_batches


class FilteredDockQDataset(DockQDataset):
    """Wrapper that substitutes another decoy of the SAME complex for a failed sample.

    ``DockQDataset.get`` returns None for a structure it cannot parse or featurise
    rather than raising, so one malformed file cannot kill a run -- in practice that is
    a handful of files in the whole manifest, hit once per epoch each. Something must
    take the failed sample's place, because the batch it belongs to was built by
    ``GroupedComplexBatchSampler`` as chunks of >=2 decoys per complex, and that
    grouping is what gives the within-complex ranking term any gradient at all.

    The replacement is drawn from the failed entry's own complex, never from the dataset
    at large. A global draw would land in some other complex, where the ranking loss
    regroups it by ``native_id`` into a singleton and drops it -- so the chunk would
    silently rank N-1 decoys instead of N, and an unrelated complex would be
    oversampled. Staying inside the complex keeps the chunk's size and identity intact.

    A complex whose decoys are all unusable raises: at that point there is no honest
    substitute, and the message names the file so the row can be dropped from the
    manifest.
    """

    #: Substitution attempts before giving up on an index.
    max_retries = 10

    def __init__(self, *args, **kwargs):
        """Build the dataset and prepare the per-complex index, filled in on first use."""
        super().__init__(*args, **kwargs)
        self._complex_to_indices: Optional[Dict[str, List[int]]] = None
        self._reported: set = set()

    def _siblings(self, idx: int) -> List[int]:
        """Return every dataset index belonging to ``idx``'s complex, itself included.

        Built once per worker process on first failure, so a run with no unparseable
        structure never pays for it. Keyed with ``_native_id_from_path`` -- the same
        derivation ``GroupedComplexBatchSampler`` and ``data.native_id`` use, so
        "same complex" means the same thing in all three places.
        """
        if self._complex_to_indices is None:
            mapping: Dict[str, List[int]] = {}
            for i, entry in enumerate(self.entries):
                mapping.setdefault(_native_id_from_path(entry["pdb_path"]), []).append(i)
            self._complex_to_indices = mapping
        return self._complex_to_indices[_native_id_from_path(self.entries[idx]["pdb_path"])]

    def _report_once(self, idx: int, exc: Optional[Exception]) -> None:
        """Name a failing structure the first time it is seen, then stay quiet.

        The same bad file is hit once per epoch for the length of the run, and the
        substitution is otherwise invisible -- one line per file is what makes it
        possible to prune the manifest afterwards.
        """
        if idx in self._reported:
            return
        self._reported.add(idx)
        reason = f": {type(exc).__name__}: {exc}" if exc is not None else ""
        print(
            f"Warning: {self.entries[idx]['pdb_path']} could not be featurised{reason}. "
            "Substituting another decoy of the same complex."
        )

    def __getitem__(self, idx):
        """Get item ``idx``, falling back to other decoys of its complex if it fails."""
        origin = idx
        tried: set = set()

        for _ in range(self.max_retries):
            try:
                data = super().__getitem__(idx)
            except Exception as exc:  # noqa: BLE001 -- one bad structure must not kill the run
                self._report_once(idx, exc)
                data = None
            else:
                if data is not None:
                    return data
                self._report_once(idx, None)  # get() already printed the underlying cause

            tried.add(idx)
            candidates = [i for i in self._siblings(origin) if i not in tried]
            if not candidates:
                break
            idx = random.choice(candidates)

        raise RuntimeError(
            f"No usable decoy for complex {_native_id_from_path(self.entries[origin]['pdb_path'])}: "
            f"{len(tried)} of its structure(s) failed to featurise, including "
            f"{self.entries[origin]['pdb_path']}. Drop them from the manifest."
        )


class DockQDataModule(pl.LightningDataModule):
    """
    DataModule for DockQ predictor training.

    Handles data splitting, dataset creation, and dataloader configuration.
    """

    def __init__(
        self,
        entries: List[Dict],
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        interface_cutoff: float = 8.0,
        knn_k: int = 16,
        random_seed: int = 42,
        cache_graphs: bool = True,
        graph_cache_dir: Optional[str] = None,
        group_by_complex: bool = False,
        decoys_per_complex: int = 8,
    ):
        """
        Initialize DataModule.

        Parameters
        ----------
        entries : List[Dict]
            List of data entries with pdb_path, model_source, chain_str and dockq
        val_fraction : float
            Fraction of PDB IDs for validation
        test_fraction : float
            Fraction of PDB IDs for testing
        batch_size : int
            Batch size for dataloaders. With ``group_by_complex`` this is
            decoys-per-batch, not complexes-per-batch.
        num_workers : int
            Number of worker processes for data loading. Training here is
            dataloader-bound, not GPU-bound, so this is the throughput knob.
        pin_memory : bool
            Whether to pin memory for faster GPU transfer
        interface_cutoff : float
            Distance cutoff for interface residues (Angstroms)
        knn_k : int
            Number of nearest neighbors for k-NN graph
        random_seed : int
            Random seed for data splitting and for the grouped sampler
        cache_graphs : bool
            Whether datasets cache built graphs in memory. Turn OFF when using a
            shared on-disk cache with many workers, so worker RAM stays flat.
        graph_cache_dir : str, optional
            Directory for an on-disk graph cache shared by all splits. Builds each
            graph once and reads it back on later epochs without per-worker RAM
            growth -- the memory-safe alternative to ``cache_graphs``.
        group_by_complex : bool
            Pack each training batch with whole chunks of same-complex decoys.
            REQUIRED for the within-complex ranking loss: with random batching a
            batch almost never holds two decoys of one complex, and the ranking
            term is then a silent no-op.
        decoys_per_complex : int
            Decoys per complex per batch when ``group_by_complex`` is set.
        """
        super().__init__()

        self.entries = entries
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.group_by_complex = group_by_complex
        self.decoys_per_complex = decoys_per_complex
        self.interface_cutoff = interface_cutoff
        self.knn_k = knn_k
        self.random_seed = random_seed
        self.cache_graphs = cache_graphs
        self.graph_cache_dir = graph_cache_dir

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Store splits for later access
        self.train_entries = None
        self.val_entries = None
        self.test_entries = None

    def setup(self, stage: Optional[str] = None):
        """
        Split data and create datasets.

        Called on every GPU in distributed training.

        Parameters
        ----------
        stage : str, optional
            'fit', 'validate', 'test', or 'predict'
        """
        # Split data only if not already split
        if self.train_entries is None:
            self.train_entries, self.val_entries, self.test_entries = create_data_splits(
                self.entries,
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
                random_seed=self.random_seed,
            )

        # Graph/feature settings shared by every split.
        dataset_kwargs = dict(
            interface_cutoff=self.interface_cutoff,
            knn_k=self.knn_k,
            cache_graphs=self.cache_graphs,
            graph_cache_dir=self.graph_cache_dir,
        )

        # Create datasets based on stage
        if stage == "fit" or stage is None:
            self.train_dataset = FilteredDockQDataset(self.train_entries, **dataset_kwargs)
            self.val_dataset = FilteredDockQDataset(self.val_entries, **dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_dataset = FilteredDockQDataset(self.test_entries, **dataset_kwargs)

    def train_dataloader(self):
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup('fit') must be called before requesting train dataloader")

        common = dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Keep workers alive across epochs so each worker's in-memory graph
            # cache (DockQDataset._cache) is reused instead of rebuilt every epoch.
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_init_loader_worker if self.num_workers > 0 else None,
        )

        # Grouped batching packs multiple decoys of the same complex into each
        # batch so the within-complex ranking loss has gradient. batch_sampler is
        # mutually exclusive with batch_size/shuffle, so branch here.
        if self.group_by_complex:
            self._train_sampler = GroupedComplexBatchSampler(
                self.train_entries,
                batch_size=self.batch_size,
                decoys_per_complex=self.decoys_per_complex,
                seed=self.random_seed,
            )
            return DataLoader(self.train_dataset, batch_sampler=self._train_sampler, **common)

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, **common)

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup('fit') must be called before requesting val dataloader")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_init_loader_worker if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("DataModule.setup('test') must be called before requesting test dataloader")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_init_loader_worker if self.num_workers > 0 else None,
        )

    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"{self.__class__.__name__}("
            f"n_entries={len(self.entries)}, "
            f"batch_size={self.batch_size}, "
            f"val_fraction={self.val_fraction:.0%}, test_fraction={self.test_fraction:.0%})"
        )
