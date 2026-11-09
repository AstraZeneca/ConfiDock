"""PyTorch Lightning datamodule for Boltz confidence fine-tuning."""

from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import BoltzFinetuneDataset


class BoltzFinetuneDataModule(pl.LightningDataModule):
    """DataModule wrapper around train/validation Boltz fine-tuning datasets."""

    def __init__(
        self,
        structure_dir: str,
        processed_dir: str,
        train_manifest: str,
        val_manifest: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_tokens: int = 512,
        max_atoms: int = 4096,
    ) -> None:
        super().__init__()
        self.structure_dir = structure_dir
        self.processed_dir = processed_dir
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.max_tokens = int(max_tokens)
        self.max_atoms = int(max_atoms)

        self.train_dataset: BoltzFinetuneDataset | None = None
        self.val_dataset: BoltzFinetuneDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets for the fit stage."""
        if stage in (None, "fit"):
            self.train_dataset = BoltzFinetuneDataset(
                structure_dir=self.structure_dir,
                processed_dir=self.processed_dir,
                split_manifest=self.train_manifest,
                max_tokens=self.max_tokens,
                max_atoms=self.max_atoms,
            )
            self.val_dataset = BoltzFinetuneDataset(
                structure_dir=self.structure_dir,
                processed_dir=self.processed_dir,
                split_manifest=self.val_manifest,
                max_tokens=self.max_tokens,
                max_atoms=self.max_atoms,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader with batch size fixed to 1."""
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup('fit') must be called before requesting train dataloader")

        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: x[0],
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader with batch size fixed to 1."""
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup('fit') must be called before requesting val dataloader")

        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda x: x[0],
        )
