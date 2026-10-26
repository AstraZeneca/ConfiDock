"""Training entrypoint for Boltz confidence fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from confidock.boltz.datamodule import BoltzFinetuneDataModule
from confidock.boltz.model import BoltzFinetune


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Boltz-2 confidence module")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume training from (optional)"
    )
    return parser.parse_args()


def main() -> None:
    """Train Boltz model with Lightning."""
    args = _parse_args()
    config = OmegaConf.load(args.config)

    datamodule = BoltzFinetuneDataModule(
        structure_dir=config.paths.structure_dir,
        processed_dir=config.paths.processed_dir,
        train_manifest=config.paths.train_manifest,
        val_manifest=config.paths.val_manifest,
        num_workers=config.datamodule.num_workers,
        pin_memory=config.datamodule.pin_memory,
        max_tokens=config.dataset.max_tokens,
        max_atoms=config.dataset.max_atoms,
    )

    model_overrides = dict(OmegaConf.to_container(config.model, resolve=True))
    model_overrides["training_args"] = OmegaConf.to_container(config.training, resolve=True)
    model_overrides["validation_args"] = OmegaConf.to_container(config.validation, resolve=True)

    model = BoltzFinetune.from_pretrained(
        checkpoint_path=config.paths.pretrained_checkpoint,
        **model_overrides,
    )

    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir),
            monitor="val/confidence_loss",
            save_top_k=3,
            mode="min",
            filename="boltz-finetune-{step}-{val_confidence_loss:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_steps=config.training.max_steps,
        precision="bf16-mixed",
        gradient_clip_val=float(config.training.gradient_clip_val),
        accumulate_grad_batches=int(config.training.accumulate_grad_batches),
        val_check_interval=500,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
    )

    # Resume from checkpoint if provided
    ckpt_path = args.resume if args.resume else None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
