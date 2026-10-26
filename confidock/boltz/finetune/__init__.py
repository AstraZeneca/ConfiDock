"""Fine-tune a DockQ head on top of Boltz-2's confidence module.

Two tracks share this package, both reading the cache built by
``workflows/scripts/inference/cache_boltz2_confidence.py``:

* **Frozen-latent probe** -- run the frozen confidence module once, store the pooled
  640-wide latents, then train a 0.2M-parameter MLP over that table. An epoch is
  seconds. Entry point ``workflows/scripts/training/train_dockq_probe.py``.
* **Unfrozen fine-tune** -- train all 25M parameters of ``confidence_module``
  alongside the head, ~7.4 h/epoch on 6 GPUs. Entry point
  ``workflows/scripts/training/finetune_confidence_ddp.py``.

Measured on ``sabdab_val_easy`` the two are statistically indistinguishable on
per-native Spearman, so prefer the probe for iteration; see
``docs/boltz_dockq_head_results.md``.

Sub-modules
-----------
cache_dataset
    Entry-grouped dataset over the confidence cache, plus the token-aware decoy
    budget that keeps large entries from OOMing.
dockq_head
    The DockQ readout that consumes the confidence module's ``s_conf``/``z_conf``
    latents, its interface-aware pooling, and its loss (regression + within-entry
    ranking).
trainer
    Training / evaluation loop and the correlation report, shared by both tracks.
distributed
    Hand-rolled gradient averaging and collective-safe step counting. The unfrozen
    fine-tune cannot use ``DistributedDataParallel``: decoy chunking varies per
    entry, activation checkpointing re-runs the forward, and ~140 of 452 confidence
    tensors get no gradient from a DockQ-only loss.
latent_cache
    On-disk format for the pooled latents the probe trains on.
probe
    The probe's train/eval loop over that table.
"""
