# Boltz-2 Confidence Head Input Caching

This guide explains how to cache confidence head inputs during Boltz-2 inference and use them to fine-tune the confidence head efficiently.

For what the fine-tuned head actually achieved on `sabdab_val_easy` — and how it
compares against Boltz-2's own confidence metrics and the GVP GNN — see
[boltz_dockq_head_results.md](boltz_dockq_head_results.md).

## Overview

Fine-tuning the Boltz-2 confidence head traditionally requires:
1. Running the full trunk (MSA processing + structure prediction)
2. Computing confidence metrics
3. Backpropagating through the confidence head only

The **trunk computation is expensive** and identical across fine-tuning epochs if you're only updating the confidence head. This caching system allows you to:
- **Run inference once** and cache the trunk outputs
- **Fine-tune the confidence head** using only cached activations
- **Save ~95% of compute time** during fine-tuning

## Architecture

Based on the actual Boltz-2 model code:

```
┌─────────────────────────────────────────────────────────────────┐
│ Boltz-2 Model                                                   │
│                                                                 │
│  ┌──────────────────┐                                          │
│  │ Input Embedder   │  (processes sequences, MSA)              │
│  └────────┬─────────┘                                          │
│           │                                                     │
│  ┌────────▼─────────┐                                          │
│  │  Template Module │  (optional, uses templates)              │
│  └────────┬─────────┘                                          │
│           │                                                     │
│  ┌────────▼─────────┐                                          │
│  │   MSA Module     │  (processes MSA)                         │
│  └────────┬─────────┘                                          │
│           │                                                     │
│  ┌────────▼─────────┐                                          │
│  │ Pairformer Stack │  (main trunk, expensive)                 │
│  │  Recycling: 0-3  │                                          │
│  └────────┬─────────┘                                          │
│           │                                                     │
│           ├──────────► s (token repr) ──────┐                  │
│           ├──────────► z (pair repr)  ──────┤                  │
│           │                                  │                  │
│  ┌────────▼──────────────┐                  │                  │
│  │ Distogram Module      │                  │                  │
│  └───────────────────────┘                  │                  │
│           │                                  │                  │
│  ┌────────▼──────────────┐                  │                  │
│  │ Diffusion Conditioning│                  │                  │
│  └────────┬──────────────┘                  │                  │
│           │                                  │                  │
│  ┌────────▼──────────────┐                  │                  │
│  │  Structure Module     │ (atom diffusion) │                  │
│  │  (AtomDiffusion)      │                  │                  │
│  └────────┬──────────────┘                  │                  │
│           │                                  │                  │
│           └── x_pred (coords) ──────────────┤                  │
│                                              │                  │
│           ┌──────────────────────────────────┘                  │
│           │  ◄─── CACHE THESE ACTIVATIONS                      │
│           │                                                     │
│  ┌────────▼──────────────────────────────┐                     │
│  │     Confidence Module                 │ (trainable)         │
│  │  Inputs:                              │                     │
│  │   - s_inputs (input embeddings)       │                     │
│  │   - s (token repr from trunk)         │                     │
│  │   - z (pair repr from trunk)          │                     │
│  │   - x_pred (predicted coordinates)    │                     │
│  │   - pred_distogram_logits             │                     │
│  │   - feats (masks, indices, etc.)      │                     │
│  │  Outputs:                             │                     │
│  │   - plddt, pde, resolved              │                     │
│  │   - pae, ptm, iptm (if alpha_pae > 0) │                     │
│  └───────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

The confidence module is called with these keyword arguments:
```python
confidence_module(
    s_inputs=s_inputs.detach(),          # Input embeddings
    s=s.detach(),                        # Token representations
    z=z.detach(),                        # Pair representations
    x_pred=sample_atom_coords.detach(),  # Predicted structure
    feats=feats,                         # Input features dict
    pred_distogram_logits=pdistogram.detach(),
    multiplicity=diffusion_samples,      # Number of samples
    run_sequentially=False,              # Whether to run samples sequentially
    use_kernels=True,                    # Use optimized kernels
)
```
## Why the trunk only runs once per entry

**The trunk never sees coordinates.** In `boltz/model/modules/trunkv2.py` the only
coordinate references are `feats["template_cb"]`/`feats["template_ca"]` inside the
template modules, which this pipeline does not use. So for a given entry `s`/`z`
are a function of sequence + MSA alone and are identical for every decoy of that
entry.

That gives the layout:

| scope | count | contents |
| --- | --- | --- |
| per entry | 3,260 | `s_inputs`, `s`, `z`, and 18 `feats` tensors |
| per decoy | 108,041 | `x_pred` coordinates + presence mask, ~33 KiB |

Measured on an A100 for a 600-token entry: the trunk takes 38.4 s and the head
1.25 s/decoy, so a 33-decoy entry costs ~80 s instead of ~1322 s (~16.5x).

## Scoring semantics

The model runs with `skip_run_structure = True`. At `boltz2.py:750` that makes the
head receive `feats["coords"].repeat_interleave(diffusion_samples, 0)` rather than
a freshly diffused structure, so **the head scores exactly the decoy supplied**.
This keeps each cached record in 1:1 correspondence with that decoy's known DockQ
label.

Passing decoys as YAML `templates` would not: Boltz would re-predict a structure,
and the cached inputs would describe that new prediction rather than the structure
the label was computed from.

## Usage

Decoys are collected from all four prediction sources under `sabdab_train`
(`boltz2_r1`, `esmfold2_r1`, `protenix_r1`, `perturbed_structures`) and joined to
their existing `dockq_scores.csv` labels.

Single GPU:

```bash
poetry run python workflows/scripts/inference/cache_boltz2_confidence.py \
    --output_dir /data/confidock/sabdab_train/confidence_cache
```

All 6 GPUs with automatic restart (one shard per GPU, round-robin over entries):

```bash
./workflows/scripts/inference/cache_boltz2_confidence_auto_restart.sh \
    /data/confidock/sabdab_train/confidence_cache
```

The wrapper defaults to every GPU `nvidia-smi` reports; pass a list to restrict it
(`... /path/to/cache 0,1,2`), and any trailing arguments are forwarded to the
Python script. Runs are resumable: an entry whose `scores/<entry>.jsonl` already
holds all its decoys is skipped, so a killed shard resumes at the entry it was
interrupted on.

This reuses the `processed/` directory from the earlier `boltz2_r1` grid run, which
already holds structures, records, MSAs, constraints and the manifest for all
entries. That is deliberate: inheriting Boltz's own tokenisation guarantees the
atom ordering that `x_pred` has to match.

## Cost

| item | size |
| --- | --- |
| entry caches (fp16 `z`, ~0.262 KiB per token²) | 148.8 GiB |
| decoy coordinates (fp32) | 3.9 GiB |
| **total** | **152.7 GiB** |

Budget at least ~160 GiB of free space on the volume you point `--output_dir` at; a home
directory or container-local disk will not hold it.

Wall-clock: ~40 h on one GPU, ~7.1 h across 6 shards (shards span 6.2-7.1 h).

Token counts across the 3,260 entries: mean 405, p95 600, max 600.

## Output layout

```
{output_dir}/
    run_config.shard<N>.json     # full configuration of each shard
    entries/<entry>.pt           # s_inputs, s, z, feats
    decoys/<entry>/<id>.npz      # x_pred + presence mask + metadata
    scores/<entry>.jsonl         # confidence metrics + DockQ per decoy
    manifest.shard<N>.jsonl      # one line per completed entry
    failures.jsonl               # per-decoy errors, if any
    logs/shard<N>.log            # wrapper logs
```

## Loading a cache for fine-tuning

```python
import torch
from confidock.boltz.inference.cache_runner import (
    build_x_pred, feats_for_confidence, run_confidence_head)
from confidock.boltz.inference.confidence_cache import load_entry_cache

cache = load_entry_cache(path, device="cuda", dtype=torch.float32)
feats = feats_for_confidence(cache.feats, device="cuda")   # adds the coords slot
x_pred = build_x_pred(coords, feats, device="cuda")        # pads the atom axis
out = run_confidence_head(model, s_inputs=cache.s_inputs, s=cache.s, z=cache.z,
                          feats=feats, coords=x_pred)
```

Two shape details the helpers exist to handle:

* `coords` as stored is **unpadded** (one row per `structure.atoms`), while the
  featurizer pads the atom axis (e.g. 2752 slots for 2725 real atoms). Passing the
  unpadded array raises `Expected size for first two dimensions of batch2 tensor`.
* `CONFIDENCE_FEAT_KEYS` deliberately excludes `coords`, since each decoy brings
  its own; `feats_for_confidence` restores a zero placeholder of the right shape.

`pred_distogram_logits` is deliberately **not** stored: it is one linear layer on
`z` (`distogram_module(z)[:, :, :, 0]`), recomputed on load, which saves ~71 GiB.

The head's real outputs are `plddt`, `pae`, `pde`, `ptm`, `iptm`, `protein_iptm`,
`ligand_iptm`, `resolved_logits`, the `*_logits` variants, and the `complex_*`
scalars. There is no `dockq_pred` output — DockQ is the *label*, carried in
`scores/<entry>.jsonl` and in each decoy npz's metadata.

## Storage decisions

`feats` is compacted before writing. The three one-hot matrices `atom_to_token`,
`token_to_rep_atom` and `r_set_to_rep_atom` (plus the 5-wide
`contact_conditioning`) are `int64` and 21.5 MiB *each* at 600 tokens — 220 GiB
across the dataset, more than `z` itself. Every row sums to 1, except
`atom_to_token`'s padded-atom rows which are all-zero, so they are stored as index
vectors with `-1` marking an empty row. Measured: 84 MiB → 4.9 MiB per entry,
verified lossless.

## Correctness checks performed

* **The split is exact.** Trunk-once + head-per-decoy reproduces the full
  `model.forward` bit-exactly (max |Δ| = 0.0) with a fixed seed. The seed matters:
  `trunkv2.py:632` subsamples the MSA with `torch.randperm`, so `--seed` is applied
  before the trunk on every entry.
* **The cache round-trips.** Re-scoring 56 decoys from the on-disk cache alone
  matches the run-time scores to max |Δ| = 0.00035 (`complex_ipde`); most metrics
  agree to ~1e-5. The residual is fp16 storage of `z`; `--store_dtype float32`
  doubles the entry cache to remove it.
* **`feats` compaction is lossless**, verified exact on 4 entries.
* **Decoy atom mapping.** 2,000 sampled decoys (500 per source) mapped with zero
  failures; boltz2/esmfold/protenix coverage 1.00000, perturbed mean 0.98709.
* **Data completeness.** All 108,041 labelled decoys resolve to existing files and
  all 3,260 entries have a processed structure.
* **Sharding and resume.** A 2-GPU wrapper run scored 112 decoys with zero overlap
  and zero failures; re-running the same shard skipped both completed entries.

## Traps found while building this (all fixed)

These were silent-corruption risks rather than crashes, so they are worth knowing:

* **`x_pred` is positional.** The head consumes coordinates in Boltz's canonical
  atom order, so an off-by-one atom shift is silently wrong rather than an error.
  `decoy_coords.py` keys on canonical atom names per chain instead of trusting
  file order.
* **Two-character PDB chain ids.** Ids like `Dx`/`Hc` overflow the single-column
  chain field and shift coordinates one column right; 25/150 boltz2 PDBs are
  affected, and gemmi mis-parses them too (residue numbers like `-456527`).
  Detected via `line[22].isalpha()`.
* **Protenix adds terminal `OXT` atoms** and renames chains to A/B/C.
* **Mid-chain insertions** in some esmfold/protenix/perturbed structures — handled
  with `difflib.SequenceMatcher` so unmatched residues are marked absent rather
  than shifting everything after them.
* **ESMFold CIFs return 0 models** from `gemmi.read_structure`; the `_atom_site.`
  loop is read directly via `gemmi.cif.read`.
* **Low coverage on perturbed decoys** (min 0.50) is genuine unresolved side chains
  in the experimental natives, not a mapping bug — residue counts match exactly.
  Absent atoms are collapsed onto the residue centre so they do not sit at the
  origin and poison `cdist`.
* **`feats` keys cannot be found by grepping.** The head's submodules (`rel_pos`,
  `contact_conditioning`, the atom encoder) read 7 keys that never appear as a
  literal `feats["..."]` in `confidencev2.py`. The list of 18 was derived by
  wrapping `feats` in a dict subclass recording `__getitem__` during a real call; a
  grep-derived list fails with `KeyError: 'residue_index'`.
* **fp16 coordinates are too coarse.** At ~100 Å magnitudes fp16 quantises to
  ~0.016 Å, which moved `complex_ipde` by up to 0.058 and `iptm` by 0.005 — the
  same order as the signal being learned. Decoy coordinates are stored as fp32.
* **Protenix decoy ids collided.** Its seed lives in the path (`seed_<seed>/`) but
  it reuses `sample_0.cif` across seeds, so 19,555 of 26,075 rows shared a
  condition + filename. `collect_decoys` now includes the seed in the id and raises
  on any duplicate rather than letting decoys overwrite each other.
* **`np.savez_compressed` appends `.npz`** unless the name already ends in it, so a
  `foo.npz.tmp` temp file silently became `foo.npz.tmp.npz` and the atomic rename
  failed for every decoy.

## Module map

Building the cache:

| file | role |
| --- | --- |
| `confidock/boltz/inference/decoy_coords.py` | map a decoy structure onto Boltz's canonical atom order |
| `confidock/boltz/inference/decoy_sources.py` | enumerate decoys + DockQ labels across the four sources |
| `confidock/boltz/inference/cache_runner.py` | split `Boltz2.forward` at the trunk/head boundary |
| `confidock/boltz/inference/confidence_cache.py` | on-disk cache format (compaction, atomic writes) |
| `workflows/scripts/inference/cache_boltz2_confidence.py` | the CLI |
| `workflows/scripts/inference/cache_boltz2_confidence_auto_restart.sh` | multi-GPU sharding + restart |

Consuming it. Two tracks train a DockQ head on these inputs; both share the library
in `confidock/boltz/finetune/`.

| file | role |
| --- | --- |
| `confidock/boltz/finetune/cache_dataset.py` | entry-grouped dataset + the token-aware decoy budget |
| `confidock/boltz/finetune/dockq_head.py` | the head, its interface-aware pooling, and its loss |
| `confidock/boltz/finetune/trainer.py` | train/eval loop and the correlation report |
| `confidock/boltz/finetune/distributed.py` | hand-rolled gradient averaging (why not DDP: see below) |
| `confidock/boltz/finetune/latent_cache.py` | on-disk format for the probe's pooled latents |
| `confidock/boltz/finetune/probe.py` | the probe's train/eval loop over that table |

**Track A -- unfrozen fine-tune.** All 25M parameters of `confidence_module` train
alongside the head, ~7.4 h/epoch on 6 GPUs.

```bash
workflows/scripts/training/finetune_confidence_ddp.sh runs/confidence_ft_v3 6 --epochs 3
```

`finetune_confidence_ddp.sh` is a `torchrun` wrapper over
`finetune_confidence_ddp.py`. Despite the name this is **not**
`DistributedDataParallel`: decoy chunking varies per entry, activation checkpointing
re-runs the forward, and ~140 of 452 confidence tensors get no gradient from a
DockQ-only loss -- each breaks a DDP assumption independently, so gradients are
averaged by hand in `distributed.py`. Two learning rates, because the randomly
initialised head (`--lr`, 1e-3) and the pretrained module (`--confidence_lr`, 1e-5)
start in very different places. `finetune_dockq_head.py` is the single-GPU variant;
`finetune_after_cache.sh` waits for a cache to finish, verifies the artifacts on
disk rather than trusting an exit code, then launches the run.

**Track B -- frozen-latent probe.** Run the frozen module once, store the pooled
640-wide latents, then train a 0.2M-parameter MLP over that table. An epoch is
seconds.

```bash
workflows/scripts/training/extract_dockq_latents_all_gpus.sh \
    $VOLUME/sabdab_val_easy/dockq_latents 0,1,2,3,4,5 \
    --cache_dir $VOLUME/sabdab_val_easy/confidence_cache

poetry run python workflows/scripts/training/train_dockq_probe.py \
    --latent_dir      $VOLUME/sabdab_train/dockq_latents \
    --eval_latent_dir $VOLUME/sabdab_val_easy/dockq_latents \
    --output_dir runs/dockq_probe_v3 --epochs 20 --dropout 0.4
```

`queue_val_easy_latents.sh` is the queued form of the extraction step.

**Prefer Track B.** On `sabdab_val_easy` the two are statistically
indistinguishable on per-native Spearman (Δ = −0.008, P = 0.19) while an epoch costs
8 seconds instead of ~7.4 GPU-hours. Track A's purpose was to establish that
unfreezing 25M parameters buys nothing, and that question is now answered --
see [boltz_dockq_head_results.md](boltz_dockq_head_results.md).
