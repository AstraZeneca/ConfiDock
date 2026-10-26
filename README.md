![Python version](https://img.shields.io/badge/python-3.10–3.12-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

# ConfiDock

**ConfiDock** is a lightweight graph neural network that predicts **DockQ** directly from a
co-folded antibody–antigen structure and its confidence outputs, with no native reference
needed at inference and no dependence on the co-folding model's internal representations.
Because one model scores all backbones, it can rank a decoy pool pooled across generators
without per-backbone calibration — which native confidence scores, each calibrated to their
own model, cannot do.

Each complex is a residue-level graph; node and edge features combine geometry with
per-residue and pairwise confidence signals. Residues near a cross-chain partner are flagged
as interface nodes. Features pass through three edge-conditioned MPNN layers, and node
embeddings are pooled by a multi-head gated set-attention readout restricted to interface
nodes, then regressed to a scalar in [0, 1] by an MLP with a final sigmoid.

This repository also contains the **matched fine-tuning control** — a DockQ head on Boltz-2's
own confidence module, with full trunk access — and the evaluation code behind the reported
numbers, including true-DockQ computation, per-interface metrics and ranking reports.

Decoys are generated with **Boltz-2**, **Protenix-v2** and **ESMFold2**.

> **Note on weights and data.** Neither model checkpoints nor datasets are shipped with
> this repository. SAbDab inputs, MSAs and Boltz-2 weights live wherever you put them; see
> [Configuration](#configuration).

--------
## Quick Start

1. **Install dependencies and set up the environment:**
   ```bash
   make init
   ```
   This will:
   - Install Python dependencies via Poetry
   - Set up pre-commit hooks
   - Install MMseqs2 for sequence analysis

   Everything needed to train and run the DockQ predictor, and to score with DockQ, is in
   the default install. Two parts are deliberately **not**:

   - **ESMFold2 decoy generation** needs `poetry install --with esm` on Python 3.12, and
     an `esm` distribution you can reach. The `esm` group is optional precisely so that
     the rest of the package installs without it — every `esm` import in the codebase is
     function-local.
   - **Protenix decoy generation** uses its own Poetry project in
     [`protenix-inference/`](protenix-inference/), because it needs a different Python.

   Neither is required to use ConfiDock on structures you already have.

2. **Verify MMseqs2 installation:**
   ```bash
   tools/mmseqs/bin/mmseqs version
   ```

3. **Evaluate predictions with DockQ:**

    This is where the walkthrough starts, because it is where the subtleties are. Getting
    here means you have decoys and their native references: dataset assembly, model-input
    preparation and the grid runners for all three backbones live under
    [`workflows/scripts/`](workflows/scripts/) with `--help` on each, and are not walked
    through here.

    Do **not** substitute an off-the-shelf DockQ call. Multi-chain complexes, multiple
    interfaces per complex and chain-role assignment all change the number materially, and
    those decisions are what [`confidock/evaluation/`](confidock/evaluation/) implements.

    Compute DockQ scores comparing predictions to native structures:

    **Grid mode (for multi-condition inference):**
    ```bash
    # Boltz2 grid
    poetry run python workflows/scripts/analysis/compute_dockq_scores.py \
        --model boltz \
        --grid_dir /path/to/grid/output \
        --native_dir /path/to/native/pdbs \
        --json_dir /path/to/jsons \
        --output_csv /path/to/output/dockq_scores.csv

    # Protenix grid
    poetry run python workflows/scripts/analysis/compute_dockq_scores.py \
        --model protenix \
        --grid_dir /path/to/protenix/grid \
        --native_dir /path/to/native/pdbs \
        --json_dir /path/to/jsons \
        --output_csv /path/to/output/dockq_scores.csv

    # ESMFold2 grid
    poetry run python workflows/scripts/analysis/compute_dockq_scores.py \
        --model esmfold2 \
        --grid_dir /path/to/esmfold2/grid \
        --native_dir /path/to/native/pdbs \
        --json_dir /path/to/jsons \
        --output_csv /path/to/output/dockq_scores.csv
    ```

    This generates:
    - `dockq_scores.csv`: Complex-level scores (one row per prediction, used for plotting)
    - `dockq_scores_per_interface.csv`: Detailed per-interface scores
    - Multiple plots: boxplot, violin, heatmap, diversity, etc.

    **Optional: Add diversity metrics and additional plots:**
    ```bash
    # Post-process to add diversity metrics and generate more plots
    poetry run python workflows/scripts/analysis/add_diversity_metrics.py \
        /path/to/output/dockq_scores.csv \
    ```

    **Key options:**
    - `--model`: Choose `boltz` (PDB), `protenix` (CIF), or `esmfold2` (CIF)
    - `--grid_dir`: the `output_dir` from your grid-inference run
    - `--native_dir`: PDB files of ground truth entries
    - `--json_dir`: Sequence JSON files for accurate chain matching (recommended), same as above if using processed sabdab data

### DockQ predictor

Step 3 produces decoys and their true DockQ scores. Steps 4–6 train the **DockQ
predictor** on that output: a graph neural network that scores a decoy *without* a native structure, so it can rank candidate poses at prediction time when the answer is not available. It is trained and selected for ranking decoys **within** one
native complex (picking the best pose for a target) rather than for absolute DockQ accuracy, and that choice runs through the loss, the batching and every metric below.

4. **Build a training manifest from the scored predictions:**

    The predictor learns from a **manifest**: a flat CSV where one row is one decoy.
    Step 3's `dockq_scores.csv` files are the input, one per model source.

    ```bash
    poetry run python workflows/scripts/analysis/sample_dockq_for_training.py \
        --boltz2-csv   /path/to/boltz2/dockq_scores.csv \
        --esmfold2-csv /path/to/esmfold2/dockq_scores.csv \
        --protenix-csv /path/to/protenix/dockq_scores.csv \
        --stratify-antigen-clusters /path/to/mmseqs/antigen_cluster.tsv \
        --n-total 3500 \
        --output data/dockq_predictor/dockq_training_manifest.csv
    ```

5. **Train the DockQ predictor:**

    ```bash
    poetry run python workflows/scripts/training/train_simple_gnn_from_manifest.py \
        --config configs/dockq_predictor/dockq_gnn.yaml \
        --manifest data/dockq_predictor/dockq_training_manifest.csv \
        --output-dir checkpoints/dockq_gnn \
        --wandb-project confidock-dockq-predictor \
        --run-name dockq-gnn-listnet
    ```

    Every hyperparameter lives in the config; the flags only say where the data is, where
    checkpoints go, and whether to log to W&B, so two runs of one config are the same
    experiment. `configs/dockq_predictor/dockq_gnn.yaml` is a self-contained config example.

    - **`data.group_by_complex: true` is required.** The within-complex ranking term is
      exactly 0 unless a batch holds ≥2 decoys of the same complex, and random batching
      over thousands of complexes almost never does. Turn it off and the loss silently
      degrades to plain Huber — with no error and a perfectly plausible loss curve.
    - **`callbacks.monitor: val/per_native_spearman_mean`**, not `val/loss`. The
      deployment question is "given one complex's decoys, can we pick the best one", and
      absolute-error metrics improve while within-complex ordering does not.
    - **`data.graph_cache_dir`** builds each graph once and reads it back on later epochs.
      Point it at a **fresh** directory whenever a feature computation changes. Budget for
      it: ~2.9 MB per graph, so a 107k-row manifest is ~300 GB, and the cold build is
      CPU-bound at ~0.36 graphs/s per worker (~1.7 h at 48 workers, ~13 h at 8).
    - **`data.num_workers`** is the throughput knob, because training is
      **dataloader-bound, not GPU-bound**. Set it to the machine's real core budget. Once the cache is warm
      the workers sit in read-wait rather than computing, so past that point the storage bandwidth is the ceiling and more workers buy nothing.

    A short smoke test of the data and model wiring, without training anything:

    ```bash
    poetry run python workflows/scripts/training/train_simple_gnn_from_manifest.py \
        --config configs/dockq_predictor/dockq_gnn.yaml \
        --manifest data/dockq_predictor/dockq_training_manifest.csv \
        --fast-dev-run
    ```

    **Running it as a batch job.** A full-size run is hours long, so run it under
    whatever batch scheduler you have rather than in an interactive shell.


6. **Score a held-out set:**

    ```bash
    poetry run python workflows/scripts/training/predict_dockq_from_manifest.py \
        --config configs/dockq_predictor/dockq_gnn.yaml \
        --checkpoint checkpoints/dockq_gnn/checkpoints/epoch027-score0.6541.ckpt \
        --manifest data/dockq_predictor/val_hard_manifest.csv \
        --output-dir results/dockq_gnn/val_hard \
        --prefix val_hard/
    ```

    Pass **the config the checkpoint was trained with**: the checkpoint holds only
    weights, so the architecture and the featurisation settings (`interface_cutoff`,
    `knn_k`) are read from the config. A mismatched config raises on the state-dict
    shapes rather than silently mis-scoring.

    Every row is scored  (no splitting, no sampling) producing `predictions.csv` (the
    manifest rows plus a `dockq_pred` column) and `metrics.json`. Add `--no-graph-cache`
    for a one-off set: a single pass gains nothing from writing a cache.

    **Reading the metrics.** `per_native_spearman_mean` pools every complex, including the
    near-flat ones where a Spearman over decoys that all score alike is noise around zero,
    so it reads far lower than the model's useful ranking ability. The scoring summary
    therefore also prints the **DockQ-std breakdown**, and the high-variance bin is the
    one that answers the deployment question:

    ```
      DockQ-std bin   n_cplx  n_decoys  per-native rho   global rho
      0.00_0.02          113      5058         -0.0320       0.1189
      ...
      0.20_1.01          230     10314          0.5845       0.7216
      std_ratio                            0.5780
    ```

7. **Score new predicted structures with the trained model:**

    Steps 4–6 train and validate the predictor. To *use* it — starting from one or many
    predicted structures with no native to compare against — the entry point is
    `workflows/scripts/inference/predict_dockq.py`, which needs no ground-truth labels and
    returns a rank per pose within each complex:

    ```bash
    poetry run python workflows/scripts/inference/predict_dockq.py \
        --checkpoint  "$MODEL_DIR/epoch034-score0.6579.ckpt" \
        --config      "$MODEL_DIR/config_used.yaml" \
        --structures '/data/poses/7y8j_HLA_*.pdb' \
        --model-source boltz2 --chains HLA \
        --output results/scores.csv
    ```

    For the whole path — fetching the model from the registry, filename conventions, CDR
    sidecars for complexes outside SAbDab, and what the scores do and do not mean — see
    [Scoring predicted structures with the DockQ predictor](docs/dockq_predictor_inference.md).
    That covers pulling the model from the W&B registry, the file-naming and CDR metadata
    the featuriser needs, building the manifest, and how far to trust the scores.


### Boltz-2 confidence fine-tune (the matched control)

This section is the **control**, not the headline model. It exists to answer one
question: does the interface-aware graph of steps 4–6 capture ranking signal that a
scalar confidence head cannot, *even when that head has full access to the co-folding
model's trunk*? Giving the baseline more information than ConfiDock gets is the point.

Step 3 gives decoys with true DockQ labels. There are two ways to score a decoy
*without* a native: train a GNN over the structure (steps 4–6), or re-use the
confidence module Boltz-2 already has. This section is the second one. The
481.5M-parameter trunk is frozen and its outputs are cached once per entry; what
trains is Boltz-2's 25.0M-parameter `confidence_module` plus a new 0.20M-parameter
`DockQHead` that predicts DockQ from the confidence latents. The target, the
composite loss and the metric suite are deliberately the same as the GNN's, so the
two are comparable rather than merely both plausible.

Before starting you need: labelled decoys and their `dockq_scores.csv` files from
step 3, a manifest CSV per dataset (step 4), the Boltz-2 weights and CCD (the
`boltz-cache` under `/data`), and **~160 GiB of volume** for the
trunk cache.

Deeper write-ups: [`docs/boltz_confidence_caching.md`](docs/boltz_confidence_caching.md)
for the caching pipeline and its correctness checks,
[`docs/boltz_dockq_head_results.md`](docs/boltz_dockq_head_results.md) for what the
head actually achieved against Boltz-2's own metrics and against the GNN.

8. **Cache the trunk once per entry:**

    ```bash
    # Single GPU
    poetry run python workflows/scripts/inference/cache_boltz2_confidence.py \
        --root       /path/to/sabdab_train \
        --output_dir /path/to/volume/sabdab_train/confidence_cache \
        --boltz_cache /path/to/boltz-cache

    # All GPUs, one shard per GPU, restarting any shard that dies
    workflows/scripts/inference/cache_boltz2_confidence_auto_restart.sh \
        /path/to/volume/sabdab_train/confidence_cache

    # A validation set: its directory layout differs and it has no perturbed
    # decoys, so drive it from the manifest instead of walking the root
    workflows/scripts/inference/cache_boltz2_confidence_auto_restart.sh \
        /path/to/volume/sabdab_val_easy/confidence_cache \
        0,1,2,3,4,5 \
        --root /path/to/sabdab_val_easy \
        --manifest_csv data/dockq_predictor/val_easy_manifest.csv
    ```

    The trunk never sees coordinates, so `s`/`z` are identical for every decoy of an
    entry — running the trunk once and scoring all ~33 decoys against it is
    bit-exact against the full `model.forward`.
    The confidence head runs with `skip_run_structure=True` so it scores *the decoy
    supplied*, keeping each cached record in 1:1 correspondence with its DockQ label.


9. **Backfill the DockQ sub-scores into the cache:**

    ```bash
    # --dry_run first: it reports the match rate and the DockQ agreement
    # without touching a byte
    poetry run python workflows/scripts/data/backfill_dockq_components.py \
        --cache_dir  /path/to/volume/sabdab_train/confidence_cache \
        --manifest   data/dockq_predictor/dockq_training_manifest.csv \
        --latent_dir /path/to/volume/sabdab_train/dockq_latents \
        --dry_run
    ```

    **This is not optional for the composite head.** A cache built by walking the
    dataset root records only `dockq`, while the head is supervised on `fnat`,
    `irmsd` and `lrmsd` as well, and `EntryGroupedCacheDataset` *drops* every decoy
    whose components are missing. Skip this and training reports zero usable decoys
    with a log line rather than an error. Pass `--manifest` more than once when
    several manifests cover one dataset.

    Nothing is recomputed: the sub-scores already exist in the manifests (DockQ was
    computed from them) and the join is on the structure path, verified to
    `max |manifest.dockq - cache.dockq| = 0` before anything is written. Score
    JSONLs and, if `--latent_dir` is given, the pooled-latent shards are rewritten in
    place via temp file + rename.

10. **Optional but cheap first: the frozen-latent probe.**

    With the confidence module frozen and in `eval()`, the pooled latents are a
    deterministic function of `(entry, decoy)`, so they can be extracted once and
    reused. An epoch then costs seconds against a ~300 MiB table instead of ~13.5 h
    against the live module, which makes this the right place to sweep the loss
    weights — and the control that says whether unfreezing 25M parameters bought
    anything.

    ```bash
    # Extract, one shard per GPU (~15 min on 6 GPUs; resumable like the cache)
    workflows/scripts/training/extract_dockq_latents_all_gpus.sh \
        /path/to/volume/sabdab_train/dockq_latents

    # Train the head on the latents
    poetry run python workflows/scripts/training/train_dockq_probe.py \
        --latent_dir      /path/to/volume/sabdab_train/dockq_latents \
        --eval_latent_dir /path/to/volume/sabdab_val_easy/dockq_latents \
        --output_dir runs/dockq_probe_v1 --epochs 40
    ```

    Extract latents for the external set too if you want to compare against it: the
    internal holdout comes out of the *training* cache, whose decoy pool includes
    `perturbed` (per-entry DockQ std ~0.29 against ~0.11 on `val_easy`), so a probe
    number from that holdout is not comparable to a fine-tune number on `val_easy`.
    Model selection stays on the internal holdout and the external set is scored
    once, at the selected epoch — selecting on it would make it a second validation
    set. `workflows/scripts/analysis/probe_epoch_curve.py` scores both sets at every
    epoch, as a diagnostic only, for the "would more training help" question.

11. **Fine-tune the confidence module + DockQ head:**

    ```bash
    # Smoke test first — one GPU, 20 batches, ~15 min
    workflows/scripts/training/finetune_confidence_ddp.sh runs/ft_smoke 1 \
        --cache_dir /path/to/volume/sabdab_train/confidence_cache \
        --max_entries 40 --limit_train_batches 20 --limit_val_batches 10 \
        --epochs 1 --no_wandb

    # The real run: one process per GPU
    workflows/scripts/training/finetune_confidence_ddp.sh runs/confidence_ft_listnet 6 \
        --cache_dir          /path/to/volume/sabdab_train/confidence_cache \
        --external_cache_dir /path/to/volume/sabdab_val_easy/confidence_cache \
        --epochs 5 \
        --lr 1e-3 --confidence_lr 1e-5 \
        --rank_type listnet --rank_tau 0.10 --lambda_rank 0.2 \
        --lambda_component 0.0 --huber_delta 0.1 \
        --wandb_run_name confidence_ft_listnet
    ```

    The wrapper is `torchrun --nproc_per_node=<NPROC>` plus `nice`; gradients are
    averaged by hand after backward rather than through DDP, because decoy chunking
    varies per entry, activation checkpointing re-runs the forward, and a DockQ-only
    loss leaves ~140 of the module's 452 tensors without gradient — each of which
    independently breaks a DDP assumption.

    **The flags that decide the result:**

    - **Two learning rates.** `--lr` (1e-3) trains the randomly-initialised head;
      `--confidence_lr` (1e-5) nudges the pretrained module. A single LR big enough
      for the head destroys the module in a few steps.
    - **The ranking term is weighted by `lambda_rank/rank_tau`, not `lambda_rank`.**
      At ratio 10 (`0.5/0.05`) both this model and the GNN got their best
      high-variance ranking but collapsed the prediction scale and with it global
      Spearman. Ratio 2 (`0.2/0.10`) is the settled setting.
    - **`--rank_type listnet`** is listwise cross-entropy over an entry's decoys and
      leaves tied labels tied; `spearman` argsorts ties into an invented order
      (10–18% of chunks on the GNN side).
    - **`--huber_delta 0.1`.** At the old default of 1.0 the Huber term is plain MSE:
      DockQ lives in [0, 1] so the error can never exceed the delta.
    - **`--freeze_confidence`** trains only the head — the in-trainer version of the
      step-13 control.
    - **`--external_cache_dir` is scored once, after training, with the best epoch's
      weights**, and is refused if it overlaps the training split. It never replaces
      the validation set: selecting on it would stop the number being
      out-of-distribution. `--eval_cache_dir` is the one that *does* replace the
      internal holdout.

    **Memory** is the other half of the flag list, and the defaults are calibrated
    together for an A100-40GB: `--forward_chunk 1` with `--amp_dtype bf16` fits 8
    decoys at 600 tokens in 29.4 GiB where `chunk=4` OOMs at 4; peak memory is
    quadratic in token count, so `--token_aware_batching` sizes each entry's batch
    from its tokens (a flat cap OOM'd 49% of batches, all of them the large
    complexes, i.e. it trained on small complexes only); `--min_decoys_on_oom 1`
    keeps the largest entries in the run when even a pair does not fit. Raising
    `--forward_chunk` invalidates the `--memory_limit_gib 34.0` calibration.

    **Cost:** ~7.5 h per epoch on 6 A100-40GBs, plus ~30 min for a full 763-entry
    `val_easy` pass at the end. Read the internal-val table after epoch 1 before
    paying for the rest.

    **Outputs** in `--output_dir`: `config.json` and `split.json` (what the run was),
    `history.json` and `report_epoch<N>.json` (the trajectory),
    `confidence_ft_best.pt` + `predictions_best.csv` + `report_best.json` (selected
    on internal per-entry Spearman), and `report_external.json` +
    `predictions_external.csv` for the OOD pass. W&B project defaults to
    `confidock-boltz_ft`; `--no_wandb` turns it off.

    **Running it as a batch job.** A full run is a day and a half on most of a node,
    so it belongs under a batch scheduler. Two things are load-bearing and easy to get
    wrong. First, the trunk cache must be on storage the batch job can actually mount
    concurrently — if you built it on a single-writer volume attached to an
    interactive session, stage it somewhere shared first and pass `--cache_dir`
    explicitly. Second, validate the environment in the first minute rather than eight
    hours in: import torch/boltz/`confidock`, assert CUDA is visible, run one NCCL
    collective, and run the loss's unit tests before the real work starts.

12. **Score a finished checkpoint on another held-out set:**

    ```bash
    poetry run torchrun --nproc_per_node=2 \
        workflows/scripts/training/finetune_confidence_ddp.py \
        --epochs 0 \
        --ft_checkpoint      runs/confidence_ft_listnet/confidence_ft_best.pt \
        --cache_dir          /path/to/volume/sabdab_train/confidence_cache \
        --external_cache_dir /path/to/volume/sabdab_val_hard/confidence_cache \
        --hidden 256 --dropout 0.1 --val_fraction 0.1 --split_seed 0 \
        --output_dir runs/confidence_ft_listnet_valhard
    ```

    `--epochs 0` with `--ft_checkpoint` loads the weights and goes straight to the
    OOD pass — one forward over 306 entries / 13,655 decoys took ~30 min on 2 GPUs.
    Three arguments must
    match the training run: `--hidden`/`--dropout` or the state dict fails to load,
    and the same `--cache_dir`/`--val_fraction`/`--split_seed` so the split is
    reproduced and the "external set overlaps the train split" guard is meaningful.

    **Reading the report.** Every table holds `dockq_pred` alongside all nine Boltz-2
    confidence metrics computed on exactly the same rows, plus `ORACLE`. Read
    `per_entry_rankable` (natives whose true-DockQ spread is at least
    `RANKABLE_MIN_STD = 0.05`) and the per-std-bin breakdown, not the ungated
    `per_entry`: on `val_easy` a third of natives have a near-flat DockQ profile, so
    ranking them measures noise, and a predictor can win the ungated mean purely by
    behaving better on those. From `runs/confidence_ft_listnet` on `val_easy`
    (33,879 decoys / 763 natives):

    ```
      metric              global   per_entry   per_entry_rankable
      dockq_pred          0.6118      0.1411               0.2503
      confidence_score    0.6145      0.0959               0.1887
      complex_plddt       0.5439      0.0858               0.1770
      iptm                0.5332      0.0074               0.0460
    ```

    So the fine-tuned head clearly out-ranks Boltz-2's own metrics *within* a native
    while matching them globally — and still trails the ListNet GNN of steps 4–6.

    To put both models in one table, computed by one code path:

    ```bash
    poetry run python workflows/scripts/analysis/compare_ft_vs_gnn.py
    ```

    Its run locations are hard-coded at the top of the file (`SETS`, `GNN_RUNS`);
    edit them for your runs. Two rules matter more than anything else the script
    does: compare against the **ListNet** GNN, whose ranking objective matches the
    fine-tune's, and keep the scope **pooled** — one Spearman per native over all
    decoys from all folding models. A native confidence metric only ever ranks within
    its own model, so a per-model native number is a different, harder task and is
    not comparable to these.


--------
## Configuration

Paths that depend on where ConfiDock is deployed are read from the environment, never
baked into a module (see [`confidock/paths.py`](confidock/paths.py)):

| Variable | Meaning | Default |
|---|---|---|
| `CONFIDOCK_BOLTZ_CACHE` | Boltz-2 weights and CCD files | `$XDG_CACHE_HOME/boltz` (`~/.cache/boltz`) |
| `CONFIDOCK_SABDAB_JSON_DIR` | `os.pathsep`-separated dirs of SAbDab sidecar JSONs (chain typing, CDR labels) | unset — no CDR metadata |

Every path shown as `/data/...` in the examples below is a placeholder for your own
storage layout. Training and inference scripts also accept all of these as explicit flags,
which take precedence over the environment.

Weights & Biases logging is optional and off unless a project is given. Credentials are
read from `WANDB_API_KEY` in the environment, or from a gitignored `wandb_env` file at the
repository root; `--no_wandb` disables tracking outright.

--------
## Project organisation

```
confidock/                 Library.
  boltz/                     Boltz-2 confidence fine-tuning, inference and caching.
  dockq_predictor/           Standalone DockQ-predictor GNN (data, models, losses, training).
  esmfold2/, protenix/       Co-folding backend wrappers.
  evaluation/                DockQ, metrics, bias, diversity, epitope and sequence analysis.
  paths.py, msa.py, rcsb.py, plot.py
workflows/scripts/         Pipeline entry points.
  processing/                Dataset building and model-input preparation.
  inference/                 Grid runners, confidence caching, DockQ scoring.
  training/                  Fine-tuning, latent extraction, probes.
  analysis/                  True-DockQ computation, correlations, model comparison.
  data/                      In-place cache maintenance (DockQ sub-score backfill).
configs/                   Hydra/YAML configs for grids and training.
protenix-inference/        Separate Poetry project for Protenix (needs its own Python).
docs/                      Method and results documentation.
tests/                     CPU-only test suite; no weights, no network.
data/, models/             Empty by design; git-ignored contents.
```

--------
## Utilities

| Command | Effect |
|---|---|
| `make init` | Create the Poetry environment, install pre-commit hooks, install MMseqs2 into `tools/mmseqs/`. |
| `make install-mmseqs` | Install MMseqs2 only. Add it with `export PATH="$PWD/tools/mmseqs/bin:$PATH"` or pass `--mmseqs_bin tools/mmseqs/bin/mmseqs`. |
| `make clean` | Remove `__pycache__` and `.pyc`/`.pyo` files. |

## Testing

```bash
poetry install
poetry run pytest -m "not slow"
```

The suite runs on CPU with no network access and no model weights. Tests that need a real
checkpoint are marked `slow` and excluded above.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributors are listed in [AUTHORS.md](AUTHORS.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
