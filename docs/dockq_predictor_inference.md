# Scoring predicted structures with the DockQ predictor

The DockQ predictor answers: **given several candidate poses of the same
antibody–antigen complex, which one is closest to the truth?** It scores a structure
without seeing a native, so it can be used at prediction time when the answer is not
available. It is trained and selected for ordering poses *within* one complex, so treat
`dockq_pred` as a ranking key, not as a calibrated DockQ value (see
[What the scores mean](#what-the-scores-mean)).

This page is the inference path only. Producing decoys and training the model are steps
1–10 of the [main README](../README.md#dockq-predictor).

---

## 0. Get the model

Weights are not shipped with this repository. You need **two** files, and they travel
together:

```
epoch034-score0.6579.ckpt   the weights
config_used.yaml            the config they were trained with
```

Both are required. The checkpoint holds only weights, so the architecture and the
featurization settings (`interface_cutoff`, `knn_k`) are read from the config. A
mismatched config raises on the state-dict shapes rather than silently mis-scoring, which
is the behaviour you want — but it does mean you cannot score with a checkpoint whose
config you have lost.

If you trained your own model with `train_simple_gnn_from_manifest.py`, both files are in
its output directory already.

---

## 1. Name the structure files

One file per pose, PDB or CIF. **The filename is load-bearing**: the complex key each
pose is grouped under is the first two `_`-separated tokens of the stem.

```
7y8j_HLA_seed42_sample0.pdb   ->  complex "7y8j_HLA"
7y8j_HLA_sample_0.cif         ->  complex "7y8j_HLA"
mypose.pdb                    ->  complex "mypose"   <- every file its own complex
```

Poses you want ranked against each other must share that prefix; poses of different
complexes must not. Use `{id}_{chains}_sample_{n}` and this takes care of itself.

Per-residue pLDDT is read from the **B-factor column** of the structure, and is one of
the 18 node features. If your writer leaves B-factors at 0 or 1, that feature block is
constant and carries no information — the model still runs, and still scores everything
slightly worse.

## 2. Supply CDR metadata

CDR labels come from exactly one place: a SAbDab-style JSON located **by complex name**,
holding `h_seq`/`l_seq` plus 1-based CDR position lists in `h_cdr`/`l_cdr`. For a
complex in the SAbDab snapshot this is found automatically and you can skip this step.
For anything else — a de-novo design, a target outside the snapshot, a generated decoy
grid — no JSON is found and CDR labelling silently returns nothing.

So for structures outside SAbDab, build the sidecars first. This derives CDRs from
ANARCI's Chothia numbering and writes the JSON the loader already knows how to read
(requires the `ANARCI` executable on PATH):

```bash
poetry run python workflows/scripts/inference/build_cdr_sidecar_json.py \
    --structures '/data/poses/*.pdb' \
    --out-dir cdr_json
```

One JSON per complex, not per pose — poses share a sequence, so ANARCI runs once per
distinct chain rather than once per file. Complexes with no immunoglobulin heavy chain
are reported and skipped, and existing JSONs are left alone unless you pass
`--overwrite`. On a validation complex the generated CDRs reproduce the SAbDab reference
exactly and restore all 18 node columns.

Then hand that directory to the scorer with `--cdr-json-dir cdr_json` (step 3). The flag
writes `CONFIDOCK_SABDAB_JSON_DIR`, which is what the featuriser actually reads and what
the dataloader workers inherit, so `export CONFIDOCK_SABDAB_JSON_DIR=$(realpath cdr_json)`
does the same job — the flag just keeps the dependency in the command rather than in your
shell history, which matters when the scoring runs as a submitted job. Both accept
several directories, `:`-separated, searched in order with first match winning — so
sidecars for complexes missing from your SAbDab snapshot add coverage rather than
shadowing the snapshot's own entries.

**Check that it took.** Every scoring run prints the search path, and a directory that
does not exist is a hard error rather than a silent miss:

```
CDR metadata search path (first match wins):
  /work/cdr_json  (12 json)
  /data/sabdab/interfaces  (14592 json)
```

A `WARNING: no CDR metadata JSON found for N complex(es)` after that means the sidecars
are not being found for those complexes — with the directory ruled out, the usual cause
is a filename that does not reduce to the JSON's name (step 1).

If you would rather write the JSONs yourself, the schema is:

```json
{
  "h_seq": "EVQLVESGGG...",
  "l_seq": "DIQMTQSPSS...",
  "h_cdr": {"1": [26, 27, 28, 29, 30, 31, 32], "2": [52, 53, 54, 55, 56], "3": [95, 96, 97]},
  "l_cdr": {"1": [24, 25, 26], "2": [50, 51], "3": [89, 90, 91]}
}
```

The positions are 1-based indices into `h_seq`/`l_seq`, **not** residue numbers —
predicted structures are numbered 1..N, so numbering schemes cannot be assumed. The
sequence is aligned to the structure at load time, so terminal truncation and indels
between the two are tolerated.

## 3. Score

```bash
MODEL_DIR=artifacts/ConfiDock_GNN:v0   # whatever artifact.download() returned in step 0

poetry run python workflows/scripts/inference/predict_dockq.py \
    --checkpoint  "$MODEL_DIR/epoch034-score0.6579.ckpt" \
    --config      "$MODEL_DIR/config_used.yaml" \
    --structures '/data/poses/7y8j_HLA_*.pdb' \
    --model-source boltz2 \
    --chains HLA \
    --cdr-json-dir cdr_json \
    --output results/scores.csv
```

`--structures` takes files, globs, or directories (walked recursively). `--cdr-json-dir`
is the sidecar directory from step 2, and can be dropped for complexes in the SAbDab
snapshot. The other two annotations are global to the run, because a glob carries
neither:

- **`--model-source`** — which co-folder produced the poses, one of `boltz2`,
  `esmfold2`, `protenix`. This is a 3-way one-hot edge feature, so it has no safe
  default and is required. **Any other value silently becomes all-zeros**, a combination
  the model never saw in training; if your poses come from something else, either name
  the closest of the three or accept that the block is dead for every row (harmless for
  ranking *within* one co-folder, since it is then constant). The script warns.
- **`--chains`** — SAbDab-style chain string, e.g. `HLA`: `H`/`Hc` is heavy,
  `L`/`K`/`Lc`/`Kc` is light, everything else is antigen. Optional, but supply it.
  Without it chain roles are guessed — from the metadata JSON by chain length, then
  ANARCI, then a bare length heuristic — and a wrong guess types your antibody as
  antigen and quietly wrecks the features.

Also worth knowing:

- A GPU is used if one is visible, otherwise it runs on CPU (`--device` to force it).
  Featurisation, not the network, is the cost: the model is 423K parameters. Tune
  `--num-workers`.
- Poses that fail to parse or featurise are dropped with a warning and reported as
  `Scored N/M structure(s)`, rather than aborting the run — check that count.
- No ground-truth `dockq` is involved anywhere. **If you do have labels** and want to
  know how good the model is rather than which pose to take, use
  `workflows/scripts/training/predict_dockq_from_manifest.py` instead: it requires a true
  `dockq` per row and reports the full metric set against it.

### Mixed batches: use a manifest

When poses come from more than one co-folder, or the chain strings differ per complex,
pass a CSV instead of `--structures`:

```csv
pdb_path,model_source,chains
/data/poses/7y8j_HLA_seed42_sample0.pdb,boltz2,HLA
/data/poses/5x8m_HLB_sample_0.cif,protenix,HLB
```

```bash
poetry run python workflows/scripts/inference/predict_dockq.py \
    --checkpoint "$MODEL_DIR/epoch034-score0.6579.ckpt" \
    --config     "$MODEL_DIR/config_used.yaml" \
    --manifest   poses.csv \
    --output     results/scores.csv
```

Only `pdb_path` is required; `chains` may be omitted, and `model_source` may come from
either the column or the flag (the flag wins). Extra columns are carried through to the
output untouched, so keep whatever you want to group or filter by later — including a
`dockq` column if you happen to have one, which is passed along and never read.
`--cdr-json-dir` applies to this mode identically; the CDR lookup is by filename, not by
anything in the manifest.

## 4. Read the output

`results/scores.csv` is your input rows plus three columns: `dockq_pred`, the `complex`
key, and `rank` within that complex (1 = best). The run also prints the best pose per
complex. For a single complex with a handful of poses, that is the whole job.

```python
import pandas as pd

df = pd.read_csv("results/scores.csv")
best = df[df["rank"] == 1]                       # the pose to take forward per complex
top5 = df[df["rank"] <= 5].sort_values(["complex", "rank"])
```

## What the scores mean

**Use them to order, not to report.** The model is selected on within-complex Spearman,
and the ranking term compresses the prediction scale as a side effect: on held-out
`val_easy` the predictions have std 0.197 against a true spread of 0.282 (0.135 vs 0.315
on `val_hard`). A `dockq_pred` of 0.45 does not mean DockQ ≈ 0.45.

**Expect nothing when the poses are all alike.** Per-native Spearman on held-out data is
0.60 for complexes whose poses genuinely differ (true DockQ std ≥ 0.2) and around zero —
`-0.02` — for the near-flat ones. That is not a defect being hidden: if every pose of a
complex is equally good, there is no ordering to recover, and the ranking you get is
noise. This is also why the pooled per-native figure for the registered model reads 0.265
while the useful regime reads 0.60.

**Held-out top-1 hit rates** (registered model, `val_easy` / `val_hard`), for calibrating
expectations of "take the model's best pose":

| threshold | top-1 | top-5 |
|---|---|---|
| DockQ ≥ 0.23 (acceptable) | 0.51 / 0.51 | 0.56 / 0.55 |
| DockQ ≥ 0.49 (medium) | 0.39 / 0.43 | 0.45 / 0.48 |
| DockQ ≥ 0.80 (high) | 0.18 / 0.21 | 0.23 / 0.26 |

## Reference

| what | where |
|---|---|
| Scoring (no labels) | `workflows/scripts/inference/predict_dockq.py` |
| CDR sidecar builder | `workflows/scripts/inference/build_cdr_sidecar_json.py` |
| Evaluation (with labels) | `workflows/scripts/training/predict_dockq_from_manifest.py` |
| Feature definitions | `confidock/dockq_predictor/data/feature_utils.py` (18 node / 96 edge) |
| Graph construction | `confidock/dockq_predictor/data/graph_builder.py` |
| CDR / chain-type lookup | `confidock/dockq_predictor/data/chain_mapping_cif.py` |
| CDR search path | `--cdr-json-dir`, then `CONFIDOCK_SABDAB_JSON_DIR` |
| Chain-string parsing | `confidock/dockq_predictor/data/chain_mapping.py` |
