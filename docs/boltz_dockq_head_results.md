# A DockQ Head on Boltz-2 Confidence Latents: Results on `sabdab_val_easy`

Can a learned head read Boltz-2's confidence latents and rank decoys better than the
confidence metrics Boltz-2 already outputs? **Yes — but the margin is small, and a
GVP GNN over the structures beats it by three times as much.**

This report is the evaluation half of the work described in
[boltz_confidence_caching.md](boltz_confidence_caching.md), which covers the caching
pipeline that makes head-only training affordable.

---

## 1. What was compared

Every predictor below was scored on **exactly the same rows**: 33,879 decoys across
763 antibody–antigen natives from `data/dockq_predictor/val_easy_manifest.csv`. Nothing
was subsampled and no predictor saw a different subset, so the differences are
attributable to the predictors rather than to the evaluation set. The join is verified
rather than assumed — see §7.

| predictor | what it is | trained on |
|---|---|---|
| `gvp` | GVP GNN over residue graphs, 3-seed mean (`checkpoints/ablation_gvp_feature_loo/full-s{0,1,2}`) | `sabdab_train` |
| `boltz_ft` | DockQ head + **unfrozen** 25M-param confidence module, 3 epochs DDP (`runs/confidence_ft_v3`) | `sabdab_train` confidence cache |
| `boltz_probe` | DockQ head (0.2M params) on **frozen** pooled latents (`runs/dockq_probe_v2_valeasy`) | `sabdab_train` latents |
| `confidence_score` | Boltz-2's own composite, `(4·complex_plddt + iptm)/5` (`boltz/model/models/boltz2.py:1085`) | — (pretrained) |
| `complex_plddt`, `iptm`, `ptm`, … | Boltz-2 re-scoring each decoy with its pretrained confidence head | — (pretrained) |
| `own_ptm`, `own_iptm` | the **generator's** self-reported scores, from the manifest | — (pretrained) |

`val_easy` is a holdout for all three learned predictors. The probe's internal split
shares zero entries with it (the trainer fails loudly on overlap,
`train_dockq_probe.py:172-182`), and the GVP ablation and the fine-tune both used it as
their external evaluation set.

### Metrics

* **global** — one Spearman over all 33,879 decoys pooled. Inflated by between-native
  variance: a metric scores well here just by separating easy targets from hard ones.
* **per-native** — Spearman within each native, then averaged over natives. *This is
  the number that matters*: given several models of one target, can you pick the good
  one? Everything else is a proxy for it.
* **rankable** — the same average over the 514 of 763 natives whose true-DockQ spread
  is at least `RANKABLE_MIN_STD = 0.05`. The other 249 have a near-flat DockQ profile
  (median per-native std across all of `val_easy` is 0.112), so ranking them measures
  noise; averaging that noise in depresses every predictor roughly uniformly.
* **top-1 DockQ** — mean true DockQ of the decoy each predictor would actually pick.
  The decision-relevant number, bounded above by ORACLE (0.4958) and below by
  pick-at-random (0.2233).

---

## 2. Headline table

33,879 decoys, 763 natives, `val_easy`. Sorted by per-native Spearman.

| predictor | global | per-native | rankable | top-1 DockQ |
|---|---|---|---|---|
| **GVP GNN** (3-seed) | 0.6554 | **0.2679** | **0.3539** | **0.3570** |
| Boltz frozen probe | 0.5042 | 0.1538 | 0.2080 | 0.2308 |
| Boltz ft (25M unfrozen) | 0.6221 | 0.1456 | 0.2263 | 0.2202 |
| **Boltz-2 `confidence_score`** | 0.6145 | 0.0959 | 0.1887 | 0.2083 |
| complex_plddt | 0.5439 | 0.0858 | 0.1770 | 0.2019 |
| complex_iplddt | 0.5440 | 0.0843 | 0.1789 | 0.1990 |
| own_iptm *(generator's)* | 0.5994 | 0.0379 | 0.1049 | 0.2104 |
| own_ptm *(generator's)* | 0.5653 | 0.0260 | 0.0916 | 0.1878 |
| iptm | 0.5332 | 0.0074 | 0.0460 | 0.2034 |
| ptm | 0.5181 | −0.0015 | 0.0380 | 0.1927 |
| complex_pde | −0.4949 | −0.0036 | −0.0519 | 0.1378 |
| complex_ipde | −0.5413 | −0.0149 | −0.0465 | 0.1423 |
| **ORACLE** (per-native max) | — | — | — | 0.4958 |
| **pick-at-random** (mean DockQ) | — | — | — | 0.2233 |

`pde`/`ipde` are predicted *error*, so their negative correlations are correct in sign
and their magnitudes are what to read.

Full joined table: `results/dockq_predictor_comparison_val_easy.csv` (§7 regenerates it).

---

## 3. Findings

### 3.1 A head on the latents does beat Boltz-2's own score — significantly

Paired bootstrap over natives, 10,000 resamples, per-native Spearman:

| comparison | Δ | 95% CI | P(Δ>0) |
|---|---|---|---|
| GVP − confidence_score | **+0.1721** | [+0.1431, +0.2013] | 1.000 |
| GVP − boltz_ft | +0.1224 | [+0.0938, +0.1514] | 1.000 |
| GVP − boltz_probe | +0.1142 | [+0.0859, +0.1421] | 1.000 |
| boltz_probe − confidence_score | +0.0579 | [+0.0354, +0.0815] | 1.000 |
| boltz_ft − confidence_score | +0.0497 | [+0.0304, +0.0692] | 1.000 |
| boltz_ft − boltz_probe | −0.0082 | [−0.0274, +0.0101] | 0.186 |

So the original question — *is better DockQ correlation achievable than Boltz-2's
output metrics?* — resolves to **yes, by +0.05 to +0.06 per-native**, with the CI
comfortably clear of zero. But the same bootstrap puts the GVP +0.12 ahead of either
Boltz head, so the confidence latents are the weaker substrate for this task.

On the rankable subset the ordering is unchanged but the probe's edge over
`confidence_score` narrows to +0.0193 (CI [−0.0084, +0.0461], P=0.914) — i.e. the
frozen probe's advantage comes disproportionately from the near-flat natives, whereas
the fine-tune keeps a clear +0.0376 (CI [+0.0138, +0.0619]) where ranking is real.

### 3.2 Unfreezing 25M parameters bought nothing

`boltz_ft` − `boltz_probe` = −0.0082, CI [−0.0274, +0.0101], P(>0) = 0.186. The 25M-param
DDP fine-tune (~7.4 h/epoch on 6×A100, 3 epochs) is statistically indistinguishable
from the 0.2M-param MLP over precomputed latents (epoch = 8 seconds). On the rankable
subset it is nominally ahead (+0.0183, CI [−0.0041, +0.0415]) and it does win clearly on
global Spearman (0.6221 vs 0.5042), but not on the axis used for model selection.

**Implication:** capacity is not the binding constraint. Iterate with the probe.

### 3.3 `iptm` carries essentially no within-native signal

`iptm` scores 0.5332 global and **0.0074** per-native. That is the between-native
inflation in its purest form: `iptm` knows which *targets* are easy and almost nothing
about which *decoy* of a given target is good. `ptm` is worse (−0.0015).

This retroactively validates the design decision to add a **separate** DockQ head
rather than repurposing the `iptm` output — there was no within-native signal there to
repurpose.

### 3.4 Two of the three learned heads barely beat random selection

Read the top-1 column against pick-at-random (0.2233):

* GVP 0.3570 — **+0.134**, a real improvement
* boltz_probe 0.2308 — +0.008
* boltz_ft 0.2202 — **−0.003, below random**
* confidence_score 0.2083 — −0.015, below random

This is a harsher framing than the Spearman columns give, and it is the framing that
matters if the deliverable is "hand over one model per target." A positive per-native
Spearman does not imply useful top-1 selection: a predictor can get the middle of the
ranking broadly right and still put a mediocre decoy first. Only the GVP converts its
correlation into selection.

### 3.5 The composite beats each of its components

`confidence_score` (0.0959 per-native) > `complex_plddt` (0.0858) > `iptm` (0.0074).
Mixing pLDDT and iptm helps, even at Boltz's hand-chosen 4:1 weighting. Weak evidence
that a *learned* mixture — including of the GVP and Boltz predictions — is worth trying.

### 3.6 Per-generator breakdown

Per-native Spearman, decoys split by which tool generated them:

| generator | rows | natives | mean DockQ | GVP | boltz_ft | boltz_probe | confidence_score |
|---|---|---|---|---|---|---|---|
| boltz2 | 11,085 | 740 | 0.164 | 0.1237 | 0.0889 | 0.0790 | 0.1219 |
| esmfold2 | 11,400 | 763 | 0.251 | 0.1837 | 0.1298 | 0.1599 | 0.1145 |
| protenix | 11,394 | 763 | 0.253 | 0.2094 | 0.1451 | 0.1206 | 0.1523 |

Two things stand out. **`confidence_score` is the best predictor of Boltz-2's own
decoys** (0.1219, essentially tied with the GVP's 0.1237 and ahead of both learned
Boltz heads) — Boltz's confidence head is calibrated to its own generator, and that
calibration does not transfer. And **within-generator per-native correlations are far
lower than the pooled figures** for every predictor, because pooling three generators
adds a between-generator axis that is easy to get right.

---

## 4. Does more training help? No.

The probe was originally trained for 40 epochs with a single external evaluation at the
internally-selected epoch, which cannot answer the question. Latents are on disk, so
five configurations × 120 epochs with external evaluation at **every** epoch cost
minutes.

| config | best ext per-native | @epoch | mean ep61–120 | best rankable | best top-1 | corr(internal, external) |
|---|---|---|---|---|---|---|
| base (wd 1e-4, drop 0.1) | 0.1813 | 22 | 0.1477 | 0.2608 | 0.2344 | +0.403 |
| wd 1e-2 | 0.1800 | 9 | 0.1439 | 0.2427 | 0.2373 | +0.249 |
| wd 1e-1 | 0.1722 | 50 | 0.1338 | 0.2577 | 0.2415 | +0.332 |
| dropout 0.4 | **0.1854** | 116 | **0.1717** | 0.2482 | 0.2428 | **−0.679** |
| wd 1e-2 + drop 0.4 | 0.1758 | 27 | 0.1439 | 0.2468 | 0.2333 | +0.117 |

Three conclusions:

1. **The probe overfits by epoch ~10–20, not 40.** Base run, external per-native by
   phase: 0.1521 (ep 1–20) → 0.1598 (ep 21–60) → 0.1477 (ep 61–120), while train loss
   falls monotonically 0.579 → 0.384. Rankable is starker: peaks 0.2608 at **epoch 6**,
   ends at 0.1797. Tripling the epochs makes it worse.
2. **Regularisation does not lift the ceiling.** The five configs span 0.172–0.185
   against an epoch-to-epoch noise floor of 0.0086–0.0166 (std of external per-native
   over epochs 20–120). Total variation across the sweep is roughly one noise unit. The
   ceiling is set by the representation, not the optimisation. Dropout 0.4 does flatten
   the late-epoch decay (0.1717 vs 0.1477) and is the better default if this is rerun,
   but it buys ~0.02.
3. **The fine-tune has already flattened too.** Per-native by epoch: 0.0010 → 0.0873 →
   0.1424 → 0.1456. The third epoch added **+0.003** for ~7.4 GPU-hours, while train
   loss was still falling (0.5591 → 0.5200 → 0.5036). Not converged in loss, but
   converged on the axis that matters — and the probe sweep shows what lies on the far
   side of that flattening.

Best case from more epochs is ~0.185 per-native against the GVP's 0.268. Even the most
favourable extra training closes about a third of the gap at the *bottom* of the
bootstrap CI. It does not change any conclusion in §3.

### 4.1 The internal selector barely tracks `val_easy`

Correlation between internal-holdout per-native Spearman and external `val_easy`
per-native Spearman across epochs is only **+0.403** (base) and **−0.679** (dropout
0.4) — actively anti-correlated in the latter. The originally-selected epoch 12 scored
0.1538 externally where epoch 22 scored 0.1813, so **~0.03 of the probe's apparent
deficit was bad epoch selection rather than model capacity**.

The cause is that the two sets measure different things. The internal split is a holdout
of the *train* cache, whose decoy pool includes `perturbed` rigid-body decoys — median
per-native DockQ std 0.29, ORACLE 0.87. `val_easy` has no perturbed decoys: std 0.11,
ORACLE 0.50. Ranking a spread-out decoy set is a different problem from ranking a
tight one.

Worth fixing regardless of anything else here, and free: hold out a `val_easy`-like
split (no perturbed decoys) for model selection. Note that selecting epochs *on*
`val_easy` would convert the holdout into a second validation set and invalidate every
number in §2.

---

## 5. Bottom line

* **The pipeline works and the head does what it was built to do.** A DockQ head on
  cached confidence latents beats Boltz-2's own `confidence_score` by +0.05–0.06
  per-native Spearman, with P(Δ>0) = 1.000 over a 763-native paired bootstrap.
* **It is not competitive with the GVP GNN**, which leads by +0.12 per-native and is the
  only predictor tested that selects meaningfully better than random (top-1 0.357 vs
  0.223).
* **Neither more capacity nor more epochs closes that gap.** Unfreezing 25M params:
  Δ = −0.008, P = 0.19. Six hundred additional probe epochs across five
  regularisation settings: best 0.185 vs 0.268.

The constraint is the representation. Pooled confidence latents are a global summary of
a structure; DockQ is a property of one interface. That is the likely reason a graph
over residues, with explicit interface geometry, extracts more.

## 6. Suggested next steps

In descending value-per-unit-effort:

1. **Ensemble GVP + Boltz predictions.** Both are already joined on identical rows in
   `results/dockq_predictor_comparison_val_easy.csv`; §3.5 and §3.6 hint the signals are complementary
   (`confidence_score` is the best boltz2-decoy predictor while the GVP is best on
   protenix/esmfold2). Cost: minutes. Tells you whether the latents carry anything the
   graph does not.
2. **Feed Boltz latents into the GVP as node features** rather than training a head on
   them alone — keeps the interface-local inductive bias and adds whatever the trunk
   knows. Note the ESM-C precedent: pretrained per-residue embeddings *hurt* GVP
   ranking (0.53 vs 0.635), so this is not a foregone win.
3. **Fix the selector** (§4.1): a `val_easy`-like internal split with no perturbed
   decoys. Recovers ~0.03 for free and makes every future probe number trustworthy.
4. **Interface-local pooling of the latents** instead of the current global pool, if the
   head is pursued further — this targets the representation gap directly rather than
   optimising around it.

Not recommended: more fine-tune epochs, or a larger head. Both are measured above and
neither pays.

## 7. Reproducing

Every table in §2, §3.1 and §3.6 is produced by one script:

```bash
poetry run python workflows/scripts/analysis/compare_dockq_predictors_val_easy.py \
    --output_csv results/dockq_predictor_comparison_val_easy.csv
```

It reconstructs the cache-style `decoy_id` from manifest paths so the GVP and Boltz
families land on the same rows, and **aborts** unless all 33,879 keys match and the two
independently-recorded true-DockQ columns agree to 1e-6 (they agree to 3e-8). Do not
hand-join these CSVs: the GVP ablation keys on the absolute `pdb_path` while the Boltz
caches key on `source__condition__[seed__]stem`, and a partial join silently scores a
predictor on a smaller, easier subset.

The §4 epoch curves:

```bash
CUDA_VISIBLE_DEVICES=0 poetry run python \
    workflows/scripts/analysis/probe_epoch_curve.py --epochs 120 --tag base
```

Artifacts:

| what | where |
|---|---|
| joined per-decoy table (all 12 predictors) | `results/dockq_predictor_comparison_val_easy.csv` |
| 120-epoch sweep curves | `results/probe_curves/probe_curve_{base,wd1e2,wd1e1,drop04,wd1e2drop04}.json` |
| probe run + external report | `runs/dockq_probe_v2_valeasy/report_external{,_rankable}.json` |
| fine-tune v3 + per-epoch reports | `runs/confidence_ft_v3/report_epoch{0,1,2,3}.json` |
| GVP per-decoy predictions | `checkpoints/ablation_gvp_feature_loo/full-s{0,1,2}/full/full_val_easy_predictions.csv` |
| GVP 3-seed summary | `checkpoints/ablation_gvp_feature_loo/analysis_wandb/ablation_summary_complete.csv` |
| `val_easy` latents (86 MB, 763 shards) | `$VOLUME/sabdab_val_easy/dockq_latents/latents` |
| `sabdab_train` confidence cache (157 GB) | `$VOLUME/sabdab_train/confidence_cache` |

`$VOLUME` is whatever storage you pointed the caching step at. **None of these artifacts
are in the repository** — `runs/`, `checkpoints/` and `$VOLUME` are all gitignored, and
the caches are 157 GB. They are recorded here so the numbers above are traceable to
what produced them; regenerate with the commands in this section.

Retrain the probe, evaluating `val_easy` at the selected epoch:

```bash
poetry run python workflows/scripts/training/train_dockq_probe.py \
    --latent_dir      $VOLUME/sabdab_train/dockq_latents \
    --eval_latent_dir $VOLUME/sabdab_val_easy/dockq_latents \
    --output_dir runs/dockq_probe_v3 --epochs 20 --dropout 0.4
```

Extract latents for a new cache (6 GPUs, ~50 min for 763 entries):

```bash
workflows/scripts/training/extract_dockq_latents_all_gpus.sh \
    $VOLUME/sabdab_val_easy/dockq_latents 0,1,2,3,4,5 \
    --cache_dir $VOLUME/sabdab_val_easy/confidence_cache
```

### A metric-definition caveat

`boltz.finetune.trainer.correlation_report` and
`dockq_predictor.training.metrics` differ in two edge cases: the former excludes
natives with fewer than 3 decoys and undefined correlations from the mean, the latter
counts them as 0.0. On `val_easy` neither branch fires — no native has fewer than 3
decoys (minimum 30, median 45) and none has constant true DockQ — and the two
implementations were verified to agree to 4 decimal places across every metric. On a
set that does contain such natives they will diverge, with `correlation_report`
reporting the higher number. `RANKABLE_MIN_STD = 0.05` is duplicated in both modules
and must be changed in both.

### Comparability caveats

* The GVP column is a 3-seed mean of per-decoy predictions; the Boltz columns are
  single runs. Per-seed GVP per-native was 0.2651 / 0.2798 / 0.2453 (seed std 0.014),
  so the +0.12 gap is an order of magnitude larger than seed variance — but the Boltz
  runs have no seed estimate of their own.
* `own_ptm` / `own_iptm` are each generator's self-reported scores and so are not a
  single scorer: the `ptm`/`iptm` rows are Boltz-2 re-scoring every decoy uniformly,
  which is the like-for-like comparison. This distinction is why figures here differ
  from the GVP ablation's `esmfold2_ptm` (0.7216, ESMFold's own pTM) versus Boltz-2's
  re-score of the same structures (0.6313).
* `ligand_iptm` is NaN throughout — these are protein-only complexes — and is omitted.
