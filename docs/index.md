# ConfiDock

DockQ-aware confidence prediction for antibody–antigen complexes.

Co-folding models report confidence in their own prediction. A design campaign needs
something different: an estimate of **DockQ** — whether the predicted interface is right.
ConfiDock provides two routes to that estimate and the evaluation machinery to compare
them.

- **Boltz-2 confidence fine-tune** — a DockQ head on a frozen Boltz-2 trunk, trained on
  cached confidence latents.
- **DockQ-predictor GNN** — a standalone graph network over the predicted complex, with no
  co-folding model needed at inference time.

Decoys come from Boltz-2, Protenix-v2 or ESMFold2, each with a grid runner for sweeping
MSA mode, seeds and sampling settings.

## Where to start

| Page | Covers |
|---|---|
| [DockQ predictor inference](dockq_predictor_inference.md) | Running the GNN on predicted structures, CDR sidecar metadata, search paths. |
| [Boltz-2 confidence caching](boltz_confidence_caching.md) | Building the cached-latent dataset the fine-tune trains on. |

Installation, the end-to-end quick start and the `CONFIDOCK_*` environment variables are
in the [repository README](https://github.com/AstraZeneca/ConfiDock#readme).
