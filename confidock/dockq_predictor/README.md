# DockQ Predictor - Simple GNN

A lightweight Graph Neural Network to predict DockQ scores for antibody-antigen complexes.

**Built with PyTorch Lightning** for scalable training with multi-GPU support, mixed precision, and comprehensive logging.

## Overview

This model takes a single predicted complex structure (from Boltz-2, ESMFold2, or Protenix) and predicts:
- **DockQ score** (0-1, higher is better)

Currently implements a simple baseline architecture for quick prototyping and establishing baseline performance.

**Note**: This baseline only predicts DockQ, not the individual components (Fnat, iRMSD, LRMSD).

## Architecture - Simple GNN

1. **Node Encoder**: Projects 24-dim residue features to hidden dim (128D default)
2. **Edge Encoder**: Projects 21-dim edge features to edge dim (64D default)
3. **Graph Attention Layers**: 3 GAT layers with 4 attention heads
4. **Interface Pooling**: Average pooling over interface residues
5. **DockQ Head**: MLP predicting DockQ score

### Node Features (24-dim)
- Amino acid one-hot (21)
- Is antibody flag (1)
- pLDDT confidence (1)
- Reserved (1)

### Edge Features (21-dim)
- Distance RBF encoding (16)
- PAE value (1)
- Cross-chain flag (1)
- Model source one-hot (3)

## Quick Start

### Training from Manifest

```bash
# Prepare your data manifest CSV with columns:
# pdb_path,pae_path,model_source,dockq,fnat,irmsd,lrmsd,chains
# /path/to/complex1.pdb,/path/to/pae1.json,boltz2,0.75,0.65,1.2,2.5,HLA

# Train SimpleGNN
python workflows/scripts/training/train_simple_gnn_from_manifest.py \
    --manifest data/dockq_predictor/dockq_training_manifest.csv \
    --config configs/dockq_predictor/simple_gnn.yaml

# Multi-GPU training (automatic with PyTorch Lightning)
python workflows/scripts/training/train_simple_gnn_from_manifest.py \
    --manifest data/dockq_predictor/dockq_training_manifest.csv \
    --config configs/dockq_predictor/simple_gnn.yaml \
    --gpus 2
```

### Building Training Manifest

The training manifest is built from the per-model DockQ score CSVs produced by
the grid inference pipeline (boltz2 / esmfold2 / protenix). For each model,
`sample_dockq_for_training.py` samples complexes and keeps **all** decoys per
complex (every grid condition x seed/sample), then resolves each decoy's
predicted structure and PAE file path and writes a validated manifest.

```bash
python workflows/scripts/analysis/sample_dockq_for_training.py \
    --boltz2-csv   /path/to/boltz2_r1/dockq/dockq_scores.csv \
    --esmfold2-csv /path/to/esmfold2_r1/dockq/dockq_scores.csv \
    --protenix-csv /path/to/protenix_r1/dockq/dockq_scores.csv \
    --boltz2-dir   /path/to/boltz2_r1 \
    --esmfold2-dir /path/to/esmfold2_r1 \
    --protenix-dir /path/to/protenix_r1 \
    --n-samples 200 \
    --output data/dockq_predictor/dockq_training_manifest_multidecoy.csv
```

`--n-samples` is the number of complexes sampled **per model** (all decoys of
each are kept). Pass `--models` to restrict to a subset of model sources.


## Directory Structure

```
dockq_predictor/
├── configs/
│   └── simple_gnn.yaml      # SimpleGNN configuration
├── data/
│   ├── dataset.py           # PyTorch Geometric Dataset
│   ├── datamodule.py        # Lightning DataModule
│   ├── graph_builder.py     # Residue-level graph construction
│   ├── chain_mapping*.py    # Chain type identification
│   ├── feature_utils_simple.py  # Simplified feature extraction
│   └── pae_utils.py         # PAE loading and normalization
├── models/
│   └── simple_gnn.py        # SimpleGNN Lightning model
└── losses/
    └── losses.py            # DockQLoss (Huber + MRL + uncertainty)
```

## Loss Function

**Simple Huber Loss** for robust DockQ regression:
- `L = huber_loss(pred_dockq, true_dockq, delta=0.1)`
- Delta of 0.1 provides robust handling of outliers

**Note**: The full DockQLoss (with Matryoshka + uncertainty components) is defined in `losses/losses.py` but not currently used by SimpleGNN. The SimpleGNN uses direct Huber loss for simplicity. Can also rethink loss learning as a ranking loss soon.
