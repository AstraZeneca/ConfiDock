# Protenix Inference Environment

This directory contains a separate Poetry environment for running Protenix 2.0.0 inference.

## Why a separate environment?

Protenix has specific dependency requirements that conflict with the main project environment:
- Requires Python 3.11-3.12 (main project uses 3.10)
- Uses a local development install of Protenix from `../protenix-src`
- Has minimal dependencies optimized for inference only

## Setup

### Quick Start (PyPI Installation)

Use the official PyPI release for standard inference:

```bash
cd protenix-inference
poetry install
```

That's it! This installs Protenix v2.0.0 from PyPI with all dependencies.

**Note:** For large-scale inference, you may want to use Option 2 below for performance improvements.

---

### Advanced: Local Clone with Custom Modifications

For large-scale inference workloads, you can clone Protenix locally and apply performance/stability patches:

Clone the Protenix source to apply performance and stability improvements:

**Step 1:** Clone Protenix source

```bash
# From the repository root
git clone https://github.com/bytedance/Protenix.git protenix-src
cd protenix-src
git checkout v2.0.0  # Tested version
cd ..
```

**Step 2:** Apply custom modifications (optional but recommended)

Edit `protenix-src/configs/configs_inference.py`:
```python
"num_workers": 4,  # Change from 0 for faster data loading
```

Edit `protenix-src/runner/inference.py` after line 505 to add memory cleanup:
```python
# Add explicit cleanup to prevent OOM
if 'prediction' in locals():
    del prediction
if 'data' in locals():
    del data
gc.collect()
torch.cuda.empty_cache()
```

**Step 3:** Update `pyproject.toml` to use local install

Edit `protenix-inference/pyproject.toml` and change:
```toml
protenix = "^2.0.0"
```
to:
```toml
protenix = { path = "../protenix-src", develop = true }
```

**Step 4:** Install the environment

```bash
cd protenix-inference
poetry install
```

This will:
- Create a Python 3.11+ virtual environment
- Install Protenix from `../protenix-src` in development mode with your modifications
- Install minimal inference dependencies (pyyaml, biopython, click)

### Verification

```bash
poetry run protenix --help
```

If this works, you're ready to run Protenix inference!

## Usage

The inference scripts automatically use this environment when running Protenix:

```bash
# From repo root
workflows/scripts/inference/protenix_grid_ddp.sh \
    /path/to/inputs.json \
    /path/to/output \
    configs/protenix_grid_conditions_P1P2.yaml
```

The script will:
1. `cd` into `protenix-inference/`
2. Run `poetry run python` to use this environment
3. Add the repo root to `PYTHONPATH` so it can import from `confidock`

## Dependencies

- **protenix**: Local development install from `../protenix-src`
- **pyyaml**: Configuration file parsing
- **biopython**: Sequence and structure handling
- **click**: CLI utilities

## Troubleshooting

**Error: `protenix-src` not found during `poetry install`**

Make sure you cloned Protenix to the correct location:
```bash
ls ../protenix-src  # Should show Protenix files
```

**Error: Python version incompatible**

Protenix requires Python 3.11-3.12. Check your version:
```bash
python --version
```

If needed, use pyenv or conda to install a compatible Python version before running `poetry install`.

**Error: Protenix CLI not found**

The grid scripts automatically find the Protenix CLI. If you need to test it manually:
```bash
cd protenix-inference
poetry run protenix --help
```
