#!/usr/bin/env python
"""Score predicted structures with a trained DockQ predictor -- no ground truth needed.

    poetry run python workflows/scripts/inference/predict_dockq.py \
        --checkpoint  model/epoch034-score0.6579.ckpt \
        --config      model/config_used.yaml \
        --structures '/data/poses/7y8j_HLA_*.pdb' \
        --model-source boltz2 \
        --chains HLA \
        --output scores.csv

This is the deployment entry point: given one or many predicted poses and no native to
compare against, it returns a ``dockq_pred`` per pose and a rank within each complex.

Its counterpart ``workflows/scripts/training/predict_dockq_from_manifest.py`` is for
EVALUATION -- it requires a true ``dockq`` per row and reports the full metric set
against it. Use that when you have labels and want to know how good the model is; use
this when you have no labels and want to know which pose to take forward.

WHAT THE OUTPUT IS FOR
----------------------
The model is trained and selected on within-complex ranking, and its ranking term
compresses the prediction scale as a side effect, so ``dockq_pred`` is a good ordering
key and a poorly calibrated DockQ estimate. Rank within a complex and take the top; do
not report the value as a DockQ.

Poses are grouped into complexes by ``_native_id_from_path`` -- the first two
'_'-separated tokens of the filename -- because that is what the model itself groups by.
Files that should be ranked against each other must share that prefix.

Complexes outside the SAbDab snapshot need a CDR metadata sidecar, or 8 of the 18 node
features are silently all-zero. Build them with
``workflows/scripts/inference/build_cdr_sidecar_json.py`` and pass ``--cdr-json-dir``;
the run prints the directories it searched and warns about complexes with no metadata.

The config must be the one the checkpoint was trained with: the checkpoint holds only
weights, so the architecture and the featurisation settings (``interface_cutoff``,
``knn_k``) are read from the config. A mismatched config raises on the state-dict shapes
rather than silently mis-scoring. The W&B registry ships the two together.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from confidock.dockq_predictor.data.chain_mapping_cif import find_sabdab_json, sabdab_json_search_dirs
from confidock.dockq_predictor.data.dataset import DockQDataset, _native_id_from_path
from confidock.dockq_predictor.models import SimpleGNN

# The folding models the edge feature block one-hots. Anything else lands on the
# all-zeros vector, which is a combination the model never saw in training.
KNOWN_MODEL_SOURCES = ("boltz2", "esmfold2", "protenix")

_SUFFIXES = (".cif", ".mmcif", ".pdb", ".ent")


class _IndexedDataset(DockQDataset):
    """``DockQDataset`` that tags each graph with its row index.

    A structure that fails to featurise yields None and is dropped from its batch, so
    dataloader position is not a reliable key to join predictions back onto rows.
    Carrying the index on the graph makes the join exact.
    """

    def get(self, idx: int):
        """Build the graph for row ``idx``, tagged with that index."""
        data = super().get(idx)
        if data is not None:
            # NOT "entry_index": PyG's Data.__inc__ adds the cumulative node count to
            # any attribute whose name contains "index" when batching, which would turn
            # these row numbers into garbage.
            data.entry_row = torch.tensor([idx], dtype=torch.long)
        return data


def _collate_skipping_failures(samples: List) -> Optional[Batch]:
    """Batch the graphs that built, dropping the Nones."""
    kept = [s for s in samples if s is not None]
    return Batch.from_data_list(kept) if kept else None


def iter_structure_files(patterns: List[str]) -> List[Path]:
    """Expand globs / directories / plain paths into a sorted list of structure files."""
    found: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            for suffix in _SUFFIXES:
                found.extend(path.rglob(f"*{suffix}"))
        elif path.exists():
            found.append(path)
        else:
            from glob import glob

            found.extend(Path(p) for p in glob(pattern, recursive=True))
    keep = [p for p in found if p.suffix.lower() in _SUFFIXES]
    return sorted(set(keep))


def rows_from_structures(patterns: List[str], model_source: str, chains: Optional[str]) -> pd.DataFrame:
    """Build the input table from structure paths, with one model_source / chains for all."""
    files = iter_structure_files(patterns)
    if not files:
        raise SystemExit(f"No .cif/.pdb files matched {patterns}")
    return pd.DataFrame(
        {
            "pdb_path": [str(p) for p in files],
            "model_source": model_source,
            "chains": chains if chains is not None else pd.NA,
        }
    )


def rows_from_manifest(manifest: Path, model_source: Optional[str], chains: Optional[str]) -> pd.DataFrame:
    """Build the input table from a CSV, filling in anything the CLI overrides.

    Only ``pdb_path`` is required. A ``dockq`` column is neither needed nor read -- it is
    carried through to the output untouched if present, so a labelled manifest can be
    scored here too without the labels influencing anything.
    """
    df = pd.read_csv(manifest)
    if "pdb_path" not in df.columns:
        raise SystemExit(f"{manifest} has no 'pdb_path' column (columns: {list(df.columns)})")

    if model_source is not None:
        df["model_source"] = model_source
    elif "model_source" not in df.columns:
        raise SystemExit(
            f"{manifest} has no 'model_source' column and --model-source was not given. "
            f"It selects a one-hot edge feature; pass one of {KNOWN_MODEL_SOURCES}."
        )

    if chains is not None:
        df["chains"] = chains
    elif "chains" not in df.columns:
        df["chains"] = pd.NA

    return df.reset_index(drop=True)


def apply_cdr_json_dirs(dirs: Optional[List[Path]]) -> None:
    """Put ``--cdr-json-dir`` at the front of the CDR metadata search path.

    Exported into the environment rather than threaded down as an argument, because the
    lookup happens deep inside the featuriser: ``chain_mapping_cif.find_sabdab_json``
    reads ``CONFIDOCK_SABDAB_JSON_DIR`` at call time. Setting it in this process means
    the dataloader workers inherit it too.

    Prepended, not replaced: an existing export still contributes, so a sidecar
    directory for a handful of complexes adds to the shared snapshot rather than hiding
    the rest of it.
    """
    if not dirs:
        return

    # A mistyped directory is precisely the failure this flag exists to prevent -- it
    # would otherwise look identical to having built no sidecars at all.
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        raise SystemExit(
            f"--cdr-json-dir: not a directory: {missing}. Build the sidecars first with "
            "workflows/scripts/inference/build_cdr_sidecar_json.py."
        )

    current = os.environ.get("CONFIDOCK_SABDAB_JSON_DIR", "")
    ordered = [str(d.resolve()) for d in dirs] + ([current] if current else [])
    os.environ["CONFIDOCK_SABDAB_JSON_DIR"] = os.pathsep.join(ordered)


def print_cdr_search_path() -> None:
    """Print the directories CDR metadata is looked up in, with the JSON count in each.

    The wiring is otherwise invisible at the call site -- nothing on the command line
    hints that CDR features depend on an environment variable -- so the run states where
    it looked and what it found.
    """
    print("CDR metadata search path (first match wins):")
    for directory in sabdab_json_search_dirs():
        path = Path(directory)
        if path.is_dir():
            print(f"  {directory}  ({sum(1 for _ in path.glob('*.json'))} json)")
        else:
            print(f"  {directory}  (does not exist)")


def warn_about_features(df: pd.DataFrame) -> None:
    """Report the two ways this run can silently lose feature blocks.

    Both failure modes are invisible at runtime -- the model scores a graph with dead
    columns and returns a plausible number -- so they are worth a paragraph on stderr
    rather than a line in a log nobody reads.
    """
    unknown = sorted(set(df["model_source"].dropna()) - set(KNOWN_MODEL_SOURCES))
    if unknown:
        print(
            f"WARNING: model_source {unknown} is not one of {list(KNOWN_MODEL_SOURCES)}. "
            "That edge feature block will be all-zeros, which the model never saw in "
            "training. Harmless for ranking within one source (it is then constant), "
            "but do not compare scores across sources on this basis.",
            file=sys.stderr,
        )

    if df["chains"].isna().all():
        print(
            "WARNING: no 'chains' given, so heavy/light/antigen roles will be guessed "
            "(metadata JSON by chain length, then ANARCI, then a length heuristic). A "
            "wrong guess types the antibody as antigen and wrecks the features. Pass "
            "--chains, e.g. HLA.",
            file=sys.stderr,
        )

    missing = sorted({_native_id_from_path(p) for p in df["pdb_path"] if find_sabdab_json(p) is None})
    if missing:
        shown = ", ".join(missing[:5]) + (f", ... (+{len(missing) - 5})" if len(missing) > 5 else "")
        print(
            f"WARNING: no CDR metadata JSON found for {len(missing)} complex(es): {shown}.\n"
            "  CDR labels will be all-zero, killing 8 of the 18 node features and 4 edge\n"
            "  features that were live during training. Build sidecars first, then point\n"
            "  this run at them:\n"
            "    python workflows/scripts/inference/build_cdr_sidecar_json.py \\\n"
            "        --structures <same paths> --out-dir cdr_json\n"
            "    ... and re-run with --cdr-json-dir cdr_json",
            file=sys.stderr,
        )


def load_model(checkpoint: Path, config: dict) -> SimpleGNN:
    """Construct the model from the config and load the checkpoint's weights.

    Constructed-then-loaded rather than ``load_from_checkpoint``: the model takes a
    single ``config`` dict, which is not a signature Lightning can reconstruct from
    saved hparams.

    Loaded with ``weights_only=True``, so a checkpoint from an untrusted source cannot
    execute code on load. Nothing on this path needs more: the config comes from the
    yaml, not from the checkpoint's hparams.
    """
    model = SimpleGNN(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    print(f"Loaded {checkpoint} (epoch {state.get('epoch', '?')}, step {state.get('global_step', '?')})")
    return model


def score(
    df: pd.DataFrame,
    model: SimpleGNN,
    config: dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    """Run the model over every structure and return the rows that scored.

    ``dockq`` is set to 0.0 in the dataset entries because ``DockQDataset`` attaches it
    as the training label ``y_dockq``. It is never read on the forward path, so the value
    is arbitrary -- but it must be present, which is exactly why the evaluation script
    cannot be used unlabelled.
    """
    data_cfg = config.get("data", {})
    entries = [
        {
            "pdb_path": row["pdb_path"],
            "model_source": row["model_source"],
            "dockq": 0.0,  # unused placeholder label, see docstring
            "chain_str": row["chains"] if pd.notna(row["chains"]) else None,
        }
        for _, row in df.iterrows()
    ]

    dataset = _IndexedDataset(
        entries=entries,
        interface_cutoff=float(data_cfg.get("interface_cutoff", 6.0)),
        knn_k=int(data_cfg.get("knn_k", 16)),
        cache_graphs=False,
        graph_cache_dir=None,  # a single pass gains nothing from writing a cache
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_skipping_failures,
        pin_memory=False,
    )

    model = model.to(device).eval()
    rows: List[int] = []
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue
            batch = batch.to(device)
            preds.append(model(batch)["dockq"].detach().cpu())
            rows.extend(batch.entry_row.cpu().tolist())
            if (batch_idx + 1) % 25 == 0:
                print(f"  scored {len(rows)}/{len(dataset)}", flush=True)

    if not preds:
        raise SystemExit("No structure could be featurised; nothing to score.")

    scored = df.iloc[rows].copy()
    scored["dockq_pred"] = torch.cat(preds).numpy()
    n_missing = len(dataset) - len(scored)
    failed = f" ({n_missing} failed to featurise)" if n_missing else ""
    print(f"Scored {len(scored)}/{len(dataset)} structure(s){failed}")
    return scored


def rank_within_complex(scored: pd.DataFrame) -> pd.DataFrame:
    """Add the ``complex`` key and a 1-based rank within it, best first."""
    scored = scored.copy()
    scored["complex"] = [_native_id_from_path(p) for p in scored["pdb_path"]]
    scored["rank"] = scored.groupby("complex")["dockq_pred"].rank(ascending=False, method="first").astype(int)
    return scored.sort_values(["complex", "rank"]).reset_index(drop=True)


def print_summary(ranked: pd.DataFrame, max_complexes: int = 20) -> None:
    """Print the best pose per complex -- the answer for the common single-complex case."""
    best = ranked[ranked["rank"] == 1]
    print("\n" + "=" * 78)
    print(f"Best pose per complex ({len(best)} complex/complexes, {len(ranked)} pose(s) scored)")
    print("=" * 78)
    print(f"  {'complex':<24s} {'dockq_pred':>10s}  {'n_poses':>7s}  best pose")
    counts = ranked.groupby("complex").size()
    for _, row in best.head(max_complexes).iterrows():
        name = Path(row["pdb_path"]).name
        print(f"  {row['complex']:<24s} {row['dockq_pred']:>10.4f}  {counts[row['complex']]:>7d}  {name}")
    if len(best) > max_complexes:
        print(f"  ... and {len(best) - max_complexes} more (see the output CSV)")
    print("-" * 78)
    print("  dockq_pred ranks poses; it is NOT a calibrated DockQ value. Ordering within a")
    print("  complex is what the model was trained and selected for.")
    print("=" * 78)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained .ckpt to score with")
    parser.add_argument("--config", type=Path, required=True, help="The config the checkpoint was trained with")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--structures",
        nargs="+",
        metavar="PATH",
        help="Structure files, globs, or directories (walked recursively) to score",
    )
    source.add_argument(
        "--manifest",
        type=Path,
        help="CSV with a pdb_path column (plus optional model_source / chains); no dockq needed",
    )

    parser.add_argument(
        "--model-source",
        help=(
            f"Which folding model produced the poses, one of {KNOWN_MODEL_SOURCES}. Required "
            "with --structures; overrides the manifest column when both are given."
        ),
    )
    parser.add_argument(
        "--chains",
        help="SAbDab-style chain string applied to every pose, e.g. HLA (H/Hc=heavy, L/K/Lc/Kc=light, rest=antigen)",
    )
    parser.add_argument(
        "--cdr-json-dir",
        type=Path,
        nargs="+",
        metavar="DIR",
        help=(
            "Directory/-ies of CDR metadata JSONs (from build_cdr_sidecar_json.py) to search "
            "before the shared SAbDab snapshot. Equivalent to exporting "
            "CONFIDOCK_SABDAB_JSON_DIR, which is also honoured; use this to keep the "
            "dependency visible in the command."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dockq_scores.csv"), help="Output CSV (default: %(default)s)"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override the config's data.batch_size")
    parser.add_argument("--num-workers", type=int, default=4, help="Featurisation workers (default: %(default)s)")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Where to run the network (default: %(default)s -- cuda if visible)",
    )
    args = parser.parse_args(argv)

    if args.structures and args.model_source is None:
        parser.error(
            "--model-source is required with --structures: it selects a one-hot edge "
            f"feature and has no safe default. Pass one of {KNOWN_MODEL_SOURCES}."
        )
    return args


def main(argv: Optional[List[str]] = None) -> None:
    """Score structures end to end and write the ranked CSV."""
    args = _parse_args(argv)

    with open(args.config) as fh:
        config: Dict = yaml.safe_load(fh)

    if args.manifest is not None:
        df = rows_from_manifest(args.manifest, args.model_source, args.chains)
        print(f"Manifest {args.manifest}: {len(df)} structure(s)")
    else:
        df = rows_from_structures(args.structures, args.model_source, args.chains)
        print(f"{len(df)} structure(s) matched")

    n_complexes = len({_native_id_from_path(p) for p in df["pdb_path"]})
    print(f"  grouped into {n_complexes} complex/complexes by filename prefix")

    apply_cdr_json_dirs(args.cdr_json_dir)
    print_cdr_search_path()
    warn_about_features(df)

    model = load_model(args.checkpoint, config)
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )
    print(f"Scoring on {device}")

    batch_size = args.batch_size or int(config.get("data", {}).get("batch_size", 128))
    scored = score(df, model, config, batch_size, args.num_workers, device)
    ranked = rank_within_complex(scored)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)
    print_summary(ranked)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
