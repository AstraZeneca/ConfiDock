#!/usr/bin/env python
r"""Write a copy of a set's ESMFold2 input JSONs with an explicit ``msaPath`` per protein chain.

The ESMFold2 arm normally carries no MSA field at all: ``run_esmfold2_grid.py`` hashes each
sequence at run time and looks for ``<msa_dir>/<sha16>.a3m``. That works, but the MSA is then a
property of whichever directory the job mounted rather than of the input, and a chain whose
``.a3m`` is missing folds MSA-free with only a warning.

This script resolves every chain up front and records the result::

    <root>/esmfold2_inputs/esmfold2_negatives_shard0_of8.json
        -> <root>/esmfold2_inputs_msa/esmfold2_negatives_shard0_of8.json

with each protein chain gaining::

    "msaPath": "/.../sabdab_msas/<sha16>.a3m"

Chains with no ``.a3m`` are left without the key on purpose, so they keep the existing
"fold MSA-free, warn about it" behaviour rather than pointing at a file that isn't there --
``resolve_chain_a3m`` raises on a dangling ``msaPath``. The coverage report says how many
chains that affects, split by antibody vs antigen.

Unlike the Protenix equivalent there is no staging tree: ESM takes a single unpaired
alignment, so the ``.a3m`` in ``--msa-dir`` is referenced directly and nothing is copied
or symlinked.

Usage::

    python workflows/scripts/processing/bake_esmfold2_msa_paths.py \
        --root /path/to/set_root
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from confidock.esmfold2.inference.runner import MSA_PATH_KEY, resolve_a3m  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_MSA_DIR = Path("/data/sabdab/msas")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Set root holding esmfold2_inputs/")
    parser.add_argument("--in-dir", type=Path, default=None, help="Default: <root>/esmfold2_inputs")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <root>/esmfold2_inputs_msa")
    parser.add_argument("--msa-dir", type=Path, default=DEFAULT_MSA_DIR, help="Flat <sha16>.a3m directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output JSONs")
    return parser.parse_args()


def bake_entry(entry: dict, msa_dir: Path, cache: dict[str, Path | None]) -> tuple[dict, int, int]:
    """Add ``msaPath`` to every protein chain of one entry that has an ``.a3m``.

    Parameters
    ----------
    entry : dict
        ESMFold2 input entry
    msa_dir : Path
        Flat hash-named MSA directory
    cache : dict
        Memo of {sequence: resolved path or None}, shared across entries

    Returns
    -------
    tuple[dict, int, int]
        (new entry, chains with an MSA, chains without)
    """
    out = {k: v for k, v in entry.items() if k != "sequences"}
    sequences = []
    found = missing = 0

    for chain in entry["sequences"]:
        c = dict(chain)
        if str(c.get("type", "")).lower() == "protein":
            seq = str(c["sequence"])
            if seq not in cache:
                cache[seq] = resolve_a3m(seq, msa_dir)
            path = cache[seq]
            if path is not None:
                c[MSA_PATH_KEY] = str(path)
                found += 1
            else:
                c.pop(MSA_PATH_KEY, None)
                missing += 1
        sequences.append(c)

    out["sequences"] = sequences
    return out, found, missing


def report_coverage(entries: list[dict], baked: list[dict]) -> None:
    """Log MSA coverage over chains, split by antibody-like vs antigen chain IDs.

    The split mirrors the H/L-prefix fallback in ``run_esmfold2_grid._is_antibody_chain``
    (chain IDs here are ``Hc``/``Lc``/``H``/``L`` for binders). It is a reporting heuristic
    only -- nothing in the written JSON depends on it, and the runner prefers the SAbDab TSV
    over the same heuristic when deciding which chains ``antigen_only`` covers.

    Parameters
    ----------
    entries : list of dict
        Original entries
    baked : list of dict
        Entries with msaPath added
    """
    total: Counter[str] = Counter()
    have: Counter[str] = Counter()
    for entry in baked:
        for chain in entry["sequences"]:
            if str(chain.get("type", "")).lower() != "protein":
                continue
            kind = "antibody" if str(chain.get("id", ""))[:1].upper() in {"H", "L"} else "antigen"
            total[kind] += 1
            have[kind] += MSA_PATH_KEY in chain
    for kind in sorted(total):
        LOGGER.info(
            "MSA coverage, %s chains: %d/%d have an .a3m (%d would fold MSA-free)",
            kind,
            have[kind],
            total[kind],
            total[kind] - have[kind],
        )
    LOGGER.info("entries: %d in, %d out", len(entries), len(baked))


def msa_paths(entries: list[dict]) -> list[tuple[str, str]]:
    """Every ``(entry name, msaPath)`` this shard names, in file order.

    Paired with the entry name rather than returned bare because the only thing done with a bad path is
    to report it, and "some shard has a dangling msaPath" is not enough to find the entry that carries it.
    Chains that were left without the key are simply absent -- that is the intended state for a sequence
    with no ``.a3m``, not something to verify.

    Parameters
    ----------
    entries : list of dict
        Entries re-read from one written shard

    Returns
    -------
    list[tuple[str, str]]
        (entry name, path) per chain naming an MSA
    """
    return [
        (str(entry.get("name")), str(chain[MSA_PATH_KEY]))
        for entry in entries
        for chain in entry["sequences"]
        if chain.get(MSA_PATH_KEY) is not None
    ]


def verify_shard(name: str, entries: list[dict], n_entries: int) -> int:
    """Check one re-read shard and return how many ``msaPath`` values it resolves.

    Both checks are fatal rather than warnings: this runs after the writes, and the whole point of baking
    the paths in is that the fold job can trust them. A shard that is short an entry or names a file that
    is not there would otherwise be found by 8 GPU jobs at once, hours later.

    Parameters
    ----------
    name : str
        Shard file name, for the messages
    entries : list of dict
        Entries re-read from the written shard
    n_entries : int
        Entry count the writer believes it wrote

    Returns
    -------
    int
        Number of ``msaPath`` values, all of which exist
    """
    if len(entries) != n_entries:
        sys.exit(f"{name}: wrote {n_entries} entries but re-read {len(entries)}")
    paths = msa_paths(entries)
    for entry_name, path in paths:
        # The emptiness test is not redundant with ``exists()``: ``Path("")`` is ``Path(".")``, which
        # exists, so a blank ``msaPath`` would verify clean here and then fail at fold time with an
        # IsADirectoryError from whichever of the 8 jobs reached it first.
        if not path or not Path(path).exists():
            sys.exit(f"{name}: entry {entry_name} has dangling {MSA_PATH_KEY}: {path!r}")
    return len(paths)


def verify(out_dir: Path, expected: dict[str, int]) -> None:
    """Re-read the written shards and assert every ``msaPath`` resolves.

    Parameters
    ----------
    out_dir : Path
        Directory of written shard JSONs
    expected : dict
        Mapping {shard_name: entry_count} the writer believes it wrote
    """
    for name, n_entries in expected.items():
        entries = json.loads((out_dir / name).read_text(encoding="utf-8"))
        n_paths = verify_shard(name, entries, n_entries)
        LOGGER.info("verified %s: %d entries, %d msaPath values all resolve", name, n_entries, n_paths)


def main() -> None:
    """Resolve each chain's .a3m once and write input JSONs that name it explicitly."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    in_dir = args.in_dir or args.root / "esmfold2_inputs"
    out_dir = args.out_dir or args.root / "esmfold2_inputs_msa"

    if not in_dir.is_dir():
        sys.exit(f"Input directory {in_dir} does not exist")
    if not args.msa_dir.is_dir():
        sys.exit(f"MSA directory {args.msa_dir} is not readable")

    shards = sorted(in_dir.glob("*.json"))
    if not shards:
        sys.exit(f"No *.json found in {in_dir}")
    existing = [s.name for s in shards if (out_dir / s.name).exists()]
    if existing and not args.overwrite:
        sys.exit(f"{out_dir} already holds {len(existing)} of these shards; pass --overwrite to replace")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Path | None] = {}
    written: dict[str, int] = {}
    all_in: list[dict] = []
    all_out: list[dict] = []

    for shard in shards:
        entries = json.loads(shard.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            sys.exit(f"{shard} is not a list of entries -- is this an ESMFold2 input JSON?")
        baked, found, missing = [], 0, 0
        for entry in entries:
            e, f, m = bake_entry(entry, args.msa_dir, cache)
            baked.append(e)
            found += f
            missing += m
        (out_dir / shard.name).write_text(json.dumps(baked, indent=2), encoding="utf-8")
        written[shard.name] = len(baked)
        all_in.extend(entries)
        all_out.extend(baked)
        LOGGER.info("wrote %s (%d chains with MSA, %d without)", out_dir / shard.name, found, missing)

    LOGGER.info("unique sequences resolved: %d", len(cache))
    report_coverage(all_in, all_out)
    verify(out_dir, written)
    LOGGER.info("done: %d shards in %s", len(written), out_dir)


if __name__ == "__main__":
    main()
