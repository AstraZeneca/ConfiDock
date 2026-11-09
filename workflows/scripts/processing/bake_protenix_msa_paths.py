#!/usr/bin/env python
r"""Write a copy of a set's Protenix input shards with the MSA paths baked into the JSON.

Input shard JSONs carry no MSA field: they are templates. ``run_protenix_grid.py`` stages
the ``.a3m`` files at launch, injects the MSA block into an in-memory copy, and hands
Protenix a *temp* JSON -- so the file on disk is never the file the model reads, and the
MSA only exists for the duration of the run.

This script does that staging once, up front, and writes the result::

    <root>/protenix_inputs/protenix_negatives_shard0_of8.json   (template, no MSA)
        -> <root>/protenix_inputs_msa/protenix_negatives_shard0_of8.json   (paths baked in)

Each ``proteinChain`` gains Protenix's current MSA keys, pointing into a staging tree shared by
all shards::

    "pairedMsaPath":   "<root>/protenix_msa_staging/<sha16>/pairing.a3m"
    "unpairedMsaPath": "<root>/protenix_msa_staging/<sha16>/non_pairing.a3m"

The output is directly runnable by ``runner/inference.py --input_json_path ... --use_msa true``
with no ``--msa_dir``, and Protenix will neither re-search MSAs nor rewrite the JSON.

Two things to know before using the output
------------------------------------------
* **``pairing.a3m`` is a query-only dummy.** ``stage_msa_dirs`` writes it so heteromer mode runs
  without crashing; effective pairing depth is zero. That matches every Protenix run in this
  project -- it is the existing convention, not a shortcut introduced here.
* **Sequences with no ``.a3m`` get a query-only ``non_pairing.a3m``**, again matching
  ``run_protenix_grid.py``. The path is still written, deliberately: Protenix's
  ``need_msa_search`` reads a *missing* path as a reason to launch an MMseqs2 search, so omitting
  it would turn a known-MSA-free chain into a live search.

By default the staged ``non_pairing.a3m`` is a **symlink** into ``--msa-dir``, which costs no disk
but makes the JSONs only as portable as that mount. ``--copy-a3m`` materialises them instead
(e.g. 1.4 GB for a set with 804 unique sequences), giving a staging tree
that stands alone.

Usage::

    python workflows/scripts/processing/bake_protenix_msa_paths.py \
        --root /path/to/set_root
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from confidock.protenix.msa import inject_msa_paths, stage_msa_dirs  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_MSA_DIR = Path("/data/sabdab/msas")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Set root holding protenix_inputs/")
    parser.add_argument("--in-dir", type=Path, default=None, help="Default: <root>/protenix_inputs")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <root>/protenix_inputs_msa")
    parser.add_argument("--staging-dir", type=Path, default=None, help="Default: <root>/protenix_msa_staging")
    parser.add_argument("--msa-dir", type=Path, default=DEFAULT_MSA_DIR, help="Flat <sha16>.a3m directory")
    parser.add_argument(
        "--copy-a3m",
        action="store_true",
        help="Copy each .a3m into the staging tree instead of symlinking, making it self-contained",
    )
    parser.add_argument(
        "--no-pairing",
        action="store_true",
        help=(
            "Reference only the unpaired MSA. Equivalent downstream -- the staged pairing.a3m is a "
            "query-only dummy and RawMsa.from_a3m seeds every alignment with the query anyway -- but "
            "it stops the JSON implying a pairing depth that does not exist"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output JSONs")
    return parser.parse_args()


def load_shards(in_dir: Path) -> dict[Path, list[dict]]:
    """Read every ``*.json`` in *in_dir* as a list of Protenix entries.

    Parameters
    ----------
    in_dir : Path
        Directory of shard JSONs

    Returns
    -------
    dict
        Mapping {shard_path: entries}
    """
    shards = sorted(in_dir.glob("*.json"))
    if not shards:
        sys.exit(f"No *.json found in {in_dir}")

    loaded: dict[Path, list[dict]] = {}
    for shard in shards:
        entries = json.loads(shard.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            sys.exit(f"{shard} is not a list of entries -- is this a Protenix input JSON?")
        loaded[shard] = entries
        LOGGER.info("%s: %d entries", shard.name, len(entries))
    return loaded


def materialise_symlinks(seq_to_dir: dict[str, Path]) -> int:
    """Replace symlinked ``non_pairing.a3m`` files with real copies.

    Parameters
    ----------
    seq_to_dir : dict
        Mapping {sequence: staged directory}

    Returns
    -------
    int
        Number of files copied
    """
    copied = 0
    for per_seq_dir in seq_to_dir.values():
        link = per_seq_dir / "non_pairing.a3m"
        if not link.is_symlink():
            continue
        source = link.resolve()
        tmp = per_seq_dir / "non_pairing.a3m.tmp"
        shutil.copyfile(source, tmp)
        tmp.replace(link)
        copied += 1
    return copied


def count_real_msas(seq_to_dir: dict[str, Path]) -> tuple[int, int]:
    """Split staged sequences into real-MSA vs query-only.

    A query-only ``non_pairing.a3m`` holds exactly two lines (``>query`` and the sequence),
    which is how a staged miss is told apart from a staged hit after the fact.

    Parameters
    ----------
    seq_to_dir : dict
        Mapping {sequence: staged directory}

    Returns
    -------
    tuple[int, int]
        (real, query_only)
    """
    real = 0
    query_only = 0
    for per_seq_dir in seq_to_dir.values():
        text = (per_seq_dir / "non_pairing.a3m").read_text(encoding="utf-8")
        if len(text.strip().splitlines()) > 2:  # noqa: PLR2004
            real += 1
        else:
            query_only += 1
    return real, query_only


def protein_chains(entries: list[dict]) -> list[tuple[str, dict]]:
    """Every ``(entry name, proteinChain)`` in one shard, in file order.

    Paired with the entry name because the only thing done with a bad chain is to report it, and "some
    shard has a dangling pairedMsaPath" is not enough to find the entry that carries it. Non-protein
    chains are absent: they have no sequence to align and Protenix asks for no MSA key on them.

    Parameters
    ----------
    entries : list of dict
        Entries re-read from one written shard

    Returns
    -------
    list[tuple[str, dict]]
        (entry name, proteinChain body) per protein chain
    """
    return [
        (str(entry.get("name")), chain["proteinChain"])
        for entry in entries
        for chain in entry["sequences"]
        if chain.get("proteinChain") is not None
    ]


def verify_chain(shard: str, entry_name: str, chain: dict, required_keys: tuple[str, ...]) -> None:
    """Check one protein chain carries every required MSA key and that each path exists.

    A missing key and a dangling path are reported differently because they mean different things: the
    first is this script failing to bake, the second is the staging tree not being where the JSON says.

    Parameters
    ----------
    shard : str
        Shard file name, for the messages
    entry_name : str
        Name of the entry holding this chain
    chain : dict
        A ``proteinChain`` body
    required_keys : tuple of str
        MSA path keys this chain must carry and resolve
    """
    for key in required_keys:
        path = chain.get(key)
        if path is None:
            sys.exit(f"{shard}: entry {entry_name} chain missing {key}")
        # The emptiness test is not redundant with ``exists()``: ``Path("")`` is ``Path(".")``, which
        # exists, so a blank path would verify clean here and then send Protenix to read a directory.
        if not path or not Path(path).exists():
            sys.exit(f"{shard}: entry {entry_name} has dangling {key}: {path!r}")


def verify_shard(shard: str, entries: list[dict], n_entries: int, required_keys: tuple[str, ...]) -> None:
    """Check one re-read shard: the entry count, then every protein chain in it.

    Both checks are fatal rather than warnings. This runs after the writes, and the point of baking the
    paths in is that Protenix can be handed the file on disk with ``--use_msa true`` and no ``--msa_dir``;
    a path that is not there is read by ``need_msa_search`` as a reason to launch a live MMseqs2 search,
    so the failure mode is not a crash but a run that quietly does something much slower and different.

    Parameters
    ----------
    shard : str
        Shard file name, for the messages
    entries : list of dict
        Entries re-read from the written shard
    n_entries : int
        Entry count the writer believes it wrote
    required_keys : tuple of str
        MSA path keys every protein chain must carry and resolve
    """
    if len(entries) != n_entries:
        sys.exit(f"{shard}: wrote {n_entries} entries but re-read {len(entries)}")
    for entry_name, chain in protein_chains(entries):
        verify_chain(shard, entry_name, chain, required_keys)


def verify(out_dir: Path, expected: dict[str, int], required_keys: tuple[str, ...]) -> None:
    """Re-read the written shards and assert every protein chain resolves to an existing MSA.

    Parameters
    ----------
    out_dir : Path
        Directory of written shard JSONs
    expected : dict
        Mapping {shard_name: entry_count} the writer believes it wrote
    required_keys : tuple of str
        MSA path keys every protein chain must carry and resolve
    """
    for name, n_entries in expected.items():
        entries = json.loads((out_dir / name).read_text(encoding="utf-8"))
        verify_shard(name, entries, n_entries, required_keys)
        LOGGER.info("verified %s: %d entries, all chains resolve", name, n_entries)


def main() -> None:
    """Stage MSAs once and write shard JSONs with the paths baked in."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    in_dir = args.in_dir or args.root / "protenix_inputs"
    out_dir = args.out_dir or args.root / "protenix_inputs_msa"
    staging_dir = args.staging_dir or args.root / "protenix_msa_staging"

    if not in_dir.is_dir():
        sys.exit(f"Input directory {in_dir} does not exist")
    if not args.msa_dir.is_dir():
        sys.exit(f"MSA directory {args.msa_dir} is not readable")

    shards = load_shards(in_dir)
    existing = [s.name for s in shards if (out_dir / s.name).exists()]
    if existing and not args.overwrite:
        sys.exit(f"{out_dir} already holds {len(existing)} of these shards; pass --overwrite to replace")

    # Stage across all shards at once so the per-sequence directories are shared, not duplicated.
    all_entries = [e for entries in shards.values() for e in entries]
    LOGGER.info("staging MSAs for %d entries into %s", len(all_entries), staging_dir)
    seq_to_dir = stage_msa_dirs(all_entries, args.msa_dir, staging_dir)

    if args.copy_a3m:
        LOGGER.info("materialising symlinks (--copy-a3m)")
        LOGGER.info("copied %d .a3m files into the staging tree", materialise_symlinks(seq_to_dir))

    real, query_only = count_real_msas(seq_to_dir)
    LOGGER.info(
        "staged %d unique sequences: %d with a real MSA, %d query-only",
        len(seq_to_dir),
        real,
        query_only,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for shard, entries in shards.items():
        baked = inject_msa_paths(entries, seq_to_dir, include_paired=not args.no_pairing)
        (out_dir / shard.name).write_text(json.dumps(baked, indent=2), encoding="utf-8")
        written[shard.name] = len(baked)
        LOGGER.info("wrote %s", out_dir / shard.name)

    required = ("unpairedMsaPath",) if args.no_pairing else ("pairedMsaPath", "unpairedMsaPath")
    verify(out_dir, written, required)
    LOGGER.info("done: %d shards in %s", len(written), out_dir)


if __name__ == "__main__":
    main()
