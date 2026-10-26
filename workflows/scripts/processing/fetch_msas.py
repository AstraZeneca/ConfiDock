r"""Pre-fetch MSAs for all Boltz-2/Protenix YAML inputs and patch the files in-place.

Uses boltz's bundled ColabFold API client (``boltz.data.msa.mmseqs2``) so no
separate colabfold installation is needed.

Sequences are deduplicated before submission: identical sequences across
different entries share a single cached ``.a3m`` file.  Chains whose YAML
``msa`` field is already set to a file path are skipped.

After fetching, each YAML's ``msa: empty`` fields are replaced with the
absolute path to the corresponding ``.a3m`` file so boltz can load them
directly without hitting the server again.

YAML input mode (patches YAMLs in-place)::

    python workflows/scripts/processing/fetch_msas.py \\
        --input_dir  data/sabdab_CROP_SIZE_600_postJune2023_boltz2_inputs \\
        --msa_dir    data/sabdab_msas \\
        --batch_size 64

JSON input mode (reads SAbDab JSON files directly; no YAML patching)::

    python workflows/scripts/processing/fetch_msas.py \\
        --json_dir  /path/to/sabdab_CROP_SIZE_600 \\
        --msa_dir   data/sabdab_msas \\
        --batch_size 64

The JSON mode reads ``h_seq`` (heavy), ``l_seq`` (light), and ``t_seq``
(antigen) from each ``.json`` file and fetches MSAs for all unique sequences
not already cached in ``--msa_dir``.  A manifest TSV mapping each sequence
hash to its ``.a3m`` path is written to ``--msa_dir/manifest.tsv``.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import yaml
from boltz.data.msa.mmseqs2 import run_mmseqs2

from confidock.msa import (
    a3m_path as _a3m_path,
)
from confidock.msa import (
    is_valid_a3m as _is_valid_a3m,
)
from confidock.msa import (
    resolve_from_local_npz as _resolve_from_local_npz,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_chains_from_json(json_dir: Path) -> list[dict]:
    """Return one record per protein chain from SAbDab CROP_SIZE_600 JSON files.

    Reads ``h_seq`` (heavy), ``l_seq`` (light), and ``t_seq`` (antigen/target)
    from each JSON.  All non-empty sequences are returned; deduplication
    happens downstream in :func:`_deduplicate`.
    """
    records: list[dict] = []
    for json_path in sorted(json_dir.glob("*.json")):
        pdb_id = json_path.stem[:4].lower()
        try:
            data = json.loads(json_path.read_text())
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not parse %s: %s – skipping", json_path.name, exc)
            continue
        for chain_type, field in [("H", "h_seq"), ("L", "l_seq"), ("A", "t_seq")]:
            seq = data.get(field) or ""
            if seq:
                records.append(
                    {
                        "yaml_path": None,  # no YAML to patch
                        "chain_id": chain_type,
                        "sequence": seq,
                        "pdb_id": pdb_id,
                    }
                )
    return records


def _has_valid_cached_msa(msa_field: str | object) -> bool:
    """Check if msa field points to a valid cached MSA file."""
    if msa_field == "empty":
        return False
    a3m_path = Path(str(msa_field))
    return a3m_path.exists() and a3m_path.stat().st_size >= 10  # noqa: PLR2004


def _extract_chains_from_yaml(yaml_path: Path) -> list[dict]:
    """Extract chain records from a single YAML file."""
    pdb_id = yaml_path.stem[:4].lower()
    try:
        data = yaml.safe_load(yaml_path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML root is not a mapping")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not parse %s: %s – skipping", yaml_path.name, exc)
        return []

    records: list[dict] = []
    for entry in data.get("sequences", []):
        p = entry.get("protein")
        if p is None:
            continue

        msa_field = p.get("msa", "empty")
        if _has_valid_cached_msa(msa_field):
            continue  # already has a valid cached MSA

        records.append(
            {
                "yaml_path": yaml_path,
                "chain_id": p["id"],
                "sequence": p["sequence"],
                "pdb_id": pdb_id,
            }
        )
    return records


def _collect_chains(input_dir: Path) -> list[dict]:
    """Return one record per protein chain across all YAML files.

    Each record::

        {
            "yaml_path": Path,
            "chain_id":  str,
            "sequence":  str,
            "pdb_id":    str,
        }

    Chains whose ``msa`` field is already a file path are omitted.
    """
    records: list[dict] = []
    for yaml_path in sorted(input_dir.glob("*.yaml")):
        records.extend(_extract_chains_from_yaml(yaml_path))
    return records


def _deduplicate(records: list[dict]) -> tuple[list[str], dict[str, list[dict]]]:
    """Return unique sequences and a map from sequence → list of records."""
    seq_to_records: dict[str, list[dict]] = {}
    for rec in records:
        seq_to_records.setdefault(rec["sequence"], []).append(rec)
    return list(seq_to_records.keys()), seq_to_records


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _needs_fetch(seq: str, msa_dir: Path) -> bool:
    """Check if a sequence needs to be fetched (no valid cached .a3m exists)."""
    p = _a3m_path(msa_dir, seq)
    return not p.exists() or p.stat().st_size < 10  # noqa: PLR2004


def _resolve_from_npz_if_available(
    unique_seqs: list[str],
    records: list[dict],
    msa_dir: Path,
    npz_msa_dir: Path | None,
) -> None:
    """Resolve sequences from local npz files if available."""
    if npz_msa_dir is None or not npz_msa_dir.exists():
        return

    seq_to_pdb_id = {rec["sequence"]: rec["pdb_id"] for rec in records if _needs_fetch(rec["sequence"], msa_dir)}
    resolved = _resolve_from_local_npz(
        [s for s in unique_seqs if _needs_fetch(s, msa_dir)],
        seq_to_pdb_id,
        npz_msa_dir,
        msa_dir,
    )
    LOGGER.info("Resolved %d sequences from local npz files", len(resolved))


def _fetch_batch_online(
    batch: list[str],
    batch_idx: int,
    tmpdir: str,
    msa_dir: Path,
    use_env: bool,
) -> int:
    """Fetch a single batch of sequences online. Returns count of fetched MSAs."""
    try:
        result = run_mmseqs2(
            batch,
            prefix=str(Path(tmpdir) / f"batch{batch_idx}"),
            use_env=use_env,
            use_filter=True,
            use_pairing=False,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Batch %d failed: %s – skipping", batch_idx + 1, exc)
        return 0

    fetched = 0
    for seq, a3m_text in zip(batch, result):
        if not _is_valid_a3m(a3m_text):
            LOGGER.warning("Skipping empty/invalid a3m for sequence %.20s…", seq)
            continue
        out = _a3m_path(msa_dir, seq)
        out.write_text(a3m_text)
        fetched += 1

    return fetched


def _try_patch_chain(protein_entry: dict, sequence: str, msa_dir: Path) -> bool:
    """Try to patch a single chain's MSA. Returns True if patched."""
    a3m = _a3m_path(msa_dir, sequence)
    if a3m.exists() and a3m.stat().st_size >= 10:  # noqa: PLR2004
        protein_entry["msa"] = str(a3m.resolve())
        return True
    return False


def _patch_yaml_file(
    yaml_path: Path,
    recs: list[dict],
    msa_dir: Path,
) -> int:
    """Patch a single YAML file with MSA paths. Returns number of chains patched."""
    data = yaml.safe_load(yaml_path.read_text()) or {}
    if not isinstance(data, dict):
        LOGGER.warning("YAML root is not a mapping in %s – skipping patching", yaml_path.name)
        return 0

    # Build lookup for O(1) chain_id access
    chain_id_to_rec = {rec["chain_id"]: rec for rec in recs}

    patched_chains = 0
    changed = False

    for entry in data.get("sequences", []):
        p = entry.get("protein")
        if p is None:
            continue

        chain_id = p["id"]
        if chain_id in chain_id_to_rec:
            if _try_patch_chain(p, p["sequence"], msa_dir):
                changed = True
                patched_chains += 1

    if changed:
        yaml_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, width=10_000))

    return patched_chains


def fetch_msas(
    input_dir: Path,
    msa_dir: Path,
    batch_size: int = 64,
    use_env: bool = True,
    patch_yamls: bool = True,
    npz_msa_dir: Path | None = None,
) -> None:
    """Fetch MSAs for sequences in YAML files and optionally patch them."""
    msa_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Scanning YAML files in %s …", input_dir)
    records = _collect_chains(input_dir)
    if not records:
        LOGGER.info("Nothing to fetch – all chains already have MSAs.")
        return

    unique_seqs, _ = _deduplicate(records)

    # ── Step 1: resolve from local npz files ────────────────────────────────────
    _resolve_from_npz_if_available(unique_seqs, records, msa_dir, npz_msa_dir)

    to_fetch = [s for s in unique_seqs if _needs_fetch(s, msa_dir)]
    LOGGER.info(
        "Found %d chains across %d unique sequences (will fetch %d MSAs online)",
        len(records),
        len(unique_seqs),
        len(to_fetch),
    )

    # ── Step 2: fetch remaining online ────────────────────────────────────────
    if to_fetch:
        batches = [to_fetch[i : i + batch_size] for i in range(0, len(to_fetch), batch_size)]
        LOGGER.info("Submitting %d sequences in %d batch(es) of ≤%d", len(to_fetch), len(batches), batch_size)

        with tempfile.TemporaryDirectory(prefix="boltz_msa_") as tmpdir:
            fetched = 0
            for batch_idx, batch in enumerate(batches):
                LOGGER.info("Batch %d/%d (%d sequences) …", batch_idx + 1, len(batches), len(batch))
                fetched += _fetch_batch_online(batch, batch_idx, tmpdir, msa_dir, use_env)

        LOGGER.info("Saved %d new .a3m files to %s", fetched, msa_dir)

    if not patch_yamls:
        return

    # ── Step 3: patch YAMLs in-place ──────────────────────────────────────────
    yaml_to_records: dict[Path, list[dict]] = {}
    for rec in records:
        yaml_to_records.setdefault(rec["yaml_path"], []).append(rec)

    patched_yamls = 0
    patched_chains = 0
    for yaml_path, recs in yaml_to_records.items():
        count = _patch_yaml_file(yaml_path, recs, msa_dir)
        if count > 0:
            patched_yamls += 1
            patched_chains += count

    LOGGER.info(
        "Patched %d chains across %d YAML files with local MSA paths",
        patched_chains,
        patched_yamls,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-fetch MSAs for Boltz-2 YAML inputs or SAbDab JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing Boltz-2 YAML input files (patches YAMLs in-place)",
    )
    src.add_argument(
        "--json_dir",
        type=Path,
        help="Directory containing SAbDab CROP_SIZE_600 JSON files (fetch-only, no patching)",
    )
    parser.add_argument(
        "--msa_dir",
        type=Path,
        required=True,
        help="Directory to write cached .a3m files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of sequences per API submission",
    )
    parser.add_argument(
        "--npz_msa_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of pre-computed Boltz MSA .npz files to "
            "resolve sequences locally before fetching online"
        ),
    )
    parser.add_argument(
        "--no_env",
        action="store_true",
        help="Disable environmental sequences (use_env=False)",
    )
    parser.add_argument(
        "--no_patch",
        action="store_true",
        help="Fetch and save .a3m files but do NOT patch the YAML files",
    )
    return parser.parse_args()


def _fetch_msas_from_json(
    json_dir: Path,
    msa_dir: Path,
    batch_size: int,
    use_env: bool,
    npz_msa_dir: Path | None,
) -> None:
    """JSON mode: read sequences from SAbDab JSON files and fetch MSAs."""
    if not json_dir.exists():
        raise SystemExit(f"JSON directory not found: {json_dir}")

    LOGGER.info("JSON mode: scanning %s …", json_dir)
    records = _collect_chains_from_json(json_dir)
    if not records:
        LOGGER.info("No sequences found – nothing to do.")
        return

    unique_seqs, _ = _deduplicate(records)
    msa_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: resolve from local npz ─────────────────────────────────────
    if npz_msa_dir is not None and npz_msa_dir.exists():
        seq_to_pdb_id = {
            rec["sequence"]: rec["pdb_id"] for rec in records if _needs_fetch_standalone(rec["sequence"], msa_dir)
        }
        resolved = _resolve_from_local_npz(
            [s for s in unique_seqs if _needs_fetch_standalone(s, msa_dir)],
            seq_to_pdb_id,
            npz_msa_dir,
            msa_dir,
        )
        LOGGER.info("Resolved %d sequences from local npz files", len(resolved))

    to_fetch = [s for s in unique_seqs if _needs_fetch_standalone(s, msa_dir)]
    LOGGER.info(
        "Found %d chains, %d unique sequences, %d need fetching online",
        len(records),
        len(unique_seqs),
        len(to_fetch),
    )

    # ── Step 2: fetch online ───────────────────────────────────────────────
    batches = [to_fetch[i : i + batch_size] for i in range(0, len(to_fetch), batch_size)]
    LOGGER.info("Submitting %d sequences in %d batch(es)", len(to_fetch), len(batches))

    fetched = 0
    with tempfile.TemporaryDirectory(prefix="boltz_msa_") as tmpdir:
        for batch_idx, batch in enumerate(batches):
            LOGGER.info("Batch %d/%d (%d sequences) …", batch_idx + 1, len(batches), len(batch))
            fetched += _fetch_batch_online(batch, batch_idx, tmpdir, msa_dir, use_env)

    LOGGER.info("Saved %d new .a3m files to %s", fetched, msa_dir)

    # ── Step 3: write manifest ─────────────────────────────────────────────
    _write_manifest(unique_seqs, msa_dir)


def _write_manifest(unique_seqs: list[str], msa_dir: Path) -> None:
    """Write manifest TSV mapping seq_hash to a3m_path."""
    manifest_path = msa_dir / "manifest.tsv"
    LOGGER.info("Writing manifest to %s …", manifest_path)
    with manifest_path.open("w") as fh:
        fh.write("seq_hash\ta3m_path\n")
        for seq in unique_seqs:
            p = _a3m_path(msa_dir, seq)
            if p.exists():
                fh.write(f"{p.stem}\t{p.resolve()}\n")
    LOGGER.info("Done.")


def main() -> None:
    """CLI entrypoint for fetching MSAs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    if args.json_dir is not None:
        _fetch_msas_from_json(
            json_dir=args.json_dir,
            msa_dir=args.msa_dir,
            batch_size=args.batch_size,
            use_env=not args.no_env,
            npz_msa_dir=args.npz_msa_dir,
        )
    else:
        # ── YAML mode: existing behaviour ─────────────────────────────────────
        fetch_msas(
            input_dir=args.input_dir,
            msa_dir=args.msa_dir,
            batch_size=args.batch_size,
            use_env=not args.no_env,
            patch_yamls=not args.no_patch,
            npz_msa_dir=args.npz_msa_dir,
        )


def _needs_fetch_standalone(seq: str, msa_dir: Path) -> bool:
    """Return True if *seq* does not yet have a valid cached .a3m in *msa_dir*."""
    p = _a3m_path(msa_dir, seq)
    return not p.exists() or p.stat().st_size < 10  # noqa: PLR2004


if __name__ == "__main__":
    main()
