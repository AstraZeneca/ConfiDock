r"""Build a non-redundant SAbDab training set with leak protection against validation.

This script implements the training-data workflow described in the project notes:

1. Coarse quality filtering (resolution/method/antigen/heavy-chain constraints)
2. Temporal split into train/validation candidate pools
3. Intra-training deduplication at interface level
4. Cross-set homology filtering (train vs validation)
5. Optional CDR-H3 leakage audit
6. Optional antigen-family stratification report

The script expects:

* A SAbDab summary TSV (with standard columns such as ``pdb``, ``date``,
  ``resolution``, ``method``, ``antigen_type``, ``Hchain``)
* A directory of interface JSON files where each file contains at least
  ``h_seq``, ``l_seq`` (optional), ``t_seq`` and optionally ``h_cdr``.

The current repository already contains such interface JSON files under
``data/sabdab_CROP_SIZE_600_*``.

Examples
--------
Build train/validation manifests only (no MMseqs2-dependent steps)::

    poetry run python workflows/scripts/processing/build_sabdab_training_set.py \\
        --summary_tsv /path/to/sabdab_summary.tsv \\
        --interface_json_dir /path/to/sabdab_interfaces \\
        --output_dir data/sabdab_trainset_pipeline \\
        --cutoff_date 2023-06-01 \\
        --skip_mmseqs

Full pipeline with MMseqs2 (recommended)::

    poetry run python workflows/scripts/processing/build_sabdab_training_set.py \\
        --summary_tsv /path/to/sabdab_summary.tsv \\
        --interface_json_dir /path/to/sabdab_interfaces \\
        --output_dir data/sabdab_trainset_pipeline \\
        --cutoff_date 2023-06-01
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# File pattern constants
PDB_FILE_PATTERN = "*.pdb"


@dataclass(frozen=True)
class PipelineThresholds:
    """Configurable thresholds for the training-set pipeline."""

    max_resolution_train: float = 4.0
    max_chain_count: int = 300
    min_antigen_len: int = 16
    train_ab_min_id: float = 0.9
    train_ag_min_id: float = 0.7
    cross_min_id: float = 0.6
    cross_ab_min_id: float = 1.0
    hard_cross_ab_min_id: float = 0.9
    cross_min_cov: float = 0.9
    cdr_h3_min_id: float = 0.9
    cdr_h3_min_cov: float = 0.8
    allow_peptide_antigens: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a non-redundant SAbDab training set with validation leak protection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary_tsv", type=Path, required=True, help="SAbDab summary TSV")
    parser.add_argument(
        "--interface_json_dir",
        type=Path,
        required=True,
        help="Directory containing interface JSON files with h_seq/l_seq/t_seq",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--cutoff_date",
        type=str,
        required=True,
        help="Temporal split cutoff (YYYY-MM-DD); <= cutoff => train candidate",
    )
    parser.add_argument(
        "--pdb_dir",
        type=Path,
        default=None,
        help=(
            "Optional PDB directory for chain-count and structural checks "
            "(defaults to --interface_json_dir if not specified)"
        ),
    )

    parser.add_argument("--max_resolution_train", type=float, default=4.0)
    parser.add_argument("--max_chain_count", type=int, default=300)
    parser.add_argument("--min_antigen_len", type=int, default=16)
    parser.add_argument(
        "--allow_peptide_antigens",
        action="store_true",
        help="Keep peptide antigens during coarse filtering.",
    )
    parser.add_argument("--train_ab_min_id", type=float, default=0.8)
    parser.add_argument("--train_ag_min_id", type=float, default=0.6)
    parser.add_argument(
        "--cross_min_id", type=float, default=0.4, help="Min sequence identity for cross-set antigen homology filtering"
    )
    parser.add_argument(
        "--cross_ab_min_id",
        type=float,
        default=1.0,
        help="Min sequence identity for cross-set antibody (H/L) homology filtering; "
        "default 1.0 keeps only exact matches",
    )
    parser.add_argument(
        "--hard_cross_ab_min_id",
        type=float,
        default=0.8,
        help="Min antibody (H/L) identity for the hard validation set (antigen + antibody filtering)",
    )
    parser.add_argument("--cross_min_cov", type=float, default=0.8)
    parser.add_argument("--cdr_h3_min_id", type=float, default=0.9)
    parser.add_argument("--cdr_h3_min_cov", type=float, default=0.8)

    parser.add_argument(
        "--keep_cluster_members_with_weights",
        action="store_true",
        help="Keep all members within each interface cluster and assign inverse-size weights",
    )
    parser.add_argument(
        "--run_cdr_h3_audit",
        action="store_true",
        help="Run optional CDR-H3 leakage audit train vs validation",
    )
    parser.add_argument(
        "--skip_mmseqs",
        action="store_true",
        help="Skip MMseqs2-dependent dedup/homology stages (for dry setup checks)",
    )
    parser.add_argument(
        "--mmseqs_bin",
        default=str(Path(__file__).resolve().parents[3] / "tools" / "mmseqs" / "bin" / "mmseqs"),
        help="MMseqs2 executable path (defaults to tools/mmseqs/bin/mmseqs installed by 'make init')",
    )
    parser.add_argument("--mmseqs_threads", type=int, default=8)
    parser.add_argument("--mmseqs_sensitivity", type=float, default=7.5)

    parser.add_argument(
        "--antigen_family_tsv",
        type=Path,
        default=None,
        help="Optional TSV/CSV with columns: pdb,family for antigen family audit",
    )

    args = parser.parse_args()

    # Default pdb_dir to interface_json_dir if not specified
    if args.pdb_dir is None:
        args.pdb_dir = args.interface_json_dir

    return args


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.upper() == "NA":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _parse_slash_date(date_str: str) -> date | None:
    """Parse M/D/YY format date (e.g. 10/15/25)."""
    parts = date_str.split("/")
    if len(parts) != 3:
        return None
    try:
        month = int(parts[0])
        day = int(parts[1])
        year = int(parts[2])
        if year < 100:
            year += 2000 if year <= 69 else 1900
        return date(year, month, day)
    except (ValueError, OverflowError):
        return None


def _parse_sabdab_date(value: str | None) -> date | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.upper() == "NA":
        return None

    # Most SAbDab exports use M/D/YY (e.g. 10/15/25), but some use ISO-like formats.
    if "/" in stripped:
        return _parse_slash_date(stripped)

    if "-" in stripped:
        try:
            return date.fromisoformat(stripped)
        except ValueError:
            return None

    return None


def _parse_summary_row(row: dict[str, str]) -> tuple[str, dict[str, object]] | None:
    """Parse a single SAbDab summary row into (pdb_id, metadata) tuple."""
    pdb = (row.get("pdb") or "").strip().lower()
    if not pdb:
        return None

    return pdb, {
        "date": _parse_sabdab_date(row.get("date")),
        "resolution": _safe_float(row.get("resolution")),
        "method": (row.get("method") or "").strip().upper(),
        "antigen_type": (row.get("antigen_type") or "").strip().lower(),
        "has_heavy": (row.get("Hchain") or "").strip().upper() not in {"", "NA"},
    }


def _aggregate_metadata(existing: dict[str, object], new: dict[str, object]) -> None:
    """Aggregate conservatively across multiple antibody rows per PDB (in-place)."""
    new_date = new.get("date")
    if new_date is not None and (existing["date"] is None or new_date < existing["date"]):
        existing["date"] = new_date

    new_res = new.get("resolution")
    if new_res is not None and (existing["resolution"] is None or new_res < existing["resolution"]):
        existing["resolution"] = new_res

    existing["method"] = " ".join(sorted({str(existing["method"]), str(new["method"])})).strip()
    existing["antigen_type"] = " ".join(sorted({str(existing["antigen_type"]), str(new["antigen_type"])})).strip()
    existing["has_heavy"] = bool(existing["has_heavy"]) or bool(new.get("has_heavy"))


def _load_summary_by_pdb(summary_tsv: Path) -> dict[str, dict[str, object]]:
    by_pdb: dict[str, dict[str, object]] = {}

    with summary_tsv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            parsed = _parse_summary_row(row)
            if parsed is None:
                continue

            pdb, metadata = parsed
            if pdb not in by_pdb:
                by_pdb[pdb] = metadata
            else:
                _aggregate_metadata(by_pdb[pdb], metadata)

    LOGGER.info("Loaded summary metadata for %d PDB entries", len(by_pdb))
    return by_pdb


def _extract_cdr_h3(h_seq: str, h_cdr: dict | None) -> str | None:
    if not h_seq or not h_cdr or "3" not in h_cdr:
        return None
    indices = h_cdr.get("3")
    if not isinstance(indices, list) or not indices:
        return None

    aa = []
    for i in indices:
        try:
            pos = int(i) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= pos < len(h_seq):
            aa.append(h_seq[pos])
    return "".join(aa) if aa else None


def _load_interfaces(interface_json_dir: Path, summary_by_pdb: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    interfaces: list[dict[str, object]] = []
    skipped_no_metadata = 0
    skipped_malformed = 0

    for path in sorted(interface_json_dir.glob("*.json")):
        stem = path.stem
        pdb = stem.split("_", maxsplit=1)[0].lower()
        meta = summary_by_pdb.get(pdb)
        if meta is None:
            skipped_no_metadata += 1
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Skipping malformed JSON: %s", path)
            skipped_malformed += 1
            continue

        h_seq = str(payload.get("h_seq") or "").strip().upper()
        l_seq = str(payload.get("l_seq") or "").strip().upper()
        t_seq = str(payload.get("t_seq") or "").strip().upper()
        h_cdr = payload.get("h_cdr") if isinstance(payload.get("h_cdr"), dict) else None

        interfaces.append(
            {
                "interface_id": stem,
                "pdb": pdb,
                "h_seq": h_seq,
                "l_seq": l_seq,
                "t_seq": t_seq,
                "cdr_h3": _extract_cdr_h3(h_seq, h_cdr),
                "date": meta.get("date"),
                "resolution": meta.get("resolution"),
                "method": meta.get("method"),
                "antigen_type": meta.get("antigen_type"),
                "has_heavy": bool(meta.get("has_heavy")),
                "source_json": str(path),
                "sample_weight": 1.0,
            }
        )

    LOGGER.info("Loaded %d interface JSON entries", len(interfaces))
    if skipped_no_metadata > 0:
        LOGGER.info("Skipped %d interfaces due to missing PDB metadata", skipped_no_metadata)
    if skipped_malformed > 0:
        LOGGER.info("Skipped %d interfaces due to malformed JSON", skipped_malformed)
    return interfaces


def _chain_count_from_pdb_file(pdb_path: Path) -> int | None:
    try:
        from Bio.PDB import PDBParser  # noqa: PLC0415
    except ImportError:
        return None
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("model", str(pdb_path))
        return len(list(structure[0].get_chains()))
    except Exception:
        return None


def _extract_pdb_id_from_filename(stem: str) -> str:
    """Extract PDB ID from filename stem (first underscore-delimited segment)."""
    return stem.split("_", maxsplit=1)[0].lower()


def _load_pdb_chain_counts(pdb_dir: Path | None) -> dict[str, int]:
    if pdb_dir is None or not pdb_dir.exists():
        return {}

    counts: dict[str, int] = {}
    for pdb_file in sorted(pdb_dir.glob(PDB_FILE_PATTERN)):
        pdb = _extract_pdb_id_from_filename(pdb_file.stem)
        if pdb in counts:
            continue
        count = _chain_count_from_pdb_file(pdb_file)
        if count is not None:
            counts[pdb] = count
    LOGGER.info("Loaded chain counts for %d PDB entries", len(counts))
    return counts


def _check_antigen_type(is_protein: bool, is_peptide: bool, allow_peptide: bool) -> str | None:
    """Check if antigen type is acceptable. Returns drop reason or None."""
    if allow_peptide and not (is_protein or is_peptide):
        return "non_protein_antigen"
    if not allow_peptide and not is_protein:
        return "non_protein_antigen"
    return None


def _check_antigen_length(t_seq: str, is_peptide: bool, thresholds: PipelineThresholds) -> str | None:
    """Check if antigen length meets threshold. Returns drop reason or None."""
    is_too_short = len(t_seq) < thresholds.min_antigen_len
    is_allowed_peptide = thresholds.allow_peptide_antigens and is_peptide
    if is_too_short and not is_allowed_peptide:
        return "peptide_antigen" if is_peptide else "short_antigen"
    return None


def _check_coarse_filter_reasons(
    rec: dict[str, object],
    chain_counts: dict[str, int],
    thresholds: PipelineThresholds,
) -> list[str]:
    """Return list of drop reasons for a single interface record."""
    reasons: list[str] = []
    resolution = rec.get("resolution")
    method = str(rec.get("method") or "").upper()
    antigen_type = str(rec.get("antigen_type") or "").lower()
    is_protein_antigen = "protein" in antigen_type
    is_peptide_antigen = "peptide" in antigen_type
    h_seq = str(rec.get("h_seq") or "")
    t_seq = str(rec.get("t_seq") or "")
    pdb = str(rec.get("pdb"))

    if not bool(rec.get("has_heavy")) or not h_seq:
        reasons.append("missing_heavy_chain")

    antigen_reason = _check_antigen_type(is_protein_antigen, is_peptide_antigen, thresholds.allow_peptide_antigens)
    if antigen_reason:
        reasons.append(antigen_reason)

    if "NMR" in method:
        reasons.append("nmr_structure")

    if resolution is None or float(resolution) > thresholds.max_resolution_train:
        reasons.append("resolution_cutoff")

    length_reason = _check_antigen_length(t_seq, is_peptide_antigen, thresholds)
    if length_reason:
        reasons.append(length_reason)

    chain_count = chain_counts.get(pdb)
    if chain_count is not None and chain_count > thresholds.max_chain_count:
        reasons.append("excessive_chain_count")

    return reasons


def _coarse_filter(
    interfaces: list[dict[str, object]],
    chain_counts: dict[str, int],
    thresholds: PipelineThresholds,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    kept: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    reason_counts: dict[str, int] = {}

    for rec in interfaces:
        reasons = _check_coarse_filter_reasons(rec, chain_counts, thresholds)

        if reasons:
            out = dict(rec)
            out["drop_reason"] = ";".join(reasons)
            dropped.append(out)
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        else:
            kept.append(rec)

    LOGGER.info("Coarse filter: kept=%d dropped=%d", len(kept), len(dropped))
    return kept, dropped, reason_counts


def _split_temporal(
    records: list[dict[str, object]], cutoff: date
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train, val, unknown = [], [], []
    for rec in records:
        dep = rec.get("date")
        if not isinstance(dep, date):
            unknown.append(rec)
        elif dep <= cutoff:
            train.append(rec)
        else:
            val.append(rec)
    LOGGER.info(
        "Temporal split: train_candidates=%d val_candidates=%d unknown_date=%d",
        len(train),
        len(val),
        len(unknown),
    )
    return train, val, unknown


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _to_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for name, seq in records:
            if not seq:
                continue
            fh.write(f">{name}\n{seq}\n")


def _run_mmseqs(command: list[str]) -> None:
    """Run MMseqs2 command with validated arguments.

    S603: subprocess call is safe - shell=False (implicit), command list
    is constructed from validated Path objects and numeric parameters only.
    """
    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)  # noqa: S603


def _cluster_mmseqs(
    fasta: Path,
    out_prefix: Path,
    min_id: float,
    cov: float,
    sensitivity: float,
    threads: int,
    mmseqs_bin: str,
) -> dict[str, str]:
    tmp = out_prefix.parent / f"tmp_{out_prefix.name}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _run_mmseqs(
        [
            mmseqs_bin,
            "easy-cluster",
            str(fasta),
            str(out_prefix),
            str(tmp),
            "--min-seq-id",
            str(min_id),
            "-c",
            str(cov),
            "--cov-mode",
            "0",
            "-s",
            str(sensitivity),
            "--threads",
            str(threads),
        ]
    )

    cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
    mapping: dict[str, str] = {}
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs cluster output not found: {cluster_tsv}")
    with cluster_tsv.open(encoding="utf-8") as fh:
        for line in fh:
            rep, member = line.strip().split("\t")
            mapping[member] = rep
    return mapping


def _search_mmseqs(
    query_fasta: Path,
    target_fasta: Path,
    out_tsv: Path,
    min_id: float,
    cov: float,
    sensitivity: float,
    threads: int,
    mmseqs_bin: str,
) -> set[str]:
    tmp = out_tsv.parent / f"tmp_{out_tsv.stem}"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    _run_mmseqs(
        [
            mmseqs_bin,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(out_tsv),
            str(tmp),
            "--min-seq-id",
            str(min_id),
            "-c",
            str(cov),
            "--cov-mode",
            "0",
            "-s",
            str(sensitivity),
            "--threads",
            str(threads),
        ]
    )

    hits: set[str] = set()
    if not out_tsv.exists():
        return hits
    with out_tsv.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            qid = line.split("\t", maxsplit=1)[0]
            hits.add(qid)
    return hits


def _fallback_exact_clusters(records: list[tuple[str, str]]) -> dict[str, str]:
    by_seq: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for name, seq in records:
        rep = by_seq.setdefault(seq, name)
        mapping[name] = rep
    return mapping


def _interface_chain_records(records: list[dict[str, object]], role: str) -> list[tuple[str, str]]:
    key = {"H": "h_seq", "L": "l_seq", "A": "t_seq", "CDR": "cdr_h3"}[role]
    out: list[tuple[str, str]] = []
    for rec in records:
        seq = str(rec.get(key) or "")
        if not seq:
            continue
        out.append((f"{rec['interface_id']}|{role}", seq))
    return out


def _cluster_chains(
    train_records: list[dict[str, object]],
    output_dir: Path,
    thresholds: PipelineThresholds,
    mmseqs_enabled: bool,
    mmseqs_bin: str,
    threads: int,
    sensitivity: float,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Cluster H, L, and A chains. Returns (h_cluster, l_cluster, a_cluster) mappings."""
    h_records = _interface_chain_records(train_records, "H")
    l_records = _interface_chain_records(train_records, "L")
    a_records = _interface_chain_records(train_records, "A")

    _to_fasta(h_records, output_dir / "mmseqs" / "train_heavy.fasta")
    _to_fasta(l_records, output_dir / "mmseqs" / "train_light.fasta")
    _to_fasta(a_records, output_dir / "mmseqs" / "train_antigen.fasta")

    if mmseqs_enabled:
        h_cluster = _cluster_mmseqs(
            output_dir / "mmseqs" / "train_heavy.fasta",
            output_dir / "mmseqs" / "heavy",
            thresholds.train_ab_min_id,
            thresholds.cross_min_cov,
            sensitivity,
            threads,
            mmseqs_bin,
        )
        l_cluster = _cluster_mmseqs(
            output_dir / "mmseqs" / "train_light.fasta",
            output_dir / "mmseqs" / "light",
            thresholds.train_ab_min_id,
            thresholds.cross_min_cov,
            sensitivity,
            threads,
            mmseqs_bin,
        )
        a_cluster = _cluster_mmseqs(
            output_dir / "mmseqs" / "train_antigen.fasta",
            output_dir / "mmseqs" / "antigen",
            thresholds.train_ag_min_id,
            thresholds.cross_min_cov,
            sensitivity,
            threads,
            mmseqs_bin,
        )
    else:
        LOGGER.warning("MMseqs2 disabled: falling back to exact-sequence clustering")
        h_cluster = _fallback_exact_clusters(h_records)
        l_cluster = _fallback_exact_clusters(l_records)
        a_cluster = _fallback_exact_clusters(a_records)

    return h_cluster, l_cluster, a_cluster


def _group_by_interface_cluster(
    train_records: list[dict[str, object]],
    h_cluster: dict[str, str],
    l_cluster: dict[str, str],
    a_cluster: dict[str, str],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Group records by interface cluster (H+L+A combination)."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    for rec in train_records:
        iid = str(rec["interface_id"])
        h_key = h_cluster.get(f"{iid}|H", f"{iid}|H")
        l_key = l_cluster.get(f"{iid}|L", f"{iid}|L") if rec.get("l_seq") else "NO_LIGHT"
        a_key = a_cluster.get(f"{iid}|A", f"{iid}|A")
        antibody_cluster = f"{h_key}::{l_key}"
        grouped[(antibody_cluster, a_key)].append(rec)

    return grouped


def _select_cluster_representatives(
    grouped: dict[tuple[str, str], list[dict[str, object]]],
    keep_members_with_weights: bool,
) -> list[dict[str, object]]:
    """Select representative(s) from each cluster."""
    deduped: list[dict[str, object]] = []

    for members in grouped.values():
        if keep_members_with_weights:
            weight = 1.0 / len(members)
            for m in members:
                out = dict(m)
                out["sample_weight"] = weight
                out["cluster_size"] = len(members)
                deduped.append(out)
            continue

        def _score_key(x: dict[str, object]) -> tuple[float, int]:
            res = x.get("resolution")
            res_num = float(res) if isinstance(res, (int, float)) else math.inf
            length = len(str(x.get("h_seq") or "")) + len(str(x.get("t_seq") or ""))
            return (res_num, -length)

        best = min(members, key=_score_key)
        out = dict(best)
        out["cluster_size"] = len(members)
        out["sample_weight"] = 1.0
        deduped.append(out)

    return deduped


def _apply_intra_train_dedup(
    train_records: list[dict[str, object]],
    thresholds: PipelineThresholds,
    output_dir: Path,
    mmseqs_enabled: bool,
    mmseqs_bin: str,
    threads: int,
    sensitivity: float,
    keep_members_with_weights: bool,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if not train_records:
        return [], {}

    h_cluster, l_cluster, a_cluster = _cluster_chains(
        train_records, output_dir, thresholds, mmseqs_enabled, mmseqs_bin, threads, sensitivity
    )

    cluster_stats = {
        "h_clusters": len(set(h_cluster.values())),
        "l_clusters": len(set(l_cluster.values())),
        "a_clusters": len(set(a_cluster.values())),
        "interface_clusters": 0,  # filled below
    }

    grouped = _group_by_interface_cluster(train_records, h_cluster, l_cluster, a_cluster)
    deduped = _select_cluster_representatives(grouped, keep_members_with_weights)

    cluster_stats["interface_clusters"] = len(grouped)

    LOGGER.info(
        "Intra-train dedup: input=%d output=%d",
        len(train_records),
        len(deduped),
    )
    return deduped, cluster_stats


def _find_cross_set_homology_hits(
    train_records: list[dict[str, object]],
    val_records: list[dict[str, object]],
    role: str,
    min_id: float,
    thresholds: PipelineThresholds,
    output_dir: Path,
    mmseqs_bin: str,
    threads: int,
    sensitivity: float,
    label: str,
) -> set[str]:
    """Search for homologous sequences between train and val for a given role."""
    q = _interface_chain_records(val_records, role)
    t = _interface_chain_records(train_records, role)
    if not q or not t:
        return set()

    suffix = f"_{label}" if label else ""
    q_fa = output_dir / "mmseqs" / f"cross_val_{role}{suffix}.fasta"
    t_fa = output_dir / "mmseqs" / f"cross_train_{role}{suffix}.fasta"
    out_tsv = output_dir / "mmseqs" / f"cross_{role}{suffix}.m8"

    _to_fasta(q, q_fa)
    _to_fasta(t, t_fa)

    hits = _search_mmseqs(
        q_fa,
        t_fa,
        out_tsv,
        min_id,
        thresholds.cross_min_cov,
        sensitivity,
        threads,
        mmseqs_bin,
    )

    return {h.split("|", maxsplit=1)[0] for h in hits}


def _partition_by_homology(
    val_records: list[dict[str, object]],
    dropped_ids: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Partition validation records into kept and dropped based on homology."""
    kept: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []

    for rec in val_records:
        if str(rec["interface_id"]) in dropped_ids:
            out = dict(rec)
            out["drop_reason"] = "cross_set_homology"
            dropped.append(out)
        else:
            kept.append(rec)

    return kept, dropped


def _apply_cross_set_filter(
    train_records: list[dict[str, object]],
    val_records: list[dict[str, object]],
    thresholds: PipelineThresholds,
    output_dir: Path,
    mmseqs_bin: str,
    threads: int,
    sensitivity: float,
    cross_ab_min_id_override: float | None = None,
    label: str = "",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Remove validation entries that are too similar to any training entry.

    Returns (val_kept, val_dropped). Training records are not modified.
    """
    if not train_records or not val_records:
        return val_records, []

    dropped_ids: set[str] = set()

    # Val sequences are queries; train sequences are targets.
    # Use separate identity thresholds: antigen uses cross_min_id (the main leakage
    # risk), antibody H/L uses cross_ab_min_id (default 1.0 = exact match only).
    ab_min_id = cross_ab_min_id_override if cross_ab_min_id_override is not None else thresholds.cross_ab_min_id
    role_min_id = {
        "H": ab_min_id,
        "L": ab_min_id,
        "A": thresholds.cross_min_id,
    }

    for role in ["H", "L", "A"]:
        min_id = role_min_id[role]
        if min_id > 1.0:
            LOGGER.info("Cross-set filtering for role=%s skipped (min_id=%.2f > 1.0)", role, min_id)
            continue

        hits = _find_cross_set_homology_hits(
            train_records, val_records, role, min_id, thresholds, output_dir, mmseqs_bin, threads, sensitivity, label
        )
        dropped_ids.update(hits)

    kept, dropped = _partition_by_homology(val_records, dropped_ids)

    LOGGER.info("Cross-set filtering: val dropped=%d val kept=%d", len(dropped), len(kept))
    return kept, dropped


def _apply_cdr_h3_audit(
    train_records: list[dict[str, object]],
    val_records: list[dict[str, object]],
    thresholds: PipelineThresholds,
    output_dir: Path,
    mmseqs_bin: str,
    threads: int,
    sensitivity: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    q = _interface_chain_records(train_records, "CDR")
    t = _interface_chain_records(val_records, "CDR")
    if not q or not t:
        LOGGER.info("CDR-H3 audit skipped: no CDR-H3 sequences available")
        return train_records, []

    q_fa = output_dir / "mmseqs" / "cdr_h3_train.fasta"
    t_fa = output_dir / "mmseqs" / "cdr_h3_val.fasta"
    out_tsv = output_dir / "mmseqs" / "cdr_h3_cross.m8"
    _to_fasta(q, q_fa)
    _to_fasta(t, t_fa)

    hits = _search_mmseqs(
        q_fa,
        t_fa,
        out_tsv,
        thresholds.cdr_h3_min_id,
        thresholds.cdr_h3_min_cov,
        sensitivity,
        threads,
        mmseqs_bin,
    )

    drop_ids = {h.split("|", maxsplit=1)[0] for h in hits}
    kept, dropped = [], []
    for rec in train_records:
        if str(rec["interface_id"]) in drop_ids:
            out = dict(rec)
            out["drop_reason"] = "cdr_h3_homology"
            dropped.append(out)
        else:
            kept.append(rec)
    LOGGER.info("CDR-H3 audit: dropped=%d kept=%d", len(dropped), len(kept))
    return kept, dropped


def _write_family_report(records: list[dict[str, object]], family_map: dict[str, str], out_path: Path) -> None:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        fam = family_map.get(str(rec["pdb"]))
        if fam:
            counts[fam] += 1
    rows = [{"family": fam, "count": cnt} for fam, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    _write_csv(out_path, rows)


def _print_threshold_summary(thresholds: PipelineThresholds, mmseqs_enabled: bool) -> None:
    print("\nThresholds used:")
    print("  [01 coarse filter]")
    print(f"    max_resolution_train : {thresholds.max_resolution_train}")
    print(f"    max_chain_count      : {thresholds.max_chain_count}")
    print(f"    min_antigen_len      : {thresholds.min_antigen_len}")
    print(f"    allow_peptide_antigens : {thresholds.allow_peptide_antigens}")
    if mmseqs_enabled:
        print("  [03 intra-train dedup]")
        print(f"    train_ab_min_id  : {thresholds.train_ab_min_id}  (H and L chain clustering)")
        print(f"    train_ag_min_id  : {thresholds.train_ag_min_id}  (antigen clustering)")
        print(f"    coverage         : {thresholds.cross_min_cov}")
        print("  [04 cross-set filtering]")
        ab_note = " (disabled — antigen-only easy val)" if thresholds.cross_ab_min_id > 1.0 else ""
        print(f"    cross_ab_min_id      : {thresholds.cross_ab_min_id}  (H/L easy val){ab_note}")
        print(f"    hard_cross_ab_min_id : {thresholds.hard_cross_ab_min_id}  (H/L hard val)")
        print(f"    cross_min_id         : {thresholds.cross_min_id}  (antigen; both val sets)")
        print(f"    coverage             : {thresholds.cross_min_cov}")
        print("  [05 CDR-H3 audit]")
        print(f"    cdr_h3_min_id    : {thresholds.cdr_h3_min_id}")
        print(f"    cdr_h3_min_cov   : {thresholds.cdr_h3_min_cov}")
    else:
        print("  [03-05 MMseqs2 steps skipped]")


def _print_pipeline_summary(stages: list[tuple[str, int, int | None]]) -> None:
    print("\nPipeline summary:")
    for label, kept, dropped in stages:
        if dropped is None:
            print(f"- {label}: {kept}")
        else:
            print(f"- {label}: kept={kept} dropped={dropped}")


def _load_family_map(path: Path) -> dict[str, str]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            pdb = (row.get("pdb") or "").strip().lower()
            family = (row.get("family") or "").strip()
            if pdb and family:
                mapping[pdb] = family
    return mapping


def _copy_pdbs_for_split(
    records: list[dict[str, object]],
    pdb_dir: Path | None,
    output_subdir: Path,
) -> int:
    """Copy PDB files for the given records into output_subdir.

    Returns the number of PDB files successfully copied.
    """
    if pdb_dir is None or not pdb_dir.exists():
        LOGGER.warning("PDB directory not provided or does not exist; skipping PDB copy")
        return 0

    output_subdir.mkdir(parents=True, exist_ok=True)

    # Avoid stale files from prior runs contaminating the current split.
    removed = 0
    for stale in output_subdir.glob(PDB_FILE_PATTERN):
        stale.unlink(missing_ok=True)
        removed += 1
    if removed:
        LOGGER.info("Removed %d stale PDB files from %s", removed, output_subdir)

    # Build a case-insensitive index by exact stem for strict interface_id matching.
    stem_to_path: dict[str, Path] = {}
    for pdb_file in sorted(pdb_dir.glob(PDB_FILE_PATTERN)):
        stem_to_path[pdb_file.stem.lower()] = pdb_file

    copied = 0
    missing = 0
    ids_seen: set[str] = set()

    for rec in records:
        interface_id = str(rec.get("interface_id", "")).lower()
        if not interface_id or interface_id in ids_seen:
            continue
        ids_seen.add(interface_id)

        src = stem_to_path.get(interface_id)
        if src is None:
            missing += 1
            continue

        dest = output_subdir / src.name
        shutil.copy2(src, dest)
        copied += 1

    LOGGER.info(
        "Copied %d PDB files to %s (%d interface IDs requested, %d missing)",
        copied,
        output_subdir,
        len(ids_seen),
        missing,
    )
    return copied


def _run_mmseqs_pipeline(
    train_candidates: list[dict[str, object]],
    val_candidates: list[dict[str, object]],
    thresholds: PipelineThresholds,
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, int],
]:
    """Run MMseqs2-dependent dedup and cross-set filtering stages.

    Returns: (train_dedup, val_easy_kept, val_easy_dropped, val_hard_kept,
              val_hard_dropped, train_final, train_cdr_dropped, intra_cluster_stats)
    """
    train_dedup, intra_cluster_stats = _apply_intra_train_dedup(
        train_candidates,
        thresholds,
        args.output_dir,
        mmseqs_enabled=True,
        mmseqs_bin=args.mmseqs_bin,
        threads=args.mmseqs_threads,
        sensitivity=args.mmseqs_sensitivity,
        keep_members_with_weights=args.keep_cluster_members_with_weights,
    )

    val_easy_kept, val_easy_dropped = _apply_cross_set_filter(
        train_dedup,
        val_candidates,
        thresholds,
        args.output_dir,
        args.mmseqs_bin,
        args.mmseqs_threads,
        args.mmseqs_sensitivity,
        label="easy",
    )

    val_hard_kept, val_hard_dropped = _apply_cross_set_filter(
        train_dedup,
        val_candidates,
        thresholds,
        args.output_dir,
        args.mmseqs_bin,
        args.mmseqs_threads,
        args.mmseqs_sensitivity,
        cross_ab_min_id_override=thresholds.hard_cross_ab_min_id,
        label="hard",
    )

    if args.run_cdr_h3_audit:
        train_final, train_cdr_dropped = _apply_cdr_h3_audit(
            train_dedup,
            val_easy_kept,
            thresholds,
            args.output_dir,
            args.mmseqs_bin,
            args.mmseqs_threads,
            args.mmseqs_sensitivity,
        )
    else:
        train_final = train_dedup
        train_cdr_dropped = []

    return (
        train_dedup,
        val_easy_kept,
        val_easy_dropped,
        val_hard_kept,
        val_hard_dropped,
        train_final,
        train_cdr_dropped,
        intra_cluster_stats,
    )


def _write_output_files(
    output_dir: Path,
    coarse_kept: list[dict[str, object]],
    coarse_dropped: list[dict[str, object]],
    train_candidates: list[dict[str, object]],
    val_candidates: list[dict[str, object]],
    unknown_date: list[dict[str, object]],
    train_dedup: list[dict[str, object]],
    val_easy_kept: list[dict[str, object]],
    val_easy_dropped: list[dict[str, object]],
    val_hard_dropped: list[dict[str, object]],
    train_cdr_dropped: list[dict[str, object]],
    train_final: list[dict[str, object]],
    val_hard_kept: list[dict[str, object]],
) -> None:
    """Write all CSV output files."""
    _write_csv(output_dir / "01_coarse_kept.csv", coarse_kept)
    _write_csv(output_dir / "01_coarse_dropped.csv", coarse_dropped)
    _write_csv(output_dir / "02_train_candidates.csv", train_candidates)
    _write_csv(output_dir / "02_val_candidates.csv", val_candidates)
    _write_csv(output_dir / "02_unknown_date.csv", unknown_date)
    _write_csv(output_dir / "03_train_dedup.csv", train_dedup)
    _write_csv(output_dir / "04_val_easy_dropped.csv", val_easy_dropped)
    _write_csv(output_dir / "04_val_hard_dropped.csv", val_hard_dropped)
    _write_csv(output_dir / "05_train_cdr_h3_dropped.csv", train_cdr_dropped)
    _write_csv(output_dir / "06_train_final.csv", train_final)
    _write_csv(output_dir / "06_validation_easy.csv", val_easy_kept)
    _write_csv(output_dir / "06_validation_hard.csv", val_hard_kept)


def _validate_no_leakage(
    train_final: list[dict[str, object]],
    val_easy_kept: list[dict[str, object]],
    val_hard_kept: list[dict[str, object]],
) -> None:
    """Verify no interface overlap between train and validation splits."""
    train_ids = {str(r.get("interface_id", "")) for r in train_final}
    val_easy_ids = {str(r.get("interface_id", "")) for r in val_easy_kept}
    val_hard_ids = {str(r.get("interface_id", "")) for r in val_hard_kept}
    overlap_easy = train_ids & val_easy_ids
    overlap_hard = train_ids & val_hard_ids
    if overlap_easy or overlap_hard:
        raise RuntimeError(
            f"Cross-set leakage detected: train∩val_easy={len(overlap_easy)} train∩val_hard={len(overlap_hard)}"
        )


def _write_run_summary(
    path: Path,
    args: argparse.Namespace,
    thresholds: PipelineThresholds,
    mmseqs_enabled: bool,
    pipeline_counts: dict[str, object],
    coarse_reason_counts: dict[str, int],
    cluster_stats: dict[str, int],
) -> None:
    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "thresholds": asdict(thresholds),
        "mmseqs_enabled": mmseqs_enabled,
        "pipeline_counts": pipeline_counts,
        "coarse_drop_reasons": coarse_reason_counts,
        "cluster_stats": cluster_stats if mmseqs_enabled else {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Run summary written to %s", path)


def _validate_mmseqs_availability(mmseqs_bin: str, skip_mmseqs: bool) -> bool:
    """Validate MMseqs2 availability and return whether it's enabled."""
    mmseqs_available = shutil.which(mmseqs_bin) is not None
    mmseqs_enabled = (not skip_mmseqs) and mmseqs_available

    if not skip_mmseqs and not mmseqs_available:
        raise SystemExit(f"MMseqs2 executable '{mmseqs_bin}' not found. Install MMseqs2 or rerun with --skip_mmseqs.")
    return mmseqs_enabled


def _copy_pdb_splits(
    train_final: list[dict[str, object]],
    val_easy_kept: list[dict[str, object]],
    val_hard_kept: list[dict[str, object]],
    pdb_dir: Path | None,
    output_dir: Path,
) -> None:
    """Copy PDB files for train, val_easy, and val_hard splits."""
    LOGGER.info("Copying PDB files for train/val_easy/val_hard splits...")
    _copy_pdbs_for_split(train_final, pdb_dir, output_dir / "pdbs" / "train")
    _copy_pdbs_for_split(val_easy_kept, pdb_dir, output_dir / "pdbs" / "val_easy")
    _copy_pdbs_for_split(val_hard_kept, pdb_dir, output_dir / "pdbs" / "val_hard")


def _write_family_reports(
    train_final: list[dict[str, object]],
    val_easy_kept: list[dict[str, object]],
    val_hard_kept: list[dict[str, object]],
    antigen_family_tsv: Path | None,
    output_dir: Path,
) -> None:
    """Write antigen family reports if the TSV is provided."""
    if antigen_family_tsv is not None and antigen_family_tsv.exists():
        fam_map = _load_family_map(antigen_family_tsv)
        _write_family_report(train_final, fam_map, output_dir / "07_train_family_report.csv")
        _write_family_report(val_easy_kept, fam_map, output_dir / "07_val_easy_family_report.csv")
        _write_family_report(val_hard_kept, fam_map, output_dir / "07_val_hard_family_report.csv")


def _print_detailed_breakdown(
    coarse_reason_counts: dict[str, int],
    coarse_kept_count: int,
    coarse_dropped_count: int,
    mmseqs_enabled: bool,
    intra_cluster_stats: dict[str, int],
    train_candidates_count: int,
    train_dedup_count: int,
    val_easy_kept_count: int,
    val_easy_dropped_count: int,
    val_hard_kept_count: int,
    val_hard_dropped_count: int,
) -> None:
    """Print detailed breakdown of pipeline stages."""
    print("\nDetailed breakdown:")
    print("  [01 coarse filter — drop reasons]")
    all_reasons = [
        "missing_heavy_chain",
        "non_protein_antigen",
        "nmr_structure",
        "resolution_cutoff",
        "peptide_antigen",
        "short_antigen",
        "excessive_chain_count",
    ]
    for r in all_reasons:
        n = coarse_reason_counts.get(r, 0)
        if n:
            print(f"    {r:<30}: {n}")
    print(f"    {'(total dropped)':<30}: {coarse_dropped_count}")
    print(f"    {'kept':<30}: {coarse_kept_count}")
    if mmseqs_enabled and intra_cluster_stats:
        print("  [03 intra-train dedup — clusters]")
        print(f"    {'heavy chain clusters':<35}: {intra_cluster_stats.get('h_clusters', 'n/a')}")
        print(f"    {'light chain clusters':<35}: {intra_cluster_stats.get('l_clusters', 'n/a')}")
        print(f"    {'antigen clusters':<35}: {intra_cluster_stats.get('a_clusters', 'n/a')}")
        print(f"    {'interface clusters (H+L+Ag)':<35}: {intra_cluster_stats.get('interface_clusters', 'n/a')}")
        print(f"    {'input sequences':<35}: {train_candidates_count}")
        print(f"    {'output (representatives)':<35}: {train_dedup_count}")
    if mmseqs_enabled:
        print("  [04 cross-set filtering]")
        print(f"    {'val easy kept (antigen only)':<35}: {val_easy_kept_count}")
        print(f"    {'val easy dropped':<35}: {val_easy_dropped_count}")
        print(f"    {'val hard kept (antigen+antibody)':<35}: {val_hard_kept_count}")
        print(f"    {'val hard dropped':<35}: {val_hard_dropped_count}")


def main() -> None:
    """Run the SAbDab trainset build pipeline end-to-end."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    cutoff = date.fromisoformat(args.cutoff_date)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pdb_dir is None:
        args.pdb_dir = args.interface_json_dir
        LOGGER.info("No --pdb_dir provided; defaulting to interface_json_dir: %s", args.pdb_dir)

    thresholds = PipelineThresholds(
        max_resolution_train=args.max_resolution_train,
        max_chain_count=args.max_chain_count,
        min_antigen_len=args.min_antigen_len,
        train_ab_min_id=args.train_ab_min_id,
        train_ag_min_id=args.train_ag_min_id,
        cross_min_id=args.cross_min_id,
        cross_ab_min_id=args.cross_ab_min_id,
        hard_cross_ab_min_id=args.hard_cross_ab_min_id,
        cross_min_cov=args.cross_min_cov,
        cdr_h3_min_id=args.cdr_h3_min_id,
        cdr_h3_min_cov=args.cdr_h3_min_cov,
        allow_peptide_antigens=args.allow_peptide_antigens,
    )

    mmseqs_enabled = _validate_mmseqs_availability(args.mmseqs_bin, args.skip_mmseqs)

    summary_by_pdb = _load_summary_by_pdb(args.summary_tsv)
    interfaces = _load_interfaces(args.interface_json_dir, summary_by_pdb)
    chain_counts = _load_pdb_chain_counts(args.pdb_dir)
    input_count = len(interfaces)

    coarse_kept, coarse_dropped, coarse_reason_counts = _coarse_filter(interfaces, chain_counts, thresholds)
    train_candidates, val_candidates, unknown_date = _split_temporal(coarse_kept, cutoff)

    coarse_kept_count = len(coarse_kept)
    coarse_dropped_count = len(coarse_dropped)
    train_candidates_count = len(train_candidates)
    val_candidates_count = len(val_candidates)
    unknown_date_count = len(unknown_date)

    if not train_candidates:
        raise SystemExit("No training candidates after filtering/splitting")

    if mmseqs_enabled:
        (
            train_dedup,
            val_easy_kept,
            val_easy_dropped,
            val_hard_kept,
            val_hard_dropped,
            train_final,
            train_cdr_dropped,
            intra_cluster_stats,
        ) = _run_mmseqs_pipeline(train_candidates, val_candidates, thresholds, args)

        train_dedup_count = len(train_dedup)
        train_dedup_dropped = train_candidates_count - train_dedup_count
        val_easy_kept_count = len(val_easy_kept)
        val_easy_dropped_count = len(val_easy_dropped)
        val_hard_kept_count = len(val_hard_kept)
        val_hard_dropped_count = len(val_hard_dropped)
        train_cdr_dropped_count = len(train_cdr_dropped)
    else:
        val_easy_kept = val_candidates
        val_hard_kept = val_candidates
        val_easy_dropped = []
        val_hard_dropped = []
        train_cdr_dropped = []
        train_final = train_candidates
        train_dedup = train_candidates
        train_dedup_count = len(train_candidates)
        train_dedup_dropped = 0
        val_easy_kept_count = len(val_easy_kept)
        val_easy_dropped_count = 0
        val_hard_kept_count = len(val_hard_kept)
        val_hard_dropped_count = 0
        train_cdr_dropped_count = 0
        intra_cluster_stats = {}

    train_dedup_output = train_dedup if mmseqs_enabled else train_candidates

    _write_output_files(
        args.output_dir,
        coarse_kept,
        coarse_dropped,
        train_candidates,
        val_candidates,
        unknown_date,
        train_dedup_output,
        val_easy_kept,
        val_easy_dropped,
        val_hard_dropped,
        train_cdr_dropped,
        train_final,
        val_hard_kept,
    )

    _validate_no_leakage(train_final, val_easy_kept, val_hard_kept)
    _copy_pdb_splits(train_final, val_easy_kept, val_hard_kept, args.pdb_dir, args.output_dir)
    _write_family_reports(train_final, val_easy_kept, val_hard_kept, args.antigen_family_tsv, args.output_dir)
    _print_threshold_summary(thresholds, mmseqs_enabled)

    _print_pipeline_summary(
        [
            ("Input interfaces", input_count, None),
            ("01 coarse filter", coarse_kept_count, coarse_dropped_count),
            ("02 temporal split train candidates", train_candidates_count, None),
            ("02 temporal split validation candidates", val_candidates_count, None),
            ("02 temporal split unknown-date", unknown_date_count, None),
            ("03 intra-train dedup", train_dedup_count, train_dedup_dropped),
            ("04 cross-set easy val (antigen only)", val_easy_kept_count, val_easy_dropped_count),
            ("04 cross-set hard val (antigen+antibody)", val_hard_kept_count, val_hard_dropped_count),
            ("05 CDR-H3 audit", len(train_final), train_cdr_dropped_count),
            ("06 final train", len(train_final), None),
            ("06 final validation (easy)", val_easy_kept_count, None),
            ("06 final validation (hard)", val_hard_kept_count, None),
        ]
    )

    _print_detailed_breakdown(
        coarse_reason_counts,
        coarse_kept_count,
        coarse_dropped_count,
        mmseqs_enabled,
        intra_cluster_stats,
        train_candidates_count,
        train_dedup_count,
        val_easy_kept_count,
        val_easy_dropped_count,
        val_hard_kept_count,
        val_hard_dropped_count,
    )

    _write_run_summary(
        args.output_dir / "run_summary.json",
        args,
        thresholds,
        mmseqs_enabled,
        {
            "input": input_count,
            "coarse_kept": coarse_kept_count,
            "coarse_dropped": coarse_dropped_count,
            "train_candidates": train_candidates_count,
            "val_candidates": val_candidates_count,
            "unknown_date": unknown_date_count,
            "train_dedup": train_dedup_count,
            "train_dedup_dropped": train_dedup_dropped,
            "val_easy_kept": val_easy_kept_count,
            "val_easy_dropped": val_easy_dropped_count,
            "val_hard_kept": val_hard_kept_count,
            "val_hard_dropped": val_hard_dropped_count,
            "train_cdr_dropped": train_cdr_dropped_count,
            "train_final": len(train_final),
        },
        coarse_reason_counts,
        intra_cluster_stats,
    )

    LOGGER.info(
        "Pipeline complete. train_final=%d val_easy=%d val_hard=%d output_dir=%s",
        len(train_final),
        len(val_easy_kept),
        len(val_hard_kept),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
