"""RCSB PDB REST/GraphQL utilities.

Provides helpers for fetching entry metadata from the RCSB PDB GraphQL API
and filtering collections of structure files (YAML / PDB / JSON) by their
deposition date.

Typical use::

    from confidock.rcsb import fetch_deposit_dates, filter_inputs_by_date

    dates = fetch_deposit_dates(["7ANQ", "8QF5"])
    kept, excluded, dates = filter_inputs_by_date(
        input_dir=Path("data/Unsal2026/boltz2_inputs"),
        output_dir=Path("data/Unsal2026/boltz2_inputs_post2023"),
        cutoff=date(2023, 6, 1),
    )
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
DEFAULT_EXTENSIONS: tuple[str, ...] = (".yaml", ".pdb", ".json")

_BATCH_SIZE = 100  # entry_ids per GraphQL request


# ---------------------------------------------------------------------------
# RCSB GraphQL helpers
# ---------------------------------------------------------------------------


def _graphql_query(entry_ids: list[str]) -> str:
    ids_literal = json.dumps(entry_ids)
    return f"""{{
  entries(entry_ids: {ids_literal}) {{
    rcsb_id
    rcsb_accession_info {{ deposit_date }}
  }}
}}"""


def _fetch_with_retry(req: urllib.request.Request, max_retries: int = 3) -> dict:
    """Execute HTTP request with exponential backoff retry logic.

    Returns
    -------
    dict
        Parsed JSON response.

    Raises
    ------
    urllib.error.URLError
        If all retry attempts fail.
    """
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                LOGGER.warning(
                    "RCSB request failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                LOGGER.error("RCSB request failed after %d attempts: %s", max_retries, exc)
                raise
    msg = "Unreachable code"
    raise RuntimeError(msg)


def _parse_deposit_date(raw_date: str | None) -> date | None:
    """Parse RCSB deposit_date string to a date object.

    Parameters
    ----------
    raw_date:
        ISO format date string (e.g. "2020-10-12T00:00:00.000+00:00").

    Returns
    -------
    date | None
        Parsed date in UTC, or None if input is None/invalid.
    """
    if not raw_date:
        return None
    # e.g. "2020-10-12T00:00:00.000+00:00"
    try:
        dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except ValueError:
        LOGGER.warning("Invalid date format: %s - treating as unknown", raw_date)
        return None
    return dt.astimezone(timezone.utc).date()


def _process_rcsb_entries(data: dict) -> dict[str, date | None]:
    """Extract deposit dates from RCSB GraphQL response.

    Returns
    -------
    dict
        Maps upper-cased PDB ID to deposit date (or None if missing).
    """
    result: dict[str, date | None] = {}
    for entry in data.get("data", {}).get("entries") or []:
        rcsb_id: str = entry["rcsb_id"]
        raw_date: str | None = (entry.get("rcsb_accession_info") or {}).get("deposit_date")
        parsed = _parse_deposit_date(raw_date)
        result[rcsb_id] = parsed
        if parsed is None:
            LOGGER.warning("No deposit date found for %s", rcsb_id)
    return result


def fetch_deposit_dates(pdb_ids: list[str]) -> dict[str, date | None]:
    """Return a mapping of upper-cased PDB ID → deposition date (or ``None`` if unknown).

    Queries the RCSB GraphQL API (``rcsb_accession_info.deposit_date``) in
    batches of up to *_BATCH_SIZE* IDs per request.

    Parameters
    ----------
    pdb_ids:
        Upper- or lower-cased PDB entry IDs (e.g. ``["7ANQ", "8qf5"]``).

    Returns
    -------
    dict
        Keys are upper-cased PDB IDs.  Values are :class:`datetime.date`
        objects, or ``None`` for entries that RCSB could not resolve.
    """
    result: dict[str, date | None] = {}

    for i in range(0, len(pdb_ids), _BATCH_SIZE):
        batch = [pid.upper() for pid in pdb_ids[i : i + _BATCH_SIZE]]
        LOGGER.info("Querying RCSB for %d entries (%d–%d)…", len(batch), i + 1, i + len(batch))

        payload = json.dumps({"query": _graphql_query(batch)}).encode()
        req = urllib.request.Request(
            RCSB_GRAPHQL_URL,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )

        data = _fetch_with_retry(req)
        batch_result = _process_rcsb_entries(data)
        result.update(batch_result)

    # Mark any IDs absent from the response
    for pid in pdb_ids:
        uid = pid.upper()
        if uid not in result:
            LOGGER.warning("No RCSB record returned for %s – will treat as unknown", uid)
            result[uid] = None

    return result


# ---------------------------------------------------------------------------
# Filename utilities
# ---------------------------------------------------------------------------


def pdb_id_from_stem(stem: str) -> str:
    """Return the upper-cased PDB ID from a filename stem.

    Assumes the PDB ID is the first ``_``-delimited segment, e.g.
    ``"7anq_B_A"`` → ``"7ANQ"``.
    """
    return stem.split("_")[0].upper()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _scan_input_files(input_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Scan directory for files matching the given extensions.

    Returns
    -------
    list[Path]
        Sorted list of files with matching extensions (case-insensitive).
    """
    exts = {e.lower() for e in extensions}
    return sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in exts)


def _extract_unique_pdb_ids(files: list[Path]) -> list[str]:
    """Extract unique PDB IDs from file stems.

    Returns
    -------
    list[str]
        Unique PDB IDs in order of first appearance.
    """
    seen: set[str] = set()
    unique_ids: list[str] = []
    for f in files:
        pid = pdb_id_from_stem(f.stem)
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)
    return unique_ids


def _should_keep_file(
    dep_date: date | None,
    cutoff: date,
    before: bool,
) -> bool:
    """Determine if a file should be kept based on its deposit date.

    Parameters
    ----------
    dep_date:
        Deposit date of the entry, or None if unknown.
    cutoff:
        Date threshold.
    before:
        If False, keep entries deposited after cutoff.
        If True, keep entries deposited on or before cutoff.

    Returns
    -------
    bool
        True if the file should be kept.
    """
    if dep_date is None:
        return False
    return dep_date <= cutoff if before else dep_date > cutoff


def _partition_files_by_date(
    files: list[Path],
    deposit_dates: dict[str, date | None],
    cutoff: date,
    before: bool,
) -> tuple[list[Path], list[Path]]:
    """Partition files into kept and excluded based on deposit dates.

    Returns
    -------
    tuple
        (kept, excluded) file lists.
    """
    kept: list[Path] = []
    excluded: list[Path] = []

    for f in files:
        pid = pdb_id_from_stem(f.stem)
        dep = deposit_dates.get(pid)
        if _should_keep_file(dep, cutoff, before):
            kept.append(f)
        else:
            excluded.append(f)

    return kept, excluded


def _copy_files_to_output(files: list[Path], output_dir: Path) -> None:
    """Copy files to output directory, creating it if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, output_dir / f.name)


def filter_inputs_by_date(
    input_dir: Path,
    output_dir: Path,
    cutoff: date,
    *,
    before: bool = False,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    dry_run: bool = False,
) -> tuple[list[Path], list[Path], dict[str, date | None]]:
    """Copy structure files filtered by deposition date to *output_dir*.

    Scans *input_dir* for files whose extension is in *extensions* and whose
    stem follows the ``{pdbid}_...`` naming convention.  Deposition dates are
    fetched from the RCSB GraphQL API in a single batched request.

    Parameters
    ----------
    input_dir:
        Source directory to scan.
    output_dir:
        Destination directory (created if needed, unless *dry_run*).
    cutoff:
        Date threshold for filtering.
    before:
        If ``False`` (default), keep entries deposited *strictly after* cutoff.
        If ``True``, keep entries deposited *on or before* cutoff.
    extensions:
        File extensions to consider (case-insensitive).
    dry_run:
        If ``True``, no files are copied and *output_dir* is not created.

    Returns
    -------
    tuple
        ``(kept, excluded, deposit_dates)`` where *kept* and *excluded* are
        lists of :class:`~pathlib.Path` objects from *input_dir*, and
        *deposit_dates* maps upper-cased PDB ID → :class:`~datetime.date`
        (or ``None``).
    """
    input_files = _scan_input_files(input_dir, extensions)
    if not input_files:
        LOGGER.error("No files with extensions %s found in %s", extensions, input_dir)
        return [], [], {}

    LOGGER.info("Found %d file(s) in %s", len(input_files), input_dir)

    unique_ids = _extract_unique_pdb_ids(input_files)
    LOGGER.info("Unique PDB IDs: %d", len(unique_ids))

    deposit_dates = fetch_deposit_dates(unique_ids)

    kept, excluded = _partition_files_by_date(input_files, deposit_dates, cutoff, before)

    LOGGER.info(
        "After filtering (cutoff %s): %d kept, %d excluded",
        cutoff,
        len(kept),
        len(excluded),
    )

    if not dry_run:
        _copy_files_to_output(kept, output_dir)
        LOGGER.info("Copied %d files to %s", len(kept), output_dir)

    return kept, excluded, deposit_dates
