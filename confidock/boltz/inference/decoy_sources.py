"""Enumerate decoy structures and their DockQ labels across all four sources.

Each source stores its predictions under a different directory layout and its
DockQ scores in a CSV with slightly different condition columns:

===========  =====================================================  ==============================
source       structure path                                          condition columns
===========  =====================================================  ==============================
boltz2       ``<cond>/boltz_results__staged_inputs/predictions/``     msa_mode, recycling_steps,
             ``<entry>/<model_file>``                                 step_scale
esmfold2     ``<cond>/<entry>/<model_file>``                          msa_mode, loops, steps
protenix     ``<cond>/<entry>/seed_<seed>/predictions/<model_file>``  msa_mode, cycle, step
perturbed    ``stratified/<entry>/<model_file>``                      criterion, regime
===========  =====================================================  ==============================

This module normalises all of them into :class:`DecoyRef` records carrying the
resolved path, the entry id, a stable unique id, and the DockQ label.

:func:`collect_decoys` walks those layouts and is specific to ``sabdab_train``.
:func:`collect_decoys_from_manifest` is the alternative for the validation sets,
which use ``<source>/`` rather than ``<source>_r1/``, have no ``perturbed`` source,
and already ship a CSV of resolved paths plus labels -- so there is nothing to
rediscover and no third set of resolvers to keep in sync.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = [
    "SOURCES",
    "DecoyRef",
    "SourceSpec",
    "collect_decoys",
    "collect_decoys_from_manifest",
]


@dataclass(frozen=True)
class DecoyRef:
    """One decoy structure with its label and provenance."""

    source: str
    entry: str
    path: Path
    #: Filesystem-safe id, unique across all sources.
    decoy_id: str
    dockq: float
    condition: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceSpec:
    """How to locate structures and labels for one prediction source."""

    name: str
    #: Path of the DockQ CSV, relative to the sabdab_train root.
    dockq_csv: Path
    #: Columns identifying the generation condition.
    condition_columns: tuple[str, ...]
    #: ``(root, row) -> path`` resolver for the structure file.
    resolve: Callable[[Path, Any], Path]
    #: Extra columns to carry into metadata when present.
    extra_columns: tuple[str, ...] = ()
    #: Columns that distinguish structures sharing a condition + ``model_file``
    #: and so must appear in ``decoy_id``.  Protenix puts the seed in the path
    #: (``seed_<seed>/``) but reuses ``sample_0.cif`` across seeds, so without
    #: this its ids collide and decoys overwrite each other.
    id_columns: tuple[str, ...] = ()


def _boltz2_path(root: Path, row: Any) -> Path:
    condition = f"msa={row.msa_mode}__recycling={row.recycling_steps}__scale={row.step_scale}"
    return root / "boltz2_r1" / condition / "boltz_results__staged_inputs" / "predictions" / row.entry / row.model_file


def _esmfold_path(root: Path, row: Any) -> Path:
    condition = f"msa={row.msa_mode}__loops={row.loops}__steps={row.steps}"
    return root / "esmfold2_r1" / condition / row.entry / row.model_file


def _protenix_path(root: Path, row: Any) -> Path:
    condition = f"msa={row.msa_mode}__cycle={row.cycle}__step={row.step}"
    return root / "protenix_r1" / condition / row.entry / f"seed_{row.seed}" / "predictions" / row.model_file


def _perturbed_path(root: Path, row: Any) -> Path:
    return root / "perturbed_structures" / "full_std_lt_0.10" / "stratified" / row.entry / row.model_file


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="boltz2",
        dockq_csv=Path("boltz2_r1/dockq/dockq_scores.csv"),
        condition_columns=("msa_mode", "recycling_steps", "step_scale"),
        resolve=_boltz2_path,
        extra_columns=("seed", "sample", "confidence_score", "ptm", "iptm"),
    ),
    SourceSpec(
        name="esmfold2",
        dockq_csv=Path("esmfold2_r1/dockq/dockq_scores.csv"),
        condition_columns=("msa_mode", "loops", "steps"),
        resolve=_esmfold_path,
        extra_columns=("seed", "sample", "plddt", "ptm", "iptm"),
    ),
    SourceSpec(
        name="protenix",
        dockq_csv=Path("protenix_r1/dockq/dockq_scores.csv"),
        condition_columns=("msa_mode", "cycle", "step"),
        resolve=_protenix_path,
        extra_columns=("seed", "sample", "plddt", "ptm", "iptm"),
        id_columns=("seed",),
    ),
    SourceSpec(
        name="perturbed",
        dockq_csv=Path("perturbed_structures/full_std_lt_0.10/perturbed/dockq/dockq_scores.csv"),
        condition_columns=("criterion", "regime"),
        resolve=_perturbed_path,
        extra_columns=("base_seed",),
    ),
)


def _slug(value: Any) -> str:
    return str(value).replace("/", "-").replace(" ", "").replace("=", "")


def collect_decoys(
    root: Path,
    *,
    sources: tuple[str, ...] | None = None,
    check_exists: bool = True,
) -> list[DecoyRef]:
    """Enumerate every decoy with a DockQ label under ``root``.

    Parameters
    ----------
    root
        The ``sabdab_train`` directory.
    sources
        Restrict to these source names; ``None`` means all four.
    check_exists
        Drop rows whose structure file is missing (and log how many).

    Returns
    -------
    list[DecoyRef]
        One entry per labelled decoy, sorted by ``(entry, source, decoy_id)`` so
        all decoys of an entry are contiguous -- the caching loop relies on that
        to run the trunk once per entry.
    """
    selected = [s for s in SOURCES if sources is None or s.name in sources]
    refs: list[DecoyRef] = []

    for spec in selected:
        csv_path = root / spec.dockq_csv
        if not csv_path.is_file():
            LOGGER.warning("%s: DockQ CSV not found at %s", spec.name, csv_path)
            continue

        frame = pd.read_csv(csv_path)
        missing = 0
        for row in frame.itertuples(index=False):
            path = spec.resolve(root, row)
            if check_exists and not path.is_file():
                missing += 1
                continue

            condition = "__".join(f"{col}={getattr(row, col)}" for col in spec.condition_columns)
            stem = Path(str(row.model_file)).stem
            id_parts = [spec.name, _slug(condition)]
            id_parts += [f"{col}{_slug(getattr(row, col))}" for col in spec.id_columns if hasattr(row, col)]
            id_parts.append(_slug(stem))
            decoy_id = "__".join(id_parts)
            metadata = {col: getattr(row, col) for col in spec.extra_columns if hasattr(row, col)}
            metadata["model_file"] = str(row.model_file)
            refs.append(
                DecoyRef(
                    source=spec.name,
                    entry=str(row.entry),
                    path=path,
                    decoy_id=decoy_id,
                    dockq=float(row.complex_dockq),
                    condition=condition,
                    metadata=metadata,
                )
            )

        LOGGER.info(
            "%s: %d labelled decoys (%d missing files)",
            spec.name,
            len(frame) - missing,
            missing,
        )

    # decoy_id is the on-disk filename, so a collision would silently overwrite
    # one decoy's coordinates with another's.  Fail loudly instead.
    seen: dict[tuple[str, str], Path] = {}
    for ref in refs:
        key = (ref.entry, ref.decoy_id)
        if key in seen:
            msg = f"duplicate decoy_id {ref.decoy_id!r} for entry {ref.entry}: {seen[key]} and {ref.path}"
            raise ValueError(msg)
        seen[key] = ref.path

    refs.sort(key=lambda r: (r.entry, r.source, r.decoy_id))
    return refs


def _manifest_decoy_id(row: Any, root: Path) -> str:
    """Build a unique, filesystem-safe id from a manifest row's resolved path.

    The manifest carries no ``seed`` column, but Protenix reuses ``sample_0.cif``
    across seeds and only the ``seed_<n>/`` path component distinguishes them.  So
    the id is derived from the path itself: everything below
    ``<root>/<source>/<condition>/``, minus the structural noise
    (``<entry>/``, ``predictions/``, ``boltz_results__staged_inputs/``) that is
    constant within a source and adds nothing.
    """
    path = Path(str(row.pdb_path))
    try:
        relative = path.relative_to(root / str(row.model_source))
        # parts[0] is the condition directory, already in the id via `condition`.
        tail = relative.parts[1:]
    except ValueError:
        # Path outside the expected root: fall back to the filename alone, which
        # the caller's collision check will catch if it is not enough.
        tail = (path.name,)

    noise = {str(row.entry), "predictions", "boltz_results__staged_inputs"}
    parts = [_slug(Path(p).stem) for p in tail if p not in noise]
    return "__".join([str(row.model_source), _slug(row.condition), *parts])


def collect_decoys_from_manifest(
    manifest_csv: Path,
    *,
    root: Path,
    sources: tuple[str, ...] | None = None,
    check_exists: bool = True,
) -> list[DecoyRef]:
    """Enumerate decoys from a prebuilt manifest CSV (the validation-set path).

    Parameters
    ----------
    manifest_csv
        CSV with at least ``entry``, ``pdb_path``, ``model_source``, ``condition``
        and ``dockq`` columns, e.g.
        ``data/dockq_predictor/val_easy_manifest.csv``.
    root
        Dataset root the paths sit under (``sabdab_val_easy``), used only to make
        ``pdb_path`` relative when building ids.
    sources
        Restrict to these ``model_source`` values; ``None`` means all present.
    check_exists
        Drop rows whose structure file is missing, and log how many.

    Returns
    -------
    list[DecoyRef]
        Sorted by ``(entry, source, decoy_id)`` so all decoys of an entry are
        contiguous -- the caching loop relies on that to run the trunk once per
        entry.
    """
    frame = pd.read_csv(manifest_csv)
    required = {"entry", "pdb_path", "model_source", "condition", "dockq"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        msg = f"{manifest_csv} is missing columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    if sources is not None:
        frame = frame[frame["model_source"].isin(sources)]

    # A NaN label cannot supervise anything; drop it here rather than letting it
    # reach the trainer as a silent NaN loss.
    labelled = frame["dockq"].notna()
    n_unlabelled = int((~labelled).sum())
    frame = frame[labelled]

    extra_columns = ("ptm", "iptm", "plddt", "fnat", "irmsd", "lrmsd", "pae_path")
    refs: list[DecoyRef] = []
    n_missing = 0
    for row in frame.itertuples(index=False):
        path = Path(str(row.pdb_path))
        if check_exists and not path.is_file():
            n_missing += 1
            continue
        metadata = {
            column: getattr(row, column)
            for column in extra_columns
            if hasattr(row, column) and pd.notna(getattr(row, column))
        }
        metadata["model_file"] = path.name
        refs.append(
            DecoyRef(
                source=str(row.model_source),
                entry=str(row.entry),
                path=path,
                decoy_id=_manifest_decoy_id(row, root),
                dockq=float(row.dockq),
                condition=str(row.condition),
                metadata=metadata,
            )
        )

    LOGGER.info(
        "%s: %d labelled decoys over %d entries (%d missing files, %d unlabelled)",
        manifest_csv.name,
        len(refs),
        len({r.entry for r in refs}),
        n_missing,
        n_unlabelled,
    )

    # decoy_id becomes the on-disk filename, so a collision silently overwrites
    # one decoy's coordinates with another's.  Fail loudly instead.
    seen: dict[tuple[str, str], Path] = {}
    for ref in refs:
        key = (ref.entry, ref.decoy_id)
        if key in seen:
            msg = f"duplicate decoy_id {ref.decoy_id!r} for entry {ref.entry}: {seen[key]} and {ref.path}"
            raise ValueError(msg)
        seen[key] = ref.path

    refs.sort(key=lambda r: (r.entry, r.source, r.decoy_id))
    return refs
