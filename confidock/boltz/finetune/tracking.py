"""Weights & Biases logging for the Boltz-2 DockQ head runs.

Deliberately thin, and deliberately optional.  A fine-tune is hours on six GPUs
and a probe sweep is dozens of runs in minutes; both are only interpretable if
the loss curves and the per-entry correlations end up somewhere comparable.  But
neither may *fail* because a tracker was unreachable, so every call here degrades
to a warning and the run continues writing its JSON reports as before.

Credentials come from ``WANDB_API_KEY`` in the environment, or from a gitignored
``wandb_env`` file at the repository root.  The file matters because detached jobs --
``torchrun`` under ``nohup``, or a batch-scheduler pod -- do not inherit an interactive
shell's environment.

``wandb`` is not a declared dependency: it arrives transitively via ``boltz``,
which pins 0.18.7.  Hence the import sits inside :func:`init_run` behind a broad
except -- if that transitive pin ever goes away the run should lose its charts,
not its training.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

__all__ = [
    "WANDB_ENV_PATH",
    "finish_run",
    "flatten_report",
    "init_run",
    "load_wandb_env",
    "log_metrics",
]

#: Repo root is four parents up from this file (confidock/boltz/finetune/).
WANDB_ENV_PATH = Path(__file__).resolve().parents[3] / "wandb_env"

#: Set once :func:`init_run` succeeds; the other helpers are no-ops until then.
_RUN: Any = None


def load_wandb_env(env_path: Path | None = None) -> None:
    """Load ``KEY=value`` credentials from ``env_path`` into ``os.environ``.

    Values here OVERRIDE anything already exported: a stale key left in a shell
    would otherwise win silently and file the run under the wrong team.  A missing
    file is not an error -- the ambient environment is then used as-is, which is
    how a Kubernetes secret would supply the key.

    ``None`` resolves :data:`WANDB_ENV_PATH` *at call time* rather than defaulting to
    it in the signature, where it would be frozen at import and a test's (or a
    caller's) reassignment of the module attribute would be silently ignored.
    """
    env_path = WANDB_ENV_PATH if env_path is None else env_path
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def init_run(
    project: str | None,
    *,
    name: str | None = None,
    group: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> bool:
    """Start a W&B run; returns whether logging is active.

    A falsy ``project`` disables tracking outright, which is what ``--no_wandb``
    and the unit tests rely on.  Call this on rank 0 only: six ranks logging the
    same averaged scalars would produce six runs, five of them redundant.
    """
    global _RUN
    if not project:
        LOGGER.info("W&B disabled")
        return False

    load_wandb_env()
    if not os.environ.get("WANDB_API_KEY"):
        LOGGER.warning("no WANDB_API_KEY (looked in env and %s); continuing without W&B", WANDB_ENV_PATH)
        return False

    try:
        import wandb

        _RUN = wandb.init(
            project=project,
            name=name,
            group=group,
            config=config or {},
            tags=tags or [],
            # The env file's WANDB_PROJECT is meant for the GNN ablations, so the
            # explicit `project` above has to win; entity is left to the env.
            entity=os.environ.get("WANDB_ENTITY") or None,
        )
    except Exception as exc:  # noqa: BLE001 - tracking must never fail a training run
        LOGGER.warning("W&B init failed (%s); continuing without it", exc)
        return False

    LOGGER.info("W&B run %s in project %s", getattr(_RUN, "name", "?"), project)
    return True


def log_metrics(metrics: dict[str, float], *, step: int | None = None) -> None:
    """Log scalars if a run is active; NaNs are dropped rather than sent.

    W&B renders a NaN as a gap in the chart, and the correlation report is full of
    them by design (a baseline that is constant within an entry has no per-entry
    Spearman).  Dropping them keeps the panel list free of all-empty charts.
    """
    if _RUN is None:
        return
    clean = {k: v for k, v in metrics.items() if not (isinstance(v, float) and math.isnan(v))}
    if not clean:
        return
    try:
        _RUN.log(clean, step=step)
    except Exception as exc:  # noqa: BLE001 - see init_run
        LOGGER.warning("W&B log failed (%s)", exc)


def finish_run() -> None:
    """Close the run, flushing anything still buffered."""
    global _RUN
    if _RUN is None:
        return
    try:
        _RUN.finish()
    except Exception as exc:  # noqa: BLE001 - see init_run
        LOGGER.warning("W&B finish failed (%s)", exc)
    _RUN = None


def flatten_report(report: dict[str, Any], prefix: str = "val") -> dict[str, float]:
    """Flatten a :func:`~.trainer.correlation_report` into ``prefix/section/metric``.

    The report nests three sections keyed by metric name -- ``global``,
    ``per_entry``, ``top1_mean_dockq`` -- and the Boltz baselines sit in them
    alongside ``dockq_pred``.  Keeping the baselines means every chart shows the
    number the fine-tune has to beat, on the same axes, without a second query.
    """
    out: dict[str, float] = {}
    for section in ("global", "per_entry", "top1_mean_dockq"):
        for metric, value in (report.get(section) or {}).items():
            out[f"{prefix}/{section}/{metric}"] = float(value)
    for scalar in ("n_decoys", "n_entries", "oracle_mean_dockq"):
        if scalar in report:
            out[f"{prefix}/{scalar}"] = float(report[scalar])
    return out
