"""Site-specific filesystem locations, resolved from the environment.

ConfiDock reads large inputs (Boltz-2 weights, SAbDab snapshots, cached latents) from
paths that differ per installation. Nothing here is baked into the package: each
helper reads an environment variable and falls back to a per-user cache directory, so
a checkout never carries one site's layout.

Environment variables
---------------------
``CONFIDOCK_BOLTZ_CACHE``
    Directory holding the Boltz-2 weights and CCD files. Falls back to
    ``$XDG_CACHE_HOME/boltz`` (``~/.cache/boltz``).
``CONFIDOCK_SABDAB_JSON_DIR``
    ``os.pathsep``-separated directories of SAbDab sidecar JSONs, read by
    :mod:`confidock.dockq_predictor.data.chain_mapping_cif`. No fallback.
"""

import os
from pathlib import Path

__all__ = ["cache_root", "default_boltz_cache"]


def cache_root() -> Path:
    """Return the per-user cache root (``$XDG_CACHE_HOME``, default ``~/.cache``)."""
    env = os.environ.get("XDG_CACHE_HOME")
    return Path(env).expanduser() if env else Path.home() / ".cache"


def default_boltz_cache() -> Path:
    """Return the Boltz-2 cache directory, from ``CONFIDOCK_BOLTZ_CACHE`` or the user cache.

    Resolved at call time rather than import time so tests and callers can set the
    variable after the module is imported.
    """
    env = os.environ.get("CONFIDOCK_BOLTZ_CACHE")
    return Path(env).expanduser() if env else cache_root() / "boltz"
