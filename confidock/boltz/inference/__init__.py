"""Boltz-2 inference utilities.

Sub-modules
-----------
runner
    Command building, output post-processing, and timing for Boltz-2 predict
    runs (used by ``workflows/scripts/inference/run_boltz2_predictions.py``).
inputs
    PDB parsing and Boltz-2 YAML input generation
    (used by ``workflows/scripts/processing/prepare_boltz2_inputs.py``).

Confidence-head input caching
-----------------------------
The four modules below exist so the confidence head can be trained without
re-running the trunk. The trunk never sees coordinates, so its ``s``/``z`` outputs
are a function of sequence + MSA alone and are shared by every decoy of an entry --
caching them once turns a ~1322 s 33-decoy entry into ~80 s. See
``docs/boltz_confidence_caching.md``.

cache_runner
    Split ``Boltz2.forward`` at the trunk/head boundary: run the trunk once, then
    the confidence head per decoy.
confidence_cache
    The on-disk format -- ``feats`` compaction, atomic writes, fp16 ``z``.
decoy_coords
    Map an arbitrary decoy structure onto Boltz's canonical atom order. ``x_pred``
    is positional, so a mis-mapped atom is silently wrong rather than an error.
decoy_sources
    Enumerate decoys and their DockQ labels across the prediction sources.
"""
