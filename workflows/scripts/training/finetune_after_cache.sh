#!/bin/bash

# Wait for the val_easy confidence cache to finish, verify it, then start the
# confidence-module fine-tune -- unattended.
#
# Why a gate rather than just chaining with `&&`: the caching wrapper exits 0 even
# when every entry failed (it reports per-shard give-ups but its own success is
# "all shards returned"), and a fine-tune launched against a half-written cache
# trains on a silently truncated eval set.  So this checks the artifacts on disk,
# not the exit code, and refuses to launch if they fall short.
#
# Usage:
#   workflows/scripts/training/finetune_after_cache.sh OUTPUT_DIR [EXTRA_ARGS...]
#
# Environment:
#   CACHE_DIR     val_easy cache to wait for  (default: the val_easy path below)
#   TRAIN_CACHE   training cache to fine-tune on (default: sabdab_train)
#   MIN_ENTRIES   refuse to launch below this many cached entries (default 750/763)
#   WAIT_PID      wrapper pid to wait on; default is discovered with pgrep
#   POLL          seconds between polls (default 60)
#   MAX_WAIT      give up waiting after this many seconds (default 21600 = 6h)
#   NPROC         GPUs for the fine-tune (default: all)
#
# Example:
#   nohup workflows/scripts/training/finetune_after_cache.sh \
#       runs/confidence_ft_v1 --epochs 3 > runs/confidence_ft_v1/chain.log 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

OUTPUT_DIR="${1:?Missing OUTPUT_DIR}"
shift
EXTRA_ARGS=("$@")

VOLUME="/data/confidock"
CACHE_DIR="${CACHE_DIR:-$VOLUME/sabdab_val_easy/confidence_cache}"
TRAIN_CACHE="${TRAIN_CACHE:-$VOLUME/sabdab_train/confidence_cache}"
MIN_ENTRIES="${MIN_ENTRIES:-750}"
POLL="${POLL:-60}"
MAX_WAIT="${MAX_WAIT:-21600}"

mkdir -p "$OUTPUT_DIR"
STATUS="$OUTPUT_DIR/chain_status.json"

say() { echo "[$(date '+%F %T')] $*"; }

write_status() {
    # A machine-readable breadcrumb, so a later check does not have to parse the log.
    cat > "$STATUS" <<EOF
{
  "state": "$1",
  "detail": "$2",
  "cached_entries": ${3:-0},
  "expected_entries": ${4:-0},
  "updated": "$(date -Is)"
}
EOF
}

count_entries() { ls "$CACHE_DIR/entries" 2>/dev/null | wc -l; }

say "=========================================="
say "Chained fine-tune"
say "  wait for cache: $CACHE_DIR"
say "  train cache:    $TRAIN_CACHE"
say "  output dir:     $OUTPUT_DIR"
say "  extra args:     ${EXTRA_ARGS[*]:-<none>}"
say "=========================================="

# ---- how many entries should the finished cache hold? --------------------
# Taken from the manifest that drove the caching run, so the gate cannot drift
# out of step with the dataset.
MANIFEST="$REPO_ROOT/data/dockq_predictor/val_easy_manifest.csv"
EXPECTED="$(
    cd "$REPO_ROOT" && poetry run python - "$MANIFEST" <<'PY' 2>/dev/null
import sys

import pandas as pd

frame = pd.read_csv(sys.argv[1])
print(frame[frame["dockq"].notna()]["entry"].nunique())
PY
)"
[[ "$EXPECTED" =~ ^[0-9]+$ ]] || EXPECTED=763
say "expecting $EXPECTED entries (min to proceed: $MIN_ENTRIES)"

# ---- wait for the caching run to exit ------------------------------------
# `wait` only works on children, and the caching wrapper was started from another
# shell, so poll instead.  Matching on the python shards rather than the wrapper
# covers the case where the wrapper was killed but its shards are still running.
PATTERN="cache_boltz2_confidence.py"
WAITED=0
if [[ -n "${WAIT_PID:-}" ]]; then
    say "waiting on pid $WAIT_PID"
fi

while true; do
    if [[ -n "${WAIT_PID:-}" ]]; then
        kill -0 "$WAIT_PID" 2>/dev/null || { say "pid $WAIT_PID exited"; break; }
    elif ! pgrep -f "$PATTERN" > /dev/null 2>&1; then
        say "no '$PATTERN' processes left"
        break
    fi

    if (( WAITED >= MAX_WAIT )); then
        write_status timeout "still caching after ${MAX_WAIT}s" "$(count_entries)" "$EXPECTED"
        say "TIMEOUT after ${MAX_WAIT}s; not launching"
        exit 2
    fi

    if (( WAITED % 600 == 0 )); then
        say "still caching: $(count_entries)/$EXPECTED entries"
        write_status waiting "caching in progress" "$(count_entries)" "$EXPECTED"
    fi
    sleep "$POLL"
    WAITED=$((WAITED + POLL))
done

# The shards write their last scores file and then exit; give the filesystem a
# moment so the count below is not read mid-flush.
sleep 20

# ---- verify before committing 6 GPUs to it ------------------------------
N_ENTRIES="$(count_entries)"
N_SCORES="$(ls "$CACHE_DIR/scores" 2>/dev/null | wc -l)"
N_DECOYS="$(cat "$CACHE_DIR"/scores/*.jsonl 2>/dev/null | wc -l)"
N_FAIL=0
[[ -f "$CACHE_DIR/failures.jsonl" ]] && N_FAIL="$(wc -l < "$CACHE_DIR/failures.jsonl")"

say "cache: $N_ENTRIES entries, $N_SCORES score files, $N_DECOYS decoys, $N_FAIL failures"

if (( N_ENTRIES < MIN_ENTRIES )); then
    write_status insufficient "only $N_ENTRIES entries cached" "$N_ENTRIES" "$EXPECTED"
    say "REFUSING to launch: $N_ENTRIES < $MIN_ENTRIES entries."
    say "Inspect $CACHE_DIR/logs/ and rerun the cache, or lower MIN_ENTRIES."
    exit 1
fi

if [[ ! -d "$TRAIN_CACHE/entries" ]]; then
    write_status missing_train_cache "$TRAIN_CACHE/entries not found" "$N_ENTRIES" "$EXPECTED"
    say "REFUSING to launch: training cache $TRAIN_CACHE/entries not found"
    exit 1
fi

# ---- let the GPUs drain ------------------------------------------------
# Six caching shards each hold several GiB; torchrun loading Boltz-2 on all six
# ranks at once while that memory is still mapped is the obvious way to OOM at
# step zero.  Bounded, then launch anyway: the box is shared, so an unrelated
# job could hold a GPU indefinitely and blocking forever on it is worse than
# letting torchrun try.
DRAIN_ATTEMPTS="${DRAIN_ATTEMPTS:-30}"
for attempt in $(seq 1 "$DRAIN_ATTEMPTS"); do
    BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
    (( BUSY == 0 )) && break
    if (( attempt == DRAIN_ATTEMPTS )); then
        say "WARNING: $BUSY GPU process(es) still running; launching regardless"
        break
    fi
    say "waiting for $BUSY GPU process(es) to exit (attempt $attempt/$DRAIN_ATTEMPTS)"
    sleep 20
done

write_status launching "starting fine-tune" "$N_ENTRIES" "$EXPECTED"
say "launching fine-tune"

cd "$REPO_ROOT" || exit 1
workflows/scripts/training/finetune_confidence_ddp.sh \
    "$OUTPUT_DIR" \
    ${NPROC:+"$NPROC"} \
    --cache_dir "$TRAIN_CACHE" \
    --eval_cache_dir "$CACHE_DIR" \
    "${EXTRA_ARGS[@]}"
RC=$?

if (( RC == 0 )); then
    write_status done "fine-tune finished" "$N_ENTRIES" "$EXPECTED"
    say "fine-tune finished; predictions in $OUTPUT_DIR/predictions_*.csv"
else
    write_status failed "fine-tune exited $RC" "$N_ENTRIES" "$EXPECTED"
    say "fine-tune FAILED (exit $RC); see $OUTPUT_DIR/train.log"
fi
exit $RC
