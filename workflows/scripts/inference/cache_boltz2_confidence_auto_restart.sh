#!/bin/bash

# Auto-restart wrapper for the Boltz-2 confidence-input caching run.
#
# Launches one shard per GPU and restarts any shard that dies.  Restarting is
# cheap because cache_boltz2_confidence.py is resumable: an entry whose
# scores/<entry>.jsonl already holds all of its decoys is skipped, so a retry
# resumes at the entry that was interrupted rather than from the beginning.
#
# Usage:
#   workflows/scripts/inference/cache_boltz2_confidence_auto_restart.sh OUTPUT_DIR [GPUS] [EXTRA_ARGS...]
#
#   OUTPUT_DIR  where the cache is written (needs ~150 GiB for the full dataset,
#               so put it on the volume, not /data)
#   GPUS        comma-separated GPU ids, one shard each (default: all visible)
#
# Examples:
#   ... /data/confidock/sabdab_train/confidence_cache
#   ... /path/to/cache 0,1,2 --sources boltz2 protenix
#   ... /path/to/cache 0 --max_entries 20 --skip_entry_cache
#
# Logs land in OUTPUT_DIR/logs/shard<N>.log; a shard's history is appended, so
# grep for "Attempt" to see how many restarts it took.

set -uo pipefail  # no -e: we handle non-zero exits ourselves

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

OUTPUT_DIR="${1:?Missing OUTPUT_DIR}"
shift
GPUS="${1:-}"
# Treat a leading --flag as an extra arg rather than a GPU list.
if [[ -n "$GPUS" && "$GPUS" != --* ]]; then
    shift
else
    GPUS=""
fi
EXTRA_ARGS=("$@")

if [[ -z "$GPUS" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        GPUS="$CUDA_VISIBLE_DEVICES"
    else
        N_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
        GPUS="$(seq -s, 0 $((N_GPUS - 1)))"
    fi
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_SHARDS="${#GPU_LIST[@]}"

MAX_RETRIES="${MAX_RETRIES:-10}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Boltz-2 Confidence Cache with Auto-Restart"
echo "=========================================="
echo "Repo root:    $REPO_ROOT"
echo "Output dir:   $OUTPUT_DIR"
echo "GPUs:         ${GPU_LIST[*]}  ($NUM_SHARDS shards)"
echo "Extra args:   ${EXTRA_ARGS[*]:-<none>}"
echo "Max retries:  $MAX_RETRIES per shard"
echo "Logs:         $LOG_DIR/shard<N>.log"
echo "=========================================="

# Run one shard to completion, restarting on failure.
run_shard() {
    local shard_index="$1"
    local gpu="$2"
    local log="$LOG_DIR/shard${shard_index}.log"
    local retry=0

    while true; do
        retry=$((retry + 1))
        {
            echo ""
            echo "[$(date)] Shard $shard_index on GPU $gpu — attempt $retry of $MAX_RETRIES"
            echo "========================================"
        } >> "$log"

        cd "$REPO_ROOT" || return 1
        CUDA_VISIBLE_DEVICES="$gpu" poetry run python \
            workflows/scripts/inference/cache_boltz2_confidence.py \
            --output_dir "$OUTPUT_DIR" \
            --shard_index "$shard_index" \
            --num_shards "$NUM_SHARDS" \
            "${EXTRA_ARGS[@]}" >> "$log" 2>&1

        local exit_code=$?

        if [[ $exit_code -eq 0 ]]; then
            echo "[$(date)] Shard $shard_index COMPLETED (exit 0)" >> "$log"
            echo "[$(date)] shard $shard_index done"
            return 0
        fi

        if [[ $exit_code -eq 137 ]]; then
            # 137 = 128 + 9 (SIGKILL), usually the host OOM killer
            echo "[$(date)] Shard $shard_index SIGKILL (exit 137) — likely OOM" >> "$log"
        else
            echo "[$(date)] Shard $shard_index FAILED (exit $exit_code)" >> "$log"
        fi

        if [[ $retry -ge $MAX_RETRIES ]]; then
            echo "[$(date)] Shard $shard_index gave up after $retry attempts" >> "$log"
            echo "[$(date)] shard $shard_index GAVE UP (exit $exit_code) — see $log"
            return 1
        fi

        local wait_time=$((10 * retry))  # backoff: 10s, 20s, 30s...
        echo "[$(date)] Waiting ${wait_time}s before retry..." >> "$log"
        sleep "$wait_time"
    done
}

PIDS=()
for i in "${!GPU_LIST[@]}"; do
    run_shard "$i" "${GPU_LIST[$i]}" &
    PIDS+=("$!")
    echo "[$(date)] launched shard $i on GPU ${GPU_LIST[$i]} (pid ${PIDS[-1]})"
    sleep 5  # stagger so shards do not all read the DockQ CSVs at once
done

# Kill every shard if this wrapper is interrupted, so no orphans keep a GPU busy.
trap 'echo "[$(date)] interrupted; stopping shards"; kill "${PIDS[@]}" 2>/dev/null; exit 130' INT TERM

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

echo "=========================================="
if [[ $FAILED -eq 0 ]]; then
    echo "[$(date)] All $NUM_SHARDS shards completed"
else
    echo "[$(date)] $FAILED of $NUM_SHARDS shards gave up — check $LOG_DIR"
fi
echo "Scored entries: $(ls "$OUTPUT_DIR/scores" 2>/dev/null | wc -l)"
echo "Cache size:     $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)"
echo "=========================================="

exit $(( FAILED > 0 ? 1 : 0 ))
