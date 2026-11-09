#!/bin/bash

# Extract pooled Boltz-2 confidence latents on every GPU, one shard each.
#
# Mirrors cache_boltz2_confidence_auto_restart.sh: shards are round-robin over
# entries, restarts are cheap because extract_dockq_latents.py skips any entry
# whose latents/<entry>.npz already exists.
#
# Usage:
#   workflows/scripts/training/extract_dockq_latents_all_gpus.sh LATENT_DIR [GPUS] [EXTRA_ARGS...]
#
#   LATENT_DIR  where the pooled latents are written (~300 MiB for the full
#               dataset -- three orders of magnitude smaller than the trunk cache)
#   GPUS        comma-separated GPU ids, one shard each (default: all visible)
#
# Examples:
#   ... /data/confidock/sabdab_train/dockq_latents
#   ... /path/to/latents 0,1,2 --forward_chunk 4
#
# Logs land in LATENT_DIR/logs/shard<N>.log.

set -uo pipefail  # no -e: we handle non-zero exits ourselves

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

LATENT_DIR="${1:?Missing LATENT_DIR}"
shift
GPUS="${1:-}"
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

MAX_RETRIES="${MAX_RETRIES:-5}"
# Each shard is one process doing synchronous GPU work with in-process data
# loading, so it needs ~1 core.  NICE keeps it behind interactive work; the
# graph-building job in the other window shares this box.
NICE="${NICE:-10}"
LOG_DIR="$LATENT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "DockQ latent extraction ($NUM_SHARDS shards)"
echo "=========================================="
echo "Repo root:   $REPO_ROOT"
echo "Latent dir:  $LATENT_DIR"
echo "GPUs:        ${GPU_LIST[*]}"
echo "Extra args:  ${EXTRA_ARGS[*]:-<none>}"
echo "Logs:        $LOG_DIR/shard<N>.log"
echo "=========================================="

run_shard() {
    local shard_index="$1"
    local gpu="$2"
    local log="$LOG_DIR/shard${shard_index}.log"
    local retry=0

    while true; do
        retry=$((retry + 1))
        {
            echo ""
            echo "[$(date)] Shard $shard_index on GPU $gpu - attempt $retry of $MAX_RETRIES"
            echo "========================================"
        } >> "$log"

        cd "$REPO_ROOT" || return 1
        CUDA_VISIBLE_DEVICES="$gpu" nice -n "$NICE" poetry run python -u \
            workflows/scripts/training/extract_dockq_latents.py \
            --latent_dir "$LATENT_DIR" \
            --shard_index "$shard_index" \
            --num_shards "$NUM_SHARDS" \
            "${EXTRA_ARGS[@]}" >> "$log" 2>&1

        local exit_code=$?

        if [[ $exit_code -eq 0 ]]; then
            echo "[$(date)] Shard $shard_index COMPLETED" >> "$log"
            echo "[$(date)] shard $shard_index done"
            return 0
        fi

        if [[ $exit_code -eq 137 ]]; then
            echo "[$(date)] Shard $shard_index SIGKILL (exit 137) - likely host OOM" >> "$log"
        else
            echo "[$(date)] Shard $shard_index FAILED (exit $exit_code)" >> "$log"
        fi

        if [[ $retry -ge $MAX_RETRIES ]]; then
            echo "[$(date)] Shard $shard_index gave up after $retry attempts" >> "$log"
            echo "[$(date)] shard $shard_index GAVE UP - see $log"
            return 1
        fi

        sleep $((10 * retry))
    done
}

PIDS=()
for i in "${!GPU_LIST[@]}"; do
    run_shard "$i" "${GPU_LIST[$i]}" &
    PIDS+=("$!")
    echo "[$(date)] launched shard $i on GPU ${GPU_LIST[$i]} (pid ${PIDS[-1]})"
    sleep 5  # stagger the checkpoint reads
done

trap 'echo "[$(date)] interrupted; stopping shards"; kill "${PIDS[@]}" 2>/dev/null; exit 130' INT TERM

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

echo "=========================================="
if [[ $FAILED -eq 0 ]]; then
    echo "[$(date)] All $NUM_SHARDS shards completed"
else
    echo "[$(date)] $FAILED of $NUM_SHARDS shards gave up - check $LOG_DIR"
fi
echo "Entries extracted: $(ls "$LATENT_DIR/latents" 2>/dev/null | wc -l)"
echo "Size:              $(du -sh "$LATENT_DIR" 2>/dev/null | cut -f1)"
