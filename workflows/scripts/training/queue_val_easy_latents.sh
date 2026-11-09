#!/bin/bash

# Wait for the running confidence fine-tune to exit, then extract val_easy latents.
#
# Why queued rather than run now: extraction needs all 6 GPUs and the fine-tune
# holds 36-40 GiB on every one of them, so starting now would either OOM the
# extraction or make the retry ladder in the trainer fire and corrupt the run's
# comparability.  Waiting costs nothing -- extraction is ~15 min once the cards
# are free.
#
# Why val_easy latents are needed at all: the frozen-probe baselines in
# runs/dockq_probe_v1 were computed on a holdout of the *train* cache, whose decoy
# pool includes `perturbed` (median per-entry DockQ std 0.29, ORACLE 0.87).
# val_easy has no perturbed decoys (std 0.11, ORACLE 0.50) and shares zero entries
# with that holdout, so the probe's 0.638 per-entry Spearman is not comparable to
# the fine-tune's val_easy number.  Extracting val_easy latents puts both on the
# same rows.
#
# Usage:
#   setsid nohup workflows/scripts/training/queue_val_easy_latents.sh \
#       > runs/val_easy_latents_queue.log 2>&1 & disown
#
# Knobs (all optional): WAIT_PID to wait on a specific process instead of the one
# WRAPPER_PATTERN finds, POLL_SECONDS for the polling interval, GPUS for the cards to
# extract on, and MAX_USED_MIB / GPU_WAIT_POLLS for how free those cards must be
# before extraction starts.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

VAL_EASY_ROOT="/data/confidock/sabdab_val_easy"
CACHE_DIR="$VAL_EASY_ROOT/confidence_cache"
LATENT_DIR="$VAL_EASY_ROOT/dockq_latents"

GPUS="${GPUS:-0,1,2,3,4,5}"
POLL_SECONDS="${POLL_SECONDS:-120}"
# What counts as a free card.  Not 0: a card with a few hundred MiB of someone's
# context on it still has the 40+ GiB extraction needs.
MAX_USED_MIB="${MAX_USED_MIB:-4096}"
GPU_WAIT_POLLS="${GPU_WAIT_POLLS:-10}"

# The finetune_confidence_ddp.sh wrapper, not torchrun: killing the wrapper is how
# the run is stopped by hand, and waiting on it also covers a torchrun restart.
#
# Looked up rather than written down: a literal PID is stale the moment that run
# ends, and Linux recycles PIDs, so a hard-coded number can send this script off to
# wait on an unrelated process -- or, once the number is simply gone, make it skip
# the wait and start extraction into a fine-tune that still owns all six cards.  The
# pattern goes stale in the safe direction: no match means no wrapper is running.
WRAPPER_PATTERN="${WRAPPER_PATTERN:-finetune_confidence_ddp\.sh}"
WAIT_PID="${WAIT_PID:-$(pgrep -f "$WRAPPER_PATTERN" | head -n 1)}"

if [[ -z "$WAIT_PID" ]]; then
    echo "[$(date)] queue: nothing matches /$WRAPPER_PATTERN/; no run to wait for"
elif ! kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "[$(date)] queue: pid $WAIT_PID is already gone; starting immediately"
else
    echo "[$(date)] queue: waiting on pid $WAIT_PID (polling every ${POLL_SECONDS}s)"
    # Record what is being waited on: with a hand-passed WAIT_PID this is the only
    # thing that distinguishes a correct 8-hour wait from waiting on the wrong process.
    echo "[$(date)] queue: pid $WAIT_PID is: $(ps -o args= -p "$WAIT_PID" 2>/dev/null)"
    # kill -0 rather than `wait`: the fine-tune is not a child of this shell (it
    # was launched detached from a different session), so `wait` would return
    # instantly with "not a child of this shell".
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep "$POLL_SECONDS"
    done
    echo "[$(date)] queue: pid $WAIT_PID exited"
fi

# The trainer's own teardown frees GPU memory asynchronously; give the driver a
# moment before a 6-way model load competes for it.
echo "[$(date)] queue: settling for 60s, then checking the cards"
sleep 60

# Then check the cards rather than infer them from the wait.  This is the condition
# the queue exists for, and it is checkable directly, so it is checked directly: the
# process being waited on is a proxy that can be wrong (the wrong PID, a run started
# under torchrun without the wrapper, a second tenant), while "40 GiB is in use on
# GPU 3" is the thing that would OOM the extraction or drive the trainer's retry
# ladder and cost the run its comparability.
busy_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
        awk -F', ' -v want=",$GPUS," -v limit="$MAX_USED_MIB" \
            'index(want, "," $1 ",") && $2 > limit { printf "%s(%sMiB) ", $1, $2 }'
}

for ((attempt = 1; ; attempt++)); do
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    busy="$(busy_gpus)"
    [[ -z "$busy" ]] && break
    if ((attempt >= GPU_WAIT_POLLS)); then
        echo "[$(date)] queue: giving up after $attempt checks; still in use: $busy" >&2
        exit 3
    fi
    echo "[$(date)] queue: still in use: $busy; re-checking in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
done

echo "[$(date)] queue: launching extraction over $CACHE_DIR"
workflows/scripts/training/extract_dockq_latents_all_gpus.sh \
    "$LATENT_DIR" \
    "$GPUS" \
    --cache_dir "$CACHE_DIR"
status=$?

echo "[$(date)] queue: extraction exited $status"
echo "shards written: $(ls "$LATENT_DIR/latents" 2>/dev/null | wc -l) of 763 entries"
exit $status
