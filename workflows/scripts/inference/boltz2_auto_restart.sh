#!/bin/bash

# Auto-restart wrapper for Boltz-2 grid runs
# Detects SIGKILL (exit 137) and automatic process crashes, reruns until completion

set +e  # Don't exit on error; we handle it

# Get repository root (3 levels up from workflows/scripts/inference)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

INPUT_DIR="${1:?Missing input_dir}"
OUTPUT_DIR="${2:?Missing output_dir}"
CACHE_DIR="${3:?Missing cache_dir}"
CONDITIONS_FILE="${4:?Missing conditions_file}"
SABDAB_TSV="${5:?Missing sabdab_tsv}"

MAX_RETRIES=5
RETRY_COUNT=0

echo "=========================================="
echo "Boltz-2 Grid Runner with Auto-Restart"
echo "=========================================="
echo "Input dir:    $INPUT_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo "Cache dir:    $CACHE_DIR"
echo "Conditions:   $CONDITIONS_FILE"
echo "SAbDab TSV:   $SABDAB_TSV"
echo "Max retries:  $MAX_RETRIES"
echo "=========================================="

while true; do
    RETRY_COUNT=$((RETRY_COUNT + 1))

    echo ""
    echo "[$(date)] Attempt $RETRY_COUNT of $MAX_RETRIES"
    echo "========================================"

    # Run the actual command from repo root
    cd "$REPO_ROOT"
    poetry run python workflows/scripts/inference/run_boltz2_grid.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --cache "$CACHE_DIR" \
        --conditions_file "$CONDITIONS_FILE" \
        --sabdab_tsv "$SABDAB_TSV"

    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[$(date)] COMPLETED SUCCESSFULLY (exit $EXIT_CODE)"
        exit 0
    elif [[ $EXIT_CODE -eq 137 ]]; then
        # 137 = 128 + 9 (SIGKILL)
        echo "[$(date)] !!  SIGKILL detected (exit $EXIT_CODE) — likely OOM"
        if [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
            echo "[$(date)] !!! Reached max retries. Giving up."
            exit 1
        fi
        WAIT_TIME=$((5 * RETRY_COUNT))  # Exponential backoff: 5s, 10s, 15s...
        echo "[$(date)] Waiting ${WAIT_TIME}s before retry..."
        sleep $WAIT_TIME
        continue
    elif [[ $EXIT_CODE -ne 0 ]]; then
        # Other failures
        echo "[$(date)] !!! FAILED (exit $EXIT_CODE)"
        if [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
            echo "[$(date)] Reached max retries. Giving up."
            exit 1
        fi
        WAIT_TIME=$((5 * RETRY_COUNT))
        echo "[$(date)] Waiting ${WAIT_TIME}s before retry..."
        sleep $WAIT_TIME
        continue
    fi
done
