#!/bin/bash

# Fine-tune the full Boltz-2 confidence module + DockQ head on every GPU.
#
# One torchrun process per GPU; gradients are averaged by hand after backward
# (see confidock/boltz/finetune/distributed.py for why not DDP).
#
# Usage:
#   workflows/scripts/training/finetune_confidence_ddp.sh OUTPUT_DIR [NPROC] [EXTRA_ARGS...]
#
# Examples:
#   ... runs/confidence_ft_v1
#   ... runs/confidence_ft_v1 6 --epochs 3 --confidence_lr 5e-6
#   ... runs/probe_baseline 6 --freeze_confidence   # head-only control
#
# NICE keeps this behind interactive work; the box is shared.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

OUTPUT_DIR="${1:?Missing OUTPUT_DIR}"
shift
NPROC="${1:-}"
if [[ -n "$NPROC" && "$NPROC" != --* ]]; then
    shift
else
    NPROC="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
fi

NICE="${NICE:-10}"
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/train.log"

echo "=========================================="
echo "Confidence-module fine-tune ($NPROC GPUs)"
echo "=========================================="
echo "Output dir: $OUTPUT_DIR"
echo "Extra args: ${*:-<none>}"
echo "Log:        $LOG"
echo "=========================================="

cd "$REPO_ROOT" || exit 1
nice -n "$NICE" poetry run torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="${MASTER_PORT:-29517}" \
    workflows/scripts/training/finetune_confidence_ddp.py \
    --output_dir "$OUTPUT_DIR" \
    "$@" 2>&1 | tee -a "$LOG"
