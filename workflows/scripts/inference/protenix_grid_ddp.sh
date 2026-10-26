#!/bin/bash
# Run run_protenix_grid.py with DDP (torchrun) enabled inside tmux
# This uses your existing grid script but with multi-GPU DDP acceleration

set -euo pipefail

# Get repository root (3 levels up from workflows/scripts/inference)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Configuration
INPUT_JSON="${1:-$REPO_ROOT/data/sabdab_val_hard/protenix_inputs/px_val_hard_inputs.json}"
OUTPUT_DIR="${2:-/data/confidock/sabdab_val_hard/protenix}"
CONDITIONS_FILE="${3:-$REPO_ROOT/configs/protenix_grid_conditions_P1P2_val.yaml}"

MSA_DIR="/data/sabdab/msas"
GRID_SCRIPT="$REPO_ROOT/workflows/scripts/inference/run_protenix_grid.py"

# DDP configuration
CUDA_DEVICES="0,1,2,3"  # Use all 4 GPUs
N_GPUS=4

# Tmux session name
TMUX_SESSION="protenix-grid-ddp"

echo "=========================================="
echo "Protenix Grid with DDP (tmux mode)"
echo "=========================================="
echo "Input: $INPUT_JSON"
echo "Output: $OUTPUT_DIR"
echo "Conditions: $CONDITIONS_FILE"
echo "GPUs: $CUDA_DEVICES ($N_GPUS total)"
echo "Tmux session: $TMUX_SESSION"
echo ""

# Count entries
if [ -f "$INPUT_JSON" ]; then
    n_entries=$(python3 -c "import json; print(len(json.load(open('$INPUT_JSON'))))")
    echo "Total entries: $n_entries"
    echo "Each GPU will process ~$((n_entries / N_GPUS)) entries"
else
    echo "ERROR: Input JSON not found: $INPUT_JSON"
    exit 1
fi

# Check if tmux session already exists
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo ""
    echo "ERROR: Tmux session '$TMUX_SESSION' already exists!"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $TMUX_SESSION"
    echo "  2. Kill existing session: tmux kill-session -t $TMUX_SESSION"
    echo "  3. Use a different session name"
    exit 1
fi

echo ""
echo "Creating tmux session '$TMUX_SESSION'..."
echo ""

# Create log file path
LOG_FILE="$OUTPUT_DIR/grid_run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

# Build the command using a safe, isolated tracking process with periodic logging
COMMAND="cd '$REPO_ROOT/protenix-inference' && \
export PYTHONPATH='$REPO_ROOT':\$PYTHONPATH && \
echo '=== PROFILER INITIALIZED ===' >> '$LOG_FILE' && \
poetry run python '$GRID_SCRIPT' \
    --input_jsons '$INPUT_JSON' \
    --msa_dir '$MSA_DIR' \
    --output_dir '$OUTPUT_DIR' \
    --conditions_file '$CONDITIONS_FILE' \
    --cuda_devices '$CUDA_DEVICES' \
    --num_workers 2 \
    --use_ddp \
    2>&1 | tee -a '$LOG_FILE' & \
PY_PID=\$!; \
PEAK=0; \
ITER=0; \
while kill -0 \$PY_PID 2>/dev/null; do \
  VAL=\$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0); \
  if [ \"\$VAL\" -gt \"\$PEAK\" ]; then PEAK=\$VAL; fi; \
  ITER=\$((ITER + 1)); \
  if [ \$((ITER % 60)) -eq 0 ]; then \
    HUMAN_PEAK=\$(numfmt --to=iec-i \$PEAK); \
    TIMESTAMP=\$(date '+%Y-%m-%d %H:%M:%S'); \
    echo \"[\$TIMESTAMP] [PROFILER] Current peak RAM: \$HUMAN_PEAK\" | tee -a '$LOG_FILE'; \
  fi; \
  sleep 0.5; \
done; \
HUMAN_PEAK=\$(numfmt --to=iec-i \$PEAK); \
echo \"[PROFILER] FINAL PEAK CONTAINER RAM CONSUMPTION: \$HUMAN_PEAK\" >> '$LOG_FILE' && \
echo '' && \
echo '==========================================' && \
echo 'Grid run completed!' && \
echo 'Results in: $OUTPUT_DIR' && \
echo 'Log: $LOG_FILE' && \
echo '==========================================' && \
echo '' && \
echo 'Press Enter or Ctrl+C to exit, or Ctrl+B D to detach...' && \
read"



# Create the tmux session with the command
tmux new-session -d -s "$TMUX_SESSION" bash -c "$COMMAND"

echo "✓ Tmux session '$TMUX_SESSION' created successfully!"
echo ""
echo "Commands:"
echo "  • Attach to session:  tmux attach -t $TMUX_SESSION"
echo "  • Detach from session: Ctrl+B, then D"
echo "  • Kill session:       tmux kill-session -t $TMUX_SESSION"
echo "  • List sessions:      tmux ls"
echo ""
echo "Output log: $LOG_FILE"
echo ""
echo "The grid is now running in the background."
echo "Attach to the session to see live progress!"
echo "=========================================="
