#!/bin/bash
# Multi-GPU wrapper for ESMFold2 grid inference with tmux sessions
# Creates separate tmux session for each GPU - survives SSH disconnects

set -e

# Default values
NUM_GPUS=2
GPU_IDS="4,5"  # Comma-separated GPU IDs (e.g., "4,5"), empty = use 0,1,2,...
INPUT_JSON="data/sabdab_val_hard/esmfold2_inputs/esmfold2_val_hard_inputs.json"
MSA_DIR="/data/sabdab/msas/"
OUTPUT_DIR="/data/confidock/sabdab_val_hard/esmfold2"
CONDITIONS_FILE="configs/esmfold2_grid_conditions_E2E3_val.yaml"
SABDAB_TSV="/data/sabdab/summary.tsv"
TOTAL_INPUTS=0
SEED=42
SESSION_PREFIX="esmfold2_gpu"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --input_json)
            INPUT_JSON="$2"
            shift 2
            ;;
        --msa_dir)
            MSA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --conditions_file)
            CONDITIONS_FILE="$2"
            shift 2
            ;;
        --sabdab_tsv)
            SABDAB_TSV="$2"
            shift 2
            ;;
        --total_inputs)
            TOTAL_INPUTS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --session_prefix)
            SESSION_PREFIX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_JSON" ]] || [[ -z "$MSA_DIR" ]] || [[ -z "$OUTPUT_DIR" ]] || [[ -z "$CONDITIONS_FILE" ]]; then
    echo "Usage: $0 --input_json <path> --msa_dir <path> --output_dir <path> --conditions_file <path> [options]"
    echo ""
    echo "Required:"
    echo "  --input_json PATH          Input JSON file with all entries"
    echo "  --msa_dir PATH            MSA directory"
    echo "  --output_dir PATH         Output directory"
    echo "  --conditions_file PATH    Conditions YAML file"
    echo ""
    echo "Optional:"
    echo "  --num_gpus N              Number of GPUs to use (default: 4)"
    echo "  --gpu_ids IDS             Comma-separated GPU IDs (e.g., '4,5', default: 0,1,2,...)"
    echo "  --sabdab_tsv PATH         SAbDab summary TSV"
    echo "  --total_inputs N          Total number of inputs (0 = all, default: 0)"
    echo "  --seed N                  Random seed (default: 42)"
    echo "  --session_prefix NAME     Tmux session prefix (default: esmfold2_gpu)"
    exit 1
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "ERROR: tmux is not installed. Install with: sudo apt-get install tmux"
    exit 1
fi

echo "=================================================="
echo "ESMFold2 Multi-GPU Grid Runner (tmux)"
echo "=================================================="
echo "Input JSON:        $INPUT_JSON"
echo "MSA directory:     $MSA_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo "Conditions file:   $CONDITIONS_FILE"
echo "Number of GPUs:    $NUM_GPUS"
echo "GPU IDs:           ${GPU_IDS:-auto (0,1,2,...)}"
echo "Total inputs:      $TOTAL_INPUTS (0 = all)"
echo "Seed:              $SEED"
echo "Session prefix:    $SESSION_PREFIX"
echo "=================================================="

# Get total number of entries in JSON
TOTAL_ENTRIES=$(python3 -c "import json; data=json.load(open('$INPUT_JSON')); print(len(data))")
echo "Total entries in JSON: $TOTAL_ENTRIES"

# Determine how many entries to process
if [[ $TOTAL_INPUTS -eq 0 ]]; then
    INPUTS_TO_PROCESS=$TOTAL_ENTRIES
else
    INPUTS_TO_PROCESS=$TOTAL_INPUTS
fi

echo "Will process: $INPUTS_TO_PROCESS entries"

# Calculate entries per GPU
ENTRIES_PER_GPU=$((($INPUTS_TO_PROCESS + $NUM_GPUS - 1) / $NUM_GPUS))
echo "Entries per GPU: ~$ENTRIES_PER_GPU"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create temporary directory for split JSONs
TEMP_DIR=$(mktemp -d)
echo "Temporary directory: $TEMP_DIR"

# Split the JSON into chunks
echo ""
echo "Splitting JSON into $NUM_GPUS chunks..."
python3 << EOF
import json
import random
from pathlib import Path

# Load full JSON
with open('$INPUT_JSON') as f:
    all_entries = json.load(f)

# Sample if needed
if $INPUTS_TO_PROCESS < len(all_entries):
    random.seed($SEED)
    all_entries = random.sample(all_entries, $INPUTS_TO_PROCESS)

# Split into chunks
chunk_size = $ENTRIES_PER_GPU
for i in range($NUM_GPUS):
    start_idx = i * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_entries))

    if start_idx >= len(all_entries):
        break

    chunk = all_entries[start_idx:end_idx]

    chunk_file = Path('$TEMP_DIR') / f'chunk_gpu{i}.json'
    with open(chunk_file, 'w') as f:
        json.dump(chunk, f, indent=2)

    print(f"GPU {i}: {len(chunk)} entries -> {chunk_file}")
EOF

# Build base command
BASE_CMD="HF_HUB_OFFLINE=1 poetry run python workflows/scripts/inference/run_esmfold2_grid.py"
BASE_CMD="$BASE_CMD --msa_dir $MSA_DIR"
BASE_CMD="$BASE_CMD --output_dir $OUTPUT_DIR"
BASE_CMD="$BASE_CMD --conditions_file $CONDITIONS_FILE"
BASE_CMD="$BASE_CMD --seed $SEED"

if [[ -n "$SABDAB_TSV" ]]; then
    BASE_CMD="$BASE_CMD --sabdab_tsv $SABDAB_TSV"
fi

echo ""
echo "Launching $NUM_GPUS tmux sessions..."
echo ""

# Parse GPU IDs
if [[ -n "$GPU_IDS" ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
else
    # Default: use 0,1,2,... up to NUM_GPUS-1
    GPU_ARRAY=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_ARRAY+=($i)
    done
fi

# Verify we have enough GPU IDs
if [[ ${#GPU_ARRAY[@]} -lt $NUM_GPUS ]]; then
    echo "ERROR: Not enough GPU IDs specified (got ${#GPU_ARRAY[@]}, need $NUM_GPUS)"
    exit 1
fi

# Kill any existing sessions with same prefix
for i in $(seq 0 $((NUM_GPUS - 1))); do
    SESSION_NAME="${SESSION_PREFIX}_${i}"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Killing existing session: $SESSION_NAME"
        tmux kill-session -t "$SESSION_NAME"
    fi
done

# Launch tmux sessions
SESSIONS_CREATED=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    CHUNK_FILE="$TEMP_DIR/chunk_gpu${i}.json"

    if [[ ! -f "$CHUNK_FILE" ]]; then
        echo "Skipping GPU $i (no data)"
        continue
    fi

    GPU_ID=${GPU_ARRAY[$i]}
    SESSION_NAME="${SESSION_PREFIX}_${i}"
    LOG_FILE="$OUTPUT_DIR/gpu${i}.log"

    # Build full command for this GPU
    # Navigate to repo root (3 levels up from workflows/scripts/inference)
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    GPU_CMD="cd '$REPO_ROOT' && CUDA_VISIBLE_DEVICES=$GPU_ID $BASE_CMD --input_jsons $CHUNK_FILE --device cuda:0"

    # Create tmux session and run command
    tmux new-session -d -s "$SESSION_NAME" "$GPU_CMD 2>&1 | tee $LOG_FILE"

    SESSIONS_CREATED+=("$SESSION_NAME")
    echo "✓ Created session: $SESSION_NAME (Physical GPU $GPU_ID)"

    # Small delay to avoid startup race conditions
    sleep 1
done

echo ""
echo "=================================================="
echo "All tmux sessions launched!"
echo "=================================================="
echo ""
echo "Sessions created: ${SESSIONS_CREATED[@]}"
echo ""
echo " Monitor commands:"
echo ""
echo "  # List all sessions:"
echo "  tmux ls"
echo ""
echo "  # Attach to specific GPU:"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    SESSION_NAME="${SESSION_PREFIX}_${i}"
    if [[ " ${SESSIONS_CREATED[@]} " =~ " ${SESSION_NAME} " ]]; then
        GPU_ID=${GPU_ARRAY[$i]}
        echo "  tmux attach -t $SESSION_NAME  # Physical GPU $GPU_ID"
    fi
done
echo ""
echo "  # Detach from session: Ctrl+B, then D"
echo ""
echo "  # Watch GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "  # Tail log files:"
echo "  tail -f $OUTPUT_DIR/gpu*.log"
echo ""
echo "  # Kill all sessions:"
echo "  tmux kill-session -t ${SESSION_PREFIX}_0"
for i in $(seq 1 $((NUM_GPUS - 1))); do
    echo "  tmux kill-session -t ${SESSION_PREFIX}_${i}"
done
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo "📝 Log files: $OUTPUT_DIR/gpu*.log"
echo "🗑️  Temp files: $TEMP_DIR (clean up manually after completion)"
echo ""
echo "=================================================="
echo "Sessions are running in background and will survive SSH disconnect!"
echo "=================================================="
