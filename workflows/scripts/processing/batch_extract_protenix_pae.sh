#!/bin/bash
# Batch process all Protenix condition directories to extract PAE matrices

set -e

# Default values
PROTENIX_ROOT=""
DELETE_FULL_DATA=false
DRY_RUN=false
PARALLEL_JOBS=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --protenix_root)
            PROTENIX_ROOT="$2"
            shift 2
            ;;
        --delete_full_data)
            DELETE_FULL_DATA=true
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$PROTENIX_ROOT" ]]; then
    echo "Usage: $0 --protenix_root <path> [--delete_full_data] [--dry_run] [--parallel N]"
    echo ""
    echo "Options:"
    echo "  --protenix_root PATH      Root directory containing condition subdirectories"
    echo "  --delete_full_data        Delete full_data JSON files after extraction (saves space)"
    echo "  --dry_run                 Show what would be processed"
    echo "  --parallel N              Number of condition directories to process in parallel (default: 4)"
    echo ""
    echo "Example:"
    echo "  $0 --protenix_root /path/to/protenix_r1 --delete_full_data --parallel 8"
    exit 1
fi

echo "========================================================"
echo "Protenix PAE Extraction - Batch Processing"
echo "========================================================"
echo "Root directory: $PROTENIX_ROOT"
echo "Delete full_data: $DELETE_FULL_DATA"
echo "Dry run: $DRY_RUN"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "========================================================"

# Find all condition directories (those matching pattern msa=*__cycle=*__step=*)
CONDITION_DIRS=()
while IFS= read -r -d '' dir; do
    CONDITION_DIRS+=("$dir")
done < <(find "$PROTENIX_ROOT" -maxdepth 1 -type d -name "msa=*__cycle=*__step=*" -print0 | sort -z)

if [[ ${#CONDITION_DIRS[@]} -eq 0 ]]; then
    echo "No condition directories found in $PROTENIX_ROOT"
    exit 0
fi

echo "Found ${#CONDITION_DIRS[@]} condition directories"
echo ""

# Build command
CMD="poetry run python workflows/scripts/processing/extract_protenix_pae_matrices.py"
if [[ "$DELETE_FULL_DATA" = true ]]; then
    CMD="$CMD --delete_full_data"
fi
if [[ "$DRY_RUN" = true ]]; then
    CMD="$CMD --dry_run"
fi

# Process each condition directory
export CMD
export -f process_condition 2>/dev/null || true

process_condition() {
    local condition_dir=$1
    local condition_name=$(basename "$condition_dir")

    echo "=================================================="
    echo "Processing: $condition_name"
    echo "=================================================="

    $CMD --predictions_dir "$condition_dir"

    echo ""
}

# Process in parallel using xargs
printf '%s\0' "${CONDITION_DIRS[@]}" | xargs -0 -P "$PARALLEL_JOBS" -I {} bash -c 'process_condition "$@"' _ {}

echo ""
echo "========================================================"
echo "Batch processing complete!"
echo "========================================================"

# Calculate space savings
if [[ "$DRY_RUN" = false ]] && [[ "$DELETE_FULL_DATA" = false ]]; then
    echo ""
    echo "To delete the original full_data JSON files and save space, run:"
    echo "  find $PROTENIX_ROOT -name '*_full_data_*.json' -type f -delete"
    echo ""
    echo "Or re-run with --delete_full_data flag"
fi
