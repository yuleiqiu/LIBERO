#!/bin/bash

# run_automation.sh
# Simple wrapper script for running the LIBERO automation pipeline

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TASK_ID=""
START_ID=""
END_ID=""
TASK_SUITE_NAME_MULTIPLE="libero_object"
TASK_SUITE_NAME_SINGLE="libero_object_single"
MAX_DEMOS=50
CAMERA_HEIGHT=128
CAMERA_WIDTH=128
MAX_WORKERS=1
DRY_RUN=false
BATCH_MODE=false

# Usage function
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run LIBERO automation pipeline for processing tasks."
    echo ""
    echo "Single task mode:"
    echo "  --task-id ID                 Process single task ID"
    echo ""
    echo "Batch mode:"
    echo "  --start-id ID                Starting task ID (inclusive)"
    echo "  --end-id ID                  Ending task ID (inclusive)"
    echo "  --max-workers N              Number of parallel workers (default: 1)"
    echo ""
    echo "Common options:"
    echo "  --task-suite-name-multiple NAME   Multiple task suite name (default: libero_object)"
    echo "  --task-suite-name-single NAME     Single task suite name (default: libero_object_single)"
    echo "  --max-demos N                Maximum demos per task (default: 50)"
    echo "  --camera-height N            Camera height (default: 128)"
    echo "  --camera-width N             Camera width (default: 128)"
    echo "  --dry-run                    Show what would be done without executing"
    echo "  --help                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --task-id 1                          # Process single task 1"
    echo "  $0 --start-id 0 --end-id 9              # Process tasks 0-9 sequentially"
    echo "  $0 --start-id 0 --end-id 9 --max-workers 3  # Process tasks 0-9 with 3 parallel workers"
    echo "  $0 --task-id 5 --dry-run               # Dry run for task 5"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-id)
            TASK_ID="$2"
            shift 2
            ;;
        --start-id)
            START_ID="$2"
            BATCH_MODE=true
            shift 2
            ;;
        --end-id)
            END_ID="$2"
            BATCH_MODE=true
            shift 2
            ;;
        --task-suite-name-multiple)
            TASK_SUITE_NAME_MULTIPLE="$2"
            shift 2
            ;;
        --task-suite-name-single)
            TASK_SUITE_NAME_SINGLE="$2"
            shift 2
            ;;
        --max-demos)
            MAX_DEMOS="$2"
            shift 2
            ;;
        --camera-height)
            CAMERA_HEIGHT="$2"
            shift 2
            ;;
        --camera-width)
            CAMERA_WIDTH="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ "$BATCH_MODE" == "true" ]]; then
    if [[ -z "$START_ID" ]] || [[ -z "$END_ID" ]]; then
        echo -e "${RED}Error: Both --start-id and --end-id are required for batch mode${NC}"
        exit 1
    fi
    if [[ "$START_ID" -gt "$END_ID" ]]; then
        echo -e "${RED}Error: start-id must be less than or equal to end-id${NC}"
        exit 1
    fi
elif [[ -z "$TASK_ID" ]]; then
    echo -e "${RED}Error: Either --task-id (single mode) or --start-id and --end-id (batch mode) are required${NC}"
    echo "Use --help for usage information."
    exit 1
fi

# Get script directory
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print configuration
echo -e "${BLUE}LIBERO Automation Pipeline${NC}"
echo "=========================="
if [[ "$BATCH_MODE" == "true" ]]; then
    echo -e "${BLUE}Mode:${NC} Batch (tasks $START_ID to $END_ID)"
    echo -e "${BLUE}Max workers:${NC} $MAX_WORKERS"
else
    echo -e "${BLUE}Mode:${NC} Single task $TASK_ID"
fi
echo -e "${BLUE}Task suite name for multiple:${NC} $TASK_SUITE_NAME_MULTIPLE"
echo -e "${BLUE}Task suite name for single:${NC} $TASK_SUITE_NAME_SINGLE"
echo -e "${BLUE}Max demos:${NC} $MAX_DEMOS"
echo -e "${BLUE}Camera size:${NC} ${CAMERA_WIDTH}x${CAMERA_HEIGHT}"
echo -e "${BLUE}Dry run:${NC} $DRY_RUN"
echo ""

# Build command
if [[ "$BATCH_MODE" == "true" ]]; then
    SCRIPT_NAME="batch_automation.py"
    CMD_ARGS=(
        "--start-id" "$START_ID"
        "--end-id" "$END_ID"
        "--max-workers" "$MAX_WORKERS"
    )
else
    SCRIPT_NAME="automation_pipeline.py"
    CMD_ARGS=(
        "--task-id" "$TASK_ID"
    )
fi

# Add common arguments
CMD_ARGS+=(
    "--task-suite-multiple" "$TASK_SUITE_NAME_MULTIPLE"
    "--task-suite-single" "$TASK_SUITE_NAME_SINGLE"
    "--max-demos" "$MAX_DEMOS"
    "--camera-height" "$CAMERA_HEIGHT"
    "--camera-width" "$CAMERA_WIDTH"
)

if [[ "$DRY_RUN" == "true" ]]; then
    CMD_ARGS+=("--dry-run")
fi

# Execute command
FULL_CMD="python3 \"$SCRIPT_DIR/$SCRIPT_NAME\" ${CMD_ARGS[*]}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN - Command that would be executed:${NC}"
    echo "$FULL_CMD"
    exit 0
fi

echo -e "${BLUE}Executing:${NC} $FULL_CMD"
echo ""

# Change to script directory and execute
cd "$SCRIPT_DIR"
python3 "$SCRIPT_NAME" "${CMD_ARGS[@]}"

# Capture exit code
exit_code=$?

echo ""
if [[ $exit_code -eq 0 ]]; then
    echo -e "${GREEN}Pipeline completed successfully!${NC}"
else
    echo -e "${RED}Pipeline failed with exit code $exit_code${NC}"
fi

exit $exit_code
