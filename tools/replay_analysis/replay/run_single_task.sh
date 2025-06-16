#!/bin/bash

# Single task runner script with additional options
# Usage: ./run_single_task.sh <task_id> [options]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <task_id> [--dry-run] [--verbose]"
    echo "Example: $0 1"
    echo "Example: $0 5 --dry-run"
    exit 1
fi

TASK_ID=$1
DRY_RUN=false
VERBOSE=false

# Parse additional arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Configuration
SCRIPT_PATH="$(dirname "$0")/replay_and_save_hdf5.py"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Running single task replay${NC}"
echo "Task ID: $TASK_ID"
echo "Script: $SCRIPT_PATH"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - Command that would be executed:${NC}"
    echo "python3 \"$SCRIPT_PATH\" --task-id $TASK_ID"
    exit 0
fi

# Build command
CMD="python3 \"$SCRIPT_PATH\" --task-id $TASK_ID"

if [ "$VERBOSE" = true ]; then
    echo -e "${BLUE}Executing: $CMD${NC}"
fi

# Execute
eval $CMD

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Task $TASK_ID completed successfully!${NC}"
else
    echo -e "${RED}Task $TASK_ID failed!${NC}"
    exit 1
fi
