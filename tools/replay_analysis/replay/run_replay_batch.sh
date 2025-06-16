#!/bin/bash

# Batch script to run replay_and_save_hdf5.py for multiple task IDs
# Usage: ./run_replay_batch.sh [start_id] [end_id]

# Default values
START_ID=${1:-1}   # Default start from 1
END_ID=${2:-9}     # Default end at 9

# Configuration
SCRIPT_PATH="$(dirname "$0")/replay_and_save_hdf5.py"
TASK_SUITE_MULTIPLE="libero_object"
TASK_SUITE_SINGLE="libero_object_single"
MAX_DEMOS=50

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting batch replay for task IDs ${START_ID} to ${END_ID}${NC}"
echo "Script: $SCRIPT_PATH"
echo "Task suite multiple: $TASK_SUITE_MULTIPLE"
echo "Task suite single: $TASK_SUITE_SINGLE"
echo "Max demos: $MAX_DEMOS"
echo "----------------------------------------"

# Track results
SUCCESS_COUNT=0
FAILED_TASKS=()

# Run for each task ID
for task_id in $(seq $START_ID $END_ID); do
    echo -e "${BLUE}[$(date)] Processing Task ID: $task_id${NC}"
    
    # Run the Python script
    if python3 "$SCRIPT_PATH" \
        --task-id $task_id \
        --task-suite-multiple "$TASK_SUITE_MULTIPLE" \
        --task-suite-single "$TASK_SUITE_SINGLE" \
        --max-demos $MAX_DEMOS; then
        
        echo -e "${GREEN}[$(date)] Task ID $task_id completed successfully${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}[$(date)] Task ID $task_id failed${NC}"
        FAILED_TASKS+=($task_id)
    fi
    
    echo "----------------------------------------"
done

# Summary
echo -e "${BLUE}Batch processing completed!${NC}"
echo "Successful tasks: $SUCCESS_COUNT/$((END_ID - START_ID + 1))"

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo -e "${RED}Failed task IDs: ${FAILED_TASKS[*]}${NC}"
    exit 1
else
    echo -e "${GREEN}All tasks completed successfully!${NC}"
    exit 0
fi
