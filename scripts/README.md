# LIBERO Automation Pipeline

This directory contains automation scripts to run the complete LIBERO dataset processing pipeline:

1. **Step 1**: Run `replay_and_save_hdf5.py` to generate replay HDF5 files
2. **Step 2**: Run `custom_dataset_creator.py` to process the HDF5 files  
3. **Step 3**: Run `generate_init_states.py` to create initial state files and copy them as `.pruned_init` files

## Files

- `automation_pipeline.py` - Main automation script for single tasks
- `batch_automation.py` - Batch processing script for multiple tasks
- `run_automation.sh` - Shell script wrapper for easy execution
- `replay/replay_and_save_hdf5.py` - Step 1 script for replay generation

## Quick Start

### Process a Single Task

```bash
# Process task ID 1
./run_automation.sh --task-id 1

# Process task ID 5 with custom parameters
./run_automation.sh --task-id 5 --max-demos 100 --camera-height 256 --camera-width 256
```

### Process Multiple Tasks (Batch Mode)

```bash
# Process tasks 0-9 sequentially
./run_automation.sh --start-id 0 --end-id 9

# Process tasks 0-9 with 3 parallel workers
./run_automation.sh --start-id 0 --end-id 9 --max-workers 3

# Process tasks 10-19 for different task suite
./run_automation.sh --start-id 10 --end-id 19 --task-suite-multiple libero_90 --task-suite-single libero_90_single
```

### Dry Run (Preview Commands)

```bash
# See what would be executed without running
./run_automation.sh --task-id 1 --dry-run
./run_automation.sh --start-id 0 --end-id 9 --dry-run
```

## Command Line Options

### Single Task Mode
- `--task-id ID` - Process single task ID

### Batch Mode  
- `--start-id ID` - Starting task ID (inclusive)
- `--end-id ID` - Ending task ID (inclusive) 
- `--max-workers N` - Number of parallel workers (default: 1)

### Common Options
- `--task-suite-multiple NAME` - Multiple task suite name (default: libero_object)
- `--task-suite-single NAME` - Single task suite name (default: libero_object_single)
- `--max-demos N` - Maximum demos per task (default: 50)
- `--camera-height N` - Camera height (default: 128)
- `--camera-width N` - Camera width (default: 128)
- `--dry-run` - Show what would be done without executing
- `--help` - Show help message

## Pipeline Steps

### Step 1: Replay and Save HDF5
- Loads original demonstrations from LIBERO datasets
- Replays actions in the single-task environment
- Saves replayed demonstrations as HDF5 files
- Outputs: `*_replay.hdf5` and `*_replay_results.json`

### Step 2: Custom Dataset Creator
- Processes the replay HDF5 files
- Adds camera observations and other features
- Creates final processed dataset
- Outputs: `*_demo.hdf5` in `libero/datasets/processed/`

### Step 3: Generate Initial States
- Creates initial state files for the task
- Generates `.init` files with random initial states
- Creates `.pruned_init` copies of the initial state files
- Outputs: `.init` and `.pruned_init` files in `libero/libero/init_files/`

## Output Files

### Generated Files Structure
```
replay/hdf5_output/
├── {task_name}_replay.hdf5           # Step 1: Replay demonstrations
├── {task_name}_replay_results.json   # Step 1: Processing results
└── pipeline_summary_task_{id}.json   # Individual task summary

libero/datasets/processed/
└── {task_name}_demo.hdf5             # Step 2: Processed dataset

libero/libero/init_files/{suite_name}/
├── {task_name}.init                  # Step 3: Initial states
└── {task_name}.pruned_init           # Step 3: Pruned initial states (copy)

batch_summary_{start}_to_{end}.json   # Batch processing summary
```

### Summary Files
- Individual task summaries include execution status, timing, and output file paths
- Batch summaries include statistics across all processed tasks
- JSON format for easy parsing and analysis

## Direct Python Usage

### Single Task
```python
from automation_pipeline import AutomationPipeline

pipeline = AutomationPipeline(task_id=1)
success = pipeline.run_pipeline()
```

### Batch Processing
```python
from batch_automation import BatchAutomation

batch = BatchAutomation(start_id=0, end_id=9, max_workers=3)
success = batch.run_batch()
```

## Error Handling

- Each step is validated before proceeding
- Failed steps are logged with detailed error messages
- Pipeline continues to summary generation even if later steps fail
- Batch processing continues with remaining tasks if individual tasks fail

## Performance Notes

- Sequential processing: Safest, uses less resources
- Parallel processing: Faster but uses more CPU/memory
- Recommended max_workers: 2-4 depending on system resources
- Large camera resolutions or demo counts increase processing time

## Troubleshooting

1. **Missing dependencies**: Ensure all LIBERO dependencies are installed
2. **File not found errors**: Check that source scripts exist in expected locations
3. **Permission errors**: Ensure write permissions for output directories
4. **Memory issues**: Reduce max_workers or max_demos for large tasks
5. **CUDA/GPU issues**: Scripts should work with CPU-only rendering

## Examples

### Development/Testing
```bash
# Quick test with single task and smaller settings
./run_automation.sh --task-id 0 --max-demos 10 --camera-height 64 --camera-width 64

# Test batch mode with dry run first
./run_automation.sh --start-id 0 --end-id 2 --dry-run
./run_automation.sh --start-id 0 --end-id 2
```

### Production Processing
```bash
# Process full libero_object suite
./run_automation.sh --start-id 0 --end-id 9 --max-workers 2

# Process with high quality settings
./run_automation.sh --start-id 0 --end-id 9 --camera-height 256 --camera-width 256 --max-demos 100
```
