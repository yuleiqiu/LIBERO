# LIBERO Replay and Save HDF5 Tools

This directory contains tools for replaying LIBERO demonstrations and saving them in HDF5 format compatible with `collect_demonstration.py`.

## Files

- `replay_and_save_hdf5.py` - Main Python script with command-line interface
- `run_replay_batch.sh` - Batch script to run multiple tasks (1-9)
- `run_single_task.sh` - Script to run a single task with options
- `README.md` - This documentation

## Usage

### 1. Single Task Execution

Run a single task with custom parameters:

```bash
python3 replay_and_save_hdf5.py --task-id 1
```

#### Available Parameters

- `--task-id`: Task ID to replay (default: 0)
- `--task-suite-multiple`: Multiple task suite name (default: "libero_object")
- `--task-suite-single`: Single task suite name (default: "libero_object_single")
- `--max-demos`: Maximum number of demonstrations (default: 50)
- `--camera-height`: Camera height (default: 128)
- `--camera-width`: Camera width (default: 128)

#### Examples

```bash
# Run task 5 with default settings
python3 replay_and_save_hdf5.py --task-id 5

# Run task 3 with custom demo count
python3 replay_and_save_hdf5.py --task-id 3 --max-demos 30

# Run with different task suites
python3 replay_and_save_hdf5.py --task-id 2 \
    --task-suite-multiple "libero_spatial" \
    --task-suite-single "libero_spatial_single"
```

### 2. Batch Execution (Tasks 1-9)

Run all tasks from 1 to 9:

```bash
./run_replay_batch.sh
```

Run tasks from 3 to 7:

```bash
./run_replay_batch.sh 3 7
```

### 3. Single Task with Helper Script

```bash
# Run task 1
./run_single_task.sh 1

# Dry run (see command without executing)
./run_single_task.sh 5 --dry-run

# Verbose mode
./run_single_task.sh 3 --verbose
```

## Output

### File Structure

Each task generates an HDF5 file named `{task_name}_replay.hdf5` in the `replay/hdf5_output/` directory.

### HDF5 Format

The output files follow the same format as `collect_demonstration.py`:

```
data/
├── demo_1/
│   ├── states (dataset)
│   ├── actions (dataset)
│   └── model_file (attribute)
├── demo_2/
│   └── ...
└── attributes:
    ├── date
    ├── time
    ├── repository_version
    ├── env
    ├── env_info
    ├── problem_info
    ├── bddl_file_name
    └── bddl_file_content
```

### Console Output

The scripts provide detailed progress information:

```
[INFO] Using single task suite: libero_object_single, task ID: 1, task name: pick_up_the_alphabet_soup_and_place_it_in_the_basket
Demo 00 – SUCCESS (states: 156, actions: 156)
Demo 01 – SUCCESS (states: 142, actions: 142)
...
SUMMARY: 45/50 replays succeed (90.00%)
HDF5 file saved to: replay/hdf5_output/pick_up_the_alphabet_soup_and_place_it_in_the_basket_replay.hdf5
```

## Error Handling

The scripts include comprehensive error handling:

- **Missing task suites**: Clear error messages for invalid task suite names
- **Invalid task IDs**: Graceful handling of out-of-range task IDs
- **Missing demo files**: Verification that required files exist
- **Processing failures**: Detailed error reporting for debugging

## Batch Script Features

### Color-coded Output

- 🔵 Blue: Information messages
- 🟢 Green: Success messages
- 🔴 Red: Error messages
- 🟡 Yellow: Warnings/dry-run mode

### Progress Tracking

- Real-time progress updates
- Success/failure counting
- Final summary with failed task IDs

### Parallel Execution (Optional)

To run tasks in parallel, you can modify the batch script or use:

```bash
# Run 3 tasks in parallel
for i in {1..3}; do
    ./run_single_task.sh $i &
done
wait
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure LIBERO is properly installed and in Python path
2. **Missing Files**: Verify that demo datasets are downloaded
3. **Memory Issues**: Reduce `--max-demos` for large tasks
4. **Permission Errors**: Ensure scripts are executable (`chmod +x *.sh`)

### Debug Mode

For debugging, you can add verbose output to the Python script:

```bash
python3 replay_and_save_hdf5.py --task-id 1 2>&1 | tee debug.log
```

## Performance Notes

- Each task typically takes 2-10 minutes depending on the number of demonstrations
- Memory usage scales with `--max-demos` parameter
- HDF5 files are typically 10-100 MB per task

## Integration

These tools are designed to work seamlessly with:

- LIBERO benchmark suite
- LeRobot data pipeline
- Standard HDF5 analysis tools
- Custom downstream processing scripts
