#!/usr/bin/env python3
"""
automation_pipeline.py
----------------------
Automation script to run the complete pipeline:
1. Run replay_and_save_hdf5.py to generate replay HDF5 files
2. Run custom_dataset_creator.py to process the HDF5 files
3. Run generate_init_states.py to create initial state files and copy them as .pruned_init files

This script automates the complete workflow for processing LIBERO tasks.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from libero.libero import benchmark, get_libero_path


class AutomationPipeline:
    """Main automation pipeline for LIBERO dataset processing."""

    def __init__(self, task_id: int, task_suite_name_multiple: str = "libero_object",
                 task_suite_name_single: str = "libero_object_single", max_demos: int = 50,
                 camera_height: int = 128, camera_width: int = 128):
        """
        Initialize the automation pipeline.
        
        Args:
            task_id: Task ID to process
            task_suite_name_multiple: Multiple task suite name
            task_suite_name_single: Single task suite name
            max_demos: Maximum number of demonstrations to process
            camera_height: Camera height for rendering
            camera_width: Camera width for rendering
        """
        self.task_id = task_id
        self.task_suite_multiple = task_suite_name_multiple
        self.task_suite_single = task_suite_name_single
        self.max_demos = max_demos
        self.camera_height = camera_height
        self.camera_width = camera_width
        
        # Set up paths
        self.base_dir = Path(__file__).parent
        
        self.replay_script_path = self.base_dir / "replay_and_save_hdf5.py"
        self.dataset_creator_script_path = self.base_dir / "create_dataset.py"
        self.init_states_script_path = self.base_dir / "generate_init_states.py"
        
        # Output directories
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_multiple = benchmark_dict[self.task_suite_multiple]()
        task_multiple = task_suite_multiple.get_task(self.task_id)
        task_suite_single = benchmark_dict[self.task_suite_single]()
        task_single = task_suite_single.get_task(self.task_id)

        datasets_dir = get_libero_path("datasets")
        self.replay_output_dir = Path(datasets_dir) / "replay" / task_suite_single.name
        self.processed_output_dir = Path(datasets_dir) / "processed"
        self.init_states_dir = Path(get_libero_path("init_states")) / task_single.problem_folder / task_single.init_states_file

        # Track pipeline status
        self.pipeline_results = {
            "task_id": task_id,
            "task_suite_multiple": task_suite_name_multiple,
            "task_suite_single": task_suite_name_single,
            "steps_completed": [],
            "steps_failed": [],
            "output_files": {},
            "errors": []
        }

    def log_info(self, message: str) -> None:
        """Log an info message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] INFO: {message}")

    def log_error(self, message: str) -> None:
        """Log an error message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ERROR: {message}")
        self.pipeline_results["errors"].append(f"{timestamp}: {message}")

    def run_command(self, command: List[str], description: str, cwd: Optional[str] = None) -> Tuple[bool, str]:
        """
        Run a command and return success status and output.
        
        Args:
            command: Command to run as list of arguments
            description: Description of the command for logging
            cwd: Working directory to run command in
            
        Returns:
            Tuple of (success, output)
        """
        self.log_info(f"Running: {description}")
        self.log_info(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                self.log_info(f"Successfully completed: {description}")
                return True, result.stdout
            else:
                error_msg = f"Command failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nStderr: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nStdout: {result.stdout}"
                self.log_error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Exception running command: {str(e)}"
            self.log_error(error_msg)
            return False, error_msg

    def step1_replay_and_save_hdf5(self) -> bool:
        """
        Step 1: Run replay_and_save_hdf5.py to generate replay HDF5 files.
        
        Returns:
            True if successful, False otherwise
        """
        self.log_info("=" * 60)
        self.log_info("STEP 1: Running replay_and_save_hdf5.py")
        self.log_info("=" * 60)
        
        if not self.replay_script_path.exists():
            self.log_error(f"Replay script not found: {self.replay_script_path}")
            return False
        
        command = [
            "python3",
            str(self.replay_script_path),
            "--task-id", str(self.task_id),
            "--task-suite-multiple", self.task_suite_multiple,
            "--task-suite-single", self.task_suite_single,
            "--max-demos", str(self.max_demos),
            "--camera-height", str(self.camera_height),
            "--camera-width", str(self.camera_width),
            "--output-dir", str(self.replay_output_dir)  # Add this line
        ]
        
        success, output = self.run_command(
            command, 
            f"Replay and save HDF5 for task {self.task_id}"
        )
        
        if success:
            self.pipeline_results["steps_completed"].append("replay_and_save_hdf5")
            
            # Find the generated HDF5 file
            replay_files = list(self.replay_output_dir.glob("*_replay.hdf5"))
            if replay_files:
                self.pipeline_results["output_files"]["replay_hdf5"] = str(replay_files[-1])
                self.log_info(f"Generated replay HDF5: {replay_files[-1]}")
            
            # Find the results JSON file
            json_files = list(self.replay_output_dir.glob("*_replay_results.json"))
            if json_files:
                self.pipeline_results["output_files"]["replay_results"] = str(json_files[-1])
                self.log_info(f"Generated results JSON: {json_files[-1]}")
        else:
            self.pipeline_results["steps_failed"].append("replay_and_save_hdf5")
        
        return success

    def step2_process_with_custom_dataset_creator(self) -> bool:
        """
        Step 2: Run custom_dataset_creator.py to process the HDF5 files.
        
        Returns:
            True if successful, False otherwise
        """
        self.log_info("=" * 60)
        self.log_info("STEP 2: Running custom_dataset_creator.py")
        self.log_info("=" * 60)
        
        if not self.dataset_creator_script_path.exists():
            self.log_error(f"Dataset creator script not found: {self.dataset_creator_script_path}")
            return False
            
        # Find the replay HDF5 file from step 1
        replay_hdf5_path = self.pipeline_results["output_files"].get("replay_hdf5")
        if not replay_hdf5_path or not Path(replay_hdf5_path).exists():
            self.log_error("No replay HDF5 file found from step 1")
            return False
        
        command = [
            "python3",
            str(self.dataset_creator_script_path),
            "--demo-file", replay_hdf5_path,
            "--use-camera-obs",
            "--use-actions"
        ]
        
        success, output = self.run_command(
            command,
            f"Process HDF5 file with custom dataset creator"
        )
        
        if success:
            self.pipeline_results["steps_completed"].append("custom_dataset_creator")
            
            # Find the processed HDF5 file
            processed_files = list(self.processed_output_dir.glob("*_demo.hdf5"))
            if processed_files:
                # Get the most recent file
                newest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
                self.pipeline_results["output_files"]["processed_hdf5"] = str(newest_file)
                self.log_info(f"Generated processed HDF5: {newest_file}")
        else:
            self.pipeline_results["steps_failed"].append("custom_dataset_creator")
        
        return success

    def step3_generate_init_states(self) -> bool:
        """
        Step 3: Run generate_init_states.py and create .pruned_init files.
        
        Returns:
            True if successful, False otherwise
        """
        self.log_info("=" * 60)
        self.log_info("STEP 3: Running generate_init_states.py")
        self.log_info("=" * 60)
        
        if not self.init_states_script_path.exists():
            self.log_error(f"Init states script not found: {self.init_states_script_path}")
            return False
        
        command = [
            "python3",
            str(self.init_states_script_path),
            "--task-id", str(self.task_id),
            "--benchmark", self.task_suite_single,
            "--num_init", "50",
            "--save_dir", str(self.init_states_dir.parent)
        ]
        
        success, output = self.run_command(
            command,
            f"Generate initial states for task {self.task_id}"
        )
        
        if success:
            self.pipeline_results["steps_completed"].append("generate_init_states")
            
            # Find the generated .init file
            init_files = list(self.init_states_dir.glob("**/*.init"))
            if init_files:
                # Get the most recent .init file
                newest_init_file = max(init_files, key=lambda f: f.stat().st_mtime)
                self.pipeline_results["output_files"]["init_file"] = str(newest_init_file)
                self.log_info(f"Generated init file: {newest_init_file}")
                
                # Create .pruned_init copy
                pruned_init_file = newest_init_file.with_suffix('.pruned_init')
                try:
                    shutil.copy2(newest_init_file, pruned_init_file)
                    self.pipeline_results["output_files"]["pruned_init_file"] = str(pruned_init_file)
                    self.log_info(f"Created pruned init file: {pruned_init_file}")
                except Exception as e:
                    self.log_error(f"Failed to create pruned init file: {e}")
                    return False
        else:
            self.pipeline_results["steps_failed"].append("generate_init_states")
        
        return success

    def save_pipeline_summary(self) -> None:
        """Save a summary of the pipeline execution."""
        summary_file = self.base_dir / f"pipeline_summary_task_{self.task_id}.json"
        
        # Add execution timestamp
        self.pipeline_results["execution_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.pipeline_results["total_steps"] = 3
        self.pipeline_results["completed_steps"] = len(self.pipeline_results["steps_completed"])
        self.pipeline_results["failed_steps"] = len(self.pipeline_results["steps_failed"])
        self.pipeline_results["success"] = len(self.pipeline_results["steps_failed"]) == 0
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_results, f, indent=2, ensure_ascii=False)
            self.log_info(f"Pipeline summary saved to: {summary_file}")
        except Exception as e:
            self.log_error(f"Failed to save pipeline summary: {e}")

    def run_pipeline(self) -> bool:
        """
        Run the complete automation pipeline.
        
        Returns:
            True if all steps completed successfully, False otherwise
        """
        self.log_info("Starting LIBERO automation pipeline")
        self.log_info(f"Task ID: {self.task_id}")
        self.log_info(f"Task suite multiple: {self.task_suite_multiple}")
        self.log_info(f"Task suite single: {self.task_suite_single}")
        self.log_info(f"Max demos: {self.max_demos}")
        
        start_time = time.time()
        overall_success = True
        
        # Step 1: Replay and save HDF5
        if not self.step1_replay_and_save_hdf5():
            overall_success = False
            self.log_error("Step 1 failed, stopping pipeline")
            return False
        
        # Step 2: Process with custom dataset creator
        if not self.step2_process_with_custom_dataset_creator():
            overall_success = False
            self.log_error("Step 2 failed, stopping pipeline")
            return False
        
        # Step 3: Generate init states
        if not self.step3_generate_init_states():
            overall_success = False
            self.log_error("Step 3 failed")
            # Don't return False here as we want to save summary even if this step fails
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print final summary
        self.log_info("=" * 60)
        self.log_info("PIPELINE EXECUTION SUMMARY")
        self.log_info("=" * 60)
        self.log_info(f"Task ID: {self.task_id}")
        self.log_info(f"Overall success: {overall_success}")
        self.log_info(f"Execution time: {execution_time:.2f} seconds")
        self.log_info(f"Completed steps: {self.pipeline_results['steps_completed']}")
        if self.pipeline_results['steps_failed']:
            self.log_info(f"Failed steps: {self.pipeline_results['steps_failed']}")
        
        # Print output files
        if self.pipeline_results["output_files"]:
            self.log_info("Generated files:")
            for file_type, file_path in self.pipeline_results["output_files"].items():
                self.log_info(f"  {file_type}: {file_path}")
        
        # Save pipeline summary
        self.save_pipeline_summary()
        
        return overall_success


def main():
    """Main function to parse arguments and run the automation pipeline."""
    parser = argparse.ArgumentParser(
        description='Automation pipeline for LIBERO dataset processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--task-id',
        type=int,
        default=0,
        help='Task ID to process'
    )
    parser.add_argument(
        '--task-suite-multiple',
        type=str,
        default='libero_object',
        help='Multiple task suite name'
    )
    parser.add_argument(
        '--task-suite-single',
        type=str,
        default='libero_object_single',
        help='Single task suite name'
    )
    parser.add_argument(
        '--max-demos',
        type=int,
        default=50,
        help='Maximum number of demonstrations to process'
    )
    parser.add_argument(
        '--camera-height',
        type=int,
        default=128,
        help='Camera height for rendering'
    )
    parser.add_argument(
        '--camera-width',
        type=int,
        default=128,
        help='Camera width for rendering'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = AutomationPipeline(
        task_id=args.task_id,
        task_suite_name_multiple=args.task_suite_multiple,
        task_suite_name_single=args.task_suite_single,
        max_demos=args.max_demos,
        camera_height=args.camera_height,
        camera_width=args.camera_width
    )
    
    success = pipeline.run_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
