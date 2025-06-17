#!/usr/bin/env python3
"""
batch_automation.py
-------------------
Batch script to run the automation pipeline for multiple task IDs.
This script allows processing multiple tasks in sequence or parallel.
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from automation_pipeline import AutomationPipeline


class BatchAutomation:
    """Batch automation for multiple LIBERO tasks."""
    
    def __init__(self, start_id: int = 0, end_id: int = 9, 
                 task_suite_multiple: str = "libero_object",
                 task_suite_single: str = "libero_object_single", 
                 max_demos: int = 50, camera_height: int = 128, 
                 camera_width: int = 128, max_workers: int = 1):
        """
        Initialize batch automation.
        
        Args:
            start_id: Starting task ID (inclusive)
            end_id: Ending task ID (inclusive)
            task_suite_multiple: Multiple task suite name
            task_suite_single: Single task suite name
            max_demos: Maximum number of demonstrations per task
            camera_height: Camera height for rendering
            camera_width: Camera width for rendering
            max_workers: Maximum number of parallel workers (1 = sequential)
        """
        self.start_id = start_id
        self.end_id = end_id
        self.task_suite_multiple = task_suite_multiple
        self.task_suite_single = task_suite_single
        self.max_demos = max_demos
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.max_workers = max_workers
        
        self.task_ids = list(range(start_id, end_id + 1))
        self.results = {}
        
        self.base_dir = Path(__file__).parent

    def log_info(self, message: str) -> None:
        """Log an info message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] BATCH INFO: {message}")

    def log_error(self, message: str) -> None:
        """Log an error message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] BATCH ERROR: {message}")

    def process_single_task(self, task_id: int) -> Dict[str, Any]:
        """
        Process a single task ID.
        
        Args:
            task_id: Task ID to process
            
        Returns:
            Dictionary with task results
        """
        self.log_info(f"Starting processing for task {task_id}")
        
        start_time = time.time()
        
        try:
            pipeline = AutomationPipeline(
                task_id=task_id,
                task_suite_multiple=self.task_suite_multiple,
                task_suite_single=self.task_suite_single,
                max_demos=self.max_demos,
                camera_height=self.camera_height,
                camera_width=self.camera_width
            )
            
            success = pipeline.run_pipeline()
            end_time = time.time()
            
            result = {
                "task_id": task_id,
                "success": success,
                "execution_time": end_time - start_time,
                "pipeline_results": pipeline.pipeline_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if success:
                self.log_info(f"Successfully completed task {task_id} in {result['execution_time']:.2f}s")
            else:
                self.log_error(f"Failed to complete task {task_id} after {result['execution_time']:.2f}s")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            error_result = {
                "task_id": task_id,
                "success": False,
                "execution_time": end_time - start_time,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            self.log_error(f"Exception processing task {task_id}: {e}")
            return error_result

    def run_sequential(self) -> None:
        """Run tasks sequentially."""
        self.log_info("Running tasks sequentially")
        
        for task_id in self.task_ids:
            result = self.process_single_task(task_id)
            self.results[task_id] = result

    def run_parallel(self) -> None:
        """Run tasks in parallel."""
        self.log_info(f"Running tasks in parallel with {self.max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.process_single_task, task_id): task_id 
                for task_id in self.task_ids
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    self.results[task_id] = result
                except Exception as e:
                    error_result = {
                        "task_id": task_id,
                        "success": False,
                        "error": f"Future exception: {str(e)}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.results[task_id] = error_result
                    self.log_error(f"Future exception for task {task_id}: {e}")

    def save_batch_summary(self) -> None:
        """Save a summary of the batch execution."""
        summary_file = self.base_dir / f"batch_summary_{self.start_id}_to_{self.end_id}.json"
        
        # Calculate statistics
        total_tasks = len(self.task_ids)
        successful_tasks = sum(1 for r in self.results.values() if r.get("success", False))
        failed_tasks = total_tasks - successful_tasks
        total_time = sum(r.get("execution_time", 0) for r in self.results.values())
        
        summary = {
            "batch_info": {
                "start_id": self.start_id,
                "end_id": self.end_id,
                "task_suite_multiple": self.task_suite_multiple,
                "task_suite_single": self.task_suite_single,
                "max_demos": self.max_demos,
                "max_workers": self.max_workers,
                "execution_mode": "parallel" if self.max_workers > 1 else "sequential"
            },
            "statistics": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
                "total_execution_time": total_time,
                "average_time_per_task": total_time / total_tasks if total_tasks > 0 else 0.0
            },
            "task_results": self.results,
            "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.log_info(f"Batch summary saved to: {summary_file}")
        except Exception as e:
            self.log_error(f"Failed to save batch summary: {e}")

    def print_final_summary(self) -> None:
        """Print a final summary of the batch execution."""
        self.log_info("=" * 80)
        self.log_info("BATCH EXECUTION FINAL SUMMARY")
        self.log_info("=" * 80)
        
        total_tasks = len(self.task_ids)
        successful_tasks = sum(1 for r in self.results.values() if r.get("success", False))
        failed_tasks = total_tasks - successful_tasks
        total_time = sum(r.get("execution_time", 0) for r in self.results.values())
        
        self.log_info(f"Task range: {self.start_id} to {self.end_id}")
        self.log_info(f"Total tasks: {total_tasks}")
        self.log_info(f"Successful tasks: {successful_tasks}")
        self.log_info(f"Failed tasks: {failed_tasks}")
        self.log_info(f"Success rate: {successful_tasks/total_tasks*100:.1f}%")
        self.log_info(f"Total execution time: {total_time:.2f} seconds")
        self.log_info(f"Average time per task: {total_time/total_tasks:.2f} seconds")
        
        if failed_tasks > 0:
            self.log_info("Failed task IDs:")
            failed_ids = [str(task_id) for task_id, result in self.results.items() 
                         if not result.get("success", False)]
            self.log_info(f"  {', '.join(failed_ids)}")
        
        self.log_info("=" * 80)

    def run_batch(self) -> bool:
        """
        Run the batch automation.
        
        Returns:
            True if all tasks completed successfully, False otherwise
        """
        self.log_info("Starting batch automation")
        self.log_info(f"Processing tasks {self.start_id} to {self.end_id}")
        self.log_info(f"Total tasks: {len(self.task_ids)}")
        
        batch_start_time = time.time()
        
        # Run tasks based on worker count
        if self.max_workers > 1:
            self.run_parallel()
        else:
            self.run_sequential()
        
        batch_end_time = time.time()
        batch_execution_time = batch_end_time - batch_start_time
        
        self.log_info(f"Batch completed in {batch_execution_time:.2f} seconds")
        
        # Print and save summary
        self.print_final_summary()
        self.save_batch_summary()
        
        # Return success if all tasks completed successfully
        successful_tasks = sum(1 for r in self.results.values() if r.get("success", False))
        return successful_tasks == len(self.task_ids)


def main():
    """Main function to parse arguments and run batch automation."""
    parser = argparse.ArgumentParser(
        description='Batch automation for LIBERO dataset processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--start-id',
        type=int,
        default=0,
        help='Starting task ID (inclusive)'
    )
    parser.add_argument(
        '--end-id',
        type=int,
        default=9,
        help='Ending task ID (inclusive)'
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
        help='Maximum number of demonstrations per task'
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
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Maximum number of parallel workers (1 = sequential)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without actually running'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - Commands that would be executed:")
        task_ids = list(range(args.start_id, args.end_id + 1))
        print(f"Task IDs: {task_ids}")
        print(f"Task suite multiple: {args.task_suite_multiple}")
        print(f"Task suite single: {args.task_suite_single}")
        print(f"Max demos per task: {args.max_demos}")
        print(f"Max workers: {args.max_workers}")
        print(f"Execution mode: {'parallel' if args.max_workers > 1 else 'sequential'}")
        return
    
    # Create and run batch automation
    batch = BatchAutomation(
        start_id=args.start_id,
        end_id=args.end_id,
        task_suite_multiple=args.task_suite_multiple,
        task_suite_single=args.task_suite_single,
        max_demos=args.max_demos,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        max_workers=args.max_workers
    )
    
    success = batch.run_batch()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
