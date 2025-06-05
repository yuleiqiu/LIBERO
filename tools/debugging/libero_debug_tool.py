#!/usr/bin/env python3
"""
LIBERO Debug Tool - Unified script for debugging LIBERO tasks, initial states, and demonstrations.

This tool combines functionality from multiple debugging scripts to provide a comprehensive
debugging experience for LIBERO benchmarks.
"""

# Standard library imports
import os
import argparse
import pprint
from collections import defaultdict

# Third-party imports
import numpy as np
import torch
import h5py
from termcolor import colored

# LIBERO-specific imports
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark


class LiberoDebugTool:
    """Unified debugging tool for LIBERO benchmarks."""
    
    def __init__(self, benchmark_name="libero_object"):
        self.benchmark_name = benchmark_name
        self.pp = pprint.PrettyPrinter(indent=2)
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.benchmark_instance = self.benchmark_dict[benchmark_name]()
        
        # Paths
        self.init_states_path = get_libero_path("init_states")
        self.datasets_path = get_libero_path("datasets")
        
    def print_separator(self, char="=", length=80):
        """Print a separator line."""
        print(char * length)
    
    def print_section_header(self, title):
        """Print a formatted section header."""
        print(f"\n{colored(title, 'cyan', attrs=['bold'])}")
        self.print_separator()
    
    def are_states_identical(self, state1, state2):
        """Compare two state objects (tensors, dictionaries, etc.) for equality."""
        if type(state1) != type(state2):
            return False

        # Handle dictionary case
        if isinstance(state1, dict):
            if set(state1.keys()) != set(state2.keys()):
                return False
            
            for key in state1:
                if not self.are_states_identical(state1[key], state2[key]):
                    return False
            return True
        
        # Handle tensor case
        if isinstance(state1, torch.Tensor):
            return torch.equal(state1, state2)
        
        # Handle numpy array case
        if isinstance(state1, np.ndarray):
            return np.array_equal(state1, state2)
        
        # Handle list/tuple case
        if isinstance(state1, (list, tuple)):
            if len(state1) != len(state2):
                return False
            return all(self.are_states_identical(a, b) for a, b in zip(state1, state2))
        
        # For simple types
        return state1 == state2
    
    def print_state_info(self, states, prefix=""):
        """Print information about state objects."""
        if isinstance(states, dict):
            print(f"{prefix}State is a dictionary with keys:")
            for key, value in states.items():
                if hasattr(value, "shape"):
                    print(f"{prefix}  {key}: shape={value.shape}, type={type(value)}")
                else:
                    print(f"{prefix}  {key}: type={type(value)}")
        elif hasattr(states, "shape"):
            print(f"{prefix}State shape: {states.shape}, type={type(states)}")
        else:
            print(f"{prefix}State type: {type(states)}")
    
    def show_benchmark_overview(self):
        """Show overview of available benchmarks and current benchmark info."""
        self.print_section_header("Benchmark Overview")
        print("Available benchmarks:")
        self.pp.pprint(self.benchmark_dict)
        print(f"\nCurrent benchmark: {colored(self.benchmark_name, 'green')}")
        print(f"Number of tasks: {colored(self.benchmark_instance.get_num_tasks(), 'green')}")
    
    def inspect_init_states_via_api(self, task_ids=None):
        """Load init states through benchmark API and examine their shapes."""
        self.print_section_header(f"Init States via API - {self.benchmark_name}")
        
        task_range = task_ids if task_ids else range(self.benchmark_instance.get_num_tasks())
        
        for task_id in task_range:
            task = self.benchmark_instance.get_task(task_id)
            print(f"\n{colored(f'Task {task_id}: {task.name}', 'yellow')}")
            
            try:
                init_states = self.benchmark_instance.get_task_init_states(task_id)
                print(f"  Number of init states: {len(init_states) if hasattr(init_states, '__len__') else 'N/A'}")
                self.print_state_info(init_states, "  ")
            except Exception as e:
                print(f"  {colored(f'Error loading init states: {e}', 'red')}")
    
    def compare_init_files(self, task_ids=None):
        """Compare .init and .pruned_init files for specified tasks."""
        self.print_section_header(f"Init Files Comparison - {self.benchmark_name}")
        
        task_range = task_ids if task_ids else range(self.benchmark_instance.get_num_tasks())
        
        for task_id in task_range:
            task = self.benchmark_instance.get_task(task_id)
            folder_path = os.path.join(self.init_states_path, task.problem_folder)
            
            print(f"\n{colored(f'Task {task_id}: {task.name}', 'yellow')}")
            print(f"  Problem folder: {task.problem_folder}")
            print(f"  Init states folder: {folder_path}")
            
            if not os.path.exists(folder_path):
                print(f"  {colored(f'Init states folder not found: {folder_path}', 'red')}")
                continue
            
            # Construct file paths for this specific task
            init_file = os.path.join(folder_path, f"{task.name}.init")
            pruned_init_file = os.path.join(folder_path, f"{task.name}.pruned_init")
            
            print(f"  .init file: {'✓' if os.path.exists(init_file) else '✗'} {init_file}")
            print(f"  .pruned_init file: {'✓' if os.path.exists(pruned_init_file) else '✗'} {pruned_init_file}")
            
            # Compare if both files exist
            if os.path.exists(init_file) and os.path.exists(pruned_init_file):
                try:
                    init_state = torch.load(init_file)
                    pruned_init_state = torch.load(pruned_init_file)
                    
                    print(f"  Init state shape: {init_state.shape if hasattr(init_state, 'shape') else type(init_state)}")
                    print(f"  Pruned init state shape: {pruned_init_state.shape if hasattr(pruned_init_state, 'shape') else type(pruned_init_state)}")
                    
                    is_identical = self.are_states_identical(init_state, pruned_init_state)
                    
                    if is_identical:
                        print(f"  {colored('✓ Files are IDENTICAL', 'green')}")
                    else:
                        print(f"  {colored('✗ Files are DIFFERENT', 'red')}")
                        
                        # Additional analysis for different files
                        if hasattr(init_state, 'shape') and hasattr(pruned_init_state, 'shape'):
                            if init_state.shape != pruned_init_state.shape:
                                print(f"    Shape difference: {init_state.shape} vs {pruned_init_state.shape}")
                            else:
                                # Check element-wise differences
                                if isinstance(init_state, torch.Tensor):
                                    diff_mask = init_state != pruned_init_state
                                    num_diff = torch.sum(diff_mask).item()
                                    print(f"    Number of different elements: {num_diff}")
                                elif isinstance(init_state, np.ndarray):
                                    diff_mask = init_state != pruned_init_state
                                    num_diff = np.sum(diff_mask)
                                    print(f"    Number of different elements: {num_diff}")
                                    
                except Exception as e:
                    print(f"  {colored(f'Error comparing files: {e}', 'red')}")
            else:
                missing = []
                if not os.path.exists(init_file):
                    missing.append('.init')
                if not os.path.exists(pruned_init_file):
                    missing.append('.pruned_init')
                if missing:
                    missing_files_str = ', '.join(missing)
                    print(f"  {colored(f'Missing files: {missing_files_str}', 'red')}")
    
    def check_task_demos(self, task_ids=None):
        """Check demonstration data for specified tasks."""
        self.print_section_header(f"Demo Data Check - {self.benchmark_name}")
        
        task_range = task_ids if task_ids else range(self.benchmark_instance.get_num_tasks())
        
        for task_id in task_range:
            task = self.benchmark_instance.get_task(task_id)
            print(f"\n{colored(f'Task {task_id}: {task.name}', 'yellow')}")
            
            try:
                # Check demo file
                demo_file_path = os.path.join(self.datasets_path, self.benchmark_instance.get_task_demonstration(task_id))
                
                if os.path.exists(demo_file_path):
                    with h5py.File(demo_file_path, "r") as f:
                        num_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])
                        print(f"  Demonstrations: {num_demos}")
                        
                        # Additional demo info
                        if num_demos > 0:
                            demo_key = "demo_0"
                            if demo_key in f["data"]:
                                demo_data = f["data"][demo_key]
                                demo_length = len(demo_data["actions"]) if "actions" in demo_data else "N/A"
                                print(f"  First demo length: {demo_length}")
                else:
                    print(f"  {colored('Demo file not found', 'red')}: {demo_file_path}")
                    
            except Exception as e:
                print(f"  {colored(f'Error checking demos: {e}', 'red')}")
    
    def detailed_task_analysis(self, task_id):
        """Perform detailed analysis of a specific task."""
        self.print_section_header(f"Detailed Task Analysis - Task {task_id}")
        
        if task_id >= self.benchmark_instance.get_num_tasks():
            print(f"{colored(f'Invalid task ID: {task_id}', 'red')}")
            return
        
        task = self.benchmark_instance.get_task(task_id)
        print(f"Task info:")
        self.pp.pprint(task)
        
        # Check init states files
        folder_path = os.path.join(self.init_states_path, task.problem_folder)
        init_file = os.path.join(folder_path, f"{task.name}.init")
        pruned_init_file = os.path.join(folder_path, f"{task.name}.pruned_init")
        
        print(f"\nInit state files:")
        print(f"  .init file: {'✓' if os.path.exists(init_file) else '✗'} {init_file}")
        print(f"  .pruned_init file: {'✓' if os.path.exists(pruned_init_file) else '✗'} {pruned_init_file}")
        
        # Load and compare if both exist
        if os.path.exists(init_file) and os.path.exists(pruned_init_file):
            try:
                init_state = torch.load(init_file)
                pruned_init_state = torch.load(pruned_init_file)
                
                print(f"\nInit state analysis:")
                print(f"  Original shape: {init_state.shape if hasattr(init_state, 'shape') else type(init_state)}")
                print(f"  Pruned shape: {pruned_init_state.shape if hasattr(pruned_init_state, 'shape') else type(pruned_init_state)}")
                
                is_identical = self.are_states_identical(init_state, pruned_init_state)
                print(f"  States identical: {colored('Yes' if is_identical else 'No', 'green' if is_identical else 'red')}")
                
            except Exception as e:
                print(f"  {colored(f'Error loading states: {e}', 'red')}")
        
        # Check demo data
        try:
            demo_file_path = os.path.join(self.datasets_path, self.benchmark_instance.get_task_demonstration(task_id))
            print(f"\nDemo file: {'✓' if os.path.exists(demo_file_path) else '✗'} {demo_file_path}")
            
            if os.path.exists(demo_file_path):
                with h5py.File(demo_file_path, "r") as f:
                    demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
                    print(f"  Number of demos: {len(demo_keys)}")
                    
                    if demo_keys:
                        demo_data = f["data"][demo_keys[0]]
                        print(f"  Demo structure:")
                        for key in demo_data.keys():
                            data = demo_data[key]
                            if hasattr(data, 'shape'):
                                print(f"    {key}: {data.shape}")
                            else:
                                print(f"    {key}: {type(data)}")
        
        except Exception as e:
            print(f"  {colored(f'Error checking demo: {e}', 'red')}")


def main():
    parser = argparse.ArgumentParser(description='LIBERO Debug Tool - Unified debugging for LIBERO benchmarks')
    
    # Basic options
    parser.add_argument('--benchmark', type=str, default='libero_object',
                       choices=['libero_object', 'libero_goal', 'libero_spatial', 'libero_10', 'libero_90', 'libero_object_single'],
                       help='Benchmark to debug (default: libero_object)')
    
    parser.add_argument('--task-id', type=int, nargs='+', 
                       help='Specific task ID(s) to analyze (default: all tasks)')
    
    # Analysis modes
    parser.add_argument('--overview', action='store_true',
                       help='Show benchmark overview')
    
    parser.add_argument('--init-states', action='store_true',
                       help='Inspect init states via API')
    
    parser.add_argument('--compare-files', action='store_true',
                       help='Compare .init and .pruned_init files')
    
    parser.add_argument('--check-demos', action='store_true',
                       help='Check demonstration data')
    
    parser.add_argument('--detailed', type=int, metavar='TASK_ID',
                       help='Perform detailed analysis of specific task')
    
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses (equivalent to --overview --init-states --compare-files --check-demos)')
    
    args = parser.parse_args()
    
    # Initialize debug tool
    try:
        debug_tool = LiberoDebugTool(args.benchmark)
    except Exception as e:
        print(f"{colored(f'Error initializing debug tool: {e}', 'red')}")
        return
    
    # If no specific analysis is requested, show help
    if not any([args.overview, args.init_states, args.compare_files, args.check_demos, args.detailed is not None, args.all]):
        parser.print_help()
        return
    
    # Run analyses
    try:
        if args.all or args.overview:
            debug_tool.show_benchmark_overview()
        
        if args.all or args.init_states:
            debug_tool.inspect_init_states_via_api(args.task_id)
        
        if args.all or args.compare_files:
            debug_tool.compare_init_files(args.task_id)
        
        if args.all or args.check_demos:
            debug_tool.check_task_demos(args.task_id)
        
        if args.detailed is not None:
            debug_tool.detailed_task_analysis(args.detailed)
    
    except KeyboardInterrupt:
        print(f"\n{colored('Analysis interrupted by user', 'yellow')}")
    except Exception as e:
        print(f"{colored(f'Error during analysis: {e}', 'red')}")


if __name__ == "__main__":
    main()
