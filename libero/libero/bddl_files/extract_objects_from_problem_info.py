"""
extract_objects_from_problem_info.py
This script parses BDDL files to extract and display information about objects in the problem.
It is designed to work with the LIBERO benchmark and can be used to examine tasks and their associated objects.
"""

import os
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

import pprint
import numpy as np
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract problem info from BDDL files to show objects in the task."
    )
    parser.add_argument(
        "--benchmark-name", 
        type=str, 
        default="libero_object", 
        help="Benchmark name to use (default: libero_object)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=10,
        help="Number of tasks in the benchmark (default: 10)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    benchmark_name = args.benchmark_name
    num_tasks = args.num_tasks

    # Initialize a benchmark (e.g., LIBERO_OBJECT)
    benchmark = get_benchmark(benchmark_name)()
    print(f"Examining {benchmark_name} benchmark with {benchmark.get_num_tasks()} tasks")
    print("-" * 80)
    bddl_files_default_path = get_libero_path("bddl_files")

    for task_id in range(num_tasks):
        print(f"Task ID: {task_id}")
        # print("-" * 80)
        # Get the task from the benchmark
        task = benchmark.get_task(task_id)
        bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
        # print(f"BDDL file: {bddl_file}")
        # print("-" * 80)
        parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file)
        # print("Parsed Problem:")
        # pprint.pprint(parsed_problem)

        objects_info = parsed_problem.get("objects", {})
        # pprint.pprint(objects_info)
        # print("-" * 80)

        object_list = list(objects_info.keys())
        print(f"Objects in the problem: {object_list}")
        print("-" * 80)

    # # write the object names to a txt file
    # output_file = f"{benchmark_name}_task{task_id}_objects.txt"
    # with open(output_file, "w") as f:
    #     for obj in object_list:
    #         f.write(obj + "\n")
    # print(f"Objects list saved to {output_file}")

if __name__ == "__main__":
    main()