from libero.libero import benchmark, get_libero_path
from libero.libero.utils.dataset_utils import get_dataset_info
import os
from termcolor import colored

import pprint

pp = pprint.PrettyPrinter(indent=2)

# 1. Check default paths
benchmark_root_path = get_libero_path("benchmark_root")
init_states_default_path = get_libero_path("init_states")
datasets_default_path = get_libero_path("datasets")
bddl_files_default_path = get_libero_path("bddl_files")
print("Default benchmark root path: ", benchmark_root_path)
print("Default dataset root path: ", datasets_default_path)
print("Default bddl files root path: ", bddl_files_default_path)
print("============================================================")

# 2. Check becnhmark lists
# Get a dictionary of mapping from benchmark name to benchmark class
benchmark_dict = benchmark.get_benchmark_dict()
pp.pprint(benchmark_dict)
print("============================================================")

# 3. Check benchmark instance
# initialize a benchmark
benchmark_instance = benchmark_dict["libero_object"]()
num_tasks = benchmark_instance.get_num_tasks()
# see how many tasks involved in the benchmark
print(f"{num_tasks} tasks in the benchmark {benchmark_instance.name}: ")
# Check if all the task names and their bddl file names
task_names = benchmark_instance.get_task_names()
print("The benchmark contains the following tasks:")
for i in range(num_tasks):
    task_name = task_names[i]
    task = benchmark_instance.get_task(i)
    bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
    print(f"\t {task_name}")
    if not os.path.exists(bddl_file):
        print(colored(f"[error] bddl file {bddl_file} cannot be found. Check your paths", "red"))
print("============================================================")
task = benchmark_instance.get_task(4)
pp.pprint(task)
print("============================================================")
print(f"An example of init file is named like this: {task.init_states_file}")
init_states = benchmark_instance.get_task_init_states(4)
pp.pprint(f"Init states shape: {init_states.shape}")
print("============================================================")
demo_files_path = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(4))
get_dataset_info(demo_files_path)
print("============================================================")

# for i in range(num_tasks):
#     task_name = task_names[i]
#     task = benchmark_instance.get_task(i)
#     init_states_path = os.path.join(init_states_default_path, task.problem_folder, task.init_states_file)
#     if not os.path.exists(init_states_path):
#         print(colored(f"[error] the init states {init_states_path} cannot be found. Check your paths", "red"))
# print(f"An example of init file is named like this: {task.init_states_file}")
# # Load torch init files
# init_states = benchmark_instance.get_task_init_states(0)
# # Init states in the same (num_init_rollouts, num_simulation_states)
# print(init_states.shape)
# print("============================================================")
# # Check if the demo files exist
# demo_files_path = [os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i)) for i in range(num_tasks)]
# for demo_file in demo_files_path:
#     if not os.path.exists(demo_file):
#         print(colored(f"[error] demo file {demo_file} cannot be found. Check your paths", "red"))
# pp.pprint(demo_files_path)
# print("============================================================")
# example_demo_file = demo_files_path[8]
# # Print the dataset info. We have a standalone script for doing the same thing available at `scripts/get_dataset_info.py`
# get_dataset_info(example_demo_file)