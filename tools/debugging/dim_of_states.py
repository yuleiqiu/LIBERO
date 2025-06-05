from libero.libero import benchmark
from libero.libero.envs.env_wrapper import DemoRenderEnv
import os
# import init_path
from libero.libero import benchmark, get_libero_path
import cv2
from datetime import datetime
import h5py

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_object" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
for task_id in range(task_suite.n_tasks):
    task = task_suite.get_task(task_id)
    task_name = task.name
    print("Current task name is:", task_name)
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    datasets_default_path = get_libero_path("datasets")
    demo_file = os.path.join(datasets_default_path, task_suite.get_task_demonstration(task_id))
    data = h5py.File(demo_file)["data"]
    # for demo_idx in range(50):
    #     demo_key = f"demo_{demo_idx}"
    #     demo = data[demo_key]
    #     states = demo["states"]
    #     print(f"The states shape of {demo_key} is: {states.shape}")
    demo = data["demo_0"]
    states = demo["states"]
    state = states[0]
    print(state.shape)
    exit(0)