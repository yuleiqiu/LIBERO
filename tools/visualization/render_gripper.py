
import argparse
import os
import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path


def main():
    parser = argparse.ArgumentParser(description="Visualize regions from a BDDL file in its environment.")
    parser.add_argument("--task-suite-name", type=str, required=True, help="Name of the task suite.")
    parser.add_argument("--task-id", type=int, required=True, help="ID of the task to visualize.")
    # parser.add_argument("--output-path", type=str, default="initialization_sampler.mp4", help="Path to save the visualization image.")
    args = parser.parse_args()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = args.task_suite_name
    task_id = args.task_id

    try:
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"Visualizing task: {task.name} from {task_bddl_file}")
    except KeyError:
        print(f"Could not find task suite or task with ID {task_id}.")
        return

    # Setup environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 512,
        "camera_widths": 512,
    }
    env = OffScreenRenderEnv(**env_args)
    for _ in range(10):
        obs, _, _, _ = env.step([0.] * 7)  # Don't advance physics significantly, just allow settling
    # import pprint
    # pprint.pprint(env.env.robots[0].__dict__)

    # for k, v in obs.items():
    #     print(f"{k}: {v}")
    image = obs["agentview_image"][::-1]
    imageio.imwrite("agentview_image.png", image)
    print(obs["basket_1_pos"])
    
    env.close()

if __name__ == "__main__":
    main()
