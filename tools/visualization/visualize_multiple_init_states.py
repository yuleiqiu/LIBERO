import argparse
import os
import numpy as np
import imageio
from PIL import Image

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path


def main():
    parser = argparse.ArgumentParser(description="Visualize multiple initial states of an environment in a grid.")
    parser.add_argument("--task-suite-name", type=str, required=True, help="Name of the task suite.")
    parser.add_argument("--task-id", type=int, required=True, help="ID of the task to visualize.")
    parser.add_argument("--output-path", type=str, default="multiple_init_states_grid.png", help="Path to save the composite visualization image.")
    parser.add_argument("--camera-name", type=str, default="agentview", help="Camera to render from.")
    parser.add_argument("--num-resets", type=int, default=9, help="Number of environment resets to visualize (should be a perfect square for grid layout).")
    parser.add_argument("--settle-steps", type=int, default=10, help="Number of empty steps to allow objects to settle.")
    args = parser.parse_args()

    # Ensure num_resets is a perfect square for grid layout
    grid_size = int(np.sqrt(args.num_resets))
    num_resets = grid_size * grid_size

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
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    
    frames = []
    
    # Visualize multiple initial states
    for i in range(num_resets):
        print(f"Processing reset {i+1}/{num_resets}")
        obs = env.reset()
        
        # Allow objects to settle with empty actions
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step([0.] * 7)  # Don't advance physics significantly, just allow settling
        
        # Record the current frame
        frame = obs[f"{args.camera_name}_image"][::-1].copy()
        frames.append(frame)
        print(f"Recorded frame for reset {i+1}")
    
    # Create a composite image
    if frames:
        # Calculate grid dimensions
        grid_rows = grid_size
        grid_cols = grid_size
        
        # Create a blank image for the composite
        single_image_height, single_image_width = frames[0].shape[:2]
        composite_width = grid_cols * single_image_width
        composite_height = grid_rows * single_image_height
        composite_image = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        # Place each frame in the composite image
        for i, frame in enumerate(frames):
            row = i // grid_cols
            col = i % grid_cols
            y_start = row * single_image_height
            y_end = y_start + single_image_height
            x_start = col * single_image_width
            x_end = x_start + single_image_width
            composite_image[y_start:y_end, x_start:x_end] = frame
        
        # Save the composite image
        imageio.imwrite(args.output_path, composite_image)
        print(f"Composite visualization saved to {args.output_path}")
    else:
        print("No frames were recorded.")
    
    env.close()

if __name__ == "__main__":
    main()