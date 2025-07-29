import os
import numpy as np
import imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

task_suite_name = "libero_object_grid" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark.get_benchmark(task_suite_name)()

# retrieve a specific task
for task_id in range(task_suite.n_tasks):
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_widths": 512,
        "camera_heights": 512,
    }
    env = OffScreenRenderEnv(**env_args)
    env.reset()

    for _ in range(10):
        obs, _, _, _ = env.step([0.] * 7)  # Don't advance physics, just get state

    sim = env.env.sim
    model = sim.model
    data = sim.data

    obj_of_interest = env.env.obj_of_interest.copy()
    # Remove 'basket_1' from the list of objects of interest
    destination_name = obj_of_interest[1] + "_main"  # Assuming the second object is the destination
    # obj_of_interest = [item for item in obj_of_interest if item != destination_name]
    target_object_name = obj_of_interest[0] + "_main"  # Assuming the first object is the target

    target_object_id = model.body_name2id(target_object_name)
    destination_id = model.body_name2id(destination_name)

    # Video recording setup
    video_path = "task_visualization.mp4"
    video_writer = imageio.get_writer(video_path, fps=24)

    # Control loop
    state = "MOVE_ABOVE_TARGET_OBJECT"
    open_timer = 0
    grasp_timer = 0
    release_timer = 0
    open_duration = 10  # Number of steps to keep gripper open
    grasp_duration = 10  # Number of steps to keep gripper closed for grasping
    release_duration = 10  # Number of steps to keep gripper open for releasing

    lift_start_pos = None
    noise_scale = 0.02 # Control the magnitude of the noise

    for _ in range(500):
        eef_pos = env.env._eef_xpos.copy()
        target_pos = data.body_xpos[target_object_id].copy()
        destination_pos = data.body_xpos[destination_id].copy()

        # print(f"Current state: {state}, EEF position: {eef_pos}, Target position: {target_pos}, Plate position: {plate_pos}")
        # print(f"{lift_start_pos=}")

        action = np.zeros(7)
        # State machine logic
        if state == "MOVE_ABOVE_TARGET_OBJECT":
            target_eef_pos = target_pos + np.array([0, 0, 0.25])
            target_eef_pos += np.random.normal(0, noise_scale, size=target_eef_pos.shape)
            if np.linalg.norm(eef_pos - target_eef_pos) < 0.01:
                state = "OPEN_GRIPPER"
                continue
        elif state == "OPEN_GRIPPER":
            action[-1] = -1  # Open gripper
            if open_timer < open_duration:
                open_timer += 1
            else:
                open_timer = 0
                state = "LOWER_TO_TARGET_OBJECT"
                continue
        elif state == "LOWER_TO_TARGET_OBJECT":
            target_eef_pos = target_pos + np.array([0, 0, 0.01])
            # target_eef_pos += np.random.normal(0, noise_scale, size=target_eef_pos.shape)
            if np.linalg.norm(eef_pos - target_eef_pos) < 0.01:
                state = "GRASP_TARGET_OBJECT"
                continue
        elif state == "GRASP_TARGET_OBJECT":
            action[-1] = 1  # Close gripper
            if grasp_timer < grasp_duration:
                grasp_timer += 1
            else:
                grasp_timer = 0
                # lift_start_pos = obs[f"{target_object_name}_pos"].copy()
                lift_start_pos = data.body_xpos[target_object_id].copy()
                state = "LIFT_TARGET_OBJECT"
                continue
        elif state == "LIFT_TARGET_OBJECT":
            target_eef_pos = lift_start_pos + np.array([0, 0, 0.3])
            target_eef_pos += np.random.normal(0, noise_scale, size=target_eef_pos.shape)
            if np.linalg.norm(eef_pos - target_eef_pos) < 0.02:
                state = "MOVE_ABOVE_DESTINATION"
                continue
        elif state == "MOVE_ABOVE_DESTINATION":
            target_eef_pos = destination_pos + np.array([0, 0, 0.3])
            target_eef_pos += np.random.normal(0, noise_scale, size=target_eef_pos.shape)
            if np.linalg.norm(eef_pos - target_eef_pos) < 0.02:
                state = "LOWER_TO_DESTINATION"
                continue
        elif state == "LOWER_TO_DESTINATION":
            target_eef_pos = destination_pos + np.array([0, 0, 0.1])
            target_eef_pos += np.random.normal(0, noise_scale, size=target_eef_pos.shape)
            if np.linalg.norm(eef_pos - target_eef_pos) < 0.02:
                state = "RELEASE_TARGET_OBJECT"
                continue
        elif state == "RELEASE_TARGET_OBJECT":
            action[-1] = -1  # Open gripper
            if release_timer < release_duration:
                release_timer += 1
            else:
                release_timer = 0
                state = "RETREAT"
                continue
        elif state == "RETREAT":
            target_eef_pos = eef_pos + np.array([0, 0, 0.3])
            target_eef_pos += np.random.normal(0, noise_scale, size=target_eef_pos.shape)
            if np.linalg.norm(eef_pos - target_eef_pos) < 0.02:
                break

        # Controller action
        error = target_eef_pos - eef_pos
        action[:3] = error * 5
        action[3:6] = 0 # Maintain orientation

        obs, _, success, _ = env.step(action)

        # Record frame
        frame = obs["agentview_image"][::-1]
        video_writer.append_data(frame)

        if success:
            print("Task completed successfully!")
            break

    video_writer.close()
    env.close()
    print(f"Video saved to {video_path}")
    break