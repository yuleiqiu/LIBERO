import os
import numpy as np
import imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, gripper

    def __call__(self, target_pos, destination_pos, current_eef_pos):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(target_pos, destination_pos)

        # obtain waypoints
        if self.trajectory[0]['t'] == self.step_count:
            print(self.trajectory)
            self.curr_waypoint = self.trajectory.pop(0)
        
        # Check if trajectory is complete
        if len(self.trajectory) == 0:
            # Return zero delta (stay in place) and maintain last gripper state
            xyz = np.zeros(3)
            gripper = self.curr_waypoint['gripper']
        else:
            next_waypoint = self.trajectory[0]
            # interpolate between waypoints to obtain current pose and gripper command
            target_xyz, gripper = self.interpolate(self.curr_waypoint, next_waypoint, self.step_count)
            # Inject noise
            if self.inject_noise:
                scale = 0.02
                target_xyz += np.random.uniform(-scale, scale, target_xyz.shape)
            # Calculate delta (dx, dy, dz) from current position to target position
            xyz = target_xyz - current_eef_pos

        # action = np.concatenate([xyz, [gripper]])

        self.step_count += 1
        return xyz, gripper

class PickAndPlacePolicy(BasePolicy):
    def generate_trajectory(self, target_pos, destination_pos):
        self.trajectory = [
            {"t": 0, "xyz": target_pos + np.array([0, 0, 0.2]), "gripper": -1}, # approach target object
            # {"t": 90, "xyz": target_pos + np.array([0, 0, 0.15]), "gripper": -1}, # approach target object
            {"t": 130, "xyz": target_pos + np.array([0, 0, -0.01]), "gripper": -1}, # go down
            {"t": 170, "xyz": target_pos + np.array([0, 0, -0.01]), "gripper": 1}, # close gripper
            {"t": 200, "xyz": target_pos + np.array([0, 0, 0.2]), "gripper": 1}, # move up
            {"t": 230, "xyz": destination_pos + np.array([0, 0, 0.25]), "gripper": 1}, # move to basket
            {"t": 310, "xyz": destination_pos + np.array([0, 0, 0.1]), "gripper": 1}, # go down
            {"t": 340, "xyz": destination_pos + np.array([0, 0, 0.1]), "gripper": -1}, # open gripper
            {"t": 370, "xyz": destination_pos + np.array([0, 0, 0.3]), "gripper": -1}, # move up
            {"t": 400, "xyz": destination_pos + np.array([0, 0, 0.3]), "gripper": -1}, # stay
        ]

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

    # self: env.env
    # env.env._gripper_to_target(
    #     gripper=env.env.robots[0].gripper, target=env.env.cube.root_body, target_type="body"
    # )
    # self._gripper_to_target(
    # dist = self._gripper_to_target(
    #     gripper=self.robots[0].gripper, target=self.cube.root_body, target_type="body", return_distance=True
    # )

    target_pos = data.body_xpos[target_object_id].copy()
    destination_pos = data.body_xpos[destination_id].copy()

    policy = PickAndPlacePolicy(inject_noise=True)

    success = False
    continue_recording = True
    post_success_steps = 0
    post_success_duration = 40  # Continue recording for 40 steps after success

    for _ in range(400):
        # Get current end-effector position for delta calculation
        current_eef_pos = env.env._eef_xpos.copy()

        if not success:
            # Get delta action from policy
            xyz_delta, gripper = policy(target_pos, destination_pos, current_eef_pos)
            
            # Create full action vector (dx, dy, dz, rx, ry, rz, gripper)
            action = np.zeros(7)
            action[:3] = xyz_delta*5  # Position deltas
            action[3:6] = 0  # Orientation deltas (maintain current orientation)
            action[6] = gripper  # Gripper command
        else:
            # If success, move up and keep the action as zero to maintain position
            action = np.zeros(7)
            action[2] += 0.1  # Move up slightly
            action[6] = -1  # Keep gripper open

        obs, _, success, _ = env.step(action)

        # Record frame
        frame = obs["agentview_image"][::-1]
        video_writer.append_data(frame)

        if success and continue_recording:
            post_success_steps += 1
            if post_success_steps >= post_success_duration:
                print(f"Task completed successfully!")
                print(f"Recording completed after {post_success_duration} additional steps.")
                break

    video_writer.close()
    env.close()
    print(f"Video saved to {video_path}")
    break

