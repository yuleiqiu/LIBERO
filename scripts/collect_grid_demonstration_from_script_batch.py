import argparse
import datetime
import h5py
import json
import numpy as np
import os
import shutil
import time
from typing import List
from glob import glob
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None

    def generate_trajectory(self):
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
            # print(self.trajectory)
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
            {"t": 120, "xyz": target_pos + np.array([0, 0, 0.04]), "gripper": -1}, # go down
            {"t": 170, "xyz": target_pos + np.array([0, 0, 0.04]), "gripper": 1}, # close gripper
            {"t": 200, "xyz": target_pos + np.array([0, 0, 0.2]), "gripper": 1}, # move up
            {"t": 250, "xyz": destination_pos + np.array([0, 0, 0.25]), "gripper": 1}, # move to basket
            {"t": 320, "xyz": destination_pos + np.array([0, 0, 0.1]), "gripper": 1}, # go down
            {"t": 370, "xyz": destination_pos + np.array([0, 0, 0.1]), "gripper": -1}, # open gripper
            {"t": 380, "xyz": destination_pos + np.array([0, 0, 0.2]), "gripper": -1}, # move up
            {"t": 400, "xyz": destination_pos + np.array([0, 0, 0.2]), "gripper": -1}, # stay
        ]


def collect_scripted_trajectory(env, remove_directory=[]):
    """
    Use a scripted policy to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.
    Args:
        env (MujocoEnv): environment to control
    """

    reset_success = False
    while not reset_success:
        try:
            obs = env.reset()
            reset_success = True
        except:
            continue
    env.render()

    # Retrieve target object and destination object names
    obj_of_interest = env.env.obj_of_interest.copy()
    target_object_name = obj_of_interest[0]
    destination_name = obj_of_interest[-1]

    # Get target and destination positions
    target_pos = obs[f"{target_object_name.replace('_main', '')}_pos"].copy()
    destination_pos = obs[f"{destination_name.replace('_main', '')}_pos"].copy()

    policy = PickAndPlacePolicy(inject_noise=True)

    success = False
    task_completion_hold_count = -1  # counter to collect 40 timesteps after reaching goal
    task_completed = False

    for _ in range(400):
        # Get current state
        current_eef_pos = obs["robot0_eef_pos"].copy()

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
            action[2] = 0.1  # Move up slightly
            action[6] = -1  # Keep gripper open

        # Run environment step
        obs, _, success, _ = env.step(action)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            task_completed = True
            break

        # state machine to check for having a success for 40 consecutive timesteps
        if success:
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 40  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success
    else:
        # This block executes if the for loop completes without a break, which means a timeout.
        print("Trajectory collection timed out...")

    # cleanup for end of data collection episodes
    if not task_completed:
        remove_directory.append(env.ep_directory.split("/")[-1])

    env.close()

    return task_completed


def gather_demonstrations_as_hdf5(
    directory, out_dir, env_info, problem_info, bddl_file, remove_directory=[]
):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    The strucure of the hdf5 file is as follows.
    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected
        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
        demo2 (group)
        ...
    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = bddl_file
    grp.attrs["bddl_file_content"] = str(open(bddl_file, "r", encoding="utf-8"))

    f.close()


def find_bddl_files(bddl_directory: str) -> List[str]:
    """
    Find all BDDL files in the specified directory.
    
    Args:
        bddl_directory (str): BDDL file directory path
        
    Returns:
        list: BDDL file paths
    """
    if not os.path.exists(bddl_directory):
        print(f"Error: Directory {bddl_directory} does not exist")
        return []

    # Recursively find all .bddl files
    bddl_files = glob(os.path.join(bddl_directory, "**/*.bddl"), recursive=True)
    
    if not bddl_files:
        print(f"Warning: No .bddl files found in directory {bddl_directory}")
        return []

    print(f"Found {len(bddl_files)} BDDL files:")
    for i, file in enumerate(bddl_files, 1):
        print(f"  {i}. {os.path.relpath(file, bddl_directory)}")

    return sorted(bddl_files)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser("Batch collect demonstration data from BDDL files")
    parser.add_argument(
        "--demo-dir",
        type=str,
        default="demonstration_data",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default=["Panda"],
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=10,
        help="Number of demonstrations to collect",
    )
    parser.add_argument(
        "--bddl-dir",
        type=str,
        required=True,
        help="Path to the directory containing BDDL files"
    )

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    bddl_files = find_bddl_files(args.bddl_dir)
    skipped_list = []
    for bddl_file in bddl_files:
        assert os.path.exists(bddl_file)
        problem_info = BDDLUtils.get_problem_info(bddl_file)
        task_name = bddl_file.split("/")[-1].split(".")[0]

        # Create environment
        problem_name = problem_info["problem_name"]
        domain_name = problem_info["domain_name"]
        language_instruction = problem_info["language_instruction"]
        if "TwoArm" in problem_name:
            config["env_configuration"] = args.config
        print(language_instruction)
        env = TASK_MAPPING[problem_name](
            bddl_file_name=bddl_file,
            **config,
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=args.camera,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
        )

        # Wrap this with visualization wrapper
        env = VisualizationWrapper(env)

        # Grab reference to controller config and convert it to json-encoded string
        env_info = json.dumps(config)

        # wrap the environment with data collection wrapper
        tmp_directory = args.demo_dir + "/tmp/{}_ln_{}/{}".format(
            problem_name,
            language_instruction.replace(" ", "_").strip('""'),
            str(time.time()).replace(".", "_"),
        )

        env = DataCollectionWrapper(env, tmp_directory)

        # make a new timestamped directory
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir = os.path.join(
            args.demo_dir,
            f"{problem_name}_{current_time}_"
            + bddl_file.split("/")[-1].split(".")[0],
        )

        os.makedirs(new_dir)

        # collect demonstrations
        remove_directory = []
        timeout_count = 0
        i = 0
        while i < args.num_demonstration:
            if timeout_count > 5:
                print(f"Too many timeouts, stopping collection for {task_name}...")
                skipped_list.append(bddl_file)
                break
                
            print(f"Collecting demonstration of task {task_name}: {i+1}/{args.num_demonstration}...")
            saving = collect_scripted_trajectory(env, remove_directory)
            if saving:
                gather_demonstrations_as_hdf5(
                    tmp_directory, new_dir, env_info, args, remove_directory
                )
                i += 1
                timeout_count = 0  # Reset timeout count on successful collection
            else:
                timeout_count += 1
                print(f"Timeout count: {timeout_count}/5")

    # Save environments that are skipped due to timeouts
    if skipped_list:
        skipped_file = os.path.join(args.demo_dir, "skipped_tasks.txt")
        with open(skipped_file, "w") as f:
            for item in skipped_list:
                f.write(f"{item}\n")
        print(f"Skipped tasks saved to {skipped_file}")