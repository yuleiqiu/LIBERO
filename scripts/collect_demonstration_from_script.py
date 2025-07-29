import argparse
import cv2
import datetime
import h5py
import init_path
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *


def collect_scripted_trajectory(env, problem_info, remove_directory=[]):
    """
    Use a scripted policy to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.
    Args:
        env (MujocoEnv): environment to control
        problem_info (dict): dictionary containing with problem information
    """

    reset_success = False
    while not reset_success:
        try:
            obs = env.reset()
            reset_success = True
        except:
            continue

    # ID = 2 always corresponds to agentview
    env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal

    obj_of_interest = env.env.obj_of_interest.copy()
    target_object_name = obj_of_interest[0]
    destination_name = obj_of_interest[-1]

    # Control loop
    state = "MOVE_ABOVE_TARGET_OBJECT"
    open_timer = 0
    grasp_timer = 0
    release_timer = 0
    open_duration = 10  # Number of steps to keep gripper open
    grasp_duration = 10  # Number of steps to keep gripper closed for grasping
    release_duration = 10  # Number of steps to keep gripper open for releasing

    lift_start_pos = None
    noise_scale = 0.01 # Control the magnitude of the noise

    task_completed = False
    for _ in range(500):
        # Get current state
        eef_pos = obs["robot0_eef_pos"].copy()
        target_pos = obs[f"{target_object_name.replace('_main', '')}_pos"].copy()
        # print(f"Current state: {state}, EEF position: {eef_pos}, Target position: {target_pos}")
        destination_pos = obs[f"{destination_name.replace('_main', '')}_pos"].copy()

        # print(f"Current state: {state}, EEF position: {eef_pos}, Target position: {target_pos}, Destination position: {destination_pos}")

        action = np.zeros(7)
        # State machine logic
        if state == "MOVE_ABOVE_TARGET_OBJECT":
            target_eef_pos = target_pos + np.array([0, 0, 0.2])
            target_eef_pos += np.random.normal(0, noise_scale, size=target_eef_pos.shape)
            if np.linalg.norm(eef_pos - target_eef_pos) < 0.02:
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
            target_eef_pos += np.random.normal(0, noise_scale, size=target_eef_pos.shape)
            if np.linalg.norm(eef_pos - target_eef_pos) < 0.02:
                state = "GRASP_TARGET_OBJECT"
                continue
        elif state == "GRASP_TARGET_OBJECT":
            action[-1] = 1  # Close gripper
            if grasp_timer < grasp_duration:
                grasp_timer += 1
            else:
                grasp_timer = 0
                lift_start_pos = obs[f"{target_object_name}_pos"].copy()
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
        action[:3] = error * 10
        action[3:6] = 0 # Maintain orientation

        # Run environment step
        obs, _, _, _ = env.step(action)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            task_completed = True
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success
    else:
        # This block executes if the for loop completes without a break, which means a timeout.
        print("Trajectory collection timed out, retrying...")

    # cleanup for end of data collection episodes
    if not task_completed:
        remove_directory.append(env.ep_directory.split("/")[-1])
    env.close()
    return task_completed


def gather_demonstrations_as_hdf5(
    directory, out_dir, env_info, args, remove_directory=[]
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
    grp.attrs["bddl_file_name"] = args.bddl_file
    grp.attrs["bddl_file_content"] = str(open(args.bddl_file, "r", encoding="utf-8"))

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
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
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--bddl-file", type=str)

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    # Check if we're using a multi-armed environment and use env_configuration argument if so

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    print(language_instruction)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
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
    tmp_directory = "demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )

    env = DataCollectionWrapper(env, tmp_directory)

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(
        args.directory,
        f"{domain_name}_ln_{problem_name}_{t1}_{t2}_"
        + language_instruction.replace(" ", "_").strip('""'),
    )

    os.makedirs(new_dir)

    # collect demonstrations

    remove_directory = []
    i = 0
    while i < args.num_demonstration:
        print(i)
        saving = collect_scripted_trajectory(
            env, problem_info, remove_directory
        )
        if saving:
            gather_demonstrations_as_hdf5(
                tmp_directory, new_dir, env_info, args, remove_directory
            )
            i += 1
