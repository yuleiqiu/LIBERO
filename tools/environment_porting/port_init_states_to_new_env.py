from libero.libero import benchmark
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
import os
# import init_path
from libero.libero import benchmark, get_libero_path
import cv2
from datetime import datetime
import h5py
import numpy as np
import copy


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_object" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()
bddl_files_default_path = get_libero_path("bddl_files")

# retrieve a specific task
task_id = 0 # pick alphabet soup and place it in the basket
task = task_suite.get_task(task_id)
task_name = task.name
print("Current task name is:", task_name)
task_description = task.language
task_bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
    f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# prepare the states from demo
datasets_default_path = get_libero_path("datasets")
demo_file = os.path.join(datasets_default_path, task_suite.get_task_demonstration(task_id))
data = h5py.File(demo_file)["data"]

demo_idx = 0 # select the first demo
demo_key = f"demo_{demo_idx}"
demo = data[demo_key]  # updated to use demo_key
states = demo["states"]
state = states[0] # the first state
print("the states shape is:", states.shape)

env_old_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env_old = OffScreenRenderEnv(**env_old_args)
# env_old.seed(0)
env_old.reset()
env_old.set_init_state(state)

for _ in range(5):
    obs, _, _, _ = env_old.step([0.] * 7)

# Save the original state image
img_in_old_env = obs["agentview_image"][::-1]

env_old_sim_states = env_old.get_sim_state()
assert env_old_sim_states.shape == state.shape, "The sim state shape should not match the original state shape."

# object_list = ["salad_dressing", "basket", "ketchup", "alphabet_soup", "cream_cheese", "milk", "tomato_sauce"]
target_object = "alphabet_soup"

DOF = 0
for body_name in env_old.env.sim.model.body_names:
    print(f"Body name in old env: {body_name}")
    b_id = env_old.env.sim.model.body_name2id(body_name)
    j_id = env_old.env.sim.model.body_jntadr[b_id]
    joint_type = env_old.env.sim.model.jnt_type[j_id]
    print(f"{joint_type=}")
    if joint_type == 0:
        DOF += 7;
    elif joint_type == 1:
        DOF += 4;
    elif joint_type == 2:
        DOF += 1;
    elif joint_type == 3:
        DOF += 1;
print(f"Total DOF in the environment: {DOF}")
assert DOF == env_old_sim_states.shape[0], "The DOF should match the sim state shape."
exit(0)

# body_slices = {}
for body_name in env_old.env.sim.model.body_names:
    print(f"Body name in old env: {body_name}")
    if target_object not in body_name:
        # print(f"Skipping object: {body_name}")
        continue
    else:
        print(f"Found target object: {body_name}")

        # find the first joint attached to that body
        b_id = env_old.env.sim.model.body_name2id(body_name)
        j_id = env_old.env.sim.model.body_jntadr[b_id]
        joint_type = env_old.env.sim.model.jnt_type[j_id]
        # FREE: 0; BALL: 1; SLIDE: 2; HINGE: 3
        print(f"{joint_type=}")
        # joint→qpos address and num of qpos entries:
        qpos_adr = env_old.env.sim.model.jnt_qposadr[j_id]
        continue

# num of position coords = depends on joint type (free=7, ball=4, hinge=1…)
states_of_target_object_from_demo = copy.deepcopy(state[qpos_adr:qpos_adr+7])
env_old.close()



# Create a new environment from new args with single object
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
tmp_bddl_dir = os.path.join(parent_dir, "get_started", "tmp", "pddl_files")

bddl_file_names = [
    os.path.join(tmp_bddl_dir, "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_1.bddl"),
    os.path.join(tmp_bddl_dir, "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_2.bddl"),
]

env_new_args = {
    "bddl_file_name": bddl_file_names[1],
    "camera_heights": 128,
    "camera_widths": 128
}

env_new = OffScreenRenderEnv(**env_new_args)
# env_new.seed(0)
env_new.reset()
for _ in range(5):
    obs, _, _, _ = env_new.step([0.] * 7)
env_new_sim_states = env_new.get_sim_state()
print(f"{env_new_sim_states.shape=}")

print("Initialized new environment with the target object's state.")

for body_name in env_new.env.sim.model.body_names:
    print(f"Body name in new env: {body_name}")
    if target_object not in body_name:
        # print(f"Skipping object: {body_name}")
        continue
    else:
        print(f"Found target object: {body_name}")

        # find the first joint attached to that body
        b_id = env_new.env.sim.model.body_name2id(body_name)
        j_id = env_new.env.sim.model.body_jntadr[b_id]
        # joint_type = env_new.env.sim.model.jnt_type[j_id]
        # print(joint_type)
        # joint→qpos address and num of qpos entries:
        qpos_adr = env_new.env.sim.model.jnt_qposadr[j_id]
        continue

env_new_sim_states[qpos_adr:qpos_adr+7] = states_of_target_object_from_demo
env_new.set_init_state(env_new_sim_states)
for _ in range(5):
    obs, _, _, _ = env_new.step([0.] * 7)
# Save the modified state image
img_in_new_env = obs["agentview_image"][::-1]
env_new.close()

# Create a comparison image
import matplotlib.pyplot as plt
from datetime import datetime

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Display the original state image
ax1.imshow(img_in_old_env)
ax1.set_title("Object in Original Environment")
ax1.axis("off")

# Display the modified state image
ax2.imshow(img_in_new_env)
ax2.set_title("Object in New Environment")
ax2.axis("off")

# Add a main title
plt.suptitle(f"Comparison of Environment States for '{target_object}'")

# Display the plot instead of saving
plt.tight_layout()
plt.show()

# # Save the figure
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# save_path = f"state_comparison_{target_object}_{timestamp}.png"
# plt.savefig(save_path)
# print(f"Comparison image saved to {save_path}")
# plt.close()
