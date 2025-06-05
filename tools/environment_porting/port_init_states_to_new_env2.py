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

def copy_body_state(src_env, dst_env, body_name):
    # === helpers ===
    def joint_sizes(jtype):
        # free(0) ball(1) slide(2) hinge(3)
        q = {0: 7, 1: 4, 2: 1, 3: 1}[jtype]
        v = {0: 6, 1: 3, 2: 1, 3: 1}[jtype]
        return q, v

    # --- 源环境索引 ---
    m_src = src_env.env.sim.model
    b_src = m_src.body_name2id(body_name)
    j_src = m_src.body_jntadr[b_src]
    nq_j, nv_j = joint_sizes(m_src.jnt_type[j_src])
    qadr_src = m_src.jnt_qposadr[j_src]
    vadr_src = m_src.jnt_dofadr[j_src]

    # --- 目标环境索引（假设 XML 相同；若不相同请用 joint-name → id） ---
    m_dst = dst_env.env.sim.model
    b_dst = m_dst.body_name2id(body_name)
    j_dst = m_dst.body_jntadr[b_dst]
    qadr_dst = m_dst.jnt_qposadr[j_dst]
    vadr_dst = m_dst.jnt_dofadr[j_dst]

    # --- 复制 ---
    qpos_slice = src_env.env.sim.data.qpos[qadr_src : qadr_src + nq_j].copy()
    qvel_slice = src_env.env.sim.data.qvel[vadr_src : vadr_src + nv_j].copy()

    print(f"{qpos_slice=}, {qvel_slice=}")

    dst_env.env.sim.data.qpos[qadr_dst : qadr_dst + nq_j] = qpos_slice
    dst_env.env.sim.data.qvel[vadr_dst : vadr_dst + nv_j] = qvel_slice
    dst_env.env.sim.forward()            # 同步派生量
# --- end of copy_body_state function ---

copy_body_state(env_old, env_new, "alphabet_soup_1_main")
for _ in range(5):
    obs, _, _, _ = env_new.step([0.] * 7)
# Save the modified state image
img_in_new_env = obs["agentview_image"][::-1]

env_old.close()
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

# # Add a main title
# plt.suptitle(f"Comparison of Environment States for '{target_object}'")

# Display the plot instead of saving
plt.tight_layout()
plt.show()
