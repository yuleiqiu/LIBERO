from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv, SegmentationRenderEnv
import os

from robosuite.utils.mjcf_utils import get_ids

# # 获取当前文件所在的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取父目录
# parent_dir = os.path.dirname(current_dir)
# # 构建notebooks目录的路径
# notebooks_dir = os.path.join(parent_dir, "get_started", "tmp", "pddl_files")

# bddl_file_names = [
#     os.path.join(notebooks_dir, "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_1.bddl"),
#     os.path.join(notebooks_dir, "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_2.bddl"),
# ]

benchmark_dict = benchmark.get_benchmark_dict()
benchmark_instance = benchmark_dict["libero_object"]()
bddl_files_default_path = get_libero_path("bddl_files")
task = benchmark_instance.get_task(2)
bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
print(f"BDDL file: {bddl_file}")

env_args = {
    "bddl_file_name": bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}

env = OffScreenRenderEnv(**env_args)
env.seed(10)
env.reset()

for _ in range(5):
    obs, _, _, _ = env.step([0.] * 7)

env_sim_states = env.get_sim_state()
# print(f"env_sim_states: {env_sim_states}")

# for k, v in obs.items():
#     print(k, v.shape)

# available_body_names = env.env.sim.model.body_names
# for name in available_body_names:
#     id = get_ids(env.env.sim, name, element_type="body")
#     print(f"Name: {name}, ID: {id}")

# object_states_from_obs = obs["object-state"]
# print(f"object_states shape: {object_states_from_obs.shape}")

object_states_dict = env.env.object_states_dict
print(f"# objects: {len(object_states_dict)}")
for object, state in object_states_dict.items():
    print(f"Object: {object}")
    print(f"State: {state.get_geom_state()}")


env.close()

# for bddl_idx in range(len(bddl_file_names)):
#     env_args = {
#         "bddl_file_name": bddl_file_names[bddl_idx],
#         "camera_heights": 128,
#         "camera_widths": 128
#     }

#     # 创建环境
#     env = SegmentationRenderEnv(**env_args)
#     env.seed(10)

#     # 重置环境
#     obs = env.reset()

#     # 执行动作
#     for _ in range(5):
#         obs, _, _, _ = env.step([0.] * 7)

#     # print(env.segmentation_id_mapping)
#     # print(env.instance_to_id)
#     # print(env.segmentation_robot_id)
#     # for k,v in obs.items():
#     #     print(k, v.shape)

#     object_states_from_obs = obs["object-state"]
#     print(f"BDDL file {bddl_idx+1}, object_states shape: {object_states_from_obs.shape}")

#     object_states_dict = env.env.object_states_dict
#     for object, state in object_states_dict.items():
#         print(f"Object: {object}")
#         print(f"State: {state.get_joint_state()}")

#     # 关闭环境
#     env.close()