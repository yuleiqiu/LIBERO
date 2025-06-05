from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv, SegmentationRenderEnv
import os
import numpy as np

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
object_states_dict = env.env.object_states_dict

# 打印env_sim_states向量长度
print(f"env_sim_states长度: {len(env_sim_states)}")
print(type(env_sim_states))

# 打印物体状态字典中的所有物体
print("所有物体名称:")
for obj_name in object_states_dict.keys():
    print(f"- {obj_name}")

obj_body_id = env.env.obj_body_id
print(f"{obj_body_id=}")

# 查看MuJoCo模型的qpos和qvel的尺寸
print(f"\nMuJoCo模型信息:")
mjdata = env.env.sim.data
body_xpos = mjdata.body_xpos
print(f"body_xpos尺寸: {body_xpos.shape}")
body_xquat = mjdata.body_xquat
print(f"body_xquat尺寸: {body_xquat.shape}")

states_from_mjdata = np.array([mjdata.time] + mjdata.qpos.tolist() + mjdata.qvel.tolist())
print(f"MuJoCo模型的状态向量长度: {len(states_from_mjdata)}")
print((env_sim_states == states_from_mjdata).all())
print(f"time: {mjdata.time}")
print(f"qpos尺寸: {mjdata.qpos.shape}")
print(f"qvel尺寸: {mjdata.qvel.shape}")

# # 如果env_sim_states是qpos和qvel的组合
# qpos_size = env.env.sim.data.qpos.shape[0]
# print(f"前{qpos_size}个元素可能是qpos，剩余的是qvel")

# # 尝试打印物体的关节状态
# for obj_name, obj_state in object_states_dict.items():
#     print(f"\n物体 {obj_name} 的关节状态:")
#     try:
#         joint_state = obj_state.get_joint_state()
#         print(joint_state)
#     except:
#         print("无法获取关节状态")



# # 检查每个物体状态的索引关系
# for obj_name, obj_state in object_states_dict.items():
#     # 获取物体的几何状态
#     geom_state = obj_state.get_geom_state()
#     print(f"\n物体: {obj_name}")
#     print(f"几何状态维度: {geom_state.shape if hasattr(geom_state, 'shape') else 'N/A'}")
    
#     # 尝试获取物体在模拟中的ID
#     obj_id = None
#     try:
#         # 获取物体的body ID
#         obj_id = get_ids(env.env.sim, obj_name, element_type="body")
#         print(f"Body ID: {obj_id}")
        
#         # 获取物体在状态向量中的位置
#         # 通常物体的位置和方向信息存储在状态向量的特定位置
#         body_pos = env.env.sim.data.body_xpos[obj_id]
#         body_quat = env.env.sim.data.body_xquat[obj_id]
        
#         print(f"位置: {body_pos}")
#         print(f"四元数: {body_quat}")
        
#         # 尝试在env_sim_states中找到对应值
#         pos_found = False
#         for i in range(0, len(env_sim_states), 7):  # 假设每个物体状态包含7个值
#             if (env_sim_states[i:i+3] == body_pos).all():
#                 print(f"在env_sim_states中找到位置，索引: {i}到{i+2}")
#                 pos_found = True
#                 break
                
#         if not pos_found:
#             print("在env_sim_states中未找到精确匹配的位置")
            
#     except Exception as e:
#         print(f"获取物体ID时出错: {e}")

env.close()