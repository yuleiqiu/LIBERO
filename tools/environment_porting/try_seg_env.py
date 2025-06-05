from libero.libero.envs.env_wrapper import SegmentationRenderEnv
import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录
parent_dir = os.path.dirname(current_dir)
# 构建notebooks目录的路径
notebooks_dir = os.path.join(parent_dir, "get_started", "tmp", "pddl_files")

bddl_file_names = [
    os.path.join(notebooks_dir, "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_1.bddl"),
    os.path.join(notebooks_dir, "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_2.bddl"),
]

env_args = {
    "bddl_file_name": bddl_file_names[0],
    "camera_heights": 128,
    "camera_widths": 128
}

# 创建环境
env = SegmentationRenderEnv(**env_args)
env.seed(10)

# 重置环境
obs = env.reset()

# 执行动作
for _ in range(5):
    obs, _, _, _ = env.step([0.] * 7)

# print(env.segmentation_id_mapping)
# print(env.instance_to_id)
# print(env.segmentation_robot_id)
for k,v in obs.items():
    print(k, v.shape)

# 获取分割掩码
segmentation_image = obs["agentview_segmentation_instance"]
segmentation_instances = env.get_segmentation_instances(segmentation_image)

# 获取感兴趣对象的分割
segmentation_of_interest = env.get_segmentation_of_interest(segmentation_image)

# 将分割转换为RGB图像进行可视化
rgb_segmentation = env.segmentation_to_rgb(segmentation_image)

# 关闭环境
env.close()