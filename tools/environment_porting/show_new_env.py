from libero.libero.envs import OffScreenRenderEnv
import matplotlib.pyplot as plt

import torch
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
    "bddl_file_name": bddl_file_names[1],
    "camera_heights": 128,
    "camera_widths": 128
}

# 创建保存图片的目录
save_dir = os.path.join(current_dir, "environment_images")
os.makedirs(save_dir, exist_ok=True)

# 创建一个大图用于显示所有环境
plt.figure(figsize=(25, 5))

# 遍历每个环境配置
for i in range(5):
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(i)
    
    # 重置环境
    env.reset()
    
    # 执行几个随机步骤
    for _ in range(5):
        obs, _, _, _ = env.step([0.] * 7)
    
    # 获取图像
    image = obs["agentview_image"]
    
    # 保存图像
    image_path = os.path.join(save_dir, f"environment_image_seed_{i}.png")
    plt.imsave(image_path, image[::-1])
    print(f"Saved image for seed {i} to {image_path}")
    
    # 添加到大图中
    plt.subplot(1, 5, i+1)
    plt.imshow(image[::-1])
    plt.axis('off')
    plt.title(f"Seed {i}")
    
    # 关闭环境
    env.close()

# 保存整个图表
combined_image_path = os.path.join(save_dir, "all_environments.png")
plt.tight_layout()
plt.savefig(combined_image_path)
plt.show()
print(f"Saved combined image to {combined_image_path}")