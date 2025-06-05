from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
import os
import argparse
import torch
import numpy as np

from robosuite.utils.mjcf_utils import get_ids

class GenerateInitStates:
    """Class to generate initial states for a specific task in the Libero benchmark."""

    def __init__(self, num_init, task_id, save_dir):
        self.num_init = num_init
        self.task_id = task_id
        self.save_dir = save_dir

    def generate_init_states(self):
        """Generate initial states for the specified task."""

        # 获取 benchmark 实例
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict["libero_object_single"]()
        bddl_files_default_path = get_libero_path("bddl_files")
        task = benchmark_instance.get_task(self.task_id)
        bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
        print(f"BDDL file: {bddl_file}")

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128
        }

        # 初始化用于保存状态的列表
        all_init_states = []

        print(f"开始收集 {self.num_init} 个不同seed的初始状态...")

        env = OffScreenRenderEnv(**env_args)

        for seed in range(self.num_init):
            print(f"处理 seed {seed}...")
            
            # 设置种子并重置环境
            env.seed(seed)
            env.reset()
            
            # 获取初始状态
            init_sim_state = env.get_sim_state()
            
            # 将状态转换为numpy数组并添加到列表
            state_array = np.array(init_sim_state)
            all_init_states.append(state_array)

        env.close()
        
        # 将列表转换为 (num_init, state_length) 的 ndarray
        all_init_states = np.stack(all_init_states, axis=0)
        print(f"状态数组形状: {all_init_states.shape}")

        # 创建保存目录
        save_dir = os.path.join(self.save_dir, "libero_object_single")
        os.makedirs(save_dir, exist_ok=True)

        # 保存所有状态到文件
        save_path = os.path.join(save_dir, f"{task.name}.init")
        
        # 检查文件是否已存在，如果存在则询问用户是否覆盖
        if os.path.exists(save_path):
            print(f"\n警告: 文件 {save_path} 已存在!")
            print(f"该文件包含任务 '{task.name}' 的初始状态数据。")
            while True:
                user_input = input("是否要覆盖现有文件? (y/n): ").strip().lower()
                if user_input in ['y', 'yes']:
                    print("确认覆盖现有文件...")
                    break
                elif user_input in ['n', 'no']:
                    print("操作已取消，保持现有文件不变。")
                    return  # 退出函数，不保存新文件
                else:
                    print("请输入 'y' (是) 或 'n' (否)")
                    continue

        # 使用 torch.save 保存数据
        torch.save(all_init_states, save_path)

        print(f"\n成功保存 {self.num_init} 个初始状态到: {save_path}")

        # 打印一些统计信息
        print(f"\n状态信息统计:")
        print(f"- 状态数组形状: {all_init_states.shape}")
        print(f"- 状态数组类型: {type(all_init_states)}")
        print(f"- 每个状态长度: {all_init_states.shape[1]}")

        print(f"\n文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Generate initial states for custom tasks.')

    parser.add_argument('--num_init', type=int, default=50,
                        help='Number of initial states to generate.')
    parser.add_argument('--task-id', type=int, default=0,
                        help='Task ID to generate initial states for.')
    parser.add_argument('--save_dir', type=str, default='libero/libero/init_files/',
                        help='Parent directory to save initial states.')
    
    args = parser.parse_args()

    # 创建生成器实例
    generator = GenerateInitStates(num_init=args.num_init, task_id=args.task_id, save_dir=args.save_dir)

    # 生成初始状态
    generator.generate_init_states()

if __name__ == "__main__":
    main()
