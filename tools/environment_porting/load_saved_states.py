#!/usr/bin/env python3
"""
用于加载和查看保存的初始状态的脚本
"""

import pickle
import numpy as np
import os
import glob

def load_init_states(file_path):
    """加载保存的初始状态文件"""
    with open(file_path, 'rb') as f:
        init_states = pickle.load(f)
    return init_states

def analyze_states(init_states):
    """分析状态数据的统计信息"""
    num_seeds = len(init_states)
    print(f"加载了 {num_seeds} 个seeds的初始状态")
    
    if num_seeds == 0:
        return
    
    # 获取第一个状态作为样本
    sample_state = init_states[0]
    
    print(f"\n状态结构信息:")
    print(f"- init_sim_state长度: {len(sample_state)}")
    print(f"- init_sim_state类型: {type(sample_state)}")
    
    # 分析不同seeds之间的差异
    print(f"\n不同seeds之间的状态差异分析:")
    
    # 比较sim_state的差异
    sim_states_list = [init_states[seed] for seed in range(num_seeds)]
    sim_states_array = np.array(sim_states_list)
    sim_states_std = np.std(sim_states_array, axis=0)
    
    print(f"- sim_state标准差范围: [{np.min(sim_states_std):.6f}, {np.max(sim_states_std):.6f}]")
    print(f"- sim_state平均标准差: {np.mean(sim_states_std):.6f}")

def compare_specific_seeds(init_states, seed1=0, seed2=1):
    """比较两个特定seeds的状态差异"""
    if seed1 not in init_states or seed2 not in init_states:
        print(f"Seeds {seed1} 或 {seed2} 不存在")
        return
    
    state1 = init_states[seed1]
    state2 = init_states[seed2]
    
    print(f"\n比较 seed {seed1} 和 seed {seed2}:")
    
    # 比较sim_state
    sim_diff = np.abs(state1 - state2)
    print(f"- sim_state最大差异: {np.max(sim_diff):.6f}")
    print(f"- sim_state平均差异: {np.mean(sim_diff):.6f}")
    print(f"- 不同元素的数量: {np.sum(sim_diff > 1e-10)}")
    print(f"- 相同元素的数量: {np.sum(sim_diff <= 1e-10)}")

def main():
    # 查找最新的状态文件
    save_dir = "libero/libero/init_files/libero_object_single"
    
    if not os.path.exists(save_dir):
        print(f"保存目录不存在: {save_dir}")
        return
    
    state_files = glob.glob(os.path.join(save_dir, "pick_up_the_alphabet_soup_and_place_it_in_the_basket.init"))
    
    if not state_files:
        print(f"在 {save_dir} 中没有找到状态文件")
        return
    
    # 使用最新的文件
    latest_file = max(state_files, key=os.path.getctime)
    print(f"加载状态文件: {latest_file}")
    
    # 加载状态
    init_states = load_init_states(latest_file)
    
    # 分析状态
    analyze_states(init_states)
    
    # 比较特定的seeds
    if len(init_states) >= 2:
        compare_specific_seeds(init_states, 0, 1)
        compare_specific_seeds(init_states, 0, 49)

if __name__ == "__main__":
    main()
