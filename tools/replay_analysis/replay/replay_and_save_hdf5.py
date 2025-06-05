#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
replay_and_save_hdf5.py
-----------------------
回放LIBERO演示并保存为与collect_demonstration.py相同格式的hdf5文件
"""
import os
import copy
import h5py
import json
import datetime
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv


# ---------- 配置 ----------
task_suite_name = "libero_object"
task_id         = 0                           # alphabet soup 任务
max_demos       = 50                          # 遍历数
output_dir      = os.path.join("replay", "hdf5_output")

os.makedirs(output_dir, exist_ok=True)

# ---------- 路径 ----------
benchmark_dict  = benchmark.get_benchmark_dict()
task_suite      = benchmark_dict[task_suite_name]()
task            = task_suite.get_task(task_id)

bddl_dir        = get_libero_path("bddl_files")
bddl_full       = os.path.join(bddl_dir, task.problem_folder, task.bddl_file)

current_dir     = os.path.dirname(os.path.abspath(__file__))
parent_dir      = os.path.dirname(current_dir)
tmp_bddl_dir    = os.path.join(parent_dir, "tmp", "pddl_files")
bddl_simplified = os.path.join(tmp_bddl_dir,
                               "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_2.bddl")

demo_dir        = get_libero_path("datasets")
demo_file       = os.path.join(demo_dir, task_suite.get_task_demonstration(task_id))
demo_h5         = h5py.File(demo_file, "r")["data"]

# ---------- 创建环境 ----------
cam_args        = dict(camera_heights=128, camera_widths=128)
env_full        = OffScreenRenderEnv(bddl_file_name=bddl_full,       **cam_args)
env_simple      = OffScreenRenderEnv(bddl_file_name=bddl_simplified, **cam_args)

# ---------- 工具：复制重叠状态 ----------
def joint_sizes(jt):
    return {0:(7,6), 1:(4,3), 2:(1,1), 3:(1,1)}[jt]

def copy_overlap(src_env, dst_env):
    m_s, d_s = src_env.env.sim.model, src_env.env.sim.data
    m_d, d_d = dst_env.env.sim.model, dst_env.env.sim.data

    for jid in range(m_s.njnt):
        name = m_s.joint_id2name(jid)
        if isinstance(name, bytes):
            name = name.decode()
        try:
            jid_d = m_d.joint_name2id(name)
        except ValueError:
            continue
        if jid_d < 0:
            continue

        nq, nv = joint_sizes(m_s.jnt_type[jid])
        qs, vs = m_s.jnt_qposadr[jid], m_s.jnt_dofadr[jid]
        qd, vd = m_d.jnt_qposadr[jid_d], m_d.jnt_dofadr[jid_d]
        d_d.qpos[qd:qd+nq] = d_s.qpos[qs:qs+nq]
        d_d.qvel[vd:vd+nv] = d_s.qvel[vs:vs+nv]

    # mocap（相机/道具）—— 如果模型使用了
    nmc = min(m_s.nmocap, m_d.nmocap)
    if nmc:
        d_d.mocap_pos [:nmc] = d_s.mocap_pos [:nmc]
        d_d.mocap_quat[:nmc] = d_s.mocap_quat[:nmc]

    dst_env.env.sim.forward()

def create_hdf5_from_replays(output_path, collected_demos, env_info, problem_info, bddl_file_path):
    """
    创建与collect_demonstration.py相同格式的hdf5文件
    """
    f = h5py.File(output_path, "w")
    grp = f.create_group("data")
    
    # 存储每个demo
    for demo_idx, demo_data in enumerate(collected_demos):
        if demo_data is None:
            continue
            
        ep_data_grp = grp.create_group("demo_{}".format(demo_idx + 1))
        
        # 存储model xml
        model_xml = env_simple.env.sim.model.get_xml()
        ep_data_grp.attrs["model_file"] = model_xml
        
        # 写入states和actions数据集
        ep_data_grp.create_dataset("states", data=np.array(demo_data["states"]))
        ep_data_grp.create_dataset("actions", data=np.array(demo_data["actions"]))
    
    # 写入元数据属性
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_simple.env.__class__.__name__
    grp.attrs["env_info"] = env_info
    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = bddl_file_path
    
    # 读取bddl文件内容
    with open(bddl_file_path, "r", encoding="utf-8") as f:
        bddl_content = f.read()
    grp.attrs["bddl_file_content"] = bddl_content
    
    f.close()

# ---------- 准备环境信息 ----------
# 模拟collect_demonstration.py中的config信息
controller_config = load_controller_config(default_controller="OSC_POSE")
config = {
    "robots": ["Panda"],  # 根据实际机器人调整
    "controller_configs": controller_config,  # 根据实际控制器调整
}
env_info = json.dumps(config)

# 获取问题信息（需要导入BDDLUtils）
try:
    import libero.libero.envs.bddl_utils as BDDLUtils
    problem_info = BDDLUtils.get_problem_info(bddl_simplified)
except:
    # 如果无法获取，创建基本信息
    problem_info = {
        "problem_name": "PickAndPlace",
        "domain_name": "libero_object", 
        "language_instruction": "pick the alphabet soup and place it in the basket"
    }

# ---------- 收集回放数据 ----------
collected_demos = []
success_cnt = 0

for demo_idx in range(max_demos):
    demo_key = f"demo_{demo_idx}"
    if demo_key not in demo_h5:
        print(f"[!] 数据集中没有 {demo_key}，提前结束")
        break

    demo      = demo_h5[demo_key]
    states    = demo["states"]
    actions   = demo["actions"]

    # 重置环境
    env_full.reset()
    env_full.set_init_state(states[0])

    env_simple.reset()
    copy_overlap(env_full, env_simple)

    # 收集状态和动作
    replay_states = []
    replay_actions = []
    
    # 获取初始状态
    obs, _, _, _ = env_simple.step([0.] * 7)  # 不推进物理，只获取状态
    initial_state = env_simple.env.sim.get_state().flatten()
    replay_states.append(initial_state)

    # -------- 回放动作并收集状态 --------
    done, info = False, {}
    for step_id, action in enumerate(actions):
        # 执行动作
        obs, _, done, info = env_simple.step(action)
        
        # 记录动作和新状态
        replay_actions.append(action.copy())
        current_state = env_simple.env.sim.get_state().flatten()
        replay_states.append(current_state)

        if done:
            break

    # 移除最后一个状态（保持与collect_demonstration.py一致）
    if len(replay_states) > 0:
        replay_states = replay_states[:-1]
    
    # 确保states和actions长度一致
    min_len = min(len(replay_states), len(replay_actions))
    replay_states = replay_states[:min_len]
    replay_actions = replay_actions[:min_len]
    
    if len(replay_states) > 0 and len(replay_actions) > 0:
        collected_demos.append({
            "states": replay_states,
            "actions": replay_actions
        })
        
        success = info.get("success", done)
        print(f"Demo {demo_idx:02d} – {'SUCCESS' if success else 'FAIL'} "
              f"(states: {len(replay_states)}, actions: {len(replay_actions)})")
        if success:
            success_cnt += 1
    else:
        collected_demos.append(None)
        print(f"Demo {demo_idx:02d} – EMPTY (skipped)")

# ---------- 保存为HDF5 ----------
output_path = os.path.join(output_dir, "replayed_demos.hdf5")
create_hdf5_from_replays(
    output_path, 
    collected_demos, 
    env_info, 
    problem_info, 
    bddl_simplified
)

# ---------- 统计 ----------
total_demos = len([d for d in collected_demos if d is not None])
print(f"\nSUMMARY: {success_cnt}/{total_demos} replays succeed "
      f"({success_cnt/total_demos:.2%})")
print(f"HDF5 file saved to: {output_path}")

# 验证保存的文件
with h5py.File(output_path, "r") as f:
    print(f"Saved {len(f['data'].keys())} demonstrations")
    for key in f["data"].keys():
        if key.startswith("demo_"):
            demo_grp = f["data"][key]
            print(f"  {key}: {len(demo_grp['states'])} states, {len(demo_grp['actions'])} actions")

env_full.close()
env_simple.close()