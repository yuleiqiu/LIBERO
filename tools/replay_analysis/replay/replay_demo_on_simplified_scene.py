#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
replay_50_demos.py
------------------
批量回放最多 50 条 LIBERO 演示，并报告每条成功 / 失败
"""
import os, copy
import cv2
import h5py
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

# ---------- 配置 ----------
task_suite_name = "libero_object"
task_id         = 0                           # alphabet soup 任务
max_demos       = 50                          # 遍历数
save_video      = True                       # True → 保存 mp4（大量磁盘)

video_dir       = os.path.join("replay", "videos")
if save_video:
    os.makedirs(video_dir, exist_ok=True)

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

# ---------- 创建两个环境一次即可 ----------
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
        except ValueError:            # robosuite 找不到抛错
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

# ---------- 遍历演示 ----------
success_cnt = 0
for demo_idx in range(max_demos):
    demo_key = f"demo_{demo_idx}"
    if demo_key not in demo_h5:
        print(f"[!] 数据集中没有 {demo_key}，提前结束")
        break

    demo      = demo_h5[demo_key]
    states    = demo["states"]
    actions   = demo["actions"]

    env_full.reset()
    env_full.set_init_state(states[0])

    env_simple.reset()
    copy_overlap(env_full, env_simple)

    # -------- 播放前抓首帧 --------
    obs0, _, _, _ = env_simple.step([0.] * 7)         # 不推进物理
    frame0 = obs0["agentview_image"][::-1]            # 垂直翻转

    # -------- 打开视频写入器（若启用） --------
    if save_video:
        h, w, _ = frame0.shape
        path = os.path.join(video_dir, f"demo_{demo_idx}.mp4")
        vw   = cv2.VideoWriter(path,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               20, (w, h))
        vw.write(cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))

    # -------- 回放动作并写帧 --------
    done, info = False, {}
    for step_id, action in enumerate(actions):
        obs, _, done, info = env_simple.step(action)

        if save_video:
            frame = obs["agentview_image"][::-1]
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if done:
            break

    # -------- 收尾 --------
    if save_video:
        vw.release()
        print(f"[video] saved {path}")

    success = info.get("success", done)
    print(f"Demo {demo_idx:02d} –", "SUCCESS" if success else "FAIL")
    if success:
        success_cnt += 1

# ---------- 统计 ----------
total = demo_idx + 1
print(f"\nSUMMARY: {success_cnt}/{total} replays succeed "
      f"({success_cnt/total:.2%})")

env_full.close()
env_simple.close()