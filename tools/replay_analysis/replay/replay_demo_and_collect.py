import argparse
import h5py
import json
import numpy as np
import os
import time
import datetime
from glob import glob
import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

def replay_demo_and_collect(
    original_demo_path, demo_id, new_env_config, output_dir, args
):
    """
    在新环境中重放demo并保存为相同格式的数据
    
    Args:
        original_demo_path: 原始demo文件路径
        demo_id: demo编号
        new_env_config: 新环境配置
        output_dir: 输出目录
        args: 命令行参数
    """
    # 加载原始demo数据
    with h5py.File(original_demo_path, 'r') as f:
        demo_key = f"demo_{demo_id}"
        if demo_key not in f["data"]:
            print(f"Demo {demo_id} not found in dataset")
            return False
            
        demo = f["data"][demo_key]
        states = demo["states"][:]
        actions = demo["actions"][:]
        
        # 获取原始环境信息
        original_problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
        original_bddl_file = f["data"].attrs.get("bddl_file_name", "")
        
    print(f"Replaying demo {demo_id} with {len(actions)} actions")
    
    # 创建新环境（基于新配置）
    env = create_new_environment(new_env_config, args)
    
    # 创建临时目录用于数据收集
    t1, t2 = str(time.time()).split(".")
    tmp_directory = f"replay_data/tmp/replay_demo_{demo_id}_{t1}_{t2}"
    
    # 用DataCollectionWrapper包装环境，这样会自动保存数据
    env = DataCollectionWrapper(env, tmp_directory)
    
    # 重置环境
    try:
        env.reset()
        # 尝试设置初始状态（如果支持的话）
        if hasattr(env.env, 'set_init_state'):  # 注意DataCollectionWrapper的嵌套
            env.env.set_init_state(states[0])
    except Exception as e:
        print(f"Warning: Could not set exact initial state: {e}")
    
    # 执行动作序列
    success_steps = 0
    for i, action in enumerate(actions):
        try:
            # 可能需要调整动作格式
            adjusted_action = adjust_action_for_new_env(action, new_env_config)
            obs, reward, done, info = env.step(adjusted_action)
            success_steps += 1
            
            if i % 50 == 0:
                print(f"Step {i+1}/{len(actions)}")
                
            if done:
                print(f"Task completed at step {i}")
                break
                
        except Exception as e:
            print(f"Error at step {i}: {e}")
            break
    
    env.close()
    
    # 现在将临时数据整合成HDF5格式
    success = gather_replay_demonstrations_as_hdf5(
        tmp_directory, output_dir, new_env_config, args, original_problem_info, demo_id
    )
    
    # 清理临时文件
    import shutil
    if os.path.exists(tmp_directory):
        shutil.rmtree(tmp_directory)
    
    return success and success_steps > len(actions) * 0.8  # 至少完成80%的步骤

def create_new_environment(config, args):
    """创建新环境"""
    # 根据你的新环境配置创建环境
    # 这里需要根据具体情况修改
    
    # 如果是LIBERO环境
    if "bddl_file" in config:
        problem_info = BDDLUtils.get_problem_info(config["bddl_file"])
        problem_name = problem_info["problem_name"]
        
        env_config = {
            "robots": config.get("robots", ["Panda"]),
            "controller_configs": config.get("controller_configs"),
        }
        
        if "TwoArm" in problem_name:
            env_config["env_configuration"] = config.get("env_configuration", "single-arm-opposed")
        
        env = TASK_MAPPING[problem_name](
            bddl_file_name=config["bddl_file"],
            **env_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera=config.get("camera", "agentview"),
            ignore_done=True,
            use_camera_obs=config.get("use_camera_obs", False),
            reward_shaping=True,
            control_freq=config.get("control_freq", 20),
        )
    else:
        # 其他类型的环境
        env_args = {
            "camera_heights": config.get("camera_height", 128),
            "camera_widths": config.get("camera_width", 128),
            **config.get("custom_params", {})
        }
        env = OffScreenRenderEnv(**env_args)
    
    return env

def adjust_action_for_new_env(action, env_config):
    """调整动作以适应新环境"""
    # 如果动作空间有变化，在这里进行映射
    # 目前保持原样
    return action

def gather_replay_demonstrations_as_hdf5(
    tmp_directory, out_dir, env_config, args, original_problem_info, demo_id
):
    """
    将重放的演示数据整合成HDF5格式
    基于原始的 gather_demonstrations_as_hdf5 函数修改
    """
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, f"replay_demo_{demo_id}.hdf5")
    f = h5py.File(hdf5_path, "w")
    
    # 创建数据组
    grp = f.create_group("data")
    
    num_eps = 0
    env_name = None
    
    # 处理临时目录中的数据
    if os.path.exists(tmp_directory):
        for ep_directory in os.listdir(tmp_directory):
            state_paths = os.path.join(tmp_directory, ep_directory, "state_*.npz")
            states = []
            actions = []
            
            for state_file in sorted(glob(state_paths)):
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])
                
                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])
            
            if len(states) == 0:
                continue
            
            # 删除最后一个state和第一个action以保持对齐
            del states[-1]
            assert len(states) == len(actions)
            
            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))
            
            # 保存模型XML
            xml_path = os.path.join(tmp_directory, ep_directory, "model.xml")
            if os.path.exists(xml_path):
                with open(xml_path, "r") as xml_file:
                    xml_str = xml_file.read()
                ep_data_grp.attrs["model_file"] = xml_str
            
            # 保存states和actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
    
    # 写入元数据
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name if env_name else "unknown"
    grp.attrs["env_info"] = json.dumps(env_config)
    
    # 保存原始问题信息
    grp.attrs["problem_info"] = json.dumps(original_problem_info)
    grp.attrs["original_demo_id"] = demo_id
    grp.attrs["replay_timestamp"] = time.time()
    
    # 如果有BDDL文件信息
    if "bddl_file" in env_config and os.path.exists(env_config["bddl_file"]):
        grp.attrs["bddl_file_name"] = env_config["bddl_file"]
        with open(env_config["bddl_file"], "r", encoding="utf-8") as f:
            grp.attrs["bddl_file_content"] = f.read()
    
    f.close()
    print(f"Replay demo saved to: {hdf5_path}")
    return num_eps > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-demo", type=str, required=True,
                       help="Path to original demo HDF5 file")
    parser.add_argument("--output-dir", type=str, default="replay_demonstrations",
                       help="Output directory for replay demonstrations")
    parser.add_argument("--demo-ids", nargs="+", type=int, default=[0, 1, 2],
                       help="Demo IDs to replay")
    
    # 新环境配置参数
    parser.add_argument("--new-bddl-file", type=str, help="BDDL file for new environment")
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--control-freq", type=int, default=20)
    
    args = parser.parse_args()
    
    # 加载控制器配置
    from robosuite import load_controller_config
    controller_config = load_controller_config(default_controller=args.controller)
    
    # 新环境配置
    new_env_config = {
        "bddl_file": args.new_bddl_file,
        "robots": args.robots,
        "controller_configs": controller_config,
        "camera": args.camera,
        "control_freq": args.control_freq,
        "use_camera_obs": False,  # 根据需要调整
    }
    
    # 重放指定的demos
    success_count = 0
    for demo_id in args.demo_ids:
        print(f"\n=== Replaying demo {demo_id} ===")
        success = replay_demo_and_collect(
            args.original_demo, demo_id, new_env_config, args.output_dir, args
        )
        if success:
            success_count += 1
            print(f"Demo {demo_id} replayed successfully")
        else:
            print(f"Demo {demo_id} failed to replay")
    
    print(f"\nSuccess rate: {success_count}/{len(args.demo_ids)}")