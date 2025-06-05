from libero.libero import benchmark
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
import os
from libero.libero import benchmark, get_libero_path
import cv2
from datetime import datetime
import h5py
import numpy as np


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
# print("the states shape is:", states.shape)

env_old_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env_old = OffScreenRenderEnv(**env_old_args)
env_old.reset()
env_old.set_init_state(state)

# Create a new environment from new args with single object
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
tmp_bddl_dir = os.path.join(parent_dir, "tmp", "pddl_files")

bddl_file_names = [
    os.path.join(tmp_bddl_dir, "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_1.bddl"),
    os.path.join(tmp_bddl_dir, "MY_FLOOR_SCENE_pick_the_alphabet_soup_and_place_it_in_the_basket_2.bddl"),
]

env_new_args = {
    "bddl_file_name": bddl_file_names[0],
    "camera_heights": 128,
    "camera_widths": 128
}

env_new = OffScreenRenderEnv(**env_new_args)
env_new.reset()

def copy_body_state_by_name(src_env, dst_env, body_name, atol=1e-6):
    # --------- tools ---------
    def joint_sizes(jtype):
        return {0:(7,6), 1:(4,3), 2:(1,1), 3:(1,1)}[jtype]   # free / ball / slide / hinge

    m_s, d_s = src_env.env.sim.model, src_env.env.sim.data
    m_d, d_d = dst_env.env.sim.model, dst_env.env.sim.data

    # --- body → 主 joint ---
    bid = m_s.body_name2id(body_name)
    jid = m_s.body_jntadr[bid]
    jname = m_s.joint_id2name(jid)
    if isinstance(jname, bytes):
        jname = jname.decode("utf-8")

    # --- 用名字在目标 env 反查 joint id ---
    jid_dst = m_d.joint_name2id(jname)
    assert jid_dst != -1, f"{jname} 不存在于目标环境！"

    # --- 位置 / 速度切片 ---
    nq, nv = joint_sizes(m_s.jnt_type[jid])
    qadr_s, vadr_s = m_s.jnt_qposadr[jid], m_s.jnt_dofadr[jid]
    qadr_d, vadr_d = m_d.jnt_qposadr[jid_dst], m_d.jnt_dofadr[jid_dst]

    d_d.qpos[qadr_d:qadr_d+nq] = d_s.qpos[qadr_s:qadr_s+nq]
    d_d.qvel[vadr_d:vadr_d+nv] = d_s.qvel[vadr_s:vadr_s+nv]

    dst_env.env.sim.forward()        # 更新派生量

    # --------- quick check ---------
    assert np.allclose(d_s.qpos[qadr_s:qadr_s+nq],
                       d_d.qpos[qadr_d:qadr_d+nq], atol=atol), "qpos 不一致"
    # assert np.allclose(d_s.geom_xpos[m_s.geom_name2id(body_name)],
                    #    d_d.geom_xpos[m_d.geom_name2id(body_name)], atol=atol), "xpos 不一致"
    print(f"[✓] 复制 {body_name} 完成！")

# copy_body_state(env_old, env_new, "alphabet_soup_1_main")
# assert_body_copied(env_old, env_new, "alphabet_soup_1_main")
copy_body_state_by_name(env_old, env_new, "alphabet_soup_1_main")


#------------------------------------------------#
video_folder = os.path.join("get_started", "replay", "videos")
os.makedirs(video_folder, exist_ok=True)

print(f"Processing demo {demo_idx}...")

actions = demo["actions"]
print("the actions shape is:", actions.shape)

# 创建视频写入器
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_name = f"replay_demo_{demo_idx}.mp4"
output_path = os.path.join(video_folder, output_name)
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None

# 存储用于视频的帧
frames = []

for _ in range(5):
    obs, _, _, _ = env_new.step([0.] * 7)

# 获取初始观察
frames.append(obs["agentview_image"][::-1])

# 执行动作并捕获每一帧
for i, action in enumerate(actions):
    obs, reward, done, info = env_new.step(action)
    frames.append(obs["agentview_image"][::-1])
    if i % 10 == 0:  # 每10帧打印一次进度
        print(f"处理帧 {i+1}/{len(actions)}")
    if done:
        print("任务完成")
        break

# 创建视频写入器并写入帧
if frames:
    # 获取第一帧的尺寸来设置视频
    h, w, _ = frames[0].shape
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        # OpenCV 使用 BGR 格式，而环境可能返回 RGB
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)
    
    video_writer.release()
    print(f"视频已保存到: {output_path}")

env_old.close()
env_new.close()