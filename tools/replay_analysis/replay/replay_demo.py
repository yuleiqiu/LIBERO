from libero.libero import benchmark
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
import os
# import init_path
from libero.libero import benchmark, get_libero_path
import cv2
from datetime import datetime
import h5py

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_object" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()
datasets_default_path = get_libero_path("datasets")

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
print("the task name is:", task_name)
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}

video_folder = os.path.join("get_started", "videos")
os.makedirs(video_folder, exist_ok=True)

done_count = 0

for i in range(50):
    print(f"Processing demo {i}...")
    env = OffScreenRenderEnv(**env_args)
    env.seed(i)
    env.reset()

    dataset_path = os.path.join(datasets_default_path, task_suite.get_task_demonstration(task_id))
    demo_key = f"demo_{i}"
    demo = h5py.File(dataset_path)["data"][demo_key]
    states = demo["states"]
    init_state = states[0]  # the first state
    print("the states shape is:", states.shape)
    actions = demo["actions"]
    print("the actions shape is:", actions.shape)

    obs = env.set_init_state(init_state)

    # for k, v in obs.items():
    #     print(k, v.shape)
    
    # exit(0)

    # 创建视频写入器
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"replay_demo_{i}.mp4"
    output_path = os.path.join(video_folder, output_name)
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None

    # 存储用于视频的帧
    frames = []

    # 获取初始观察
    frames.append(obs["agentview_image"][::-1])

    # 执行动作并捕获每一帧
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        frames.append(obs["agentview_image"][::-1])
        if i % 10 == 0:  # 每10帧打印一次进度
            print(f"处理帧 {i+1}/{len(actions)}")
        if done:
            print("任务完成")
            done_count += 1
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

    env.close()

print(f"完成的任务数量: {done_count}")