import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv, SegmentationRenderEnv
from robosuite.utils.mjcf_utils import get_ids

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
print(f"# objects: {len(object_states_dict)}")
for object, state in object_states_dict.items():
    print(f"Object: {object}")
    print(f"State: {state.get_geom_state()}")

env.close()