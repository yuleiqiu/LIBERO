"""
Generate BDDL files for various target object initial positions on a sparse grid.
This script dynamically registers a new scene template for each grid cell,
then generates a BDDL file under tmp/pddl_files/sparse_grid/<i>_<j>.
"""
import numpy as np
import os
from pathlib import Path
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info, get_task_info

# Base definitions for scene objects
FIXTURE_NUM_INFO = {"floor": 1}
OBJECT_NUM_INFO = {
    "alphabet_soup": 1,
    "basket": 1,
    "salad_dressing": 1,
    "cream_cheese": 1,
    "milk": 1,
    "tomato_sauce": 1,
    "butter": 1,
}
LANGUAGE_TEMPLATE = "Pick the alphabet soup and place it in the basket"
OBJECTS_OF_INTEREST = ["alphabet_soup_1", "basket_1"]
GOAL_STATES = [("In", "alphabet_soup_1", "basket_1_contain_region")]

# Grid definition
grid = np.linspace(-0.4, 0.1, 11)

def generate_for_cell(i, j, interval_x, interval_y, output_base):
    # Unique identifiers
    scene_suffix = f"cell_{i}_{j}"
    scene_type = "floor"
    class_name = f"Cell_{i}_{j}"
    # 构建动态类方法
    def __init__(self):
        # 显式调用父类构造，避免 super() 在动态类型中失效
        InitialSceneTemplates.__init__(
            self,
            workspace_name="floor",
            fixture_num_info=FIXTURE_NUM_INFO,
            object_num_info=OBJECT_NUM_INFO,
        )

    def define_regions(self):
        # bin region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0, 0.26],
                region_name="bin_region",
                target_name=self.workspace_name,
                region_half_len=0.01
            )
        )

        # workspace region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, -0.15],
                region_name="workspace_region",
                target_name=self.workspace_name,
                region_half_len=0.25
            )
        )

        # target object region for this cell
        cx = float((interval_x[0] + interval_x[1]) / 2)
        cy = float((interval_y[0] + interval_y[1]) / 2)
        half_len = float((interval_x[1] - interval_x[0]) / 2)
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[cx, cy],
                region_name="target_object_region",
                target_name=self.workspace_name,
                region_half_len=half_len
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    def init_states(self):
        states = [
            ("On", "alphabet_soup_1", "floor_target_object_region"),
            ("On", "salad_dressing_1", "floor_workspace_region"),
            ("On", "cream_cheese_1", "floor_workspace_region"),
            ("On", "milk_1", "floor_workspace_region"),
            ("On", "tomato_sauce_1", "floor_workspace_region"),
            ("On", "butter_1", "floor_workspace_region"),
            ("On", "basket_1", "floor_bin_region"),
        ]
        return states
    # 创建类并注册
    DynamicScene = type(class_name, (InitialSceneTemplates,), {
        '__init__': __init__,
        'define_regions': define_regions,
        'init_states': property(init_states)
    })
    register_mu(scene_type=scene_type)(DynamicScene)

    # from libero.libero.utils.mu_utils import MU_DICT
    # for k,v in MU_DICT.items():
    #     print(f"Registered scene: {k} -> {v.__name__}")

    register_task_info(
        LANGUAGE_TEMPLATE,
        scene_name=f"cell_{i}_{j}",
        objects_of_interest=OBJECTS_OF_INTEREST,
        goal_states=GOAL_STATES,
    )

    # prepare output folder
    out_folder = Path(output_base)
    # out_folder.mkdir(parents=True, exist_ok=True)

    bddl_file_names, failures = generate_bddl_from_task_info(folder=str(out_folder))
    return bddl_file_names, failures


def main():
    output_base = "tmp/pddl_files/sparse_grid"
    for i in range(len(grid) - 1):
        for j in range(len(grid) - 1):
            interval_x = [grid[i], grid[i + 1]]
            interval_y = [grid[j], grid[j + 1]]
            names, fails = generate_for_cell(i, j, interval_x, interval_y, output_base)
            print(f"Cell ({i},{j}): Generated files: {names}, Failures: {fails}")
        #     break
        # break


if __name__ == "__main__":
    main()
