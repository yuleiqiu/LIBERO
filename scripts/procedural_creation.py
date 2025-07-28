from libero.libero.envs.objects import get_object_dict, get_object_fn
from libero.libero.envs.predicates import get_predicate_fn_dict, get_predicate_fn
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info

import numpy as np
import pprint

# # Get a dictionary of all the objects
# object_dict = get_object_dict()
# # pprint.pprint(object_dict)
# for object_name, object_fn in object_dict.items():
#     # pprint.pprint(object_fn.__dict__)
#     print(f"Object Name: {object_name}, Object Function: {object_fn}")
#     # print(f"Horizontal Radius: {object_fn.horizontal_radius}")
#     try:
#         obj_instance = object_fn()
#         print(f'Instance type: {type(obj_instance)}')
#         print(f'Instance horizontal_radius: {obj_instance.horizontal_radius}')
#     except Exception as e:
#         print(f'Cannot instantiate: {e}')
#     # break

# # Get a dictionary of all the predicates
# predicate_dict = get_predicate_fn_dict()
# pprint.pprint(predicate_dict)
# print("=============")
# predicate_name = "on"
# pprint.pprint(get_predicate_fn(predicate_name))


# Define your own initial state distribution
@register_mu(scene_type="floor")
class FloorScene1(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "floor": 1,
        }

        object_num_info = {
            "alphabet_soup": 1,
            "basket": 1,
            "salad_dressing": 1,
            "cream_cheese": 1,
            "milk": 1,
            "tomato_sauce": 1,
            "butter": 1,
        }

        super().__init__(
            workspace_name="floor",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0, 0.25], 
                                 region_name="bin_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0
                                )
        )

        self.regions.update(
            self.get_region_dict(region_centroid_xy=[-0.15, -0.15], 
                                 region_name="workspace_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.25
                                )
        )

        self.regions.update(
            self.get_region_dict(region_centroid_xy=[-0.40, -0.35], 
                                 region_name="target_object_region", 
                                 target_name=self.workspace_name, 
                                 region_half_len=0.05/2
                                )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "alphabet_soup_1", "floor_workspace_region"),
            ("On", "salad_dressing_1", "floor_workspace_region"),
            ("On", "cream_cheese_1", "floor_workspace_region"),
            ("On", "milk_1", "floor_workspace_region"),
            ("On", "tomato_sauce_1", "floor_workspace_region"),
            ("On", "butter_1", "floor_workspace_region"),
            ("On", "basket_1", "floor_bin_region"),
        ]
        return states

# Define your own task goal
scene_name = "floor_scene1"
language = "Pick the alphabet soup and place it in the basket"
register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=["alphabet_soup_1", "basket_1"],
                    goal_states=[("In", "alphabet_soup_1", "basket_1_contain_region")]
)

# """
# The task goals will be temporarily saved in the variable `libero.libero.utils.task_generation_utils.TASK_INFO` in the format of namedtuple `libero.libero.utils.task_generation_utils.TaskInfoTuple`.
# This design aims to make it easy for batch creation of tasks.
# """

# # This is the default path to store all the pddl scene files.
# # Here we store the files in the temporary folder.
# # If you want to directly add files into the libero codebase, get the default path use the following commented lines:
# # from libero.libero import get_libero_path
# # YOUR_BDDL_FILE_PATH = get_libero_path("bddl_files")

YOUR_BDDL_FILE_PATH = "tmp/pddl_files"
bddl_file_names, failures = generate_bddl_from_task_info(folder=YOUR_BDDL_FILE_PATH)

# print(bddl_file_names)
# print("Encountered some failures: ", failures)

# # Read the content of the BDDL file
# with open(bddl_file_names[0], "r") as f:
#     content = f.read()
# print(content)