import numpy as np
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info

# List of grocery items
grocery_items = [
    "alphabet_soup", 
    "salad_dressing", 
    "cream_cheese", 
    "milk", 
    "tomato_sauce", 
    "butter", 
    "orange_juice", 
    "chocolate_pudding", 
    "bbq_sauce", 
    "ketchup"
]

@register_mu(scene_type="floor")
class MyFloorScene(InitialSceneTemplates):
    def __init__(self):
        # Always include the floor and basket
        fixture_num_info = {
            "floor": 1,
        }
        
        # Initialize with 7 objects total - 1 target, 1 basket, and 5 others
        object_num_info = {
            "basket": 1,
            "alphabet_soup": 1,
        }
        
        super().__init__(
            workspace_name="floor",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info
        )
    
    def define_regions(self):
        # Define bin region (where basket will be placed)
        self.regions.update(
            self.get_region_dict(region_centroid_xy=[0.0, 0.26], 
                               region_name="bin_region", 
                               target_name=self.workspace_name, 
                               region_half_len=0.01)
        )
        
        # Define target object region
        self.regions.update(
            self.get_region_dict(region_centroid_xy=[-0.12, -0.24], 
                               region_name="target_object_region", 
                               target_name=self.workspace_name, 
                               region_half_len=0.025)
        )
        
        # Define other object regions
        # other_object_positions = [
        #     [0.05, -0.1],
        #     [-0.15, 0.06],
        #     [0.1, -0.2],
        #     [0.15, 0.03],
        #     [-0.2, -0.08]
        # ]
        
        # for i, pos in enumerate(other_object_positions):
        #     self.regions.update(
        #         self.get_region_dict(region_centroid_xy=pos, 
        #                            region_name=f"other_object_region_{i}", 
        #                            target_name=self.workspace_name, 
        #                            region_half_len=0.025)
        #     )

        self.regions.update(
            self.get_region_dict(region_centroid_xy=[5.0, 5.0], 
                                region_name="alphabet_soup_init_region", 
                                target_name=self.workspace_name, 
                                region_half_len=2.5)
        )
        
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)
    
    @property
    def init_states(self):
        states = [
            ("On", "basket_1", "floor_bin_region"),
            ("On", "alphabet_soup_1", "alphabet_soup_init_region"),
        ]
        # Other states for objects will be added when specific target object is chosen
        return states

# Generate BDDL files for each grocery item as target
# for target_item in grocery_items:
#     # Create language instruction
#     language = f"Pick the {target_item.replace('_', ' ')} and place it in the basket"
    
#     # Register task for this target item
#     register_task_info(
#         language,
#         scene_name="my_floor_scene",
#         objects_of_interest=[f"{target_item}_1", "basket_1"],
#         goal_states=[("In", f"{target_item}_1", "basket_1_contain_region")]
#     )

target_item = "alphabet_soup"  # Example target item
# Create language instruction
language = f"Pick the {target_item.replace('_', ' ')} and place it in the basket"

# Register task for this target item
register_task_info(
    language,
    scene_name="my_floor_scene",
    objects_of_interest=[f"{target_item}_1", "basket_1"],
    goal_states=[("In", f"{target_item}_1", "basket_1_contain_region")]
)

# Generate BDDL files
# bddl_file_path = "libero/libero/bddl_files/libero_object"
bddl_file_path = "get_started/tmp/pddl_files"
bddl_file_names, failures = generate_bddl_from_task_info(folder=bddl_file_path)

print(f"Generated {len(bddl_file_names)} BDDL files")
print(f"Failures: {failures}")