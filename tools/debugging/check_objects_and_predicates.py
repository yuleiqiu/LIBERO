from libero.libero.envs.objects import get_object_dict, get_object_fn
from libero.libero.utils.mu_utils import get_scene_class, get_scene_dict

import pprint

pp = pprint.PrettyPrinter(indent=2)

scene_name = "floor"
scene = get_scene_class(scene_name)()
pp.pprint(scene)
# scene_dict = get_scene_dict(scene_type="floor")
# pp.pprint(scene_dict)
# print("=============================================")

# Get a dictionary of all the objects
object_dict = get_object_dict()
pp.pprint(object_dict)
print("=============================================")

# Get the object class of a specific object
category_name = "cream_cheese"
object_cls = get_object_fn(category_name)
print(category_name, ": defined in the class ", object_cls)
print("=============================================")


from libero.libero.envs.predicates import get_predicate_fn_dict, get_predicate_fn

predicate_dict = get_predicate_fn_dict()
pp.pprint(predicate_dict)
print("=============================================")
predicate_name = "on"
print(get_predicate_fn(predicate_name))
print("=============================================")
