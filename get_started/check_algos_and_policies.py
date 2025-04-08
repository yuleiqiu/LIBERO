import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pprint
from libero.lifelong.algos import get_algo_list
from libero.lifelong.models import get_policy_list


pp = pprint.PrettyPrinter(indent=2)

pp.pprint("Available algorithms:")
pp.pprint(get_algo_list())

pp.pprint("Available policies:")
pp.pprint(get_policy_list())