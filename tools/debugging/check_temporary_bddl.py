# This is the default path to store all the pddl scene files. Here we store the files in the temporary folder. If you want to directly add files into the libero codebase, get the default path use the following commented lines:
# from libero.libero import get_libero_path
# YOUR_BDDL_FILE_PATH = get_libero_path("bddl_files")

import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录
parent_dir = os.path.dirname(current_dir)
# 构建notebooks目录的路径
notebooks_dir = os.path.join(parent_dir, "notebooks", "tmp", "pddl_files")

bddl_file_names = [
    os.path.join(notebooks_dir, "KITCHEN_SCENE1_your_language_1.bddl"),
    os.path.join(notebooks_dir, "KITCHEN_SCENE1_your_language_2.bddl"),
]

# bddl_file_names = [
#     "../notebooks/tmp/pddl_files/KITCHEN_SCENE1_your_language_1.bddl",
#     "../notebooks/tmp/pddl_files/KITCHEN_SCENE1_your_language_2.bddl",
# ]
with open(bddl_file_names[0], "r") as f:
    content = f.read()
print(content)