from libero.libero.benchmark import print_benchmark

print_benchmark()

from libero.libero.benchmark import get_benchmark_dict

# 打印所有可用的 benchmark 名称
get_benchmark_dict(help=True)

from libero.libero.benchmark import get_benchmark

# 替换为您的 benchmark 名称
my_benchmark_name = "libero_object_single" 

try:
    # 尝试获取 benchmark 类
    benchmark_class = get_benchmark(my_benchmark_name)
    
    # 实例化 benchmark
    benchmark = benchmark_class()
    
    # 打印基本信息
    print(f"Benchmark name: {benchmark.name}")
    print(f"Number of tasks: {benchmark.n_tasks}")
    print(f"Task names: {benchmark.get_task_names()}")
    
    print("✅ Benchmark created successfully!")
except Exception as e:
    print(f"❌ Error creating benchmark: {e}")

    # 获取第一个任务的 BDDL 文件路径
if benchmark.n_tasks > 0:
    bddl_path = benchmark.get_task_bddl_file_path(0)
    
    # 检查文件是否存在
    import os
    if os.path.exists(bddl_path):
        print(f"✅ BDDL file exists: {bddl_path}")
    else:
        print(f"❌ BDDL file not found: {bddl_path}")