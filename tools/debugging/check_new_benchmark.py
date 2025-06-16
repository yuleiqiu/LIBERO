#!/usr/bin/env python3
"""
LIBERO Benchmark Checker

This script validates a LIBERO benchmark by:
1. Printing available benchmarks
2. Creating a benchmark instance
3. Checking task information and BDDL file existence
"""

import os
from libero.libero.benchmark import print_benchmark, get_benchmark_dict, get_benchmark


def main():
    """Main function to check LIBERO benchmark functionality."""
    
    # Print all available benchmarks
    print("=== Available Benchmarks ===")
    print_benchmark()
    print("\n=== Benchmark Details ===")
    get_benchmark_dict(help=True)
    
    # Configure benchmark to test
    benchmark_name = "libero_object_single"  # 替换为您的 benchmark 名称
    
    try:
        # Get and instantiate benchmark
        print(f"\n=== Testing Benchmark: {benchmark_name} ===")
        benchmark_class = get_benchmark(benchmark_name)
        benchmark = benchmark_class()
        
        # Print basic information
        print(f"Benchmark name: {benchmark.name}")
        print(f"Number of tasks: {benchmark.n_tasks}")
        print(f"Task names: {benchmark.get_task_names()}")
        
        # Check BDDL files for the first task
        if benchmark.n_tasks > 0:
            print(f"\n=== Checking BDDL Files ===")
            bddl_path = benchmark.get_task_bddl_file_path(0)
            
            if os.path.exists(bddl_path):
                print(f"✅ BDDL file exists: {bddl_path}")
            else:
                print(f"❌ BDDL file not found: {bddl_path}")
        else:
            print("⚠️ No tasks found in benchmark")
        
        print("✅ Benchmark validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during benchmark validation: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()