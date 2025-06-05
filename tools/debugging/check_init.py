import os
import torch
import numpy as np
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from collections import defaultdict

def inspect_init_states_from_benchmark():
    """Load init states through benchmark API and examine their shapes."""
    # Initialize a benchmark (e.g., LIBERO_OBJECT)
    benchmark_name = "libero_object"  # Change to any benchmark you want to examine
    benchmark = get_benchmark(benchmark_name)()
    
    print(f"Examining {benchmark_name} benchmark with {benchmark.get_num_tasks()} tasks")
    print("-" * 80)
    
    # Iterate through all tasks in the benchmark
    for i in range(benchmark.get_num_tasks()):
        task = benchmark.get_task(i)
        print(f"Task {i}: {task.name}")
        
        # Load init states using the benchmark method
        init_states = benchmark.get_task_init_states(i)
        
        # Print information about the loaded init states
        if isinstance(init_states, dict):
            print("  Init states is a dictionary with keys:")
            for key, value in init_states.items():
                if hasattr(value, "shape"):
                    print(f"    {key}: shape={value.shape}, type={type(value)}")
                else:
                    print(f"    {key}: type={type(value)}")
        elif hasattr(init_states, "shape"):
            print(f"  Init states shape: {init_states.shape}, type={type(init_states)}")
        else:
            print(f"  Init states type: {type(init_states)}")
        print()

def are_states_identical(state1, state2):
    """Compare two state objects (tensors, dictionaries, etc.) for equality."""
    if type(state1) != type(state2):
        return False

    # Handle dictionary case
    if isinstance(state1, dict):
        if set(state1.keys()) != set(state2.keys()):
            return False
        
        for key in state1:
            if not are_states_identical(state1[key], state2[key]):
                return False
        return True
    
    # Handle tensor case
    if isinstance(state1, torch.Tensor):
        return torch.equal(state1, state2)
    
    # Handle numpy array case
    if isinstance(state1, np.ndarray):
        return np.array_equal(state1, state2)
    
    # Handle list/tuple case
    if isinstance(state1, (list, tuple)):
        if len(state1) != len(state2):
            return False
        return all(are_states_identical(a, b) for a, b in zip(state1, state2))
    
    # For simple types
    return state1 == state2

def inspect_init_states_directly():
    """Directly load all init files from the directory."""
    # Get the path to init_states directory
    init_states_dir = get_libero_path("init_states")
    
    # Specify the subfolder (e.g., libero_object)
    subfolder = "libero_object"  # Change to examine different folders
    folder_path = os.path.join(init_states_dir, subfolder)
    
    print(f"Examining all init files in {folder_path}")
    print("-" * 80)
    
    # Group files by their base name (without extension)
    file_groups = defaultdict(dict)
    for filename in os.listdir(folder_path):
        if filename.endswith('.init') or filename.endswith('.pruned_init'):
            base_name = filename.rsplit('.', 1)[0]
            ext = filename.rsplit('.', 1)[1]
            file_groups[base_name][ext] = filename
    
    # Compare matching .init and .pruned_init files
    print("Comparing .init and .pruned_init files:")
    print("-" * 80)
    
    for base_name, files in file_groups.items():
        if 'init' in files and 'pruned_init' in files:
            print(f"Comparing {files['init']} and {files['pruned_init']}...")
            
            init_file = os.path.join(folder_path, files['init'])
            pruned_init_file = os.path.join(folder_path, files['pruned_init'])
            
            try:
                init_state = torch.load(init_file)
                pruned_init_state = torch.load(pruned_init_file)
                
                is_identical = are_states_identical(init_state, pruned_init_state)
                
                if is_identical:
                    print(f"  ✓ Files are IDENTICAL")
                else:
                    print(f"  ✗ Files are DIFFERENT")
            except Exception as e:
                print(f"  ! Error comparing files: {e}")
            print()
    
    # Continue with original functionality
    print("\nExamining individual files:")
    print("-" * 80)
    
    init_files = os.listdir(folder_path)
    for filename in init_files:
        file_path = os.path.join(folder_path, filename)
        print(f"Loading {filename}")
        
        # Load the init state
        init_states = torch.load(file_path)
        
        # Print information about the loaded init states
        if isinstance(init_states, dict):
            print("  Init states is a dictionary with keys:")
            for key, value in init_states.items():
                if hasattr(value, "shape"):
                    print(f"    {key}: shape={value.shape}, type={type(value)}")
                else:
                    print(f"    {key}: type={type(value)}")
        elif hasattr(init_states, "shape"):
            print(f"  Init states shape: {init_states.shape}, type={type(init_states)}")
        else:
            print(f"  Init states type: {type(init_states)}")
        print()

if __name__ == "__main__":
    print("Method 1: Using benchmark API")
    inspect_init_states_from_benchmark()
    
    print("\nMethod 2: Directly loading files")
    inspect_init_states_directly()