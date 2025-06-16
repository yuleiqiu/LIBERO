"""
Generate initial states for LIBERO benchmark tasks.

This script generates and saves initial simulation states for specified tasks
in the LIBERO benchmark, which can be used for consistent environment initialization
during training or evaluation.
"""

import os
import argparse
import torch
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

class GenerateInitStates:
    """
    Generator for initial states in LIBERO benchmark tasks.
    
    This class handles the generation and saving of initial simulation states
    for a specific task, providing multiple random seeds for diverse starting
    configurations.
    """

    def __init__(self, num_init: int, task_id: int, save_dir: str):
        """
        Initialize the state generator.
        
        Args:
            num_init: Number of initial states to generate
            task_id: ID of the task to generate states for
            save_dir: Directory to save the generated states
        """
        self.num_init = num_init
        self.task_id = task_id
        self.save_dir = save_dir

    def _get_task_and_env_config(self):
        """
        Get task configuration and environment arguments.
        
        Returns:
            tuple: (task, env_args) containing task object and environment config
        """
        # Get benchmark instance
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict["libero_object_single"]()
        bddl_files_default_path = get_libero_path("bddl_files")
        task = benchmark_instance.get_task(self.task_id)
        bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
        print(f"BDDL file: {bddl_file}")

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128
        }
        
        return task, env_args

    def _collect_initial_states(self, env_args: dict) -> np.ndarray:
        """
        Collect initial states from the environment using different seeds.
        
        Args:
            env_args: Environment configuration arguments
            
        Returns:
            np.ndarray: Array of initial states with shape (num_init, state_length)
        """
        # Initialize list to store states
        all_init_states = []

        print(f"Starting collection of {self.num_init} initial states with different seeds...")

        env = OffScreenRenderEnv(**env_args)

        for seed in range(self.num_init):
            print(f"Processing seed {seed}...")
            
            # Set seed and reset environment
            env.seed(seed)
            env.reset()
            
            # Get initial state
            init_sim_state = env.get_sim_state()
            
            # Convert state to numpy array and add to list
            state_array = np.array(init_sim_state)
            all_init_states.append(state_array)

        env.close()
        
        # Convert list to (num_init, state_length) ndarray
        all_init_states = np.stack(all_init_states, axis=0)
        print(f"State array shape: {all_init_states.shape}")
        
        return all_init_states

    def _prompt_file_overwrite(self, save_path: str, task_name: str) -> bool:
        """
        Prompt user whether to overwrite existing file.
        
        Args:
            save_path: Path to the file that would be overwritten
            task_name: Name of the task for display purposes
            
        Returns:
            bool: True if user wants to overwrite, False otherwise
        """
        print(f"\nWarning: File {save_path} already exists!")
        print(f"This file contains initial state data for task '{task_name}'.")
        
        while True:
            user_input = input("Do you want to overwrite the existing file? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                print("Confirmed overwriting existing file...")
                return True
            elif user_input in ['n', 'no']:
                print("Operation cancelled, keeping existing file unchanged.")
                return False
            else:
                print("Please enter 'y' (yes) or 'n' (no)")
                continue

    def _save_states_to_file(self, all_init_states: np.ndarray, task_name: str) -> str:
        """
        Save states to file with user confirmation for overwrite.
        
        Args:
            all_init_states: Array of initial states to save
            task_name: Name of the task for filename
            
        Returns:
            str: Path where the file was saved, or None if cancelled
        """
        # Create save directory
        save_dir = os.path.join(self.save_dir, "libero_object_single")
        os.makedirs(save_dir, exist_ok=True)

        # Save all states to file
        save_path = os.path.join(save_dir, f"{task_name}.init")
        
        # Check if file already exists and prompt for overwrite
        if os.path.exists(save_path):
            if not self._prompt_file_overwrite(save_path, task_name):
                return None

        # Save data using torch.save
        torch.save(all_init_states, save_path)
        return save_path

    def _print_statistics(self, all_init_states: np.ndarray, save_path: str):
        """
        Print statistics about the generated and saved states.
        
        Args:
            all_init_states: The generated state array
            save_path: Path where the file was saved
        """
        print(f"\nSuccessfully saved {self.num_init} initial states to: {save_path}")

        # Print statistics
        print(f"\nState information statistics:")
        print(f"- State array shape: {all_init_states.shape}")
        print(f"- State array type: {type(all_init_states)}")
        print(f"- Each state length: {all_init_states.shape[1]}")

        print(f"\nFile size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")

    def generate_init_states(self):
        """
        Main method to generate initial states for the specified task.
        
        This method orchestrates the entire process of generating, collecting,
        and saving initial states for the task.
        """
        # Get task and environment configuration
        task, env_args = self._get_task_and_env_config()
        
        # Collect initial states from environment
        all_init_states = self._collect_initial_states(env_args)
        
        # Save states to file
        save_path = self._save_states_to_file(all_init_states, task.name)
        
        # Print statistics if file was saved successfully
        if save_path is not None:
            self._print_statistics(all_init_states, save_path)

def main():
    """
    Main function to parse arguments and run the initial state generator.
    """
    parser = argparse.ArgumentParser(
        description='Generate initial states for LIBERO benchmark tasks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--num_init', 
        type=int, 
        default=50,
        help='Number of initial states to generate with different seeds.'
    )
    parser.add_argument(
        '--task-id', 
        type=int, 
        default=0,
        help='Task ID to generate initial states for.'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='libero/libero/init_files/',
        help='Parent directory to save initial states files.'
    )
    
    args = parser.parse_args()

    # Create generator instance
    generator = GenerateInitStates(
        num_init=args.num_init, 
        task_id=args.task_id, 
        save_dir=args.save_dir
    )

    # Generate initial states
    generator.generate_init_states()

if __name__ == "__main__":
    main()
