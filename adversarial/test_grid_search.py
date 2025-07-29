"""
Test script for the adversarial grid search experiment.

This script demonstrates how to run the grid search experiment with proper
paths and configurations that work with your existing codebase structure.
"""

import os
import sys
import torch

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grid_search_experiment import run_adversarial_grid_search_experiment
import utils

def test_grid_search_experiment():
    """
    Test the grid search experiment function with a small configuration
    to verify everything works correctly.
    """
    
    print("Testing Adversarial Grid Search Experiment")
    print("="*50)
    
    # Configure paths - adjust these to match your actual paths
    MODEL_PATH = "/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/models/natural/149_checkpoint.pt"
    DATA_PATH = "/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/datasets"
    SAVE_PATH = "/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/adversarial/test_grid_search_results"
    
    print(f"Model path: {MODEL_PATH}")
    print(f"Data path: {DATA_PATH}")
    print(f"Save path: {SAVE_PATH}")
    
    # Check if paths exist
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model path does not exist: {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your trained model")
        return False
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data path does not exist: {DATA_PATH}")
        print("Please update DATA_PATH to point to your dataset directory")
        return False
    
    try:
        # Load dataset
        print("\nLoading dataset...")
        dataset, train_loader, test_loader = utils.load_dataset(DATA_PATH)
        print("Dataset loaded successfully")
        
        # Run experiment with reduced parameters for testing
        print("\nRunning grid search experiment...")
        print("Note: Using reduced num_random_images=10 for faster testing")
        
        results = run_adversarial_grid_search_experiment(
            model_path=MODEL_PATH,
            data_path=DATA_PATH,
            loader=test_loader,
            save_path=SAVE_PATH,
            num_random_images=10,  # Reduced for testing
            verbose=True
        )
        
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        # Print some basic statistics
        grid_results = results['grid_results']
        print(f"Total parameter combinations tested: {len(grid_results)}")
        
        if grid_results:
            # Show a sample result
            sample_key = list(grid_results.keys())[0]
            sample_result = grid_results[sample_key]
            
            print(f"\nSample result ({sample_key}):")
            print(f"  Average SSIM: {sample_result['avg_ssim']:.4f}")
            print(f"  Label flip rate: {sample_result['label_flip_rate']:.2f}%")
            print(f"  Attack success: {sample_result['attack_success']}")
        
        # Show files created
        print(f"\nFiles created in {SAVE_PATH}:")
        if os.path.exists(SAVE_PATH):
            for file in os.listdir(SAVE_PATH):
                file_path = os.path.join(SAVE_PATH, file)
                file_size = os.path.getsize(file_path)
                print(f"  {file} ({file_size} bytes)")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during experiment: {e}")
        print("Please check your paths and ensure all dependencies are available")
        import traceback
        traceback.print_exc()
        return False

def run_full_experiment():
    """
    Run the full experiment with the complete configuration.
    
    This is the function you would call for your actual research experiment.
    """
    
    print("Running Full Adversarial Grid Search Experiment")
    print("="*60)
    
    # Configure paths - adjust these to match your actual paths
    MODEL_PATH = "/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/models/natural/149_checkpoint.pt"
    DATA_PATH = "/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/datasets"
    SAVE_PATH = "/afs/csail.mit.edu/u/h/hectorxm/distribution-shift/adversarial/full_grid_search_results"
    
    # Load dataset
    dataset, train_loader, test_loader = utils.load_dataset(DATA_PATH)
    
    # Run full experiment
    results = run_adversarial_grid_search_experiment(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        loader=test_loader,
        save_path=SAVE_PATH,
        num_random_images=100,  # Full configuration
        verbose=True
    )
    
    print("Full experiment completed successfully!")
    return results

if __name__ == "__main__":
    # Run test first
    success = test_grid_search_experiment()
    
    if success:
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("You can now run the full experiment using run_full_experiment()")
        print("="*60)
        
        # Uncomment the line below to run the full experiment immediately
        # run_full_experiment()
    else:
        print("\nTest failed. Please fix the errors before running the full experiment.") 