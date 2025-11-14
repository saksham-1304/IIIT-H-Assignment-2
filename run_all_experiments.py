"""
Script to run all experiments sequentially.
"""

import subprocess
import sys
import os

def run_experiment(config_path, experiment_name):
    """
    Run a single experiment.
    
    Args:
        config_path: Path to configuration file
        experiment_name: Name of the experiment
    """
    print("\n" + "="*100)
    print(f"Starting {experiment_name} experiment...")
    print("="*100 + "\n")
    
    cmd = [
        sys.executable,
        "train.py",
        "--config", config_path,
        "--experiment", experiment_name
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n{experiment_name} experiment completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError running {experiment_name} experiment: {e}")
        return False
    
    return True

def main():
    """Run all three experiments."""
    
    experiments = [
        ("configs/config_underfit.json", "underfit"),
        ("configs/config_bestfit.json", "bestfit"),
        ("configs/config_overfit.json", "overfit")
    ]
    
    print("\n" + "="*100)
    print("NEURAL LANGUAGE MODEL TRAINING - ALL EXPERIMENTS")
    print("="*100)
    print("\nThis script will train three models:")
    print("1. Underfitting model (small capacity)")
    print("2. Best-fit model (optimal capacity + regularization)")
    print("3. Overfitting model (large capacity, no regularization)")
    print("\nNote: This may take several hours depending on your hardware.")
    print("="*100 + "\n")
    
    # Check if config files exist
    for config_path, _ in experiments:
        if not os.path.exists(config_path):
            print(f"Error: Configuration file not found: {config_path}")
            return
    
    # Run experiments
    results = []
    for config_path, experiment_name in experiments:
        success = run_experiment(config_path, experiment_name)
        results.append((experiment_name, success))
    
    # Print summary
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)
    for experiment_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{experiment_name:15s} - {status}")
    print("="*100 + "\n")
    
    # Check outputs
    print("Generated outputs:")
    for experiment_name, success in results:
        if success:
            print(f"\n{experiment_name.capitalize()}:")
            print(f"  - Checkpoint: checkpoints/{experiment_name}_best_model.pt")
            print(f"  - Loss Plot: plots/{experiment_name}_loss_plot.png")
            print(f"  - Perplexity Plot: plots/{experiment_name}_perplexity_plot.png")
            print(f"  - Results: outputs/{experiment_name}_results.json")
    
    print("\nAll experiments completed! Check the plots/ directory for visualizations.")

if __name__ == "__main__":
    main()
