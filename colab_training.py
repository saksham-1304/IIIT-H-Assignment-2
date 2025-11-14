# Neural Language Model Training - Google Colab
# IIIT Hyderabad Assignment 2

## Instructions:
# 1. Upload this notebook to Google Colab
# 2. Runtime > Change runtime type > GPU (T4)
# 3. Run cells in order
# 4. Download results at the end

# Cell 1: Clone Repository
!git clone https://github.com/saksham-1304/IIIT-H-Assignment-2.git
%cd IIIT-H-Assignment-2

# Cell 2: Install Dependencies
!pip install torch torchvision numpy matplotlib

# Cell 3: Verify Setup
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test imports
from models.lstm import create_model
from data.preprocessing import TextPreprocessor
print("✓ All imports successful!")

# Cell 4: Quick Test (Optional - 2 minutes)
# Run this first to verify everything works
!python train.py --config configs/config_underfit.json --experiment underfit

# Cell 5: Run All Experiments (45-60 minutes)
# This will train all three models
!python run_all_experiments.py

# Cell 6: Evaluate All Models
!python evaluate.py --checkpoint checkpoints/underfit_best_model.pt --config configs/config_underfit.json --experiment underfit
!python evaluate.py --checkpoint checkpoints/bestfit_best_model.pt --config configs/config_bestfit.json --experiment bestfit
!python evaluate.py --checkpoint checkpoints/overfit_best_model.pt --config configs/config_overfit.json --experiment overfit

# Cell 7: Generate Sample Text
!python generate_text.py --checkpoint checkpoints/bestfit_best_model.pt --config configs/config_bestfit.json --prompt "It is a truth universally acknowledged"
!python generate_text.py --checkpoint checkpoints/bestfit_best_model.pt --config configs/config_bestfit.json --prompt "Elizabeth Bennet"
!python generate_text.py --checkpoint checkpoints/bestfit_best_model.pt --config configs/config_bestfit.json --prompt "Mr Darcy"

# Cell 8: View Results
import json
import os

experiments = ['underfit', 'bestfit', 'overfit']
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"{'Experiment':<15} {'Train PPL':<15} {'Val PPL':<15} {'Test PPL':<15}")
print("-"*70)

for exp in experiments:
    results_path = f'outputs/{exp}_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            train_ppl = results.get('train_perplexity', 0)
            val_ppl = results.get('val_perplexity', 0)
            test_ppl = results.get('test_perplexity', 0)
            print(f"{exp.capitalize():<15} {train_ppl:<15.2f} {val_ppl:<15.2f} {test_ppl:<15.2f}")
print("="*70)

# Cell 9: Display Plots
from IPython.display import Image, display
import os

print("\n=== TRAINING LOSS PLOTS ===\n")
for exp in experiments:
    plot_path = f'plots/{exp}_loss_plot.png'
    if os.path.exists(plot_path):
        print(f"\n{exp.upper()} Loss Plot:")
        display(Image(filename=plot_path))

# Cell 10: Download Results
# Compress all results for download
!zip -r results.zip checkpoints/ plots/ outputs/

print("\n✓ Results compressed to results.zip")
print("Download it: Files panel (left) > results.zip > right-click > Download")

# Cell 11: Optional - Upload to Google Drive
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!mkdir -p /content/drive/MyDrive/IIIT-H-Assignment-2-Results
!cp results.zip /content/drive/MyDrive/IIIT-H-Assignment-2-Results/
!cp -r checkpoints /content/drive/MyDrive/IIIT-H-Assignment-2-Results/
!cp -r plots /content/drive/MyDrive/IIIT-H-Assignment-2-Results/
!cp -r outputs /content/drive/MyDrive/IIIT-H-Assignment-2-Results/

print("✓ Results copied to Google Drive: IIIT-H-Assignment-2-Results/")
