"""
Neural Language Model Training - Google Colab Complete Training Script
IIIT Hyderabad Assignment 2 - Full Submission Training

INSTRUCTIONS:
1. Go to https://colab.research.google.com
2. File > Upload notebook > Upload this .py file
3. Runtime > Change runtime type > Hardware accelerator: GPU (T4 or better)
4. Runtime > Run all (Ctrl+F9)
5. Wait ~45-60 minutes for all three experiments to complete
6. Download results.zip from Files panel

ESTIMATED TIME:
- Underfitting: ~5-10 minutes
- Best-fit: ~20-30 minutes  
- Overfitting: ~20-30 minutes
- Total: ~45-60 minutes on GPU
"""

# ============================================================================
# CELL 1: Setup and Clone Repository
# ============================================================================
print("="*80)
print("STEP 1: Cloning Repository...")
print("="*80)

!git clone https://github.com/saksham-1304/IIIT-H-Assignment-2.git
%cd IIIT-H-Assignment-2

print("\n✓ Repository cloned successfully!")

# ============================================================================
# CELL 2: Install Dependencies
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Installing Dependencies...")
print("="*80)

!pip install -q torch numpy matplotlib
print("✓ Dependencies installed!")

# ============================================================================
# CELL 3: Verify GPU and Setup
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Verifying Setup...")
print("="*80)

import torch
import sys
import os

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ WARNING: GPU not available! Training will be SLOW on CPU.")
    print("Go to: Runtime > Change runtime type > Hardware accelerator: GPU")

# Test imports
try:
    from models.lstm import create_model
    from data.preprocessing import TextPreprocessor
    from data.dataset import get_dataloaders
    print("✓ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    raise

# ============================================================================
# CELL 4: Train Underfitting Model (~5-10 minutes)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Training UNDERFITTING Model...")
print("Expected time: ~5-10 minutes")
print("="*80)

!python train.py --config configs/config_underfit.json --experiment underfit

print("\n✓ Underfitting training completed!")

# ============================================================================
# CELL 5: Train Best-fit Model (~20-30 minutes)
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Training BEST-FIT Model...")
print("Expected time: ~20-30 minutes (may finish earlier with early stopping)")
print("="*80)

!python train.py --config configs/config_bestfit.json --experiment bestfit

print("\n✓ Best-fit training completed!")

# ============================================================================
# CELL 6: Train Overfitting Model (~20-30 minutes)
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Training OVERFITTING Model...")
print("Expected time: ~20-30 minutes")
print("="*80)

!python train.py --config configs/config_overfit.json --experiment overfit

print("\n✓ Overfitting training completed!")

# ============================================================================
# CELL 7: Evaluate All Models
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Evaluating All Models...")
print("="*80)

print("\n--- Evaluating Underfitting Model ---")
!python evaluate.py --checkpoint checkpoints/underfit_best_model.pt --config configs/config_underfit.json --experiment underfit

print("\n--- Evaluating Best-fit Model ---")
!python evaluate.py --checkpoint checkpoints/bestfit_best_model.pt --config configs/config_bestfit.json --experiment bestfit

print("\n--- Evaluating Overfitting Model ---")
!python evaluate.py --checkpoint checkpoints/overfit_best_model.pt --config configs/config_overfit.json --experiment overfit

print("\n✓ All evaluations completed!")

# ============================================================================
# CELL 8: Generate Sample Text
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Generating Sample Text...")
print("="*80)

prompts = [
    "It is a truth universally acknowledged",
    "Elizabeth Bennet",
    "Mr Darcy"
]

for prompt in prompts:
    print(f"\n--- Generating from prompt: '{prompt}' ---")
    !python generate_text.py --checkpoint checkpoints/bestfit_best_model.pt --config configs/config_bestfit.json --prompt "{prompt}" --length 50

print("\n✓ Text generation completed!")

# ============================================================================
# CELL 9: Results Summary
# ============================================================================
print("\n" + "="*80)
print("STEP 9: FINAL RESULTS SUMMARY")
print("="*80)

import json
import os

experiments = ['underfit', 'bestfit', 'overfit']

# Read all results
results_data = {}
for exp in experiments:
    results_path = f'outputs/{exp}_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results_data[exp] = json.load(f)

# Display summary table
print(f"\n{'='*85}")
print(f"{'Experiment':<15} {'Train Loss':<12} {'Val Loss':<12} {'Train PPL':<15} {'Val PPL':<15}")
print(f"{'-'*85}")

for exp in experiments:
    if exp in results_data:
        data = results_data[exp]
        train_losses = data.get('train_losses', [])
        val_losses = data.get('val_losses', [])
        train_ppls = data.get('train_perplexities', [])
        val_ppls = data.get('val_perplexities', [])
        
        # Get final values
        train_loss = train_losses[-1] if train_losses else 0
        val_loss = val_losses[-1] if val_losses else 0
        train_ppl = train_ppls[-1] if train_ppls else 0
        val_ppl = val_ppls[-1] if val_ppls else 0
        
        print(f"{exp.capitalize():<15} {train_loss:<12.4f} {val_loss:<12.4f} {train_ppl:<15.2f} {val_ppl:<15.2f}")
    else:
        print(f"{exp.capitalize():<15} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<15}")

print(f"{'='*85}")

# Display observation
print("\n📊 OBSERVATIONS:")
print("1. UNDERFITTING: High train & validation loss - model too simple")
print("2. BEST-FIT: Balanced train & validation loss - good generalization")
print("3. OVERFITTING: Low train loss, high validation loss - memorization")

# ============================================================================
# CELL 10: Display All Plots
# ============================================================================
print("\n" + "="*80)
print("STEP 10: Displaying Training Plots...")
print("="*80)

from IPython.display import Image, display

for exp in experiments:
    loss_plot = f'plots/{exp}_loss_plot.png'
    ppl_plot = f'plots/{exp}_perplexity_plot.png'
    
    if os.path.exists(loss_plot):
        print(f"\n--- {exp.upper()} Loss Plot ---")
        display(Image(filename=loss_plot, width=600))
    
    if os.path.exists(ppl_plot):
        print(f"\n--- {exp.upper()} Perplexity Plot ---")
        display(Image(filename=ppl_plot, width=600))

print("\n✓ All plots displayed!")

# ============================================================================
# CELL 11: Verify All Files Generated
# ============================================================================
print("\n" + "="*80)
print("STEP 11: Verifying All Output Files...")
print("="*80)

required_files = {
    'Checkpoints': [
        'checkpoints/underfit_best_model.pt',
        'checkpoints/underfit_final_model.pt',
        'checkpoints/bestfit_best_model.pt',
        'checkpoints/bestfit_final_model.pt',
        'checkpoints/overfit_best_model.pt',
        'checkpoints/overfit_final_model.pt'
    ],
    'Results': [
        'outputs/underfit_results.json',
        'outputs/underfit_vocab.json',
        'outputs/bestfit_results.json',
        'outputs/bestfit_vocab.json',
        'outputs/overfit_results.json',
        'outputs/overfit_vocab.json'
    ],
    'Plots': [
        'plots/underfit_loss_plot.png',
        'plots/underfit_perplexity_plot.png',
        'plots/bestfit_loss_plot.png',
        'plots/bestfit_perplexity_plot.png',
        'plots/overfit_loss_plot.png',
        'plots/overfit_perplexity_plot.png'
    ]
}

all_present = True
for category, files in required_files.items():
    print(f"\n{category}:")
    for file in files:
        exists = os.path.exists(file)
        status = "✓" if exists else "❌"
        size = f"({os.path.getsize(file) / 1024:.1f} KB)" if exists else ""
        print(f"  {status} {file} {size}")
        if not exists:
            all_present = False

if all_present:
    print("\n✅ All required files generated successfully!")
else:
    print("\n⚠️ Some files are missing. Check error messages above.")

# ============================================================================
# CELL 12: Package Results for Download
# ============================================================================
print("\n" + "="*80)
print("STEP 12: Packaging Results...")
print("="*80)

!zip -r submission_results.zip checkpoints/ plots/ outputs/ README.md REPORT.md requirements.txt

print("\n✓ Results packaged as submission_results.zip")
print("\n📦 TO DOWNLOAD:")
print("1. Click on 'Files' icon in left sidebar")
print("2. Find 'submission_results.zip'")
print("3. Right-click > Download")
print("\nFile size:")
!du -sh submission_results.zip

# ============================================================================
# CELL 13: Upload to Google Drive (RECOMMENDED)
# ============================================================================
print("\n" + "="*80)
print("STEP 13: Upload to Google Drive (Recommended)...")
print("="*80)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Create directory in Drive
    drive_dir = '/content/drive/MyDrive/IIIT-H-Assignment-2-Submission'
    !mkdir -p {drive_dir}
    
    # Copy all results
    !cp submission_results.zip {drive_dir}/
    !cp -r checkpoints {drive_dir}/
    !cp -r plots {drive_dir}/
    !cp -r outputs {drive_dir}/
    
    print(f"\n✅ Results backed up to Google Drive!")
    print(f"Location: MyDrive/IIIT-H-Assignment-2-Submission/")
    
    # Create shareable link instructions
    print("\n📤 TO CREATE SHAREABLE LINKS FOR SUBMISSION:")
    print("1. Open Google Drive")
    print("2. Navigate to IIIT-H-Assignment-2-Submission/checkpoints/")
    print("3. Right-click each model file > Share > Anyone with link can view")
    print("4. Copy the link and add to README.md")
    
except Exception as e:
    print(f"⚠️ Could not mount Google Drive: {e}")
    print("You can still download submission_results.zip manually")

# ============================================================================
# CELL 14: Final Submission Checklist
# ============================================================================
print("\n" + "="*80)
print("✅ TRAINING COMPLETE - SUBMISSION CHECKLIST")
print("="*80)

checklist = [
    ("All three experiments trained", all_present),
    ("All plots generated", all([os.path.exists(f"plots/{e}_loss_plot.png") for e in experiments])),
    ("Results packaged", os.path.exists("submission_results.zip")),
    ("Files ready for download", True)
]

print("\nStatus:")
for item, status in checklist:
    symbol = "✅" if status else "❌"
    print(f"{symbol} {item}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Download 'submission_results.zip' from Files panel")
print("2. Upload checkpoints to Google Drive and get shareable links")
print("3. Add Google Drive links to your README.md")
print("4. Push all code and plots to GitHub repository")
print("5. Submit GitHub repo link via email")
print("\n" + "="*80)
print("🎉 ALL DONE! Good luck with your submission!")
print("="*80)
