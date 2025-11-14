# Quick Test Script
# Run this to verify everything works before full training

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  QUICK TEST SCRIPT" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Test 1: Python Version
Write-Host "[1/8] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python not found!" -ForegroundColor Red
    exit 1
}

# Test 2: PyTorch Installation
Write-Host "`n[2/8] Checking PyTorch installation..." -ForegroundColor Yellow
python -c "import torch; print('  ✓ PyTorch version:', torch.__version__); print('  ✓ CUDA available:', torch.cuda.is_available())" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ PyTorch not installed!" -ForegroundColor Red
    Write-Host "  Run: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Test 3: Required Imports
Write-Host "`n[3/8] Checking imports..." -ForegroundColor Yellow
python -c "from models.lstm import create_model; from data.preprocessing import TextPreprocessor; from data.dataset import TextDataset; from utils import set_seed, plot_losses; print('  ✓ All imports successful')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Import failed!" -ForegroundColor Red
    exit 1
}

# Test 4: Dataset
Write-Host "`n[4/8] Checking dataset..." -ForegroundColor Yellow
if (Test-Path "dataset/Pride_and_Prejudice-Jane_Austen.txt") {
    $fileSize = (Get-Item "dataset/Pride_and_Prejudice-Jane_Austen.txt").Length / 1KB
    Write-Host "  ✓ Dataset found ($([math]::Round($fileSize, 2)) KB)" -ForegroundColor Green
} else {
    Write-Host "  ✗ Dataset not found!" -ForegroundColor Red
    exit 1
}

# Test 5: Config Files
Write-Host "`n[5/8] Checking configuration files..." -ForegroundColor Yellow
$configs = @("config_underfit.json", "config_bestfit.json", "config_overfit.json")
$allPresent = $true
foreach ($config in $configs) {
    if (Test-Path "configs/$config") {
        Write-Host "  ✓ $config" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $config missing!" -ForegroundColor Red
        $allPresent = $false
    }
}
if (-not $allPresent) { exit 1 }

# Test 6: Output Directories
Write-Host "`n[6/8] Checking/creating output directories..." -ForegroundColor Yellow
$dirs = @("checkpoints", "outputs", "plots")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✓ Created $dir/" -ForegroundColor Green
    } else {
        Write-Host "  ✓ $dir/ exists" -ForegroundColor Green
    }
}

# Test 7: Data Loading Test
Write-Host "`n[7/8] Testing data preprocessing..." -ForegroundColor Yellow
python -c @"
from data.preprocessing import TextPreprocessor
import sys

try:
    preprocessor = TextPreprocessor()
    text = preprocessor.load_text('dataset/Pride_and_Prejudice-Jane_Austen.txt')
    print(f'  ✓ Loaded {len(text.split())} words from dataset')
    
    preprocessor.build_vocab([text])
    print(f'  ✓ Built vocabulary with {len(preprocessor.vocab)} unique tokens')
    
    sys.exit(0)
except Exception as e:
    print(f'  ✗ Error: {e}')
    sys.exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Data preprocessing failed!" -ForegroundColor Red
    exit 1
}

# Test 8: Model Creation Test
Write-Host "`n[8/8] Testing model creation..." -ForegroundColor Yellow
python -c @"
from models.lstm import create_model
import torch
import json
import sys

try:
    with open('configs/config_underfit.json', 'r') as f:
        config = json.load(f)
    
    model = create_model(
        vocab_size=1000,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f'  ✓ Model created successfully')
    print(f'  ✓ Total parameters: {param_count:,}')
    
    sys.exit(0)
except Exception as e:
    print(f'  ✗ Error: {e}')
    sys.exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Model creation failed!" -ForegroundColor Red
    exit 1
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ALL TESTS PASSED! ✓" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nYou're ready to run training!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Quick test (10 min):  python train.py --config configs/config_underfit.json --experiment underfit" -ForegroundColor White
Write-Host "  2. Full training:        python run_all_experiments.py" -ForegroundColor White
Write-Host "  3. Or see:               SETUP_AND_RUN_GUIDE.md for detailed instructions`n" -ForegroundColor White
