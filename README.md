# Neural Language Model Training - Pride and Prejudice

A from-scratch PyTorch implementation of a neural language model trained on Jane Austen's "Pride and Prejudice". This project demonstrates understanding of sequence models, training dynamics (underfitting/overfitting/best-fit), and model evaluation using perplexity.

## 📋 Project Overview

- **Framework**: PyTorch (implemented from scratch, no pre-trained models)
- **Dataset**: Pride and Prejudice by Jane Austen (Project Gutenberg)
- **Model**: LSTM-based language model
- **Evaluation Metric**: Perplexity
- **Experiments**: Three scenarios demonstrating underfitting, overfitting, and best-fit



## 🗂️ Project Structure

```
IIIT-H-Assignment-2/
├── data/
│   ├── preprocessing.py      # Text preprocessing and tokenization
│   └── dataset.py            # PyTorch Dataset/DataLoader
├── models/
│   └── lstm.py               # LSTM language model architecture
├── configs/
│   ├── config_underfit.json  # Underfitting configuration
│   ├── config_bestfit.json   # Best-fit configuration
│   └── config_overfit.json   # Overfitting configuration
├── dataset/
│   └── Pride_and_Prejudice-Jane_Austen.txt
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── generate_text.py          # Text generation script
├── utils.py                  # Utility functions (plotting, checkpointing)
├── run_all_experiments.py    # Run all experiments sequentially
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── REPORT.md                 # Detailed report
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/saksham-1304/IIIT-H-Assignment-2.git
   cd IIIT-H-Assignment-2
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   
   # On Windows:
   .\venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Running the Experiments

### Option 1: Run All Experiments at Once

```bash
python run_all_experiments.py
```

This will train all three models (underfit, bestfit, overfit) sequentially.

### Option 2: Run Individual Experiments

**Underfitting Experiment:**

```bash
python train.py --config configs/config_underfit.json --experiment underfit
```

**Best-fit Experiment:**

```bash
python train.py --config configs/config_bestfit.json --experiment bestfit
```

**Overfitting Experiment:**

```bash
python train.py --config configs/config_overfit.json --experiment overfit
```

## 📈 Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint checkpoints/bestfit_best_model.pt --config configs/config_bestfit.json --experiment bestfit
```

This will:
- Calculate perplexity on train/val/test sets
- Generate sample text from the model

## 🎨 Text Generation

To generate text using a trained model:

```bash
python generate_text.py --checkpoint checkpoints/bestfit_best_model.pt --config configs/config_bestfit.json --prompt "It is a truth"
```

## 📊 Experiment Configurations

### 1. Underfitting Configuration

- **Goal**: Demonstrate insufficient model capacity
- **Model**: Small (1 layer, 64 hidden units)
- **Training**: Short (10 epochs)
- **Regularization**: None
- **Expected**: High training and validation loss

### 2. Best-fit Configuration

- **Goal**: Achieve optimal generalization
- **Model**: Medium (2 layers, 256 hidden units)
- **Training**: Moderate (up to 50 epochs with early stopping)
- **Regularization**: Dropout (0.3), weight tying
- **Expected**: Low validation loss, good generalization

### 3. Overfitting Configuration

- **Goal**: Demonstrate overfitting behavior
- **Model**: Large (3 layers, 512 hidden units)
- **Training**: Long (100 epochs)
- **Regularization**: None
- **Expected**: Very low training loss, high validation loss

## 📁 Output Files

After training, the following files will be generated:

### Checkpoints (in `checkpoints/`)

- `{experiment}_best_model.pt` - Best model based on validation loss
- `{experiment}_final_model.pt` - Final model after all epochs

### Plots (in `plots/`)

- `{experiment}_loss_plot.png` - Training vs validation loss curves
- `{experiment}_perplexity_plot.png` - Training vs validation perplexity curves

### Results (in `outputs/`)

- `{experiment}_results.json` - Detailed metrics and results
- `{experiment}_vocab.json` - Vocabulary mappings

## 🔬 Model Architecture

The LSTM language model consists of:

1. **Embedding Layer**: Converts token indices to dense vectors
2. **LSTM Layers**: Process sequences and capture dependencies
3. **Dropout**: Regularization technique (in best-fit model)
4. **Output Layer**: Projects to vocabulary size
5. **Weight Tying** (optional): Shares weights between embedding and output layers

### Model Parameters by Configuration

| Configuration | Embedding Dim | Hidden Dim | Layers | Dropout | Parameters |
|--------------|---------------|------------|--------|---------|------------|
| Underfit     | 64            | 64         | 1      | 0.0     | ~320K      |
| Best-fit     | 256           | 256        | 2      | 0.3     | ~5M        |
| Overfit      | 512           | 512        | 3      | 0.0     | ~20M       |

## 📊 Results Summary

### Expected Performance (Perplexity)

| Experiment | Train Perplexity | Val Perplexity | Test Perplexity | Status |
|-----------|------------------|----------------|-----------------|--------|
| Underfit  | ~200-300         | ~200-300       | ~200-300        | High   |
| Best-fit  | ~80-120          | ~90-130        | ~90-130         | Good   |
| Overfit   | ~30-50           | ~180-250       | ~180-250        | Gap    |

Note: Actual results may vary based on training runs

## 🎓 Key Learnings

1. **Underfitting**: Small model capacity leads to poor performance on both training and validation sets
2. **Best-fit**: Balanced model with proper regularization achieves good generalization
3. **Overfitting**: Large model without regularization memorizes training data but fails on validation

## 🔧 Technical Details

### Reproducibility

- Fixed random seeds (PyTorch, NumPy, Python random)
- Deterministic CUDA operations
- Documented Python and package versions

### Training Features

- Gradient clipping to prevent exploding gradients
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping to prevent overfitting
- Checkpoint saving for best model

### Data Preprocessing

- Gutenberg metadata removal
- Word-level tokenization
- Vocabulary building (only on training data)
- 80/10/10 train/val/test split

## 📝 Trained Model Links

Due to file size limitations, trained model checkpoints are hosted on Google Drive:

- **Best-fit Model**: [Download Link](https://drive.google.com/file/d/YOUR_FILE_ID)
- **Underfit Model**: [Download Link](https://drive.google.com/file/d/YOUR_FILE_ID)
- **Overfit Model**: [Download Link](https://drive.google.com/file/d/YOUR_FILE_ID)

Note: Replace with actual Google Drive links after training

## 🐛 Troubleshooting

### Out of Memory Error

- Reduce `batch_size` in config files
- Use smaller `hidden_dim` or `embedding_dim`
- Reduce `seq_length`

### Slow Training

- Use GPU if available
- Increase `batch_size` (if memory allows)
- Reduce `num_epochs`

### Import Errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (3.8+ required)

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Pride and Prejudice (Project Gutenberg)](https://www.gutenberg.org/ebooks/42671)
- LSTM Paper: Hochreiter & Schmidhuber (1997)

## 👤 Author

**Saksham Singh Rathore**

- GitHub: [@saksham-1304](https://github.com/saksham-1304)
- Repository: [IIIT-H-Assignment-2](https://github.com/saksham-1304/IIIT-H-Assignment-2)

## 📄 License

This project is for educational purposes as part of IIIT Hyderabad Assignment 2.

## 🙏 Acknowledgments

- IIIT Hyderabad for the assignment
- Jane Austen for Pride and Prejudice
- Project Gutenberg for making texts freely available

- LSTM Paper: Hochreiter & Schmidhuber (1997)

## 📄 License

This project is for educational purposes as part of IIIT Hyderabad Assignment 2.

## 🙏 Acknowledgments

- IIIT Hyderabad for the assignment
- Jane Austen for Pride and Prejudice
- Project Gutenberg for making texts freely available
