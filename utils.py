"""
Utility functions for model training, plotting, and reproducibility.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Dict
import os


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def plot_losses(train_losses: List[float],
                val_losses: List[float],
                save_path: str,
                title: str = "Training vs Validation Loss"):
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss plot saved to {save_path}")


def plot_perplexity(train_perplexities: List[float],
                    val_perplexities: List[float],
                    save_path: str,
                    title: str = "Training vs Validation Perplexity"):
    """
    Plot training and validation perplexities.
    
    Args:
        train_perplexities: List of training perplexities per epoch
        val_perplexities: List of validation perplexities per epoch
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_perplexities) + 1)
    
    plt.plot(epochs, train_perplexities, 'b-o', label='Training Perplexity', linewidth=2)
    plt.plot(epochs, val_perplexities, 'r-s', label='Validation Perplexity', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Perplexity plot saved to {save_path}")


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    train_loss: float,
                    val_loss: float,
                    val_perplexity: float,
                    save_path: str,
                    config: Dict):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        val_perplexity: Validation perplexity
        save_path: Path to save checkpoint
        config: Configuration dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_perplexity': val_perplexity,
        'config': config
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    checkpoint_path: str,
                    device: torch.device):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Tuple of (epoch, train_loss, val_loss, val_perplexity)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    val_perplexity = checkpoint['val_perplexity']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {epoch}, Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
    
    return epoch, train_loss, val_loss, val_perplexity


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    
    return device


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in validation loss to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return False
