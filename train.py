"""
Training script for LSTM language model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.lstm import create_model
from data.preprocessing import TextPreprocessor, split_data, create_sequences
from data.dataset import get_dataloaders
from utils import set_seed, plot_losses, plot_perplexity, save_checkpoint, get_device, EarlyStopping
import math


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                clip_grad: float = 5.0) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Language model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        clip_grad: Gradient clipping value
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(inputs)
        
        # Reshape for loss calculation
        # outputs: (batch_size, seq_length, vocab_size)
        # targets: (batch_size, seq_length)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            perplexity = math.exp(avg_loss)
            print(f"  Batch [{batch_idx + 1}/{num_batches}] - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    return total_loss / num_batches


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> float:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Language model
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model(config: dict, experiment_name: str):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of the experiment (underfit/overfit/bestfit)
    """
    # Set random seed for reproducibility
    set_seed(config['seed'])
    
    # Get device
    device = get_device()
    
    print("\n" + "="*80)
    print(f"Starting experiment: {experiment_name}")
    print("="*80)
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['plots_dir'], exist_ok=True)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    preprocessor = TextPreprocessor(
        vocab_size=config['vocab_size'],
        min_freq=config['min_freq']
    )
    
    # Clean text
    text = preprocessor.clean_gutenberg_text(config['data_path'])
    print(f"Text length: {len(text)} characters")
    
    # Tokenize
    tokens = preprocessor.tokenize(text, level=config['tokenization_level'])
    print(f"Total tokens: {len(tokens)}")
    
    # Build vocabulary on training data first (to avoid data leakage)
    total_len = len(tokens)
    train_len = int(total_len * config['train_ratio'])
    train_tokens = tokens[:train_len]
    
    preprocessor.build_vocabulary(train_tokens)
    
    # Convert all tokens to indices
    indices = preprocessor.tokens_to_indices(tokens)
    
    # Split data
    train_data, val_data, test_data = split_data(
        indices,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio']
    )
    
    # Create sequences
    print("\n2. Creating sequences...")
    train_inputs, train_targets = create_sequences(train_data, config['seq_length'])
    val_inputs, val_targets = create_sequences(val_data, config['seq_length'])
    test_inputs, test_targets = create_sequences(test_data, config['seq_length'])
    
    print(f"Train sequences: {len(train_inputs)}")
    print(f"Val sequences: {len(val_inputs)}")
    print(f"Test sequences: {len(test_inputs)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_inputs, train_targets,
        val_inputs, val_targets,
        test_inputs, test_targets,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # Create model
    print("\n3. Creating model...")
    model_config = {
        'vocab_size': len(preprocessor.word2idx),
        'embedding_dim': config['embedding_dim'],
        'hidden_dim': config['hidden_dim'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout'],
        'tie_weights': config.get('tie_weights', True)
    }
    model = create_model(model_config)
    model = model.to(device)
    
    # Save vocabulary for later use
    vocab_path = os.path.join(config['output_dir'], f'{experiment_name}_vocab.json')
    vocab_dict = {
        'word2idx': preprocessor.word2idx,
        'idx2word': preprocessor.idx2word
    }
    with open(vocab_path, 'w') as f:
        json.dump(vocab_dict, f)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 5),
        verbose=True
    )
    
    # Training loop
    print("\n4. Training model...")
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch [{epoch + 1}/{config['num_epochs']}]")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config['clip_grad'])
        train_perplexity = math.exp(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        val_perplexity = math.exp(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_perplexities.append(train_perplexity)
        val_perplexities.append(val_perplexity)
        
        print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'{experiment_name}_best_model.pt'
            )
            save_checkpoint(
                model, optimizer, epoch + 1,
                train_loss, val_loss, val_perplexity,
                best_checkpoint_path, config
            )
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Save final model
    final_checkpoint_path = os.path.join(
        config['checkpoint_dir'],
        f'{experiment_name}_final_model.pt'
    )
    save_checkpoint(
        model, optimizer, epoch + 1,
        train_loss, val_loss, val_perplexity,
        final_checkpoint_path, config
    )
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    test_loss = evaluate(model, test_loader, criterion, device)
    test_perplexity = math.exp(test_loss)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.2f}")
    
    # Save results
    results = {
        'experiment_name': experiment_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_perplexities': train_perplexities,
        'val_perplexities': val_perplexities,
        'best_val_loss': best_val_loss,
        'best_val_perplexity': math.exp(best_val_loss),
        'test_loss': test_loss,
        'test_perplexity': test_perplexity,
        'config': config
    }
    
    results_path = os.path.join(config['output_dir'], f'{experiment_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    
    # Plot losses
    plot_path = os.path.join(config['plots_dir'], f'{experiment_name}_loss_plot.png')
    plot_losses(train_losses, val_losses, plot_path, 
                title=f'{experiment_name.capitalize()} - Training vs Validation Loss')
    
    # Plot perplexities
    perplexity_plot_path = os.path.join(config['plots_dir'], f'{experiment_name}_perplexity_plot.png')
    plot_perplexity(train_perplexities, val_perplexities, perplexity_plot_path,
                    title=f'{experiment_name.capitalize()} - Training vs Validation Perplexity')
    
    print("\n" + "="*80)
    print(f"Experiment '{experiment_name}' completed!")
    print(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")
    print(f"Test perplexity: {test_perplexity:.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM Language Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (underfit/overfit/bestfit)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Train model
    train_model(config, args.experiment)
