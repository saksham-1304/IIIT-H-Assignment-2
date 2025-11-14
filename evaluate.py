"""
Evaluation script for LSTM language model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
import os
import sys
from pathlib import Path
import math

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.lstm import LSTMLanguageModel
from data.preprocessing import TextPreprocessor, split_data, create_sequences
from data.dataset import get_dataloaders
from utils import set_seed, get_device


def calculate_perplexity(model: nn.Module,
                         dataloader: DataLoader,
                         criterion: nn.Module,
                         device: torch.device) -> tuple:
    """
    Calculate perplexity on a dataset.
    
    Args:
        model: Language model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Tuple of (loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Reshape for loss calculation
            batch_size, seq_length, vocab_size = outputs.size()
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Accumulate
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def generate_text(model: nn.Module,
                  preprocessor: TextPreprocessor,
                  start_text: str,
                  max_length: int,
                  device: torch.device,
                  temperature: float = 1.0) -> str:
    """
    Generate text using the trained model.
    
    Args:
        model: Trained language model
        preprocessor: Text preprocessor with vocabulary
        start_text: Starting text for generation
        max_length: Maximum number of tokens to generate
        device: Device to run on
        temperature: Sampling temperature (higher = more random)
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Tokenize start text
    tokens = preprocessor.tokenize(start_text.lower(), level='word')
    indices = preprocessor.tokens_to_indices(tokens)
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_seq = torch.tensor([indices], dtype=torch.long).to(device)
            
            # Get predictions
            outputs, _ = model(input_seq)
            
            # Get last token's predictions
            logits = outputs[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=0)
            
            # Sample from distribution
            next_idx = torch.multinomial(probs, num_samples=1).item()
            
            # Add to generated sequence
            next_token = preprocessor.idx2word.get(next_idx, '<UNK>')
            generated.append(next_token)
            
            # Update input sequence (keep last 35 tokens for context)
            indices.append(next_idx)
            indices = indices[-35:]
            
            # Stop if we hit end token
            if next_token in ['<EOS>', '.', '!', '?'] and len(generated) > 10:
                break
    
    return ' '.join(generated)


def evaluate_model(checkpoint_path: str, config_path: str, experiment_name: str):
    """
    Evaluate a trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        experiment_name: Name of the experiment
    """
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device()
    
    print("\n" + "="*80)
    print(f"Evaluating model: {experiment_name}")
    print("="*80)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    preprocessor = TextPreprocessor(
        vocab_size=config['vocab_size'],
        min_freq=config['min_freq']
    )
    
    # Clean text
    text = preprocessor.clean_gutenberg_text(config['data_path'])
    
    # Tokenize
    tokens = preprocessor.tokenize(text, level=config['tokenization_level'])
    
    # Build vocabulary
    total_len = len(tokens)
    train_len = int(total_len * config['train_ratio'])
    train_tokens = tokens[:train_len]
    preprocessor.build_vocabulary(train_tokens)
    
    # Convert to indices
    indices = preprocessor.tokens_to_indices(tokens)
    
    # Split data
    train_data, val_data, test_data = split_data(
        indices,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio']
    )
    
    # Create sequences
    train_inputs, train_targets = create_sequences(train_data, config['seq_length'])
    val_inputs, val_targets = create_sequences(val_data, config['seq_length'])
    test_inputs, test_targets = create_sequences(test_data, config['seq_length'])
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_inputs, train_targets,
        val_inputs, val_targets,
        test_inputs, test_targets,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Create model
    print("\n2. Loading model...")
    model = LSTMLanguageModel(
        vocab_size=len(preprocessor.word2idx),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        tie_weights=config.get('tie_weights', True)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Define loss
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate on all splits
    print("\n3. Calculating perplexity...")
    
    train_loss, train_ppl = calculate_perplexity(model, train_loader, criterion, device)
    print(f"Train - Loss: {train_loss:.4f}, Perplexity: {train_ppl:.2f}")
    
    val_loss, val_ppl = calculate_perplexity(model, val_loader, criterion, device)
    print(f"Val   - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
    
    test_loss, test_ppl = calculate_perplexity(model, test_loader, criterion, device)
    print(f"Test  - Loss: {test_loss:.4f}, Perplexity: {test_ppl:.2f}")
    
    # Generate sample text
    print("\n4. Generating sample text...")
    start_texts = [
        "it is a truth universally acknowledged",
        "mr darcy",
        "elizabeth bennet"
    ]
    
    for start_text in start_texts:
        generated = generate_text(model, preprocessor, start_text, max_length=50, 
                                  device=device, temperature=0.8)
        print(f"\nStart: '{start_text}'")
        print(f"Generated: {generated}")
    
    print("\n" + "="*80)
    print("Evaluation completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LSTM Language Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name')
    
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.config, args.experiment)
