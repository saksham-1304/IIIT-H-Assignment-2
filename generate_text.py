"""
Simple text generation script using trained model.
"""

import torch
import json
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.lstm import LSTMLanguageModel
from data.preprocessing import TextPreprocessor


def load_model_and_vocab(checkpoint_path, vocab_path, device):
    """Load trained model and vocabulary."""
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    
    # Create preprocessor and load vocab
    preprocessor = TextPreprocessor()
    preprocessor.word2idx = vocab_dict['word2idx']
    preprocessor.idx2word = {int(k): v for k, v in vocab_dict['idx2word'].items()}
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = LSTMLanguageModel(
        vocab_size=len(preprocessor.word2idx),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=0.0,  # No dropout during inference
        tie_weights=config.get('tie_weights', True)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, preprocessor


def generate_text(model, preprocessor, start_text, max_length=100, 
                  temperature=0.8, device=None):
    """Generate text from a trained model."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tokenize start text
    tokens = preprocessor.tokenize(start_text.lower(), level='word')
    indices = preprocessor.tokens_to_indices(tokens)
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input (keep last 35 tokens for context)
            input_indices = indices[-35:]
            input_seq = torch.tensor([input_indices], dtype=torch.long).to(device)
            
            # Get predictions
            outputs, _ = model(input_seq)
            
            # Get last token's predictions
            logits = outputs[0, -1, :] / temperature
            
            # Apply softmax
            probs = torch.softmax(logits, dim=0)
            
            # Sample from distribution
            next_idx = torch.multinomial(probs, num_samples=1).item()
            
            # Get token
            next_token = preprocessor.idx2word.get(next_idx, '<UNK>')
            
            # Stop at sentence end
            if next_token in ['.', '!', '?']:
                generated.append(next_token)
                if len(generated) > 20:  # Minimum length
                    break
            else:
                generated.append(next_token)
            
            # Add to context
            indices.append(next_idx)
    
    return ' '.join(generated)


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--start', type=str, default="it is a truth",
                        help='Starting text for generation')
    parser.add_argument('--length', type=int, default=100,
                        help='Maximum length to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading model...")
    model, preprocessor = load_model_and_vocab(
        args.checkpoint,
        args.vocab,
        device
    )
    print("Model loaded successfully!\n")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples with start text: '{args.start}'\n")
    print("="*80)
    
    for i in range(args.num_samples):
        generated = generate_text(
            model, preprocessor, args.start,
            max_length=args.length,
            temperature=args.temperature,
            device=device
        )
        print(f"\nSample {i+1}:")
        print(generated)
        print("-"*80)


if __name__ == "__main__":
    main()
