"""
LSTM Language Model implementation in PyTorch.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMLanguageModel(nn.Module):
    """LSTM-based Language Model for next token prediction."""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 tie_weights: bool = True):
        """
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            tie_weights: Whether to tie input and output embeddings
        """
        super(LSTMLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output projection layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Weight tying: share weights between embedding and output layer
        if tie_weights:
            if embedding_dim != hidden_dim:
                raise ValueError("When tying weights, embedding_dim must equal hidden_dim")
            self.fc.weight = self.embedding.weight
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)
    
    def forward(self,
                input_seq: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_seq: Input sequence (batch_size, seq_length)
            hidden: Hidden state tuple (h, c) from previous step
            
        Returns:
            Tuple of (output logits, hidden state)
            - output: (batch_size, seq_length, vocab_size)
            - hidden: Tuple of (h, c) tensors
        """
        # Embed input tokens
        embedded = self.embedding(input_seq)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout_layer(embedded)
        
        # Pass through LSTM
        if hidden is not None:
            lstm_out, hidden = self.lstm(embedded, hidden)
        else:
            lstm_out, hidden = self.lstm(embedded)
        
        # Apply dropout to LSTM output
        lstm_out = self.dropout_layer(lstm_out)  # (batch_size, seq_length, hidden_dim)
        
        # Project to vocabulary size
        output = self.fc(lstm_out)  # (batch_size, seq_length, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (h0, c0) tensors
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


def create_model(config: dict) -> LSTMLanguageModel:
    """
    Create LSTM language model from configuration.
    
    Args:
        config: Dictionary containing model hyperparameters
        
    Returns:
        LSTMLanguageModel instance
    """
    model = LSTMLanguageModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config.get('embedding_dim', 256),
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        tie_weights=config.get('tie_weights', True)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Embedding dim: {config.get('embedding_dim', 256)}")
    print(f"Hidden dim: {config.get('hidden_dim', 512)}")
    print(f"Num layers: {config.get('num_layers', 2)}")
    print(f"Dropout: {config.get('dropout', 0.3)}")
    
    return model
