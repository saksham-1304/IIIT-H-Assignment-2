"""
Data preprocessing module for Pride and Prejudice dataset.
Handles text cleaning, tokenization, vocabulary building, and data splitting.
"""

import re
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict


class TextPreprocessor:
    """Preprocesses text data for language modeling."""
    
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        """
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a word to be included in vocab
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        
    def clean_gutenberg_text(self, filepath: str) -> str:
        """
        Remove Project Gutenberg metadata and clean the text.
        
        Args:
            filepath: Path to the Pride and Prejudice text file
            
        Returns:
            Cleaned text content
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Find start and end markers
        start_marker = "***START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE***"
        end_marker = "***END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE***"
        
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            # Extract content between markers
            text = text[start_idx + len(start_marker):end_idx]
        
        # Additional cleaning
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str, level: str = 'word') -> List[str]:
        """
        Tokenize text at word or character level.
        
        Args:
            text: Input text
            level: 'word' or 'char' for tokenization level
            
        Returns:
            List of tokens
        """
        if level == 'word':
            # Convert to lowercase and split on whitespace/punctuation
            text = text.lower()
            # Keep apostrophes but split on other punctuation
            tokens = re.findall(r"\b\w+(?:'\w+)?\b|[.,!?;]", text)
            return tokens
        elif level == 'char':
            return list(text)
        else:
            raise ValueError(f"Unknown tokenization level: {level}")
    
    def build_vocabulary(self, tokens: List[str]) -> None:
        """
        Build vocabulary from tokens.
        
        Args:
            tokens: List of tokens from training data
        """
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Start with special tokens
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.special_tokens.items()}
        
        # Add most frequent tokens
        most_common = token_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        idx = len(self.special_tokens)
        for token, count in most_common:
            if count >= self.min_freq:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Most common tokens: {most_common[:20]}")
    
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to indices using vocabulary.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of indices
        """
        unk_idx = self.word2idx['<UNK>']
        return [self.word2idx.get(token, unk_idx) for token in tokens]
    
    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        """
        Convert indices back to tokens.
        
        Args:
            indices: List of indices
            
        Returns:
            List of tokens
        """
        return [self.idx2word.get(idx, '<UNK>') for idx in indices]


def split_data(indices: List[int], 
               train_ratio: float = 0.8, 
               val_ratio: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        indices: List of token indices
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    total_len = len(indices)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    train_data = indices[:train_len]
    val_data = indices[train_len:train_len + val_len]
    test_data = indices[train_len + val_len:]
    
    print(f"\nData split:")
    print(f"Train: {len(train_data)} tokens ({len(train_data)/total_len*100:.1f}%)")
    print(f"Val: {len(val_data)} tokens ({len(val_data)/total_len*100:.1f}%)")
    print(f"Test: {len(test_data)} tokens ({len(test_data)/total_len*100:.1f}%)")
    
    return train_data, val_data, test_data


def create_sequences(data: List[int], 
                     seq_length: int = 35) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-target sequence pairs for language modeling.
    
    Args:
        data: List of token indices
        seq_length: Length of input sequences
        
    Returns:
        Tuple of (inputs, targets) as numpy arrays
    """
    num_sequences = len(data) // seq_length
    # Trim data to fit evenly into sequences
    data = data[:num_sequences * seq_length]
    
    inputs = []
    targets = []
    
    for i in range(0, len(data) - seq_length):
        input_seq = data[i:i + seq_length]
        target_seq = data[i + 1:i + seq_length + 1]
        inputs.append(input_seq)
        targets.append(target_seq)
    
    return np.array(inputs), np.array(targets)
