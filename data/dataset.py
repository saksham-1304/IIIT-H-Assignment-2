"""
PyTorch Dataset and DataLoader implementations for language modeling.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class LanguageModelDataset(Dataset):
    """PyTorch Dataset for language modeling."""
    
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Args:
            inputs: Input sequences (num_sequences, seq_length)
            targets: Target sequences (num_sequences, seq_length)
        """
        self.inputs = torch.from_numpy(inputs).long()
        self.targets = torch.from_numpy(targets).long()
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def get_dataloaders(train_inputs: np.ndarray,
                    train_targets: np.ndarray,
                    val_inputs: np.ndarray,
                    val_targets: np.ndarray,
                    test_inputs: np.ndarray,
                    test_targets: np.ndarray,
                    batch_size: int = 64,
                    shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_inputs: Training input sequences
        train_targets: Training target sequences
        val_inputs: Validation input sequences
        val_targets: Validation target sequences
        test_inputs: Test input sequences
        test_targets: Test target sequences
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = LanguageModelDataset(train_inputs, train_targets)
    val_dataset = LanguageModelDataset(val_inputs, val_targets)
    test_dataset = LanguageModelDataset(test_inputs, test_targets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nDataLoader info:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader
