"""
data_setup.py
Creates PyTorch DataLoaders for the AisleEye tray-table dataset.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    """
    Creates training and testing DataLoaders.

    Args:
        train_dir: Path to training data directory.
        test_dir: Path to testing data directory.
        transform: torchvision transforms to perform on data.
        batch_size: Number of samples per batch.
        num_workers: Number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # faster data transfer to GPU if available
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names