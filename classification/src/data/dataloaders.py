"""
dataloaders.py

Module to generate the dataloaders instances to train models
"""


import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from .dataset import ImageDataset
from .transforms import get_train_transforms, get_val_transforms


NUM_WORKERS = 4
IMG_SIZE = 224


def compute_mean_std(dataset: Dataset, batch_size: int = 32) -> tuple[float, float]:
    """
    Given a dataset, computes its mean and std.

    Mean is calculated as:
    mu = 1/N * sum_{i,j}x

    Std is calculated as:
        sigma = sqrt(1/N*sum_{i,j}x^2 - mu^2),
    since:
        sigma = sqrt(V[x])
        V[x] = E[x^2] - E[x]^2
        E[x] = mu

    Parameters
    ----------
    dataset: Dataset
        Raw dataset to compute mean and std
    batch_size: int, optional
        Batch size to compute mean and std (default = 32)

    Returns
    -------
    mean: float
        Dataset mean
    std: float
        Dataset std
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    channel_sum = 0.0
    channel_sum_sq = 0.0
    n_pixels = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        
        channel_sum += images.sum(dim=[0, 2])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2])
        
        n_pixels += images.size(0) * images.size(2)
        
    mean = channel_sum / n_pixels
    std = torch.sqrt(channel_sum_sq / n_pixels - mean ** 2)

    return mean, std


def get_dataloaders(
    df: pd.DataFrame,
    origin: str,
    fold_val: int,
    class_to_idx: dict[str, int],
    batch_size: int,
    image_col: str = "id",
    label_col: str = "label",
    fold_col: str = "fold",
    img_size: int = IMG_SIZE,
) -> dict:
    """
    Get training and validation dataloaders from dataframe:
        1. Get Train and Validation split dataframes
        2. Compute mean and std from training set
        3. Generate Train and Validation sets with transformations
        4. Generate sampler (WeightedRandomSampler) to handle class imbalance
        5. Generate dataloaders

    Parameters
    ----------
    dataframe: pd.DataFrame
        The DataFrame containing the data information
    origin: str
        Path with the input images
    fold_val: int
        Value of fold to analize in validation
    class_to_idx: dict[str, int]
        Dictionary relating the original label to ordinal encoding label
    batch_size: int
        Batch size
    image_col: str, optional
        Name of the column with the path of the images (default = "id")
    label_col: str, optional
        Name of the column with the path of the labels (default = "label")
    fold_col: str, optional
        Name of the column with the fold numbers (default = "fold")
    img_size: int, optional
        Objective img_size (default = IMG_SIZE)
    
    Returns
    -------
    dict
        Dictionary with:
        - train_loader: DataLoader
            DataLoader with training set
        - val_loader: DataLoader
            DataLoader with validation set
        - mean: float
            Mean value to normalize
        - std: float
            Std value to normalize
    """
    
    # Train / Validation split
    df_val = df[df[fold_col] == fold_val]
    df_train = df[(df[fold_col] != -1) & (df[fold_col] != fold_val)]

    # Compute mean and std from training set -> Normalize
    stats_transform = T.Compose([T.ToTensor(), T.Resize((img_size, img_size))])
    train_ds_raw = ImageDataset(df_train, origin, class_to_idx, image_col, label_col, transform=stats_transform)
    mean, std = compute_mean_std(train_ds_raw, batch_size=batch_size)

    # Train and validation datasets with transformations
    train_ds = ImageDataset(
        df_train, origin, class_to_idx, image_col, label_col,
        transform=get_train_transforms(mean, std, img_size),
    )
    val_ds = ImageDataset(
        df_val, origin, class_to_idx, image_col, label_col,
        transform=get_val_transforms(mean, std, img_size),
    )
    
    # Oversampling: Weighted Random Sampler to handle class imbalance
    train_labels = df_train[label_col].map(class_to_idx).values
    class_counts = np.bincount(train_labels, minlength=len(class_to_idx))
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "mean": mean,
        "std": std,
    }
    