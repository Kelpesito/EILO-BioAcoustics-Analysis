"""
dataset.py

Generates the Dataset instance for training
"""


from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tifffile


class ImageDataset(Dataset):
    """
    The instance Dataset for training with images (Time-Frequrncy Representations).
    Each item is obtained as follows:
        1. Obtain the image path
        2. Read the image (.tiff)
        3. If there is `transforms`, apply it
        4. Convert label from int to tensor

    Parameters
    ---------
    dataframe: pd.DataFrame
        The DataFrame containing the data information
    origin: str
        Path with the input images
    class_to_idx: dict[str, int]
        Dictionary relating the original label to ordinal encoding label
    path_col: str, optional
        Name of the column with the path of the images (default = "id")
    label_col: str, optional
        Name of the column with the path of the labels (default = "label")
    transform: transforms.Compose | None, optional
        List of transformations to apply to the raw dataset
    in_channels: int, optional
        Number of input channels
    """
    def __init__(
            self, 
            dataframe: pd.DataFrame, 
            origin: str, 
            class_to_idx: dict[str, int], 
            path_col: str = "id", 
            label_col: str = "label", 
            transform: transforms.Compose | None = None, 
            in_channels: int = 1
        ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.origin = Path(origin)
        self.class_to_idx = class_to_idx
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform
        self.in_channels = in_channels
        self.num_classes = len(class_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(self.origin / f"{row[self.path_col]}.tiff")
        image = tifffile.imread(img_path)

        if self.transform is not None:
            image = self.transform(image)
            
        label = torch.tensor(self.class_to_idx[row[self.label_col]], dtype=torch.long)

        return image, label
    