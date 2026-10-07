"""
dataset.py

Generates the Dataset instance for training
"""


from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tifffile
from tqdm.auto import tqdm


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


class CachedImageDataset(Dataset):
    """
    The instance Dataset for training with images (Time-Frequrncy Representations), with all the
    images cached in memory.

    At initialization:
        1. Read every image (.tiff), using `num_threads` threads
        2. If there is `pre_transform` (deterministic transformations), apply it
        3. Store all the images in a single tensor (N, C, H, W) and the labels in a tensor (N,)

    Each item is obtained as follows:
        1. Get the cached image
        2. If there is `transform` (random transformations, Data Augmentation), apply it

    `transform` must not modify the input tensor in-place, otherwise the cache is corrupted.

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
    pre_transform: transforms.Compose | None, optional
        Deterministic transformations, applied once when caching. Its output must be a tensor
    transform: transforms.Compose | None, optional
        Random transformations, applied to the cached tensor in each `__getitem__`
    in_channels: int, optional
        Number of input channels
    num_threads: int, optional
        Number of threads to read the images (default = 8)
    """
    def __init__(
            self,
            dataframe: pd.DataFrame,
            origin: str,
            class_to_idx: dict[str, int],
            path_col: str = "id",
            label_col: str = "label",
            pre_transform: transforms.Compose | None = None,
            transform: transforms.Compose | None = None,
            in_channels: int = 1,
            num_threads: int = 8,
        ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.origin = Path(origin)
        self.class_to_idx = class_to_idx
        self.path_col = path_col
        self.label_col = label_col
        self.pre_transform = pre_transform
        self.transform = transform
        self.in_channels = in_channels
        self.num_classes = len(class_to_idx)

        self.labels = torch.tensor(
            self.df[label_col].map(class_to_idx).values, dtype=torch.long
        )
        self.images = self._cache_images(num_threads)

    def _load_image(self, img_id) -> torch.Tensor:
        image = tifffile.imread(self.origin / f"{img_id}.tiff")

        if self.pre_transform is not None:
            image = self.pre_transform(image)

        return torch.as_tensor(image)

    def _cache_images(self, num_threads: int) -> torch.Tensor:
        ids = self.df[self.path_col].tolist()
        images = None

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(self._load_image, ids)
            for i, image in enumerate(tqdm(results, total=len(ids), desc="Caching images", leave=False)):
                # Preallocate with the shape of the first image (avoids a second copy with stack)
                if images is None:
                    images = torch.empty((len(ids), *image.shape), dtype=image.dtype)
                images[i] = image

        return images

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[idx]
