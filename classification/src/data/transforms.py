"""
transforms.py

Module with the different transformations to apply to the datasets
"""


import numpy as np
import torch
from torchvision import transforms


SILENCE_VAL = -140.0
IMG_SIZE = 224


class Normalize:
    """
    Normalization of image by batch mean and std.

    Parameters
    ----------
    mean: float, optional
        Batch mean value (default = 0.5)
    std: float, optional
        Batch std value (default = 0.5)
    """
    def __init__(self, mean: float = 0.5, std: float = 0.5):

        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


class TemporalShift:
    """
    Performs a temporal shift in the image.

    Parameters
    ----------
    max_shift: float, optional
        Maximum fraction of width to time shift (default = 0.1)
    p: float, optional
        Probability to apply the transformation (default = 0.5)
    fill_value: float, optional
        Zero-value to fill (default = SILENCE_VAL)
    """
    def __init__(self, max_shift: float = 0.1, p: float = 0.5, fill_value: float = SILENCE_VAL):
        self.max_shift = max_shift
        self.p = p
        self.fill_value = fill_value

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x

        _, h, w = x.shape

        shift = torch.randint(int(np.floor(-self.max_shift*w)), int(np.floor(self.max_shift*w)) + 1, (1,)).item()

        if shift == 0:
            return x

        x = x.clone()
        if shift > 0:
            x[:, :, shift:] = x[:, :, :-shift]
            x[:, :, :shift] = self.fill_value

        else:
            shift = abs(shift)
            x[:, :, :-shift] = x[:, :, shift:]
            x[:, :, -shift:] = self.fill_value

        return x


class TimeMask:
    """
    Applies a mask to a full time band.

    Parameters
    ----------
    max_width: float, optional
        Maximum fraction of width to the time mask (default = 0.1)
    p: float, optional
        Probability to apply the transformation (default = 0.5)
    fill_value: float, optional
        Zero-value to fill (default = SILENCE_VAL)
    """
    def __init__(self, max_width=0.1, p=0.5, fill_value=SILENCE_VAL):
        self.max_width = max_width
        self.p = p
        self.fill_value = fill_value

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x

        _, h, w = x.shape
        width = torch.randint(1, int(np.floor(self.max_width*w)) + 1, (1,)).item()
        start = torch.randint(0, max(1, w - width) + 1, (1,)).item()

        x = x.clone()
        x[:, :, start:start + width] = self.fill_value
        return x


class FreqMask:
    """
    Applies a mask to a full frequency band.

    Parameters
    ----------
    max_width: float, optional
        Maximum fraction of width to the frequency mask (default = 0.1)
    p: float, optional
        Probability to apply the transformation (default = 0.5)
    fill_value: float, optional
        Zero-value to fill (default = SILENCE_VAL)
        """
    def __init__(self, max_height: float = 0.1, p: float = 0.5, fill_value: float = SILENCE_VAL):
        self.max_height = max_height
        self.p = p
        self.fill_value = fill_value

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x

        _, h, w = x.shape
        height = torch.randint(1, int(np.floor(self.max_height*h)) + 1, (1,)).item()
        start = torch.randint(0, max(1, h - height) + 1, (1,)).item()

        x = x.clone()
        x[:, start:start + height, :] = self.fill_value
        return x


class AddGaussianNoise:
    """
    Adds gaussian noise to the image.

    Parameters
    ----------
    mean: float, optional
        Mean value of noise (default = 0.0)
    std_range: tuple[float, float]
        Min and max possible values for noise std (default = (0.01, 0.1))
    p: float, optional
        Probability to apply the transformation (default = 0.5)
    """
    def __init__(
            self, 
            mean: float = 0.0,
            std_range: tuple[float, float] = (0.01, 0.1), 
            p: float = 0.5
        ):
        self.mean = mean
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() > self.p:
            return tensor

        std = torch.empty(1).uniform_(*self.std_range).item()
        noise = torch.randn_like(tensor) * std + self.mean
        return tensor + noise


def get_train_transforms(mean: float, std: float, img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Returns the set of transformations for training sets:
    ToTensor + Resize + Normalize + DataAugmentation

    Data Augmentation:
    TemporalShift + TimeMask + FreqMask + AddGaussianNoise

    Parameters
    ----------
    mean: float
        Mean value to normalize
    std: float
        Std value to normalize
    img_size: int, optional
        Image size (default = IMG_SIZE)
    
    Returns
    -------
    transforms.Compose
        The transormations to apply
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        TemporalShift(max_shift=0.1, p=0.5, fill_value=SILENCE_VAL),
        TimeMask(max_width=0.1, p=0.5, fill_value=SILENCE_VAL),
        FreqMask(max_height=0.1, p=0.5, fill_value=SILENCE_VAL),
        Normalize(mean=mean, std=std),
        AddGaussianNoise(std_range=(0.01, 0.1), p=0.5),
    ])


def get_val_transforms(mean: float, std: float, img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Returns the set of transformations for validation / test sets:
    ToTensor + Resize + Normalize

    Parameters
    ----------
    mean: float
        Mean value to normalize
    std: float
        Std value to normalize
    img_size: int, optional
        Image size (default = IMG_SIZE)
        
    Returns
    -------
    transforms.Compose
        The transormations to apply
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        Normalize(mean=mean, std=std),
    ])
