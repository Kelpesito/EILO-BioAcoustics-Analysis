"""
build_loss.py

Util function to build the loss function
"""


import torch.nn as nn

from src.training.losses import FocalLoss


def build_loss_function(**kwargs) -> nn.Module:
    """
    Returns a loss function object by its name and arguments, if needed.

    Loss functions available:
    - "ce": CrossEntropyLoss
    - "fl": FocalLoss

    Returns
    -------
    nn.Module
        The loss function
    """
    
    match kwargs["loss_name"]:
        case "ce":
            return nn.CrossEntropyLoss()
        
        case "fl":
            return FocalLoss(kwargs["gamma_focal"])

    raise ValueError(f"Unknown loss function: {kwargs["loss_name"]}")
