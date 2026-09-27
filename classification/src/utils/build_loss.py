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

    If a "class_weights" tensor is passed, it is used to weight the loss per class to handle
    class imbalance ("weight" for CrossEntropyLoss, "alpha" for FocalLoss).

    Returns
    -------
    nn.Module
        The loss function
    """
    class_weights = kwargs.get("class_weights")

    match kwargs["loss_name"]:
        case "ce":
            return nn.CrossEntropyLoss(weight=class_weights)

        case "fl":
            return FocalLoss(kwargs["gamma_focal"], alpha=class_weights)

    raise ValueError(f"Unknown loss function: {kwargs["loss_name"]}")
