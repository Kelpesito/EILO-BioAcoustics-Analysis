"""
early_stopping.py

Early Stopping algorithm
"""


import copy
import numpy as np
import torch.nn as nn


PATIENCE = 10


class EarlyStopping:
    """
    An EarlyStopping algorith implementation.
    
    Early Stopping is a regularization method to train Deep Neural Networks to stop training when
    the model stops learning or improving its validation performance.

    Parameters
    ----------
    patience: int, optional
        Number of epochs until stop the training if the performance does not improve
        (default = PATIENCE)
    mode: str, optional
        "max" if we want to maximize the obective variable; "min" if we want to minimize it
        (default = "max")
    min_delta: float, optional
        Minimum increment of objective value to consider if the model has improved (default = 0.0)
    """
    def __init__(self, patience: int = PATIENCE, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_epoch = 0
        self.best_score = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Performs a step in EarlyStopping:
            - Sees if the model has improved or not:
                - If it has improved, save new best score, new best epoch, and new best model
                    params. Reset the epoch counter too.
                - If it has noy improved, add 1 to the epoch counter:
                    - If the epoch counter is equal to the patience, turn self.should_stop on

        Parameters
        ----------
        score: float
            Score to evaluate
        model: nn.Module
            Model to save weights
        epoch: int
            Current epoch

        Returns
        -------
        improved: bool
            Whether the model performance has improved or not. 
        """

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch

            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())

        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return improved
    