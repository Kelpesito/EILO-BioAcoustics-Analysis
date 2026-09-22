"""
build_scheduler.py

Util function to build the Learning Rate Scheduler
"""


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from src.training.callbacks.lwu_ca import lwu_ca


def build_scheduler(scheduler_name: str, optimizer: Optimizer, **kwargs) -> LRScheduler:
    match scheduler_name:
        case "lwu_ca":
            return lwu_ca(
                optimizer=optimizer,
                lr_max=kwargs["lr_max"],
                lr_min_ratio=kwargs["lr_min_ratio"],
                num_epochs=kwargs["num_epochs"],
                warmup_epochs=kwargs["warmup_epochs"],
            )

        case "rlrop":
            return ReduceLROnPlateau(
                optimizer=optimizer,
                mode=kwargs["mode"],
                factor=kwargs["factor"],
                patience=kwargs["patience"],
                threshold_mode='abs',
                cooldown=kwargs["cooldown"],
            )