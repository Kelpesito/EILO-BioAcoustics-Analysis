"""
build_scheduler.py

Util function to build the Learning Rate Scheduler
"""


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ConstantLR, SequentialLR


WARMUP_EPOCHS = 5


def build_scheduler(
    optimizer: Optimizer,
    lr_max: float,
    lr_min_ratio: float,
    num_epochs: int,
    warmup_epochs: int = WARMUP_EPOCHS,
) -> SequentialLR:
    """
    Builds the following Learning Rate Scheduler:
    - Epoch 1 to warmup_epochs: LinearLR (lr_min_ratio * lr_max -> lr_max)
    - Epoch warmup_epochs + 1 to num_epochs: CosineAnnealingLR
        (lr_max -> lr_min_ratio * lr_max)
    - Epoch num_epochs + 1 to end: ConstantLR (lr_min_ratio * lr_max)

    Parameters
    ----------
    optimizer: Optimizer
        The optimizer algorithm instance
    lr_max: float
        Maximum learning rate
    lr_min_ratio: float
        Ratio between minimum learning rate anf maximum. lr_min = lr_min_ratio * lr_max
    num_epochs: int
        Number of epochs of Linear Warm Up + Cosine Annealing
    warmup_epochs: int, optional
        Number of epochs in Linear Warm Up (default = WARMUP_EPOCHS)
    """
    if num_epochs <= warmup_epochs:
        raise ValueError(
            "num_epochs must be greater than warmup_epochs."
        )

    # Linear warm-up: lr_min_ratio * lr -> lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=lr_min_ratio,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Cosine decay: lr -> lr_min
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=lr_max * lr_min_ratio,
    )
    
    # Constant Learning rate
    constant_scheduler = ConstantLR(
        optimizer,
        factor=lr_min_ratio,
        total_iters=10**9,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
        milestones=[warmup_epochs, num_epochs],
    )

    return scheduler
