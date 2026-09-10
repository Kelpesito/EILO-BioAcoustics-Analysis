"""
build_optimizer.py

Util function to build the optimizer
"""


from torch.optim import SGD, Adam, AdamW, Optimizer


def build_optimizer(model, **kwargs) -> Optimizer:
    """
    Returns an optimpizer object by its name and arguments, if needed.

    Optimizers available:
    - "sgd": SGD
    - "adam": Adam
    - "adamw": AdamW

    Returns
    -------
    Optimizer
        The optimizer
    """
    
    match kwargs["optimizer_name"]:
        case "sgd":
            return SGD(
                model.parameters(),
                lr=kwargs["lr_max"],
                momentum=0.9,
                weight_decay=kwargs["weight_decay"],
                nesterov=True,
            )
        
        case "adam":
            return Adam(
                model.parameters(),
                lr=kwargs["lr_max"],
                weight_decay=kwargs["weight_decay"],
            )

        case "adamw":
            return AdamW(
                model.parameters(),
                lr=kwargs["lr_max"],
                weight_decay=kwargs["weight_decay"],
            )           

    raise ValueError(f"Unknown loss function: {kwargs["loss_name"]}")
