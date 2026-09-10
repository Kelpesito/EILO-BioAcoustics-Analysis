import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    r"""
    The Focal Loss implementation. 

    Focal Loss is defined as:
        FL = \sum_{i=1}^K -\alpha_i * (1-p_i)^\gamma * \log(p_i)
    
    where p_i is the model's estimated probability for the true class,
    gamma >= 0 is the focusing parameter that down-weights easy examples
    (high p_t), and alpha_t is an optional per-class weighting factor
    used to address class imbalance.

    Parameters
    ----------
    gamma: float
        Gamma hyperparameter
    alpha: torch.Tensor
        Class weights
    reduction: str, optional
        Type of reduction: "mean" or "sum" (default = "mean")
    """
    def __init__(
        self,
        gamma: float,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B] (class indices, not one-hot)
        log_probs = F.log_softmax(logits, dim=1)            # [B, C]
        probs = log_probs.exp()                             # [B, C]

        # Select the log-probability and prob of the true class for each sample
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # [B]

        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * log_pt                          # [B]

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)          # [B]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # "none"
    