"""
ema.py

Exponential Moving Average (EMA) of a monitored metric
"""


ALPHA = 0.4


class EMA:
    """
    An Exponential Moving Average implementation to smooth a noisy metric along the epochs.

    The smoothed value is defined as:
        ema_1 = x_1
        ema_t = alpha * x_t + (1 - alpha) * ema_{t-1}

    where x_t is the raw metric value at epoch t. Higher alpha values follow the raw metric
    more closely; lower alpha values smooth it more.

    Parameters
    ----------
    alpha: float, optional
        Smoothing factor, in (0, 1] (default = ALPHA)
    """
    def __init__(self, alpha: float = ALPHA):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")

        self.alpha = alpha
        self.value = None

    def step(self, x: float) -> float:
        """
        Updates the EMA with a new raw value.

        Parameters
        ----------
        x: float
            Raw metric value at the current epoch

        Returns
        -------
        float
            Updated smoothed value
        """
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value

        return self.value
