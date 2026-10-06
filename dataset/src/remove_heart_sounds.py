"""
remove_heart_sounds.py

Functions to attenuate the heart sounds (S1/S2) of respiratory sound fragments, used as the first
pre-processing step of get_rtf.py.

Adaptation of soundwaveletSWT (adaptive wavelet filter, inspired by Zhan et al., 2010, "A
wavelet-based adaptive filter for removing ECG interference in EMGdi signals"):
    - Stationary Wavelet Transform (SWT, db4) until the approximation is below ~15 Hz
    - Per level: envelope vs. local reference (median of the envelope 0.1-0.4 s at each side)
    - Events above K times the reference are attenuated down to the background level
    - Reconstruction with the details only (the < 15 Hz component is removed)

Modification: protection of long events. Detected events longer than MAX_EVENT (at any level) or
touching the fragment borders are not heart sounds (stridor, wheezes, rhonchi...) and are kept.
"""


import numpy as np
import pywt
from scipy.ndimage import uniform_filter1d, label, binary_dilation
from scipy.signal import hilbert


# Heart sound removal - SWT (db4)
WAVELET = "db4"
F_APPROX = 15  # Hz. Decomposition levels until the approximation is below F_APPROX
K = 5  # Detection threshold = K * reference
G_FAC = 1.5  # Level (times the reference) to which the detected events are attenuated
# G_FAC: inside the mask, the envelope is left at min(env, G_FAC*ref)
#   K decides WHERE to act, G_FAC HOW MUCH to attenuate
#   - G_FAC = 0   -> coefficients set to 0 (leaves gaps)
#   - G_FAC = 1   -> event left at the background median (sounds a bit muffled)
#   - G_FAC = 1.5 -> event left approx. at the respiratory sound level
#   - G_FAC = K   -> only clipped to the threshold (almost no attenuation)
#   If heart sounds remain in the "removed per level" plot -> decrease it
#   If the filtered signal shows gaps where S1/S2 were -> increase it
REF_INNER = 0.1  # s. Reference window: [REF_INNER, REF_OUTER] s at each side of the sample
REF_OUTER = 0.4  # s  (the central zone is excluded so the heart sound itself is not used as reference)
ENV_SMOOTH = 0.004  # s. Envelope smoothing
MASK_DILATION = 0.02  # s. Mask widening (10 ms at each side)
GAIN_SMOOTH = 0.005  # s. Gain smoothing
BLOCK_SIZE = 2000  # Samples per block when computing the reference
REF_STEP = 0.005  # s. The reference is computed every REF_STEP and interpolated (None -> every sample, exact)
# Protection of adventitious sounds (set max_event=None to reproduce the original algorithm)
MAX_EVENT = 0.12  # s. Detected events longer than this (at any level) are not heart sounds (S1/S2 <~90 ms)
PROTECT_MARGIN = 0.05  # s. Widening of the protected zone around the long events
EDGE = 0.02  # s. Events touching the fragment borders (real duration unknown) are also protected
EPS = np.finfo(float).eps


def local_reference(env: np.ndarray, lb: int, ub: int, step: int = 1) -> np.ndarray:
    """
    Computes, for each sample j, the median of the envelope in [j-ub, j-lb] U [j+lb, j+ub].
    Samples outside the signal are padded with NaN and ignored by the median. Computed by blocks to
    avoid a sample-by-sample loop. NaN only appear in the windows of the first/last ub samples, so
    the (slow) nanmedian is only used there and the (fast) median elsewhere.
    The reference varies slowly (median over 0.3 s), so it is only computed every `step` samples and
    linearly interpolated in between (step = 1 -> exact computation at every sample).

    Parameters
    ----------
    env: np.ndarray
        Envelope of a detail level
    lb: int
        Inner limit of the reference window (samples)
    ub: int
        Outer limit of the reference window (samples)
    step: int
        Samples between consecutive computed medians

    Returns
    -------
    ref: np.ndarray
        Reference level for each sample
    """
    n_tot = len(env)
    lw = ub - lb + 1
    padded = np.concatenate((np.full(ub, np.nan), env, np.full(ub, np.nan)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, lw)

    grid = np.unique(np.r_[np.arange(0, n_tot, step), n_tot - 1])  # Samples where the median is computed
    ref_grid = np.empty(len(grid))
    for a in range(0, len(grid), BLOCK_SIZE):
        k = np.arange(a, min(a + BLOCK_SIZE, len(grid)))
        j = grid[k]
        left = windows[j]  # Starts at j-ub
        right = windows[j + ub + lb]  # Starts at j+lb
        values = np.hstack((left, right))

        # Windows fully inside the signal (ub <= j < n_tot - ub) -> no NaN
        inside = (j >= ub) & (j < n_tot - ub)
        ref_grid[k[inside]] = np.median(values[inside], axis=1)
        # Windows partially outside the signal -> nanmedian. In fragments shorter than ~2*lb (~0.2 s)
        # both windows can be fully outside -> no data: reference left as NaN (no detection there)
        border = k[~inside]
        if len(border):
            has_data = ~np.isnan(values[~inside]).all(axis=1)
            ref_grid[border] = np.nan
            ref_grid[border[has_data]] = np.nanmedian(values[~inside][has_data], axis=1)

    if step == 1:
        return ref_grid

    return np.interp(np.arange(n_tot), grid, ref_grid)


def protect_long_events(mask: np.ndarray, levels, fs: float, N: int) -> np.ndarray:
    """
    Removes from the detection mask the events that are not heart sounds:
    - Events longer than MAX_EVENT at any level -> their time span (+- PROTECT_MARGIN) is protected
      at ALL levels (adventitious sounds split into short pieces in the low bands)
    - Events touching the fragment borders (EDGE) -> protected (their real duration is unknown)

    Parameters
    ----------
    mask: np.ndarray
        Detection mask (levels x samples, padded). Row 0 = level 1
    levels: iterable of int
        Processed detail levels (1..n)
    fs: float
        Sample frequency
    N: int
        Length of the original (unpadded) signal

    Returns
    -------
    mask: np.ndarray
        Mask without the protected events
    """
    mask = mask.copy()
    max_len = round(MAX_EVENT*fs)
    edge = round(EDGE*fs)

    # Protected time spans (union over levels)
    protected = np.zeros(mask.shape[1], dtype=bool)
    for level in levels:
        lab, n_events = label(mask[level - 1])
        if n_events == 0:
            continue
        long_ids = np.where(np.bincount(lab.ravel()) > max_len)[0][1:]  # [1:] -> skip background (0)
        border_ids = np.unique(np.concatenate((lab[:edge], lab[N - edge:])))
        protected |= np.isin(lab, np.union1d(long_ids, border_ids[border_ids > 0]))
    protected = binary_dilation(protected, np.ones(2*round(PROTECT_MARGIN*fs) + 1, dtype=bool))

    # Events overlapping a protected span are discarded (at every level)
    for level in levels:
        lab, _ = label(mask[level - 1])
        hit = np.unique(lab[protected])
        mask[level - 1] &= ~np.isin(lab, hit[hit > 0])

    return mask


def iswt_details(details: np.ndarray) -> np.ndarray:
    """
    Inverse SWT using only the detail coefficients (approximation set to 0).

    Parameters
    ----------
    details: np.ndarray
        Detail coefficients (levels x samples). Row 0 = level 1

    Returns
    -------
    signal: np.ndarray
        Reconstructed signal
    """
    n_levels = details.shape[0]
    zeros = np.zeros_like(details[0])
    coeffs = [(zeros, details[k]) for k in range(n_levels - 1, -1, -1)]  # pywt: coarsest level first

    return pywt.iswt(coeffs, WAVELET)


def remove_heart_sounds(signal: np.ndarray, fs: float, levels=None):
    """
    Attenuates heart sounds in a respiratory sound signal:
    - Decomposes the signal with the SWT (db4) until the approximation is below ~15 Hz
    - For each detail level, computes its envelope and compares it with a reference level (median
      of the envelope between 0.1 and 0.4 s at each side of the sample)
    - Where the envelope exceeds K times the reference, an event is detected
    - Events longer than MAX_EVENT (at any level) or touching the borders are protected: they are
      adventitious/respiratory sounds, not heart sounds
    - The remaining events are attenuated down to the background level
    - Reconstructs the signal using only the details -> the < 15 Hz component is removed

    Parameters
    ----------
    signal: np.ndarray
        The respiratory sound (if multichannel, only the first channel is used)
    fs: float
        Sample frequency
    levels: iterable of int, optional
        Detail levels (1..n) to process. By default, all of them

    Returns
    -------
    clean: np.ndarray
        The filtered signal (same length as signal)
    """
    signal = signal - signal.mean()
    N = len(signal)

    # Parameters (in samples)
    n_levels = round(np.log2(fs/F_APPROX)) - 1
    levels = range(1, n_levels + 1) if levels is None else levels
    lb = round(REF_INNER*fs)
    ub = round(REF_OUTER*fs)
    l_env = round(ENV_SMOOTH*fs)
    l_dil = round(MASK_DILATION*fs)
    l_gain = round(GAIN_SMOOTH*fs)
    ref_step = 1 if REF_STEP is None else max(1, round(REF_STEP*fs))

    # SWT needs a length multiple of 2^n_levels -> symmetric padding
    n_tot = int(np.ceil(N / 2**n_levels)) * 2**n_levels
    n_pad = n_tot - N
    padded = np.concatenate((signal, signal[N - n_pad:][::-1]))

    # SWT decomposition. pywt returns [(cAn, cDn), ..., (cA1, cD1)] -> reversed (row 0 = level 1)
    coeffs = pywt.swt(padded, WAVELET, level=n_levels)
    detail = np.array([cD for _, cD in coeffs[::-1]])

    detail_filtered = detail.copy()
    envelope = np.zeros((n_levels, n_tot))
    reference = np.full((n_levels, n_tot), np.nan)
    gain = np.ones((n_levels, n_tot))
    mask = np.zeros((n_levels, n_tot), dtype=bool)

    for i in range(n_levels):
        envelope[i] = uniform_filter1d(np.abs(hilbert(detail[i])), l_env)

    # Detection mask (dilated)
    for level in levels:
        i = level - 1
        reference[i] = local_reference(envelope[i], lb, ub, ref_step)
        m = envelope[i] > K*reference[i]
        mask[i] = np.convolve(m.astype(float), np.ones(l_dil), "same") > 0.5
        mask[i] &= ~np.isnan(reference[i])  # No reference (very short fragments) -> no detection (avoids NaN gains)

    # Protection of long events (adventitious sounds) and events at the borders
    if MAX_EVENT is not None:
        mask = protect_long_events(mask, levels, fs, N)

    for level in levels:
        i = level - 1
        m = mask[i]

        # Gain: envelope -> min(envelope, G_FAC*reference) inside the mask
        g = np.ones(n_tot)
        g[m] = np.minimum(1, G_FAC*reference[i, m] / np.maximum(envelope[i, m], EPS))
        gain[i] = uniform_filter1d(g, l_gain)
        detail_filtered[i] = detail[i]*gain[i]

    # Reconstruction only with the details
    clean = iswt_details(detail_filtered)[:N]

    return clean
