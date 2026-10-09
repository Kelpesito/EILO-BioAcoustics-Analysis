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

Modifications:
    - Protection of long events. Detected events longer than MAX_EVENT at a band (protected at that
      band and +- PROTECT_LEVELS) or closer than EDGE to the fragment borders are not heart sounds
      (stridor, wheezes, rhonchi...) and are kept.
    - Band splitting. The levels in SPLIT_LEVELS are split into 2 half-bands (stationary wavelet
      packet), each one processed as one more level.
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
# Protection of adventitious sounds (MAX_EVENT = None, SPLIT_LEVELS = None and REF_STEP = None -> original algorithm)
MAX_EVENT = 0.12  # s. Detected events longer than this are not heart sounds (S1/S2 <~90 ms)
PROTECT_LEVELS = 0  # A long event at band l protects bands l-PROTECT_LEVELS..l+PROTECT_LEVELS (None -> all bands)
# PROTECT_LEVELS: in the low levels the long SWT filters merge close short impulses into one long event
#   - None -> a long event at any band protects ALL bands
#   - 0    -> only the band where the long event is detected
#   - 1    -> that band and the adjacent ones (an adventitious sound spans neighbouring bands)
PROTECT_MARGIN = 0.0  # s. Widening of the protected zone around the long events and the events at the borders
EDGE = 0.0  # s. Events closer than EDGE to the fragment borders (real duration unknown) are also protected (0 -> no)
# Band splitting (stationary wavelet packet): each level in SPLIT_LEVELS is split into 2 half-bands
SPLIT_LEVELS = [3, 4, 5]  # 62-500 Hz in half-octaves -> 10 bands. e.g. range(1, 8) -> 14 bands (None -> no split)
# With split levels, PROTECT_LEVELS counts bands (half-octaves), not levels
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
    - Events longer than MAX_EVENT at band l -> their time span (+- PROTECT_MARGIN) is protected
      at bands l-PROTECT_LEVELS..l+PROTECT_LEVELS (None -> at ALL bands)
    - Events closer than EDGE to the fragment borders -> protected at ALL bands (their real duration
      is unknown)

    Parameters
    ----------
    mask: np.ndarray
        Detection mask (bands x samples, padded). Row 0 = band 1 (without split, bands = levels)
    levels: iterable of int
        Processed bands (1..n_bands)
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

    margin = np.ones(2*round(PROTECT_MARGIN*fs) + 1, dtype=bool)

    # Protected time spans: long events (per band) and events at the borders (union over bands)
    long_span = np.zeros(mask.shape, dtype=bool)  # Row l-1: long events detected at band l
    border_span = np.zeros(mask.shape[1], dtype=bool)
    for level in levels:
        lab, n_events = label(mask[level - 1])
        if n_events == 0:
            continue
        long_ids = np.where(np.bincount(lab.ravel()) > max_len)[0][1:]  # [1:] -> skip background (0)
        border_ids = np.unique(np.concatenate((lab[:edge], lab[N - edge:N])))  # [N - edge:N] -> padding excluded
        long_span[level - 1] = binary_dilation(np.isin(lab, long_ids), margin)
        border_span |= np.isin(lab, border_ids[border_ids > 0])
    border_span = binary_dilation(border_span, margin)

    # Events overlapping a protected span are discarded: long events of the bands within
    # +- PROTECT_LEVELS of the current one, events at the borders at every band
    for level in levels:
        if PROTECT_LEVELS is None:
            neighbours = list(levels)
        else:
            neighbours = [l for l in levels if abs(l - level) <= PROTECT_LEVELS]
        protected = border_span | long_span[[l - 1 for l in neighbours]].any(axis=0)
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


def split_level(detail: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits the detail coefficients of a SWT level into 2 half-bands (one more SWT step with the
    filters dilated to that level, as in the stationary wavelet packet transform). The level
    [fs/2^(level+1), fs/2^level] is split at its center frequency.
    The low-pass branch of a detail contains its UPPER half and the high-pass branch its LOWER half
    (frequency order of the wavelet packets). norm=True -> tight frame (see merge_level).

    Parameters
    ----------
    detail: np.ndarray
        Detail coefficients of the level
    level: int
        SWT level (1..n) of the detail

    Returns
    -------
    upper: np.ndarray
        Coefficients of the upper half-band
    lower: np.ndarray
        Coefficients of the lower half-band
    """
    (upper, lower), = pywt.swt(detail, WAVELET, level=1, start_level=level, norm=True)
    return upper, lower


def merge_level(upper: np.ndarray, lower: np.ndarray, level: int) -> np.ndarray:
    """
    Inverse of split_level (pywt.iswt does not accept start_level). The split is a tight frame of
    circular filters, so its inverse is its adjoint: circular correlation with the impulse response
    of each branch (computed with the FFT).

    Parameters
    ----------
    upper: np.ndarray
        Coefficients of the upper half-band
    lower: np.ndarray
        Coefficients of the lower half-band
    level: int
        SWT level (1..n) of the detail

    Returns
    -------
    detail: np.ndarray
        Detail coefficients of the level
    """
    impulse = np.zeros(len(upper))
    impulse[0] = 1
    h_upper, h_lower = split_level(impulse, level)
    detail = (np.conj(np.fft.fft(h_upper))*np.fft.fft(upper)
              + np.conj(np.fft.fft(h_lower))*np.fft.fft(lower))

    return np.real(np.fft.ifft(detail))


def split_bands(detail: np.ndarray, fs: float, split_levels) -> tuple[np.ndarray, list[dict]]:
    """
    Builds the bands to process: the SWT levels, with the levels in split_levels split into 2
    half-bands. Bands ordered from high to low frequency (as the levels).

    Parameters
    ----------
    detail: np.ndarray
        Detail coefficients (levels x samples). Row 0 = level 1
    fs: float
        Sample frequency
    split_levels: iterable of int or None
        Levels (1..n) to split. None -> no split (bands = levels)

    Returns
    -------
    bands: np.ndarray
        Coefficients of each band (bands x samples)
    info: list of dict
        For each band: level, part ("full", "upper" or "lower") and frequency range (f_low, f_high)
    """
    split_levels = set() if split_levels is None else set(split_levels)
    bands, info = [], []
    for level in range(1, detail.shape[0] + 1):
        f_low, f_high = fs/2**(level + 1), fs/2**level
        if level in split_levels:
            upper, lower = split_level(detail[level - 1], level)
            f_mid = (f_low + f_high) / 2
            bands += [upper, lower]
            info += [dict(level=level, part="upper", f_low=f_mid, f_high=f_high),
                     dict(level=level, part="lower", f_low=f_low, f_high=f_mid)]
        else:
            bands.append(detail[level - 1])
            info.append(dict(level=level, part="full", f_low=f_low, f_high=f_high))

    return np.array(bands), info


def merge_bands(bands: np.ndarray, info: list[dict]) -> np.ndarray:
    """
    Inverse of split_bands: detail coefficients of each SWT level from the bands.

    Parameters
    ----------
    bands: np.ndarray
        Coefficients of each band (bands x samples)
    info: list of dict
        Band information (see split_bands)

    Returns
    -------
    detail: np.ndarray
        Detail coefficients (levels x samples). Row 0 = level 1
    """
    n_levels = info[-1]["level"]
    detail = np.zeros((n_levels, bands.shape[1]))
    for b, band in enumerate(info):
        if band["part"] == "full":
            detail[band["level"] - 1] = bands[b]
        elif band["part"] == "upper":  # Always followed by its lower half
            detail[band["level"] - 1] = merge_level(bands[b], bands[b + 1], band["level"])

    return detail


def remove_heart_sounds(signal: np.ndarray, fs: float, levels=None):
    """
    Attenuates heart sounds in a respiratory sound signal:
    - Decomposes the signal with the SWT (db4) until the approximation is below ~15 Hz
    - Splits the levels in SPLIT_LEVELS into 2 half-bands (stationary wavelet packet)
    - For each band, computes its envelope and compares it with a reference level (median of the
      envelope between 0.1 and 0.4 s at each side of the sample)
    - Where the envelope exceeds K times the reference, an event is detected
    - Events longer than MAX_EVENT (protected at that band and +- PROTECT_LEVELS) or closer than
      EDGE to the borders are protected: they are adventitious/respiratory sounds, not heart sounds
    - The remaining events are attenuated down to the background level
    - Merges the split levels back and reconstructs the signal using only the details -> the
      < 15 Hz component is removed

    Parameters
    ----------
    signal: np.ndarray
        The respiratory sound (if multichannel, only the first channel is used)
    fs: float
        Sample frequency
    levels: iterable of int, optional
        SWT levels (1..n) to process (all their bands, if split). By default, all of them

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

    # SWT needs a length multiple of 2^n_levels (2^(n_levels+1) to split the levels) -> symmetric padding
    n_pow = n_levels + 1 if SPLIT_LEVELS else n_levels
    n_tot = int(np.ceil(N / 2**n_pow)) * 2**n_pow
    n_pad = n_tot - N
    padded = np.concatenate((signal, signal[N - n_pad:][::-1]))

    # SWT decomposition. pywt returns [(cAn, cDn), ..., (cA1, cD1)] -> reversed (row 0 = level 1)
    coeffs = pywt.swt(padded, WAVELET, level=n_levels)
    detail = np.array([cD for _, cD in coeffs[::-1]])

    # Bands to process: the levels, some of them split into 2 half-bands. From here on, "level" is
    # a band index (1..n_bands); without split, bands = levels
    detail, band_info = split_bands(detail, fs, SPLIT_LEVELS)
    n_bands = len(band_info)
    levels = [b + 1 for b, band in enumerate(band_info) if band["level"] in levels]

    detail_filtered = detail.copy()
    envelope = np.zeros((n_bands, n_tot))
    reference = np.full((n_bands, n_tot), np.nan)
    gain = np.ones((n_bands, n_tot))
    mask = np.zeros((n_bands, n_tot), dtype=bool)

    for i in range(n_bands):
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

    # Reconstruction only with the details (split levels merged back first)
    clean = iswt_details(merge_bands(detail_filtered, band_info))[:N]

    return clean
