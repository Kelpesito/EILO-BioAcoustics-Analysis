"""
calculate_rtf.py

Functions to calculate and preprocess the Time-Frequency Representations (RTF) from audio 
fragments.
"""


import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from scipy.interpolate import interp1d
from skimage.transform import resize


EPS = 1e-14

FMIN = 50
FMAX = 2000

# Espectrograma - STFT
N = 179  # Samples/window for STFT
HOP = 10  # Overlap samples for STFT
NFFT = 2048  # output samples/winndow for FFT

# Imagen de salida
IMG_SIZE = 224


def calculate_stft(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Calculates the Short-Time Fourier Transform and computes the spectrogram, in log-frequency and
    dB:
    - Calculates the STFT (Hann window - 179 samples; overlap - 10 samples; 2048 samples FFT)
    - Converts absolute scale to dB
    - Crop frequencies [50 - 2000] Hz
    - Log frequency
    
    Parameters
    ----------
    signal: np.ndarray
        The respiratory sound
    fs: float
        Sample frequency
    
    Returns
    -------
    S_log: np.ndarray
        The spectogram
    """
    # STFT calculation and spectrogram
    stft = ShortTimeFFT(hann(N), HOP, fs, mfft=NFFT, scale_to="psd")
    spectrogram = stft.spectrogram(signal)
    f = stft.f
    spectrogram_dB = 10*np.log10(spectrogram + EPS)  # Convertir a dB
    
    # Crop frequencies
    mask = (f >= FMIN) & (f <= FMAX)
    f_masked = f[mask]
    spectrogram_masked = spectrogram_dB[mask]
    
    # log frequency
    f_log = np.geomspace(f_masked.min(), f_masked.max(), len(f_masked))
    S_log = interp1d(f_masked, spectrogram_masked, axis=0, kind="linear")(f_log)
    
    return S_log


CALCULATE_RTF = {"STFT": calculate_stft}


def calculate_rtf(rtf_type: str, signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Calculates and computes the given RTF and resizes it to a constant image size (224 x 224).
    
    Parameters
    ----------
    rtf_type: str
        The desired RTF to compute
    signal: np.ndarray
        The respiratory sound
    fs: float
        Sample frequency
        
    Returns
    -------
    np.ndarray
        The desired RTF
    """
    rtf_fn = CALCULATE_RTF[rtf_type]
    rtf = rtf_fn(signal, fs)  # Calculate RTF
    rtf_resized = np.flipud(resize(rtf, (IMG_SIZE, IMG_SIZE)))  # reshape
    
    return rtf_resized.astype(np.float32)
