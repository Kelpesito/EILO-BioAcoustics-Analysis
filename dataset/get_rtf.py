"""
get_rtf.py

DATASET PREPARATION PIPELINE (5/6)
----------------------------------
Generates a folder with the desired Time-Frequency Representation (RTF) of each fragment.
Heart sounds (S1/S2) are first attenuated (see src/remove_heart_sounds.py).
If duration < 4 seconds:
    - Cyclic padding: Add the same signal n times until duration > 4 seconds
    - Select randomly a 4-second segment
If duration > 4 seconds:
    - Pass through an energy filter
    - Select the 4-second window with the most energy
 
Requires (4/5):
    dataset/dataset/audio/{x}.wav
    dataset/fragments_metadata.csv

Generates:
    dataset/dataset/{RTF}/{x}.tiff

Execution:
    python dataset/get_rtf.py -t <type>
        - type: Desired RTF files to obtain:
            - STFT    
"""


import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import os
import random

import numpy as np
from tqdm import tqdm
import soundfile as sf
import tifffile

from src.calculate_rtf import calculate_rtf
from src.remove_heart_sounds import remove_heart_sounds


SEGMENT_DURATION = 4  # seconds
SEED = 42  # Base seed for the random window selection (per-fragment seed: SEED + fragment id)
CHUNKSIZE = 1  # Fragments sent to each worker per batch

# Representaciones Tiempo-Frecuencia disponibles
RTFS = ["STFT"]

# Paths
DATASET_PATH = Path("dataset")
DATASET_DATASET_PATH = DATASET_PATH / "dataset"
FRAGMENTS_PATH = DATASET_DATASET_PATH / "audio"
SPECTROGRAMS_PATH = DATASET_DATASET_PATH / "spectrogram"

PATHS = {
    "STFT": SPECTROGRAMS_PATH
}


def create_folder(rtf_type: str) -> Path:
    """
    Given the name (key) of a folder, corresponding to a RTF, creates it and returns its path.
    
    Parameter
    ---------
    rtf_type: str
        The desired RTF type to generate
    
    Returns
    -------
    path: Path
        The folder path for the RTF images 
    """
    path = PATHS[rtf_type]
    path.mkdir(exist_ok=True)
    
    return path


def pre_process(signal, fs, rng: random.Random):
    # Filter cardiac sounds
    signal = remove_heart_sounds(signal, fs)

    # Duration normalization
    duration = len(signal)/fs
    if duration < 4:  # Cyclic padding
        n_rep = int(np.ceil(SEGMENT_DURATION/duration))  # Number of repetitions to fill more than 4 seconds
        cyclic = np.tile(signal, n_rep)  # Make repetitions of the signal

        # Select randomly a 4-seconds window
        start = rng.randint(0, len(cyclic)-1)
        end = start + SEGMENT_DURATION*fs
        pre_signal_idx = (np.arange(start, end) % len(cyclic)).astype(int)
        pre_signal = cyclic[pre_signal_idx]

    else:  # Select the most energetic window (4 seconds)
        N = int(SEGMENT_DURATION*fs)

        # Energy filter
        energy = np.convolve(signal**2, np.ones(N), mode="full")
        valid = energy[N-1 : len(signal)]  # Samples representing full windows in signal
        i_max = int(np.argmax(valid)) + N  # Most energetic window (last sample)

        ini = i_max - N
        fin = i_max
        pre_signal = signal[ini:fin]

    # Standardization
    mu = pre_signal.mean()
    sigma = pre_signal.std()
    pre_signal = (pre_signal - mu) / sigma

    return pre_signal



def process_file(file: str, rtf_type: str, path: Path) -> None:
    """
    Calculates the desired rtf of a single fragment and saves it in the corresponding folder as
    .tiff.

    Parameters
    ----------
    file: str
        Fragment file name (.wav), relative to FRAGMENTS_PATH
    rtf_type: str
        The desired RTF type to generate
    path: Path
        The folder path for the RTF images
    """
    name = Path(file).stem
    wav_file = FRAGMENTS_PATH / file

    # Open file .wav
    signal, fs = sf.read(wav_file)

    # Pre-process fragment (seeded per fragment: reproducible regardless of processing order)
    rng = random.Random(SEED + int(name))
    pre_signal = pre_process(signal, fs, rng)

    # Calculate RTF
    rtf = calculate_rtf(rtf_type, pre_signal, fs)

    # Save RTF
    tifffile.imwrite(f"{path / name}.tiff", rtf)


def get_RTFs(rtf_type: str, path: Path) -> None:
    """
    For each fragment, it calculates the desired rtf and saves them in the corresponding folder as
    .tiff. The fragments are processed in parallel, one per worker process.

    Parameters
    ----------
    rtf_type: str
        The desired RTF type to generate
    path: Path
        The folder path for the RTF images
    """
    files = os.listdir(FRAGMENTS_PATH)
    worker = partial(process_file, rtf_type=rtf_type, path=path)

    with Pool() as pool:  # As many workers as CPU cores
        results = pool.imap_unordered(worker, files, chunksize=CHUNKSIZE)
        for _ in tqdm(results, total=len(files), desc=f"Generating {rtf_type}"):
            pass

    
def main():
    print()
    parser = argparse.ArgumentParser(description="Calculate Time-Frequency representations and save them")
    parser.add_argument("-t", "--type", required=True, choices=RTFS, help="Type of RTF to generate")
    args = parser.parse_args()
    rtf = args.type
    
    # Create folder if it does not exists
    print("Setting up folder")
    path = create_folder(rtf)
    
    # Extract RTF
    get_RTFs(rtf, path)
    print()
    
if __name__ == "__main__":
    main()
    