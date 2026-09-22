"""
get_rtf.py

DATASET PREPARATION PIPELINE (5/6)
----------------------------------
Generates a folder with the desired Time-Frequency Representation (RTF) of each fragment.
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
from pathlib import Path
import os
import random

import numpy as np
from tqdm import tqdm
import soundfile as sf
import tifffile

from src.calculate_rtf import calculate_rtf


SEGMENT_DURATION = 4  # seconds

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


def pre_process(signal, fs):
    duration = len(signal)/fs
    if duration < 4:  # Cyclic padding
        n_rep = int(np.ceil(SEGMENT_DURATION/duration))  # Number of repetitions to fill more than 4 seconds
        cyclic = np.tile(signal, n_rep)  # Make repetitions of the signal

        # Select randomly a 4-seconds window
        start = random.randint(0, len(cyclic))
        end = start + SEGMENT_DURATION*fs
        pre_signal_idx = np.arange(start, end) % len(cyclic)
        pre_signal = cyclic[pre_signal_idx]

    else:  # Select the most energetic window (4 seconds)
        N = SEGMENT_DURATION*fs

        # Energy filter
        energy = np.convolve(signal**2, np.ones(N), mode="full")
        valid = energy[N-1 : len(signal)]  # Samples representing full windows in signal
        i_max = int(np.argmax(valid)) + N  # Most energetic window (last sample)

        ini = i_max - N
        fin = i_max
        pre_signal = signal[ini:fin]

    # Normalization
    mu = pre_signal.mean()
    sigma = pre_signal.std()
    pre_signal = (pre_signal - mu) / sigma

    return pre_signal



def get_RTFs(rtf_type: str, path: Path) -> None:
    """
    For each fragment, it calculates the desired rtf and saves them in the corresponding folder as
    .tiff.
    
    Parameters
    ----------
    rtf_type: str
        The desired RTF type to generate
    path: Path
        The folder path for the RTF images 
    """
    for file in tqdm(os.listdir(FRAGMENTS_PATH), desc=f"Generating {rtf_type}"):
        try:
            name = Path(file).stem
            wav_file = FRAGMENTS_PATH / file
            
            # Open file .wav
            signal, fs = sf.read(wav_file)

            # Pre-process fragment
            pre_signal = pre_process(signal, fs)
            
            # Calculate RTF
            rtf = calculate_rtf(rtf_type, pre_signal, fs)
            
            # Save RTF
            tifffile.imwrite(f"{path / name}.tiff", rtf)
        except:
            print(name)
        
    
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
    