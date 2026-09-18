"""
check_channels.py

DATASET PREPARATION PIPELINE (EXTRA)
-----------------------------------------
Iterates through the .wav files to obtain multichannel files 

Requires (1/5):
    ICBHI/ICBHI/data/wav/{x}.wav

Execution:
    python ICBHI/check_chennels.py
"""


import os
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


WAV_PATH = Path("ICBHI/ICBHI/data/wav")


def get_n_channels() -> list[str]:
    """
    Iterates through the .wav directory to obtain files with sounds in more than one channel

    Returns
    -------
    files: list[str]
        List of files containing data in more than one channel.
    """
    files = []
    for file in tqdm(os.listdir(WAV_PATH), desc="Inspecting number of channels"):
        info = sf.info(WAV_PATH / file)
        n_channels = info.channels
        if n_channels > 1:
            files.append(file)

    return files


def main():
    print()
    files = get_n_channels()
    print()

    print(f"{len(files)} files with more than one channel have been found.")
    for file in files:
        print(file)


if __name__ == "__main__":
    main()
