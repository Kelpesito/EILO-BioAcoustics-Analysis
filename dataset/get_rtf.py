"""
get_rtf.py

DATASET PREPARATION PIPELINE (5/5)
----------------------------------
Generates a folder with the desired Time-Frequency Representation (RTF) of each fragment
 
Requires (4/5):
    dataset/dataset/audio/{x}.wav
    dataset/fragments_metadata.csv

Generates:
    dataset/dataset/{RTF}/{x}.tiff

Execution:
    python dataset/get_fragments.py -t <type>
        - type: Desired RTF files to obtain:
            - STFT    
"""


import argparse
from pathlib import Path
import os

from tqdm import tqdm
import soundfile as sf
import tifffile

from calculate_rtf import calculate_rtf


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
        name = Path(file).stem
        wav_file = FRAGMENTS_PATH / file
        
        # Open file .wav
        signal, fs = sf.read(wav_file)
        
        # Calculate RTF
        rtf = calculate_rtf(rtf_type, signal, fs)
        
        # Save RTF
        tifffile.imwrite(f"{path / name}.tiff", rtf)
        
    
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
    