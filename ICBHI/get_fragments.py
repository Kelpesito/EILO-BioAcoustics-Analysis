"""
get_fragments.py

DATASET PREPARATION PIPELINE (3/3)
----------------------------------
Extracts the respiratory fragments from each record given the time annotations.
As (EXTRA) step returned none files with multichannel sounds, no longer multi-channel signal
processing is needed.

Requires (1/3):
    ICBHI/ICBHI/data/wav/{x}.wav
    ICBHI/ICBHI/data/json/{x}.json

Generates:
    ICBHI/dataset/audio/{x}.wav
    ICBHI/fragments_metadata.csv

Execution:
    python ICBHI/get_fragments.py
"""


import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from get_metadata import get_metadata_from_file


ICBHI_PATH = Path("ICBHI")
ICBHI_DATASET_PATH = ICBHI_PATH / "ICBHI/data"
WAV_PATH = ICBHI_DATASET_PATH / "wav"
JSON_PATH = ICBHI_DATASET_PATH / "json"
ICBHI_FRAGMENTS_PATH = ICBHI_PATH / "dataset"
FRAGMENTS_PATH = ICBHI_FRAGMENTS_PATH / "audio"


def create_folder_structure() -> None:
    """
    Creates the following folders:
    - ICHBI/dataset
    - ICHBI/dataset/audio
    """
    ICBHI_FRAGMENTS_PATH.mkdir(exist_ok=True)
    FRAGMENTS_PATH.mkdir(exist_ok=True)


def extract_fragments() -> list[dict]:
    """
    Iterates over the records directory and extracts the fragments by the time annotations, and
    its label annotations. For each record:
    - Load .wav and .json files
    - For each fragment: crop the signal and save some metadata
    
    Returns
    -------
    fragments_metadata: list[dict]
        List of dictionaries containing the metadata of each fragment:
        - id: int
            Fragment id
        - patient_id: str
            Patient id (ICBHI)
        - id_record: int
            Record id
        - segment: int
            Index of segment in record (starting at 0)
        - Crackle: int
            Presence/absence of crackles (presence=1, absence=0)
        - Wheeze: int
            Presence/absence of wheezes (presence=1, absence=0)
        - duration: float
            Fragment duration, in s
        """

    fragments_metadata = []
    count = 1
    for i, file in enumerate(tqdm(os.listdir(WAV_PATH), desc="Extracting fragments"), start=1):
        filename = Path(file).stem
        metadata_file = get_metadata_from_file(JSON_PATH / file)
        
        wav_file = WAV_PATH / f"{filename}.wav"
        json_file = JSON_PATH / f"{filename}.json"
        
        # Load signal 
        signal, fs = sf.read(wav_file)
        
        # Load metadata
        with open(json_file, "r") as f:
            events = json.load(f)
        
        onsets = [float(event["start"]) for event in events]  # in s
        offsets = [float(event["end"]) for event in events]  # in s
        crackles = [event["Crackle"] for event in events]
        wheezes = [event["Wheeze"] for event in events]
        
        for j, (onset, offset, crackle, wheeze) in enumerate(zip(onsets, offsets, crackles, wheezes)):
            # Obtener segmento
            onset_i = int(np.floor(onset*fs))
            offset_i = int(np.ceil(offset*fs))
            fragment = signal[onset_i:offset_i+1]
            
            # Obtener metadata
            segment_metadata_i = {
                "id": count,
                "patient_id": metadata_file["patient_id"],
                "id_record": i,
                "segment": j,
                "Crackle": crackle,
                "Wheeze": wheeze,
                "duration": len(fragment)/fs
            }
            
            # Save signal and metadata
            sf.write(FRAGMENTS_PATH / f"{count}.wav", fragment, int(fs))
            
            fragments_metadata.append(segment_metadata_i)
            
            count += 1
    
    return fragments_metadata


def save_fragments_metadata(fragments_metadata: list[dict]) -> None:
    """
    Given the list of fragments' metadata, saves it as a csv.
    
    Parameter
    ---------
    fragments_metadata: list[dict]
        List of dictionaries containing the metadata of each fragment:
        - id: int
            Fragment id
        - patient_id: str
            Patient id (ICBHI)
        - id_record: int
            Record id
        - segment: int
            Index of segment in record (starting at 0)
        - Crackle: int
            Presence/absence of crackles (presence=1, absence=0)
        - Wheeze: int
            Presence/absence of wheezes (presence=1, absence=0)
        - duration: float
            Fragment duration, in s
    """
    # Save metadata.csv
    df = pd.DataFrame(fragments_metadata)
    df.to_csv(ICBHI_PATH / "fragments_metadata.csv", index=False)


def main():
    print()
    print("Setting up folder structure\n")
    create_folder_structure()
    
    # Extract fragments from signals
    fragments_metadata = extract_fragments()
    
    print("\nSaving metadata")
    save_fragments_metadata(fragments_metadata)
    print()


if __name__ == "__main__":
    main()
