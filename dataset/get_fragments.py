"""
get_fragments.py

DATASET PREPARATION PIPELINE (4/5)
----------------------------------
Extracts the respiratory fragments from each record given the time annotations 

Requires (3/5):
    dataset/good_quality/json/{x}.json
    dataset/good_quality/wav/{x}.wav

Generates:
    dataset/dataset/audio/{x}.wav
    dataset/fragments_metadata.csv

Execution:
    python dataset/get_fragments.py
"""


from pathlib import Path
import os
import json

from tqdm import tqdm
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import pandas as pd

from get_metadata import get_metadata_from_file


DATASET_PATH = Path("dataset")
GOOD_QUALITY_PATH = DATASET_PATH / "good_quality"
GOOD_QUALITY_JSON_PATH = GOOD_QUALITY_PATH / "json"
GOOD_QUALITY_WAV_PATH = GOOD_QUALITY_PATH / "wav"

DATASET_DATASET_PATH = DATASET_PATH / "dataset"
FRAGMENTS_PATH = DATASET_DATASET_PATH / "audio"

FS = 8000.
FS_NEW = 4000.


def create_folder_structure() -> None:
    """
    Creates the following folders:
    - dataset/dataset
    - dataset/dataset/audio
    """
    DATASET_DATASET_PATH.mkdir(exist_ok=True)
    FRAGMENTS_PATH.mkdir(exist_ok=True)
    
    
def pre_process(
    signal: np.ndarray,
    fs: float = FS,
    fs_new: float = FS_NEW,
    fmin: float = 70.,
    fmax: float = 1900.
) -> np.ndarray:
    """
    Preprocesses the respiratory signal:
    - Bandpass filtering: 8th order Butterworth [`fmin` - `fmax`] Hz
    - Resample from `fs` sampling rate (in Hz) to `fs_new`
    
    Parameters
    ----------
    signal: np.ndarray
        Respiratory sound signal
    fs: float (optional)
        Original sampling rate, in Hz (default = FS = 8000.)
    fs_new: float (optional)
        New sampling rate after resampling, in Hz (default = FS_NEW = 4000.)
    fmin: float (optional)
        Min frequency for bandpass filter, in Hz (default = 70.)
    fmax: float (optional)
        Max frequency for bandpass filter, in Hz (default = 1900.)
    
    Returns
    -------
    signal_pre: np.ndarray
        The preprocessed signal
    """
    # Filtro Butterworth: 8º orden, pasa banda [70 - 1900] Hz
    b, a = butter(4, [fmin, fmax], btype="bandpass", fs=fs)
    signal_filtered = filtfilt(b, a, signal)

    # Diezmado
    signal_pre = resample_poly(signal_filtered, fs_new, fs)
    
    return signal_pre
    
    
def extract_fragments() -> list[dict]:
    """
    Iterates over the records directory and extracts the fragments by the time annotations, and
    its label annotation. For each record:
    - Load .wav and .json files
    - Pre-process signal
    - For each fragment: crop the signal and save some metadata
    
    Returns
    -------
    fragments_metadata: list[dict]
        List of dictionaries containing the metadata of each fragment:
        - id: int
            fragment id
        - name: str
            Patient id
        - age: str
            Patient's age
        - gender: str
            Whether male ("0") or female ("1")
        - position: str
            Record position
        - record_id: str
            SPRSound's record id
        - segment: int
            Index of segment in record (starting at 0)
        - label: str
            Respiratory sound annotation ("Normal", "Rhonchi", "Wheeze", "Stridor",
            "Coarse Crackle", "Fine Crackle", "Wheeze+Crackle")
        - category: str
            "Normal" if label is "Normal", else "Adventitious"
        - duration: float
            Fragment duration, in s
    """
    fragments_metadata = []
    count = 1
    patients = {}
    for file in tqdm(os.listdir(GOOD_QUALITY_JSON_PATH), desc="Extracting fragments"):
        filename = Path(file).stem
        metadata_file = get_metadata_from_file(GOOD_QUALITY_JSON_PATH / file)
        
        wav_file = GOOD_QUALITY_WAV_PATH / f"{filename}.wav"
        json_file = GOOD_QUALITY_JSON_PATH / f"{filename}.json"
        
        # Load signal 
        signal, fs = sf.read(wav_file)
        
        # Load metadata
        with open(json_file, "r") as f:
            data = json.load(f)
            events = data["event_annotation"]
        
        onsets = [float(event["start"])/1000 for event in events]  # in s
        offsets = [float(event["end"])/1000 for event in events]  # in s
        annotations = [event["type"] for event in events]
        
        # Pre-process signal
        pre_signal = pre_process(signal, fs=fs)
        for j, (onset, offset, label) in enumerate(zip(onsets, offsets, annotations)):
            # Obtener segmento
            onset_i = int(np.floor(onset*FS_NEW))
            offset_i = int(np.ceil(offset*FS_NEW))
            fragment = pre_signal[onset_i:offset_i+1]
            
            # Obtener metadata
            p = (metadata_file["patient_id"], metadata_file["age"])
            if p not in patients:
                n = len(patients)
                patients[p] = f"P{n+1}" 
            p = patients[p]
             
            segment_metadata_i = {
                "id": count,
                "name": p,
                "age": metadata_file["age"],
                "gender": metadata_file["gender"],
                "position": metadata_file["position"],
                "record_id": metadata_file["record_id"],
                "segment": j,
                "label": label,
                "category": label if label == "Normal" else "Adventitious",
                "duration": len(fragment)/FS_NEW
            }
            
            # Save signal and metadata
            sf.write(FRAGMENTS_PATH / f"{count}.wav", fragment, int(FS_NEW))
            
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
            fragment id
        - name: str
            Patient id
        - age: str
            Patient's age
        - gender: str
            Whether male ("0") or female ("1")
        - position: str
            Record position
        - record_id: str
            SPRSound's record id
        - segment: int
            Index of segment in record (starting at 0)
        - label: str
            Respiratory sound annotation ("Normal", "Rhonchi", "Wheeze", "Stridor",
            "Coarse Crackle", "Fine Crackle", "Wheeze+Crackle")
        - category: str
            "Normal" if label is "Normal", else "Adventitious"
        - duration: float
            Fragment duration, in s
    """
    # Save metadata.csv
    df = pd.DataFrame(fragments_metadata)
    df.to_csv(DATASET_PATH / "fragments_metadata.csv", index=False)
            
            
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
    