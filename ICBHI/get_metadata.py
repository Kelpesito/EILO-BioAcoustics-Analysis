"""
get_metadata.py

DATASET PREPARATION PIPELINE (2/3)
----------------------------------
Obtains a metadata file from the respiratory sound records.

Requires (1/5):
    ICBHI/ICBHI/data/wav/{x}.wav
    ICBHI/ICBHI/data/json/{x}.json

Generates:
    ICBHI/metadata.csv
        A file containing the following entries per record:
        - id
        - patient_id (ICBHI)
        - record_id (ICBHI)
        - position
        - acquisition
        - equipment
        - json_path
        - wav_path
        - duration: Duration of recording

Execution:
    python ICBHI/get_metadata.py
"""


import os
from pathlib import Path
import json

import pandas as pd
import soundfile as sf
from tqdm import tqdm


ICBHI_PATH = Path("ICBHI")
ICBHI_DATASET_PATH = ICBHI_PATH / "ICBHI/data"
WAV_PATH = ICBHI_DATASET_PATH / "wav"
JSON_PATH = ICBHI_DATASET_PATH / "json"


def get_metadata_from_file(file: Path) -> dict:
    """
    Given a file record, it returns a dictionary with its metadata.
    
    Parameter
    ---------
        file: Path
            Record file (.wav or .json)
    
    Returns
    -------
        dict
            Dictionary with the record metadata:
            - patient_id: str
                ICBHI's patient id
            - record_id: str
                ICBHI's recording index
            - position: str
                Record position
            - acquisition: str
                Acquisition mode (single channel or multichannel)
            - equipment: str
                Recording equipment
            - json_path: str
                Relative .json path
            - wav_path: str
                Relative .wav file
            - duration: float
                Signal duration, in s
    """
    filename = file.stem
    patient_id, record_id, position, acquisition, equipment = filename.split("_")

    # Duration calculation
    info = sf.info(f"{WAV_PATH / filename}.wav")
    duration = info.duration
    
    return {
        "patient_id": patient_id,
        "record_id": record_id,
        "position": position,
        "acquisition": acquisition,
        "equipment": equipment,
        "json_path": f"{filename}.json",
        "wav_path": f"{filename}.wav",
        "duration": duration,
    }


def get_metadata() -> None:
    """
    Iterates over the records folder to generate the metadata file for each record.
    """
    metadata_list = []
    for i, file in enumerate(tqdm(os.listdir(WAV_PATH), desc="Processing metadata"), start=1):
        metadata = get_metadata_from_file(WAV_PATH / file)
        metadata = {"id": i} | metadata
        metadata_list.append(metadata)
        
    df = pd.DataFrame(metadata_list)
    df.to_csv(ICBHI_PATH / "metadata.csv", index=False)
    

def main():
    print()
    get_metadata()


if __name__ == "__main__":
    main()
