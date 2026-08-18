"""
get_metadata.py

DATASET PREPARATION PIPELINE (2/5)
----------------------------------
Obtains a metadata file from the respiratory sound records.

Requires (1/5):
    dataset/full_dataset/wav/{x}.wav
    dataset/full_dataset/json/{x}.json

Generates:
    dataset/metadata.csv
        A file containing the following entries per record:
        - record_id (SPRSound)
        - patient_id (SPRSound)
        - age
        - gender
        - position
        - json_path
        - wav_path
        - Poor Quality: Whether it is a Poor Quality record (1) or not (0) 

Execution:
    python dataset/get_metadata.py
"""

import os
from pathlib import Path
import json

import pandas as pd
from tqdm import tqdm


DATASET_PATH = Path("dataset")
FULL_DATASET_PATH = DATASET_PATH / "full_dataset"
WAV_PATH = FULL_DATASET_PATH / "wav"
JSON_PATH = FULL_DATASET_PATH / "json"


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
            - record_id: str
                SPRSound's record id
            - patient_id: str
                SPRSound's patient id
            - age: str
                Patient's age
            - gender: str
                Whether male ("0") or female ("1")
            - position: str
                Record position
            - json_path: str
                Relative .json path
            - wav_path: str
                Relative .wav file
            - Poor Quality: int
                Whether the record is "Poor Quality" (1) or not (0)
    """
    filename = file.stem
    patient_id, age, gender, position, record_id = filename.split("_")
    
    with open(file, "r") as f:
        data = json.load(f)
        
        poor_quality = data.get("record_annotation", 0)
        poor_quality = 1 if poor_quality == "Poor Quality" else 0
    
    return {
        "record_id": record_id,
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "position": position,
        "json_path": f"{filename}.json",
        "wav_path": f"{filename}.wav",
        "Poor Quality": poor_quality
    }


def get_metadata() -> None:
    """
    Iterates over the records folder to generate the metadata file for each record.
    """
    metadata_list = []
    for file in tqdm(os.listdir(JSON_PATH), desc="Processing metadata"):
        metadata = get_metadata_from_file(JSON_PATH / file)
        metadata_list.append(metadata)
        
    df = pd.DataFrame(metadata_list)
    df.to_csv(DATASET_PATH / "metadata.csv", index=False)
    
    print(f"There are {df['Poor Quality'].sum()}/{len(df)} records with poor quality.\n")


def main():
    print()
    get_metadata()


if __name__ == "__main__":
    main()
    