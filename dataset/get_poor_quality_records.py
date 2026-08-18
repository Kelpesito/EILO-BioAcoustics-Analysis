"""
get_poor_quality_records.py

DATASET PREPARATION PIPELINE (3/5)
----------------------------------
Separates the files which are classified as "Poor Quality" and copy them in a folder for these
records, and copy the other files in a folder for the "Good Quality" records.

Requires (2/5):
    dataset/full_dataset/wav/{x}.wav   -> 1/5
    dataset/full_dataset/json/{x}.json -> 1/5
    dataset/metadata.csv               -> 2/5

Generates:
    dataset/poor_quality/json/{x}.json
    dataset/poor_quality/wav/{x}.wav
    dataset/good_quality/json/{x}.json
    dataset/good_quality/wav/{x}.wav

Execution:
    python dataset/get_poor_quality_records.py
"""


from pathlib import Path
import shutil

import pandas as pd
from tqdm import tqdm


DATASET_PATH = Path("dataset")
POOR_QUALITY_PATH = DATASET_PATH / "poor_quality"
POOR_QUALITY_JSON_PATH = POOR_QUALITY_PATH / "json"
POOR_QUALITY_WAV_PATH = POOR_QUALITY_PATH / "wav"
GOOD_QUALITY_PATH = DATASET_PATH / "good_quality"
GOOD_QUALITY_JSON_PATH = GOOD_QUALITY_PATH / "json"
GOOD_QUALITY_WAV_PATH = GOOD_QUALITY_PATH / "wav"
JSON_PATH = DATASET_PATH / "full_dataset" / "json"
WAV_PATH = DATASET_PATH / "full_dataset" / "wav"


def create_folder_structure() -> None:
    """
    Creates the following folders:
    - dataset/poor_quality
    - dataset/poor_quality/json
    - dataset/poor_quality/wav
    - dataset/good_quality
    - dataset/good_quality/json
    - dataset/good_quality/wav
    """
    POOR_QUALITY_PATH.mkdir(exist_ok=True)
    POOR_QUALITY_JSON_PATH.mkdir(exist_ok=True)
    POOR_QUALITY_WAV_PATH.mkdir(exist_ok=True)
    GOOD_QUALITY_PATH.mkdir(exist_ok=True)
    GOOD_QUALITY_JSON_PATH.mkdir(exist_ok=True)
    GOOD_QUALITY_WAV_PATH.mkdir(exist_ok=True)
    
    
def classify_records(df: pd.DataFrame) -> None:
    """
    Iterates over all records in metadata and copy the files to the corresponding folders:
    - If "Poor Quality" -> dataset/poor_quality
    - Else              -> dataset/good_quality
    
    Parameter
    ---------
    df: pd.DataFrame
        DataDrame containing the metadata
    """
    for _, record in tqdm(df.iterrows(), desc="Classifying records", total=len(df)):
        if record["Poor Quality"] == 1:
            destination_folder = POOR_QUALITY_PATH
        else:
            destination_folder = GOOD_QUALITY_PATH
        
        wav_file = WAV_PATH / record["wav_path"]
        json_file = JSON_PATH / record["json_path"]
        
        # Copy wav and json files
        shutil.copyfile(wav_file, destination_folder / "wav" / record["wav_path"])
        shutil.copyfile(json_file, destination_folder / "json" / record["json_path"])
    
    
def main():
    print()
    print("Setting up folder structure\n")
    create_folder_structure()
    
    df = pd.read_csv(DATASET_PATH / "metadata.csv")
    classify_records(df)
    print()


if __name__ == "__main__":
    main()
