"""
get_icbhi_dataset.py

DATASET PREPARATION PIPELINE (1/3)
----------------------------------
Extracts the records from the ICBHI folder repository.

Requires:
    ICBHI/ICBHI/data <- Download the dataset from https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
        (move all files to the folder and remove non data files like `filename_differences.txt`, 
        `filename_format.txt`, ...)

Generates:
    ICBHI/ICBHI/data/wav/{x}.wav
    ICBHI/ICBHI/data/json/{x}.json

Execution:
    python ICBHI/get_icbhi_dataset.py
"""


from pathlib import Path
import os
import shutil

import pandas as pd
from tqdm import tqdm


ICBHI_PATH = Path("ICBHI/ICBHI/data")
WAV_PATH = ICBHI_PATH / "wav"
JSON_PATH = ICBHI_PATH / "json"


def create_folder_structure() -> None:
    """
    Create the following folders:
        - ICBHI/ICBHI/wav
        - ICBHI/ICBHI/json
    """
    WAV_PATH.mkdir(exist_ok=True)
    JSON_PATH.mkdir(exist_ok=True)


def get_type_files(file_type: str) -> list[str]:
    """
    Iterate over the ICBHI data and extract the indicated file type
    
    Parameter
    ---------
        file_type: str
            Extension of the type file desired to extract (Ex: "wav")
    
    Returns
    -------
        list[Path]
            The list of files whith the extension `file_type`
    """
    file_list = []
    for file in tqdm(os.listdir(ICBHI_PATH), desc=f"Searching for files {file_type}"):
        file = Path(file)
        if file.suffix == f".{file_type}":
            file_list.append(ICBHI_PATH / file)

    return file_list


def txt2json(file: Path, destination: Path) -> None:
    """
    Extracts the information from txt file and dumps it into a json file.

    Parameters
    ----------
    file: Path
        txt file path
    destination: Path
        Destination directory
    """
    df = pd.read_csv(file, sep="\t", header=None, names=["start", "end", "Crackle", "Wheeze"])
    df.to_json(destination / f"{file.stem}.json", orient="records", indent=4)


def main():
    print()
    print("Setting up folder structure\n")
    create_folder_structure()

    # Get files
    wav_files = sorted(get_type_files("wav"))
    txt_files = sorted(get_type_files("txt"))
    print(f"Found {len(wav_files)} wav files and {len(txt_files)} txt files.\n")

    # Move .wav files to folder
    for file in tqdm(wav_files, desc="Moving .wav files"):
        shutil.move(file, WAV_PATH / file.name)

    # Get .json files from .txt and move to folder 
    for file in tqdm(txt_files, desc="Generating .json files and removing the .txt files"):
        txt2json(file, JSON_PATH)
        # Remove .txt files
        os.remove(file)

    print(f"All files copied to {ICBHI_PATH} successfully.\n")

    
if __name__ == "__main__":
    main()
