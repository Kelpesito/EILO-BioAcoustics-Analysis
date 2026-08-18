"""
get_sprsound_dataset.py

DATASET PREPARATION PIPELINE (1/5)
----------------------------------
Extracts the records from the SPRSound folder repository.

Requires:
    SPRSound <- git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git

Generates:
    dataset/full_dataset/wav/{x}.wav
    dataset/full_dataset/json/{x}.json

Execution:
    python dataset/get_sprsound_dataset.py
"""


from pathlib import Path
import os
import shutil

from tqdm import tqdm


FULL_DATASET_PATH = Path("dataset/full_dataset")
WAV_PATH = FULL_DATASET_PATH / "wav"
JSON_PATH = FULL_DATASET_PATH / "json"
SPRSOUND_PATH = Path("SPRSound")

FOLDERS = [
    SPRSOUND_PATH / "BioCAS2022", 
    SPRSOUND_PATH / "BioCAS2023", 
    SPRSOUND_PATH / "BioCAS2024",
    SPRSOUND_PATH / "BioCAS2025",
]


def create_folder_structure() -> None:
    """
    Create the following folders:
        - dataset/full_dataset
        - dataset/full_dataset/wav
        - dataset/full_dataset/json
    """
    FULL_DATASET_PATH.mkdir(exist_ok=True)
    WAV_PATH.mkdir(exist_ok=True)
    JSON_PATH.mkdir(exist_ok=True)
    
    
def get_type_files(file_type: str) -> list[Path]:
    """
    Iterate over the SPRSound data folders and extract the indicated file type
    
    Parameter
    ---------
        file_type: str
            Extension of the type file desired to extract (Ex: "json")
    
    Returns
    -------
        list[Path]
            The list of files whith the extension `file_type`
    """
    file_list = []
    for folder in tqdm(FOLDERS, desc=f"Searching for files {file_type}"):
        for root, _, files in os.walk(folder):
            root = Path(root)
            for file in files:
                if (root / file).suffix == f".{file_type}":
                    file_list.append(root / file)
    
    return sorted(file_list)
    
    
def copy_files_to_folder(file_list: list[Path], destination_folder: Path) -> None:
    """
    Copy and paste the files from the `file_list` to the `destination_folder`.
    
    Parameters
    ----------
    file_list: list[Path]
        The list of files to copy
    
    destination_folder: Path
        The objective folder
    """
    for file in tqdm(file_list, desc=f"Copying files to {destination_folder}"):
        shutil.copyfile(file, destination_folder / file.name)


def main():
    print()
    print("Setting up folder structure\n")
    create_folder_structure()
    
    wav_files = sorted(get_type_files("wav"))
    json_files = sorted(get_type_files("json"))
    print(f"Found {len(wav_files)} wav files and {len(json_files)} json files.\n")
    
    copy_files_to_folder(wav_files, WAV_PATH)
    copy_files_to_folder(json_files, JSON_PATH)
    print(f"All files copied to {FULL_DATASET_PATH} successfully.\n")


if __name__ == "__main__":
    main()
    