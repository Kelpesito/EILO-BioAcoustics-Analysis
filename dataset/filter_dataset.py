"""
filter_dataset.py

DATASET PREPARATION PIPELINE (6/6)
----------------------------------
Filters the SPRSound dataset to remove labels:
    - Coarse Crackle
    - Fine Crackle
    - Wheeze+Crackle replaced to Wheeze

Requires (4/6):
    dataset/fragments_metadata.csv

Generates:
    dataset/fragments_metadata_filtered.csv

Execution:
    python dataset/filter_dataset.py
"""


from pathlib import Path

import pandas as pd


DATASET_PATH = Path("dataset")


def main():
    print()
    # Load fragments_metadata.csv
    df = pd.read_csv(DATASET_PATH / "fragments_metadata.csv")
    df_new = df.copy()

    # Collapse Wheeze+Crackle to Wheeze
    df_new["label"] = df_new["label"].replace("Wheeze+Crackle", "Wheeze")
    # Drop rows
    df_new = df_new[~df_new["label"].isin(["Coarse Crackle", "Fine Crackle"])].reset_index(drop=True)

    print(df_new["label"].value_counts())

    # Save new .csv
    df_new.to_csv(DATASET_PATH / "fragments_metadata_filtered.csv", index=False)
    print("Saving new .csv")


if __name__ == "__main__":
    main()
