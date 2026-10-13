"""
filter_dataset.py

DATASET PREPARATION PIPELINE (6/6)
----------------------------------
Filters the SPRSound dataset:
    - By label:
        - Coarse Crackle removed
        - Fine Crackle removed
        - Wheeze+Crackle replaced to Wheeze
    - By duration: only the fragments with D_MIN <= duration <= D_MAX are kept
      (0.5 - 3.0 seconds)

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

D_MIN = 0.5  # seconds
D_MAX = 3.0  # seconds


def duration_mask(df: pd.DataFrame, d_min: float | None = None, d_max: float | None = None) -> pd.Series:
    """
    Mask for preserved fragments: 
        d_min <= duration <= d_max (None = no limit)
    
    Parameters
    ----------
    df: pd.DataFrame
        Metadata dataframe
    d_min: float | None, optional
        Minimum duration to keep, in seconds (default = None).
    d_max: float | None, optional
        Maximum duration to keep, in seconds (default = None).
    
    Returns
    -------
    keep: pd.Series
        Boolean mask for preserved fragments
    """
    keep = pd.Series(True, index=df.index)
    if d_min is not None:
        keep &= df["duration"] >= d_min
    if d_max is not None:
        keep &= df["duration"] <= d_max
    return keep


def main():
    print()
    # Load fragments_metadata.csv
    df = pd.read_csv(DATASET_PATH / "fragments_metadata.csv")
    df_new = df.copy()

    # Collapse Wheeze+Crackle to Wheeze
    df_new["label"] = df_new["label"].replace("Wheeze+Crackle", "Wheeze")
    # Drop rows
    df_new = df_new[~df_new["label"].isin(["Coarse Crackle", "Fine Crackle"])].reset_index(drop=True)

    # Duration filtering
    df_new = df_new[duration_mask(df_new, d_min=D_MIN, d_max=D_MAX)].reset_index(drop=True)

    print(f"New dataset size: {len(df_new)}\n")

    print("Label distribution:")
    print(df_new["label"].value_counts())
    print()

    # Save new .csv
    df_new.to_csv(DATASET_PATH / "fragments_metadata_filtered.csv", index=False)
    print("Saving new .csv")


if __name__ == "__main__":
    main()
