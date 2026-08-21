"""
get_splits.py

SPLIT DATA FOR CLASSIFICATION TASK
----------------------------------
From the metadata file containing information of the fragments performs the data split in
Tran/Validation/Test sets GROUPING BY patient, and stratifying, conserving the original
proportions:
- Train/Val - Test set: ~80/20
- Train - Val: 5-fold Cross-Validation ~80/20
- Final proportions: 64/16/20

Requires:
    dataset/fragments_metadata.csv
    
Generates:
    classification/splits/splits.csv
    
Execution:
    python classification/get_splits.py
"""


from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DATASET_PATH = Path("dataset")
FRAGMENTS_METADATA = DATASET_PATH / "fragments_metadata.csv"

CLASSIFICATION_PATH = Path("classification")
SPLITS_PATH = CLASSIFICATION_PATH / "splits"

N_SPLITS = 5
RANDOM_STATE = 42


def create_folder_structure() -> None:
    """
    Creates the following folder:
    - classification/splits
    """
    SPLITS_PATH.mkdir(exist_ok=True)


def do_train_val_test_split(
    df: pd.DataFrame,
    X_label: str,
    y_label: str,
    group_label: str
) -> pd.DataFrame:
    """
    Given a DataFrame, and the corresponding column names of input variable, output and group
    separation, returns the same DataFrame with a new column "fold", containing if the fragment
    belongs to test set or which fold for cross-validation:
    - -1: test set
    - 1-N_SPLITS: validation set in i-th fold
    
    Parameters
    ----------
    df: pd.DataFrame
        Data
    X_label: str
        Column name of input variable
    y_label: str
        Column name of output variable
    group_label: str
        Column name of group variable
    
    Returns
    -------
    split_df: pd.DataFrame
        The initial DataFrame with the fold column
    """
    X = df[X_label].values
    y = df[y_label].values
    groups = df[group_label].values
    
    # Train/Val - Test split (~80/20)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(sgkf.split(X, y, groups))
    
    df_train_val = df.iloc[train_val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    X_tv = df_train_val[X_label].values
    y_tv = df_train_val[y_label].values
    groups_tv = df_train_val[group_label].values
    
    # Train - Val split (5-fold Cross-Validation: ~80/20)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    df_train_val["fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(X_tv, y_tv, groups_tv), start=1):
        df_train_val.iloc[val_idx, df_train_val.columns.get_loc("fold")] = fold
        
    # Join Dataframes
    df_test["fold"] = -1
    split_df = pd.concat([df_train_val, df_test]).sort_values(X_label)
    
    return split_df


def main():
    # Create directory, if necessary
    print("\nCreating folder\n")
    create_folder_structure()
    
    # Read metadata
    print("Generating folds\n")
    df = pd.read_csv(FRAGMENTS_METADATA)

    # Train-Val-Test split (~64/16/20)
    split_df = do_train_val_test_split(df, X_label="id", y_label="label", group_label="name")
    
    print("Saving csv\n")
    # Save splits csv
    split_df.to_csv(SPLITS_PATH / "splits.csv", index=False)


if __name__ == "__main__":
    main()
    