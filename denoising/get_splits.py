"""
get_splits.py

TUNING/TEST SPLIT FOR THE PARAMETER OPTIMIZATION OF THE TRANSIENT REMOVAL FILTER (remove_noise)
-----------------------------------------------------------------------------------------------
Splits the clean signals and the artifacts of experiment 1 into tuning and test (section 0.1 of
PARAMETER_OPTIMIZATION.md): generates the mixtures of experiment 1 (get_mixes_1 of run_experiment.py, controls
removed) and takes the first of N_SPLITS = 4 folds as test (75 % tuning / 25 % test).
    - Clean signals: grouped by clean_id, stratified by label (StratifiedGroupKFold)
    - Artifacts: grouped by art_id, stratified by duration (StratifiedGroupKFold)
Crossing both splits gives the four sets of parameter_optimization.ipynb: tuning, test, new signals and new artifacts.

Requires:
    denoising/audio/{id}.wav
        The 38 clean signals and the 32 fragments the artifacts are cut from

Generates:
    denoising/splits/clean_split.csv
        clean_id, label and set (tune / test) of each clean signal
    denoising/splits/artifact_split.csv
        art_id, art_duration_ms and set (tune / test) of each artifact

Execution:
    python denoising/get_splits.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from run_experiment import get_mixes_1, load_artifacts, load_clean_signals
from src.evaluation.evaluation import FS
from src.optimization.splits import SPLITS_PATH  # denoising/splits (same folder as the one read by the notebook)


DENOISING_PATH = Path(__file__).resolve().parent  # denoising/ (independent of the cwd, as in run_experiment.py)
SIGNALS_PATH = DENOISING_PATH / "audio"

N_SPLITS = 4
RANDOM_STATE = 42


def create_folder_structure() -> None:
    """
    Creates the following folder:
    - denoising/splits
    """
    SPLITS_PATH.mkdir(exist_ok=True)


def do_tune_test_split(df: pd.DataFrame) -> tuple[set, set]:
    """
    Tuning/test split of the clean signals and of the artifacts (first fold of N_SPLITS as test):
    - Clean signals: grouped by clean_id, stratified by label
    - Artifacts: grouped by art_id, stratified by duration (art_duration_ms)

    Parameters
    ----------
    df: pd.DataFrame
        Mixtures (without controls), with the duration of the artifact (art_duration_ms)

    Returns
    -------
    groups_clean: set
        clean_id of the tuning clean signals
    groups_artifact: set
        art_id of the tuning artifacts
    """
    # Split clean signals
    sgkf_clean = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tune_clean_idx, _ = next(sgkf_clean.split(df, df["label"], df["clean_id"]))
    groups_clean = set(df.iloc[tune_clean_idx]["clean_id"])

    # Split artifacts
    sgkf_artifact = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tune_artifact_idx, _ = next(sgkf_artifact.split(df, df["art_duration_ms"], df["art_id"]))
    groups_artifact = set(df.iloc[tune_artifact_idx]["art_id"])

    return groups_clean, groups_artifact


def save_split(df: pd.DataFrame, groups_clean: set, groups_artifact: set) -> None:
    """
    Saves the tuning/test split of the clean signals and of the artifacts:
    - denoising/splits/clean_split.csv: clean_id, label, set
    - denoising/splits/artifact_split.csv: art_id, art_duration_ms, set

    Parameters
    ----------
    df: pd.DataFrame
        Mixtures (without controls), with the duration of the artifact (art_duration_ms)
    groups_clean: set
        clean_id of the tuning clean signals
    groups_artifact: set
        art_id of the tuning artifacts
    """
    clean = df.groupby("clean_id")["label"].first().reset_index()
    clean["set"] = clean["clean_id"].isin(groups_clean).map({True: "tune", False: "test"})
    clean.to_csv(SPLITS_PATH / "clean_split.csv", index=False)

    artifact = df.groupby("art_id")["art_duration_ms"].first().reset_index()
    artifact["set"] = artifact["art_id"].isin(groups_artifact).map({True: "tune", False: "test"})
    artifact.to_csv(SPLITS_PATH / "artifact_split.csv", index=False)


def main():
    # Create directory, if necessary
    print("\nCreating folder\n")
    create_folder_structure()

    # Generate metadata of mixtures
    clean_dict = load_clean_signals()
    artifact_dict = load_artifacts()
    exp_df = get_mixes_1(clean_dict, artifact_dict)
    exp_df = exp_df[exp_df["art_id"] != "Control"]
    exp_df = exp_df.astype({"art_id": int})

    # Generate splits
    print("\nSplitting\n")
    groups_clean, groups_artifact = do_tune_test_split(exp_df)
    n_clean, n_artifact = exp_df["clean_id"].nunique(), exp_df["art_id"].nunique()
    print(f"Clean signals: {len(groups_clean)} tune / {n_clean - len(groups_clean)} test")
    print(f"Artifacts:     {len(groups_artifact)} tune / {n_artifact - len(groups_artifact)} test")

    # Save splits
    print("\nSaving splits\n")
    save_split(exp_df, groups_clean, groups_artifact)


if __name__ == "__main__":
    main()
