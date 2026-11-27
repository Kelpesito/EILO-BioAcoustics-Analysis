"""
splits.py

Tuning/test split of parameter_optimization.ipynb: loads the split of the clean signals and of the artifacts saved by
get_splits.py and splits the results of an experiment into the four sets of section 3.1:
    - tune:          tuning clean signals x tuning artifacts
    - test:          test clean signals   x test artifacts
    - new_signals:   test clean signals   x tuning artifacts
    - new_artifacts: tuning clean signals x test artifacts
Each set keeps the controls (clean signal without artifact) of its clean signals.
Also loads the folds of the nested cross-validation of the tuning set (search_candidates.py), also saved by
get_splits.py.
"""

from pathlib import Path

import pandas as pd


SPLITS_PATH = Path(__file__).resolve().parents[2] / "splits"  # denoising/splits (independent of the cwd)
SETS = ["tune", "test", "new_signals", "new_artifacts"]
N_CV_FOLDS = 5  # Folds of the nested cross-validation of the tuning set
CV_FILES = ("cv_clean_folds.csv", "cv_artifact_folds.csv")  # In SPLITS_PATH


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the tuning/test split saved by get_splits.py.

    Returns
    -------
    clean_split: pd.DataFrame
        clean_id, label and set (tune / test) of each clean signal
    artifact_split: pd.DataFrame
        art_id, art_duration_ms and set (tune / test) of each artifact
    """
    clean_split = pd.read_csv(SPLITS_PATH / "clean_split.csv")
    artifact_split = pd.read_csv(SPLITS_PATH / "artifact_split.csv")
    return clean_split, artifact_split


def load_cv_folds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the folds of the nested cross-validation of the tuning set saved by get_splits.py.

    Returns
    -------
    clean_folds: pd.DataFrame
        clean_id, label and fold (1..N_CV_FOLDS) of each tuning clean signal
    artifact_folds: pd.DataFrame
        art_id, art_duration_ms and fold (1..N_CV_FOLDS) of each tuning artifact
    """
    return tuple(pd.read_csv(SPLITS_PATH / name) for name in CV_FILES)


def split_results(results_df: pd.DataFrame, clean_split: pd.DataFrame | None = None,
                  artifact_split: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """
    Splits the results of an experiment with one artifact per mixture (format of experiment_1.csv) into the four
    sets. Each set keeps the controls of its clean signals: those of the tuning clean signals in tune and
    new_artifacts, those of the test clean signals in test and new_signals.

    Parameters
    ----------
    results_df: pd.DataFrame
        Results (mixtures + controls), e.g. read_results("baseline/experiment_1.csv")
    clean_split, artifact_split: pd.DataFrame or None
        Split of the clean signals and of the artifacts (load_split). None -> loaded from SPLITS_PATH

    Returns
    -------
    sets: dict
        "tune", "test", "new_signals", "new_artifacts" -> pd.DataFrame (rows of results_df)
    """
    if clean_split is None or artifact_split is None:
        clean_split, artifact_split = load_split()
    tune_clean = set(clean_split.loc[clean_split["set"] == "tune", "clean_id"])
    tune_art = set(artifact_split.loc[artifact_split["set"] == "tune", "art_id"].astype(str))

    art_id = results_df["art_id"].astype(str)  # numbers and "Control" in the same column
    is_control = art_id == "Control"
    in_clean = results_df["clean_id"].isin(tune_clean)
    in_art = art_id.isin(tune_art)

    return {
        "tune":          results_df[in_clean & (in_art | is_control)],
        "test":          results_df[~in_clean & (~in_art | is_control)],
        "new_signals":   results_df[~in_clean & (in_art | is_control)],
        "new_artifacts": results_df[in_clean & (~in_art | is_control)],
    }
