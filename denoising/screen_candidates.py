"""
screen_candidates.py

SCREENING OF THE CANDIDATE GENERATION OF THE TRANSIENT REMOVAL FILTER (remove_noise)
-------------------------------------------------------------------------------------
Section 4.3 of parameter_optimization.ipynb (phase 1, screening): each parameter of the candidate generation is
varied alone over the values of SCREENING, with the others at their default value (remove_noise.py). For each value,
the candidates of the mixtures of experiment 1 are generated (run_candidates of get_candidates.py; not saved) and their
metrics are computed (candidate_score of src/optimization/candidate_score.py, section 4.1).
The parameters are module constants of remove_noise: each value is set on the module (rn.K = 4) and copied to the
worker processes by run_candidates, then the default value is restored.

Requires:
    denoising/results/experiments/experiment_1.csv
        Mixtures of experiment 1 (run_experiment.py -e 1)
    denoising/audio/{id}.wav
        The 38 clean signals and the 32 fragments the artifacts are cut from
    denoising/splits/{clean, artifact}_split.csv
        Tuning/test split (get_splits.py)

Generates:
    denoising/results/optimization/candidates/screening/<PARAMETER>.csv
        One row per value of the parameter: value, default (the value of remove_noise.py), the metrics (recall, load,
        w_med), the scalar diagnostics of candidate_score and the recall at each SNR (recall_<snr>). Rewritten after
        each value, so an interrupted run keeps the values already done

Execution:
    python denoising/screen_candidates.py [--params <PARAMETER> ...] [-s <set>] [-p <processes>]
        --params: parameters to vary, e.g. --params K MASK_DILATION (default: all the parameters of SCREENING)
        -s, --set: tune, test, new_signals or new_artifacts (default: tune)
        -p, --processes: number of worker processes (default: as many as CPU cores)
"""


import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from get_candidates import CANDIDATES_PATH, OPTIMIZATION_PATH, run_candidates
from run_experiment import load_artifacts, load_clean_signals
from src.algorithm import remove_noise as rn
from src.evaluation.evaluation import read_results
from src.optimization.candidate_score import candidate_score
from src.optimization.splits import SETS, split_results


# Paths
SCREENING_PATH = CANDIDATES_PATH / "screening"  # denoising/results/optimization/candidates/screening
EXPERIMENT_FILE = "experiment_1.csv"  # In denoising/results/experiments

# Values of each parameter (the default value of remove_noise.py is included in each list)
SCREENING = {
    "WAVELET": ["db1", "db2", "db4", "db6", "db8", "db10", "coif2", "coif4"],       # Orthogonal only
    "K": [1, 3, 5, 7, 9, 11, 13, 15],                                               # Detection threshold (times the reference)
    "ENV_SMOOTH": [0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064],             # s
    "MASK_DILATION": [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04],              # s (0 -> no widening)
    "REF_INNER": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],                    # s (< REF_OUTER = 0.4)
    "REF_OUTER": [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],                         # s (> REF_INNER = 0.1)
    "MAX_EVENT": [0.06, 0.08, 0.1, 0.11, 0.12, 0.13, 0.14, 0.16],                   # s
}

# Scalar diagnostics of candidate_score saved with the metrics
DIAGNOSTICS = ["recall_unweighted", "recall_w", "overflow_ms_med", "load_mix"]


def create_folder_structure() -> None:
    """
    Creates the following folders:
    - denoising/results/optimization
    - denoising/results/optimization/candidates
    - denoising/results/optimization/candidates/screening
    """
    OPTIMIZATION_PATH.mkdir(exist_ok=True)
    CANDIDATES_PATH.mkdir(exist_ok=True)
    SCREENING_PATH.mkdir(exist_ok=True)


def screening_row(value, default, metrics: dict, diagnostics: dict) -> dict:
    """
    Row of the screening table of one value of a parameter: value, whether it is the default one, metrics, scalar
    diagnostics and recall at each SNR (recall_<snr>, e.g. recall_-30).
    """
    recall_snr = diagnostics["per_snr"]["recall"]
    return {
        "value": value,
        "default": value == default,
        **metrics,
        **{key: diagnostics[key] for key in DIAGNOSTICS},
        **{f"recall_{snr:g}": recall for snr, recall in recall_snr.items()},
    }


def screen_parameter(param: str, exp_df: pd.DataFrame, clean_dict: dict, artifact_dict: dict,
                     processes: int | None = None) -> pd.DataFrame:
    """
    Screening of one parameter: for each value of SCREENING[param], sets it on remove_noise, generates the candidates
    of exp_df and computes their metrics. The default value is restored at the end (also if a run fails). The table is
    saved in SCREENING_PATH/<param>.csv after each value.

    Parameters
    ----------
    param: str
        Name of the parameter (constant of remove_noise, key of SCREENING)
    exp_df: pd.DataFrame
        Mixtures of experiment 1 (rows of experiment_1.csv)
    clean_dict, artifact_dict: dict
        Clean signals and artifacts (load_clean_signals, load_artifacts of run_experiment.py)
    processes: int or None
        Number of worker processes (None -> as many as CPU cores)

    Returns
    -------
    table: pd.DataFrame
        One row per value (see screening_row)
    """
    default = getattr(rn, param)
    path = SCREENING_PATH / f"{param}.csv"
    rows = []
    # Progress bar of the values (2nd line; the mixtures of each value below it, removed when the value ends)
    values_bar = tqdm(SCREENING[param], desc=f"{param} values", position=1, leave=False)
    for value in values_bar:
        values_bar.set_postfix_str(f"{param} = {value}" + (" (default)" if value == default else ""))
        setattr(rn, param, value)
        try:
            cand_df = run_candidates(exp_df, clean_dict, artifact_dict, processes=processes, position=2, leave=False)
        finally:
            setattr(rn, param, default)
        metrics, diagnostics = candidate_score(cand_df)
        rows.append(screening_row(value, default, metrics, diagnostics))
        tqdm.write(f"  {param} = {value}: recall = {metrics['recall']:.4f}, load = {metrics['load']:.3f}, "
                   f"w_med = {metrics['w_med']:.4f}")
        pd.DataFrame(rows).to_csv(path, index=False)
    values_bar.close()

    return pd.DataFrame(rows)


def main():
    print()
    parser = argparse.ArgumentParser(description="Screening of the candidate generation of the transient removal "
                                                 "filter (one parameter at a time)")
    parser.add_argument("--params", nargs="+", choices=list(SCREENING), default=list(SCREENING),
                        help="Parameters to vary (default: all)")
    parser.add_argument("-s", "--set", choices=SETS, default="tune",
                        help="Set of mixtures: tune, test, new_signals or new_artifacts (default: tune)")
    parser.add_argument("-p", "--processes", type=int, default=None,
                        help="Number of worker processes (default: as many as CPU cores)")
    args = parser.parse_args()

    print(f"Reading the mixtures of {EXPERIMENT_FILE}...")
    exp_df = split_results(read_results(EXPERIMENT_FILE))[args.set]
    print(f"{len(exp_df)} mixtures (controls included), set: {args.set}\n")

    print("Loading signals...")
    clean_dict = load_clean_signals()
    artifact_dict = load_artifacts()
    print(f"{len(clean_dict)} clean signals, {len(artifact_dict)} artifacts")

    create_folder_structure()
    n_runs = sum(len(SCREENING[p]) for p in args.params)
    print(f"Parameters: {', '.join(args.params)} ({n_runs} runs)\n")

    # Progress bar of the parameters (1st line); the values and the mixtures below it (see screen_parameter)
    params_bar = tqdm(args.params, desc="Parameters", position=0)
    for param in params_bar:
        params_bar.set_postfix_str(param)
        tqdm.write(f"Screening of {param}: {SCREENING[param]}")
        screen_parameter(param, exp_df, clean_dict, artifact_dict, processes=args.processes)
        tqdm.write(f"  Saved in {(SCREENING_PATH / f'{param}.csv').relative_to(Path(__file__).resolve().parent.parent)}")
    params_bar.close()


if __name__ == "__main__":  # Required by multiprocessing (spawn) on Windows
    main()
