"""
get_features.py

FEATURES OF THE CANDIDATES OF THE TRANSIENT REMOVAL FILTER (remove_noise)
-------------------------------------------------------------------------
Phase 2 of parameter_optimization.ipynb (feature definition): rebuilds the mixtures of experiment 1 (and its controls),
generates their candidates and computes the features of each one (generate_features of src/algorithm/remove_noise.py)
with the parameters of a configuration, in parallel (one worker process per CPU core by default). Each candidate is
also classified with the current click rules (find_click_events).
Configurations:
    baseline    default constants of remove_noise
    C1          baseline + parameters of the candidate generation chosen in phase 1 (C1 of the whole tuning set,
                results/optimization/candidates/search/all.json)
    C2          C1 + parameters of the features chosen in phase 2 (C2': best configuration of the run all of each group,
                results/optimization/features/search/{width, isolation, impulsiveness}/all.json)
The baselines of phase 2 are baseline and C1 (default parameters of the features, with the candidates of each one).

Requires:
    denoising/results/experiments/experiment_1.csv
        Mixtures of experiment 1 (run_experiment.py -e 1)
    denoising/audio/{id}.wav
        The 38 clean signals and the 32 fragments the artifacts are cut from
    denoising/splits/{clean, artifact}_split.csv
        Tuning/test split (get_splits.py), only if a set is given
    denoising/results/optimization/candidates/search/all.json
        C1 (search_candidates.py), only for the configurations C1 and C2
    denoising/results/optimization/features/search/{width, isolation, impulsiveness}/all.json
        C2' (search_features.py), only for the configuration C2

Generates:
    denoising/results/optimization/features/{config}/{set}.csv
        One row per candidate (mixtures and controls; a mixture without candidates has no row): identification of the
        mixture, number of mixtures of its cell (s, a, c), span of the candidate, whether it touches the zone of the
        artifact, its features and its classification with the current click rules (set = all if no set is given)

Execution:
    python denoising/get_features.py [-c <config>] [-s <set>] [-p <processes>]
        -c, --config: baseline, C1 or C2 (default: baseline)
        -s, --set: tune, test, new_signals or new_artifacts (default: all the mixtures)
        -p, --processes: number of worker processes (default: as many as CPU cores)
"""


import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from get_candidates import CANDIDATES_PATH, ID_COLUMNS, OPTIMIZATION_PATH
from run_experiment import load_artifacts, load_clean_signals
from src.algorithm import remove_noise as rn
from src.evaluation import evaluation as ev
from src.optimization.candidate_search import format_params, remove_noise_params
from src.optimization.splits import SETS, split_results


# Paths
FEATURES_PATH = OPTIMIZATION_PATH / "features"  # + /{config}
C1_FILE = CANDIDATES_PATH / "search" / "all.json"  # C1 of the whole tuning set (search_candidates.py)
EXPERIMENT_FILE = "experiment_1.csv"  # In denoising/results/experiments

# C2' of each group of features: best configuration of the run all (search_features.py), in FEATURES_PATH/search/<group>
FEATURE_GROUPS = ["width", "isolation", "impulsiveness"]

CONFIGS = ["baseline", "C1", "C2"]

# Columns of the output (besides ID_COLUMNS)
CELL = ["snr_db", "art_id", "clean_id"]  # Cell (s, a, c) of a mixture (controls: one cell per clean signal)
FEATURES = ["bands", "r_med", "r_max", "z_max", "isolation", "width_med", "width_max", "misalign", "at_border"]
RULES = ["broadband", "impulsive", "transient", "aligned", "border"]


def create_folder_structure(config: str) -> None:
    """
    Creates the following folders:
    - denoising/results/optimization
    - denoising/results/optimization/features
    - denoising/results/optimization/features/{config}

    Parameters
    ----------
    config: str
        Name of the configuration (baseline, C1 or C2)
    """
    OPTIMIZATION_PATH.mkdir(exist_ok=True)
    FEATURES_PATH.mkdir(exist_ok=True)
    (FEATURES_PATH / config).mkdir(exist_ok=True)


def configuration(config: str) -> dict:
    """
    Constants of remove_noise of a configuration (only those that differ from the defaults).

    Parameters
    ----------
    config: str
        baseline, C1 or C2

    Returns
    -------
    params: dict
        Name of the constant -> value
    """
    if config == "baseline":
        return {}
    if not C1_FILE.exists():
        raise FileNotFoundError(f"{C1_FILE} not found (run search_candidates.py --runs all)")
    with open(C1_FILE) as f:
        params = json.load(f)["c1"]["params"]
    if config == "C1":
        return params
    for group in FEATURE_GROUPS:  # C2 = C1 + C2' (the best configuration of each group)
        file = FEATURES_PATH / "search" / group / "all.json"
        if not file.exists():
            raise FileNotFoundError(f"{file} not found: C2' of {group} not chosen yet (run search_features.py -g {group} "
                                    "--runs all)")
        params = {**params, **json.loads(file.read_text(encoding="utf-8"))["best"]["params"]}
    return params


def process_mixture_features(row: dict) -> list[dict]:
    """
    Rebuilds a mixture of experiment 1 (row of experiment_1.csv; art_id = "Control" -> clean signal without
    artifact), generates its candidates and their features, and classifies them with the click rules. The signals
    are those stored by evaluation.init_worker.

    Returns
    -------
    rows: list of dict
        One per candidate: the row + cand (index of the candidate in the mixture), start and end (first and last + 1
        sample), touches_zone (the candidate touches the zone Z = [k0 - TOL, k1 + TOL) of the artifact, as in
        get_candidates.candidate_metrics; control: NaN), the features (FEATURES), whether it meets each rule
        (rule_<name>) and is_click
    """
    clean = ev._clean_dict[int(row["clean_id"])]["signal"]
    if str(row["art_id"]) == "Control":  # No artifact
        mixed, k0, k1 = clean.copy(), None, None
    else:
        art = ev._artifact_dict[int(row["art_id"])]
        k0 = int(row["k0"])
        k1 = k0 + len(art)
        mixed, _ = ev.insert_artifact(clean, art, k0, float(row["snr_db"]))  # Rebuild the mixture (snr_db: text in the .csv)

    # max_event and split_levels passed explicitly: the defaults of the function are fixed when the module is imported
    stages = rn.generate_features(mixed, ev.FS, max_event=rn.MAX_EVENT, split_levels=rn.SPLIT_LEVELS)
    _, _, _, classified = rn.find_click_events(stages["mask_protected"], stages["envelope"], stages["ratio"],
                                               stages["z"], stages["env_signal"], stages["levels"], ev.FS,
                                               stages["fwhm"], features=stages["features"])

    N = stages["N"]
    if k0 is not None:
        z0, z1 = max(0, k0 - ev.TOL), min(N, k1 + ev.TOL)
    rows = []
    for j, (cand, cls) in enumerate(zip(stages["features"], classified)):
        a, b = int(cand["a"]), int(min(cand["b"], N))
        touches = np.nan if k0 is None else (a < z1 and b > z0)
        rows.append({
            **row,
            "cand": j,
            "start": a,
            "end": b,
            "touches_zone": touches,
            **{name: cand["features"][name] for name in FEATURES},
            **{f"rule_{rule}": cls["passed"][rule] for rule in RULES},
            "is_click": cls["is_click"],
        })

    return rows


def run_features(exp_df: pd.DataFrame, clean_dict: dict, artifact_dict: dict,
                 processes: int | None = None) -> pd.DataFrame:
    """
    Features of the candidates of every mixture (and control) of exp_df, in parallel, with the current constants of
    remove_noise (copied to the workers by evaluation.init_worker).

    Parameters
    ----------
    exp_df: pd.DataFrame
        Mixtures of experiment 1 (rows of experiment_1.csv)
    clean_dict: dict
        Clean signals: clean_id -> {"signal", "id", "label"}
    artifact_dict: dict
        Artifacts: art_id -> signal
    processes: int or None
        Number of worker processes (None -> as many as CPU cores)

    Returns
    -------
    features: pd.DataFrame
        One row per candidate (see process_mixture_features), in the order of exp_df. Column n_positions: number of
        mixtures of the cell (s, a, c) of the mixture (N_{a,c}, for the weights of the metric of phase 2)
    """
    n_positions = exp_df.groupby(CELL, sort=False)["id"].transform("size").rename("n_positions")
    rows = pd.concat([exp_df[ID_COLUMNS], n_positions], axis=1).to_dict("records")
    params = {"remove_noise": ev.constants(rn), "evaluation": ev.constants(ev)}
    with Pool(processes, initializer=ev.init_worker, initargs=(clean_dict, artifact_dict, params)) as pool:
        results = list(tqdm(pool.imap(process_mixture_features, rows, chunksize=ev.CHUNKSIZE), total=len(rows),
                            desc="Mixtures"))

    return pd.DataFrame([cand for mixture in results for cand in mixture])


def main():
    print()
    parser = argparse.ArgumentParser(description="Features of the candidates of the transient removal filter in the "
                                                 "mixtures of experiment 1")
    parser.add_argument("-c", "--config", choices=CONFIGS, default="baseline",
                        help="Configuration: baseline, C1 or C2 (default: baseline)")
    parser.add_argument("-s", "--set", choices=SETS, default=None,
                        help="Set of mixtures: tune, test, new_signals or new_artifacts (default: all)")
    parser.add_argument("-p", "--processes", type=int, default=None,
                        help="Number of worker processes (default: as many as CPU cores)")
    args = parser.parse_args()

    params = configuration(args.config)
    print(f"Configuration {args.config}: {format_params(params) if params else 'default constants of remove_noise'}\n")

    print(f"Reading the mixtures of {EXPERIMENT_FILE}...")
    exp_df = ev.read_results(EXPERIMENT_FILE)
    if args.set is not None:
        exp_df = split_results(exp_df)[args.set]
    set_name = args.set or "all"
    print(f"{len(exp_df)} mixtures (controls included), set: {set_name}\n")

    print("Loading signals...")
    clean_dict = load_clean_signals()
    artifact_dict = load_artifacts()
    print(f"{len(clean_dict)} clean signals, {len(artifact_dict)} artifacts\n")

    with remove_noise_params(params):
        feat_df = run_features(exp_df, clean_dict, artifact_dict, processes=args.processes)

    create_folder_structure(args.config)
    path = FEATURES_PATH / args.config / f"{set_name}.csv"
    feat_df.to_csv(path, index=False)
    print(f"\n{len(feat_df)} candidates saved in {path.relative_to(Path(__file__).resolve().parent.parent)}")


if __name__ == "__main__":  # Required by multiprocessing (spawn) on Windows
    main()
