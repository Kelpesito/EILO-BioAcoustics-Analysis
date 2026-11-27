"""
configs.py

Configurations of the constants of the transient removal filter (remove_noise) chosen by the phases of
parameter_optimization.ipynb (PARAMETER_OPTIMIZATION.md). Used by run_experiment.py, save_figures.py and
get_features.py (option -c):
    baseline    default constants of remove_noise
    C1          baseline + parameters of the candidate generation chosen in phase 1 (C1 of the whole tuning set,
                results/optimization/candidates/search/all.json, search_candidates.py)
    C2          C1 + parameters of the features chosen in phase 2 (C2': best configuration of the run all of each group,
                results/optimization/features/search/{width, isolation, impulsiveness}/all.json, search_features.py)
"""

import json
from pathlib import Path


OPTIMIZATION_PATH = Path(__file__).resolve().parents[2] / "results" / "optimization"  # Independent of the cwd
C1_FILE = OPTIMIZATION_PATH / "candidates" / "search" / "all.json"  # C1 of the whole tuning set
FEATURE_SEARCH_PATH = OPTIMIZATION_PATH / "features" / "search"  # + /<group>/all.json: C2' of each group
FEATURE_GROUPS = ["width", "isolation", "impulsiveness"]

CONFIGS = ["baseline", "C1", "C2"]


def configuration(config: str) -> dict:
    """
    Constants of remove_noise of a configuration (only those that differ from the defaults; set them with
    candidate_search.remove_noise_params).

    Parameters
    ----------
    config: str
        baseline, C1 or C2

    Returns
    -------
    params: dict
        Name of the constant -> value
    """
    if config not in CONFIGS:
        raise ValueError(f"Unknown configuration {config} (available: {CONFIGS})")
    if config == "baseline":
        return {}
    if not C1_FILE.exists():
        raise FileNotFoundError(f"{C1_FILE} not found (run search_candidates.py --runs all)")
    params = json.loads(C1_FILE.read_text(encoding="utf-8"))["c1"]["params"]
    if config == "C1":
        return params
    for group in FEATURE_GROUPS:  # C2 = C1 + C2' (the best configuration of each group)
        file = FEATURE_SEARCH_PATH / group / "all.json"
        if not file.exists():
            raise FileNotFoundError(f"{file} not found: C2' of {group} not chosen yet (run search_features.py -g {group} "
                                    "--runs all)")
        params = {**params, **json.loads(file.read_text(encoding="utf-8"))["best"]["params"]}
    return params
