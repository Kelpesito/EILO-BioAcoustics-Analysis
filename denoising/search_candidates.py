"""
search_candidates.py

BAYESIAN SEARCH OF THE CANDIDATE GENERATION OF THE TRANSIENT REMOVAL FILTER (remove_noise)
------------------------------------------------------------------------------------------
Sections 4.4 and 4.5 of parameter_optimization.ipynb (phase 1, search and choice of C1) with a nested cross-validation:
the protocol of src/optimization/candidate_search.py (searches (a) and (b), safeguard (b2) and choice of C1) is run
    - on each of the N_CV_FOLDS = 5 folds of the tuning set (runs fold_1 ... fold_5): searches on the training mixtures
      of the fold, and its C1 (and the baseline) evaluated on the validation mixtures of the fold
    - on the whole tuning set (run all): final C1
The folds (get_splits.py) cross the folds of the tuning clean signals and of the tuning artifacts, as the tuning/test
split (fold_sets of candidate_search.py).
Each configuration evaluated = candidates of the training mixtures of the run (run_candidates of get_candidates.py) and
their metrics (score_candidates: recall and its SE, load, median w_i and diagnostics).

Requires:
    denoising/results/experiments/experiment_1.csv
        Mixtures of experiment 1 (run_experiment.py -e 1)
    denoising/audio/{id}.wav
        The 38 clean signals and the 32 fragments the artifacts are cut from
    denoising/splits/{clean, artifact}_split.csv, denoising/splits/cv_{clean, artifact}_folds.csv
        Tuning/test split and folds of the cross-validation (get_splits.py)
    denoising/results/optimization/candidates/baseline/tune.csv
        Candidates of the baseline (default constants of remove_noise) on the tuning set (get_candidates.py -s tune):
        metrics of the baseline and constraint of (b) on the median w_i

Generates:
    denoising/results/optimization/search/
        <run>.db            Optuna studies a, b (and b2) of the run (SQLite): an interrupted run continues where it
                            stopped when it is launched again
        <run>_trials.csv    One row per evaluated configuration: search, configuration, metrics, feasible
        <run>.json          Summary of the run: constraints, C1 and its metrics and those of the baseline (training
                            and, in the folds, validation). A run with its .json is skipped
        <runs>.log          Log of the execution, identical to the terminal (<runs>: the runs launched, e.g. fold_1 or
                            fold_1_fold_2_all; appended when the same runs are launched again)
        summary.csv         One row per finished run (rebuilt from the .json after each run)

Execution:
    python denoising/search_candidates.py [--runs <run> ...] [--trials-a <n>] [--trials-b <n>] [-p <processes>]
        --runs: folds 1-5 and/or all (whole tuning set) (default: 1 2 3 4 5 all). The runs are independent: they can
                be launched in separate jobs at the same time (e.g. --runs 1, --runs 2, ..., --runs all)
        --trials-a, --trials-b: configurations evaluated in (a) and in (b) (and in b2) (default: 100 and 100)
        -p, --processes: number of worker processes (default: as many as CPU cores)
        --seed: seed of the samplers (default: 42)
"""


import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from get_candidates import CANDIDATES_PATH, OPTIMIZATION_PATH, run_candidates
from run_experiment import load_artifacts, load_clean_signals
from src.evaluation.evaluation import read_results
from src.optimization import candidate_search as cs
from src.optimization.splits import N_CV_FOLDS, load_cv_folds, load_split, split_results


# Paths
SEARCH_PATH = OPTIMIZATION_PATH / "search"  # denoising/results/optimization/search
BASELINE_FILE = CANDIDATES_PATH / "baseline" / "tune.csv"  # get_candidates.py -s tune
EXPERIMENT_FILE = "experiment_1.csv"  # In denoising/results/experiments

RUNS = [str(k) for k in range(1, N_CV_FOLDS + 1)] + ["all"]
LOG_METRICS = ["recall", "load", "w_med"]  # Metrics of the summary of the cross-validation


# ====================================================================================================================
# Helpers
# ====================================================================================================================

def create_folder_structure() -> None:
    """
    Creates the following folders:
    - denoising/results/optimization
    - denoising/results/optimization/search
    """
    OPTIMIZATION_PATH.mkdir(exist_ok=True)
    SEARCH_PATH.mkdir(exist_ok=True)


class TqdmHandler(logging.Handler):
    """Logging handler that writes with tqdm.write, so the messages do not break the progress bars"""

    def emit(self, record):
        tqdm.write(self.format(record))


def get_logger(name: str) -> logging.Logger:
    """Logger of the execution: the same messages, with the same format, to the terminal and to SEARCH_PATH/<name>.log"""
    logger = logging.getLogger(f"search_candidates.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(SEARCH_PATH / f"{name}.log", encoding="utf-8"), TqdmHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def n_mixtures(df: pd.DataFrame) -> str:
    """Number of mixtures, controls, clean signals and artifacts of a set, as text"""
    art_id = df["art_id"].astype(str)
    is_control = art_id == "Control"
    return (f"{(~is_control).sum()} mixtures + {is_control.sum()} controls "
            f"({df['clean_id'].nunique()} signals x {art_id[~is_control].nunique()} artifacts)")


def format_metrics(metrics: dict, reference: dict | None = None) -> str:
    """recall ± SE, load and median w_i as text; with reference, also the difference with it"""
    text = (f"recall = {metrics['recall']:.4f} ± {metrics['recall_se']:.4f}, load = {metrics['load']:.2f}, "
            f"w_med = {metrics['w_med']:.3f}")
    if reference is not None:
        text += (f"  (Δ vs baseline: recall {metrics['recall'] - reference['recall']:+.4f}, "
                 f"load {metrics['load'] - reference['load']:+.2f}, w_med {metrics['w_med'] - reference['w_med']:+.3f})")
    return text


def make_evaluate(exp_df: pd.DataFrame, clean_dict: dict, artifact_dict: dict, processes: int | None):
    """evaluate(params) -> metrics of the candidates of exp_df with the constants of remove_noise params"""
    def evaluate(params):
        with cs.remove_noise_params(params):
            cand_df = run_candidates(exp_df, clean_dict, artifact_dict, processes=processes, position=2, leave=False)
        return cs.score_candidates(cand_df)
    return evaluate


def write_summary() -> pd.DataFrame:
    """summary.csv: one row per finished run (from the .json of the runs)"""
    rows = []
    for run in ["fold_" + r for r in RUNS[:-1]] + ["all"]:
        path = SEARCH_PATH / f"{run}.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text(encoding="utf-8"))
        row = {"run": run, "safeguard": s["safeguard"], "recall_min": s["recall_min"], "w_min": s["w_min"],
               **(s["c1"]["params"] if s["c1"] else {})}
        for part in ("train", "validation"):
            for who in ("c1", "baseline"):
                metrics = s[part][who] if s.get(part) and s[part].get(who) else {}
                row.update({f"{part}_{who}_{m}": metrics.get(m, np.nan) for m in cs.METRICS})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(SEARCH_PATH / "summary.csv", index=False)
    return summary


def log_cv_summary(summary: pd.DataFrame, logger: logging.Logger) -> None:
    """Validation of the folds: C1 of each fold and mean ± SD of its validation metrics and of their differences"""
    if summary.empty or "validation_c1_recall" not in summary:
        return
    folds = summary[summary["run"].str.startswith("fold_")].dropna(subset=["validation_c1_recall"])
    if folds.empty:
        return
    logger.info(f"=== Cross-validation: {len(folds)} of {N_CV_FOLDS} folds finished ===")
    params = folds.set_index("run")[[p for p in cs.SEARCH_PARAMS if p in folds]].copy()
    for name in ("ENV_SMOOTH", "MASK_DILATION", "REF_INNER", "REF_OUTER", "MAX_EVENT"):
        params[name] = (1000*params[name]).round(1)  # ms
    params["K"] = params["K"].round(2)
    logger.info("C1 of each fold (times in ms):\n" + params.to_string())
    for m in LOG_METRICS:
        c1, base = folds[f"validation_c1_{m}"], folds[f"validation_baseline_{m}"]
        logger.info(f"Validation {m}: C1 {c1.mean():.4f} ± {c1.std():.4f}, baseline {base.mean():.4f} ± {base.std():.4f}, "
                    f"Δ {(c1 - base).mean():+.4f} ± {(c1 - base).std():.4f} (mean ± SD over the folds)")


# ====================================================================================================================
# One run
# ====================================================================================================================

def run_one(run: str, train_df: pd.DataFrame, validation_df: pd.DataFrame | None, base_train: pd.DataFrame,
            base_validation: pd.DataFrame | None, clean_dict: dict, artifact_dict: dict, args, runs_bar: tqdm,
            logger: logging.Logger) -> None:
    """
    Search, choice of C1 and validation of one run (a fold or the whole tuning set); saves <run>_trials.csv and
    <run>.json
    """
    logger.info(f"=== Run {run} ===")
    logger.info(f"Training: {n_mixtures(train_df)}")
    if validation_df is not None:
        logger.info(f"Validation: {n_mixtures(validation_df)}")
    baseline_train = cs.score_candidates(base_train)
    logger.info(f"Baseline (training): {format_metrics(baseline_train)}")

    # Progress bar of the configurations of each search (2nd line) and log of each one
    state = {"bar": None, "search": None, "best": None, "t0": time.time()}

    def on_stage(search, info):
        if state["bar"] is not None:
            state["bar"].close()
        state["search"], state["best"] = search, None
        runs_bar.set_postfix_str(f"{run}, search {search}")
        if search == "a":
            logger.info(f"(a) Maximum recall: {info['n_trials']} configurations ({info['done']} already evaluated)")
        else:
            label = "(b) Minimum load" if search == "b" else "Safeguard (b2): (b) again"
            logger.info(f"{label}: {info['n_trials']} configurations ({info['done']} already evaluated); "
                        f"recall_max = {info['recall_max']:.4f}, SE = {info['se']:.4f} -> recall >= "
                        f"{info['recall_min']:.4f}, w_med >= {info['w_min']:.3f}")
            state["constraints"] = (info["recall_min"], info["w_min"])
        state["bar"] = tqdm(total=info["n_trials"], initial=info["done"], desc=f"Configurations ({search})",
                            position=1, leave=False)

    def on_trial(study, trial):
        m = trial.user_attrs
        seconds = trial.duration.total_seconds() if trial.duration else float("nan")
        logger.info(f"[{run} {study.study_name} {len(cs.own_trials(study)):3d}] {format_metrics(m)} | {seconds:.0f} s")
        if state["search"] == "a":
            if state["best"] is None or m["recall"] > state["best"]:
                state["best"] = m["recall"]
            state["bar"].set_postfix_str(f"best recall {state['best']:.4f}")
        elif cs.is_feasible(trial, *state["constraints"]):
            if state["best"] is None or m["load"] < state["best"]:
                state["best"] = m["load"]
            state["bar"].set_postfix_str(f"best feasible load {state['best']:.2f}")
        state["bar"].update(1)

    evaluate = make_evaluate(train_df, clean_dict, artifact_dict, args.processes)
    result = cs.run_search(evaluate, f"sqlite:///{(SEARCH_PATH / f'{run}.db').as_posix()}",
                           w_base=baseline_train["w_med"], n_trials_a=args.trials_a, n_trials_b=args.trials_b,
                           seed=args.seed, on_trial=on_trial, on_stage=on_stage)
    state["bar"].close()

    trials = cs.trials_table(result["trials"], result["recall_min"], result["w_min"])
    trials.to_csv(SEARCH_PATH / f"{run}_trials.csv", index=False)
    n_search = {search: int(n) for search, n in trials["search"].value_counts().items()}
    logger.info(f"Evaluated configurations: {n_search}; feasible (constraints of (b)): {int(trials['feasible'].sum())}"
                f"{'; safeguard used' if result['safeguard'] else ''}")

    # C1: metrics on the training mixtures and, in the folds, on the validation mixtures (and those of the baseline)
    c1 = result["c1"]
    summary = {"run": run, "recall_max": result["recall_max"], "se": result["se"], "recall_min": result["recall_min"],
               "w_min": result["w_min"], "safeguard": result["safeguard"], "n_trials": n_search, "c1": None,
               "train": {"baseline": baseline_train}, "validation": None}
    if c1 is None:
        logger.info("No configuration meets the constraints of (b): no C1")
    else:
        params = cs.trial_params(c1)
        summary["c1"] = {"search": c1.user_attrs["search"], "trial": c1.number, "params": params}
        summary["train"]["c1"] = {m: c1.user_attrs[m] for m in cs.METRICS}
        logger.info(f"C1 ({c1.user_attrs['search']}, trial {c1.number}): {cs.format_params(params)}")
        logger.info(f"C1 (training): {format_metrics(c1.user_attrs, baseline_train)}")
        if validation_df is not None:
            runs_bar.set_postfix_str(f"{run}, validation")
            baseline_validation = cs.score_candidates(base_validation)
            c1_validation = make_evaluate(validation_df, clean_dict, artifact_dict, args.processes)(params)
            summary["validation"] = {"c1": c1_validation, "baseline": baseline_validation}
            logger.info(f"Baseline (validation): {format_metrics(baseline_validation)}")
            logger.info(f"C1 (validation): {format_metrics(c1_validation, baseline_validation)}")
    logger.info(f"Run {run} finished in {(time.time() - state['t0'])/60:.0f} min")

    (SEARCH_PATH / f"{run}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ====================================================================================================================
# Main
# ====================================================================================================================

def main():
    parser = argparse.ArgumentParser(description="Bayesian search of the candidate generation (phase 1), with a "
                                                 "nested cross-validation")
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=RUNS,
                        help="Folds (1-5) and/or all (whole tuning set) (default: all of them)")
    parser.add_argument("--trials-a", type=int, default=cs.N_TRIALS_A,
                        help=f"Configurations evaluated in (a) (default: {cs.N_TRIALS_A})")
    parser.add_argument("--trials-b", type=int, default=cs.N_TRIALS_B,
                        help=f"Configurations evaluated in (b) and in b2 (default: {cs.N_TRIALS_B})")
    parser.add_argument("-p", "--processes", type=int, default=None,
                        help="Number of worker processes (default: as many as CPU cores)")
    parser.add_argument("--seed", type=int, default=cs.RANDOM_STATE, help="Seed of the samplers (default: 42)")
    args = parser.parse_args()
    runs = ["all" if r == "all" else f"fold_{r}" for r in args.runs]

    create_folder_structure()
    logger = get_logger("_".join(runs))
    logger.info(f"=== Bayesian search of the candidate generation: runs {', '.join(runs)} ===")
    logger.info(f"Searches: (a) {args.trials_a} and (b) {args.trials_b} configurations per run; seed {args.seed}")

    logger.info(f"Reading the mixtures of {EXPERIMENT_FILE}, the split, the folds and the candidates of the baseline...")
    clean_split, artifact_split = load_split()
    exp_df = split_results(read_results(EXPERIMENT_FILE), clean_split, artifact_split)["tune"]
    clean_folds, artifact_folds = load_cv_folds()
    baseline_df = pd.read_csv(BASELINE_FILE)
    logger.info(f"Tuning set: {n_mixtures(exp_df)}")
    for k in range(1, N_CV_FOLDS + 1):
        logger.info(f"  Fold {k}: {(clean_folds['fold'] == k).sum()} clean signals, "
                    f"{(artifact_folds['fold'] == k).sum()} artifacts")

    logger.info("Loading signals...")
    clean_dict = load_clean_signals()
    artifact_dict = load_artifacts()
    logger.info(f"{len(clean_dict)} clean signals, {len(artifact_dict)} artifacts")

    logger.info(f"Baseline (whole tuning set): {format_metrics(cs.score_candidates(baseline_df))}")
    logger.info(f"Results in {SEARCH_PATH.relative_to(Path(__file__).resolve().parent.parent)}")

    # Progress bar of the runs (1st line); the configurations (2nd) and the mixtures (3rd) below it
    runs_bar = tqdm(runs, desc="Cross-validation", position=0)
    for run in runs_bar:
        runs_bar.set_postfix_str(run)
        if (SEARCH_PATH / f"{run}.json").exists():
            logger.info(f"Run {run} already finished ({run}.json): skipped")
            continue
        if run == "all":
            train_df, validation_df, base_train, base_validation = exp_df, None, baseline_df, None
        else:
            fold = int(run.removeprefix("fold_"))
            train_df, validation_df = cs.fold_sets(exp_df, clean_folds, artifact_folds, fold)
            base_train, base_validation = cs.fold_sets(baseline_df, clean_folds, artifact_folds, fold)
        run_one(run, train_df, validation_df, base_train, base_validation, clean_dict, artifact_dict, args, runs_bar,
                logger)
        log_cv_summary(write_summary(), logger)
    runs_bar.close()

    for handler in logger.handlers:
        handler.close()


if __name__ == "__main__":  # Required by multiprocessing (spawn) on Windows
    main()
