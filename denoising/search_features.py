"""
search_features.py

BAYESIAN SEARCH OF THE PARAMETERS OF THE FEATURES OF THE TRANSIENT REMOVAL FILTER (remove_noise)
-----------------------------------------------------------------------------------------------
Sections 5.2.2, 5.3.2 and 5.4.2 of parameter_optimization.ipynb (phase 2, search of the parameters of each group of
features) with a nested cross-validation: the protocol of src/optimization/feature_search.py (default configuration +
TPE, choice of the configuration with the highest metric) is run, for one group of features,
    - on each of the N_CV_FOLDS = 5 folds of the tuning set (runs fold_1 ... fold_5): search on the training mixtures
      of the fold, and its best configuration (and the default one) evaluated on the validation mixtures of the fold
    - on the whole tuning set (run all): C2' of the group
The folds are those of phase 1 (fold_sets of candidate_search.py). The candidates are those of C1 and do not change:
each configuration evaluated = features of the group of the candidates of the training mixtures of the run (only the
mixtures: the controls are not used by the metric) and their metric (feature_score.py, section 5.1):
    - Width (BG_FACTOR, WIDTH_FLOOR_DB, EXTEND): candidate generation of each mixture (C1) and widths
    - Isolation (ISO_WIN): envelope of each mixture and isolation of the spans of the candidates of C1
    - Impulsiveness (Z_WIN, AR_ORDER): impulsiveness score of each mixture and z_max of the spans of the candidates of C1

Requires:
    denoising/results/optimization/features/C1/tune.csv
        Features of the candidates of C1 on the tuning set (get_features.py -c C1 -s tune): mixtures, spans of the
        candidates, whether they touch the zone of the artifact and the features with the default parameters (baseline)
    denoising/results/optimization/candidates/search/all.json
        C1 (search_candidates.py)
    denoising/audio/{id}.wav
        The 38 clean signals and the 32 fragments the artifacts are cut from
    denoising/splits/cv_{clean, artifact}_folds.csv
        Folds of the cross-validation (get_splits.py)

Generates:
    denoising/results/optimization/features/search/{group}/
        <run>.db           Optuna study of the run (SQLite): an interrupted run continues where it stopped when it is
                            launched again
        <run>_trials.csv    One row per evaluated configuration: trial, baseline (default configuration), configuration,
                            metrics
        <run>.json          Summary of the run: best configuration (C2' of the group in the run) and its metrics and those
                            of the baseline (training and, in the folds, validation). A run with its .json is skipped
        <runs>.log          Log of the execution, identical to the terminal (appended when the same runs are launched
                            again)
        summary.csv         One row per finished run (rebuilt from the .json after each run)

Execution:
    python denoising/search_features.py -g <group> [--runs <run> ...] [--trials <n>] [-p <processes>] [--seed <n>]
        -g, --group: width, isolation or impulsiveness
        --runs: folds 1-5 and/or all (whole tuning set) (default: 1 2 3 4 5 all). The runs are independent: they can
                be launched in separate jobs at the same time
        --trials: configurations evaluated per run, the default one included (default: 60 width, 20 isolation,
                  40 impulsiveness)
        -p, --processes: number of worker processes (default: as many as CPU cores)
        --seed: seed of the sampler (default: 42)
"""


import argparse
import json
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from get_features import FEATURES_PATH, configuration
from run_experiment import load_artifacts, load_clean_signals
from src.algorithm import remove_noise as rn
from src.evaluation import evaluation as ev
from src.optimization import candidate_search as cs
from src.optimization import feature_score as fsc
from src.optimization import feature_search as fs
from src.optimization.splits import N_CV_FOLDS, load_cv_folds


# Paths
SEARCH_PATH = FEATURES_PATH / "search"  # + /{group}
C1_FEATURES_FILE = FEATURES_PATH / "C1" / "tune.csv"  # get_features.py -c C1 -s tune

GROUPS = {"width": "Width", "isolation": "Isolation", "impulsiveness": "Impulsiveness"}  # Option -> group
RUNS = [str(k) for k in range(1, N_CV_FOLDS + 1)] + ["all"]
CHUNKSIZE = {"Width": 2, "Isolation": 64, "Impulsiveness": 8}  # Mixtures sent to a worker at a time


# ====================================================================================================================
# Features of a group (worker processes)
# ====================================================================================================================

def group_values(task: tuple) -> list[dict]:
    """
    Features of a group of the candidates of one mixture, with the current constants of remove_noise (those of C1 and
    of the configuration, copied to the workers by evaluation.init_worker). The signals are those stored by
    evaluation.init_worker.

    Parameters
    ----------
    task: tuple
        (row, spans, group): row of the table of the candidates of C1 (identification of the mixture), spans (start,
        end) of its candidates in the order of the table, and the group (key of feature_score.FEATURE_GROUPS)

    Returns
    -------
    values: list of dict
        One per candidate, in the order of spans: {feature: value} of the features of the group
    """
    row, spans, group = task
    clean = ev._clean_dict[int(row["clean_id"])]["signal"]
    art = ev._artifact_dict[int(row["art_id"])]
    mixed, _ = ev.insert_artifact(clean, art, int(row["k0"]), float(row["snr_db"]))  # Rebuild the mixture

    if group == "Width":  # Candidate generation (envelope and reference of each band) and widths
        # max_event and split_levels passed explicitly: the defaults of the function are fixed when it is imported
        stages = rn.generate_candidates(mixed, ev.FS, max_event=rn.MAX_EVENT, split_levels=rn.SPLIT_LEVELS)
        ratio = stages["envelope"] / np.where(np.isnan(stages["reference"]), np.inf,
                                              np.maximum(stages["reference"], rn.EPS))  # As generate_features
        fwhm = rn.band_resolution(ev.FS, stages["n_levels"],
                                  None if rn.SPLIT_LEVELS is None else tuple(rn.SPLIT_LEVELS), stages["l_env"])
        env_signal = rn.signal_envelope(stages["signal"], ev.FS)
        names = fsc.FEATURE_GROUPS[group]
        cands = rn.candidate_features(stages["mask_protected"], stages["envelope"], ratio, None, env_signal,
                                      stages["levels"], ev.FS, fwhm, names=names)
        found = [(int(c["a"]), int(min(c["b"], stages["N"]))) for c in cands]
        if found != list(spans):
            raise RuntimeError(f"Mixture {row['id']}: the candidates differ from those of {C1_FEATURES_FILE.name} "
                               "(are the constants of C1 the same?)")
        return [{name: c["features"][name] for name in names} for c in cands]

    signal = rn.prepare_signal(mixed)
    if group == "Isolation":  # Envelope of the signal and isolation of the spans
        env_signal = rn.signal_envelope(signal, ev.FS)
        return [{"isolation": rn.span_isolation(env_signal, a, b, ev.FS)} for a, b in spans]

    # Impulsiveness: score of each sample and its maximum in each span (+- Z_TOLERANCE)
    z = rn.impulsiveness(signal, ev.FS)[0]
    z_near = rn.impulsiveness_near(z, len(z), ev.FS)
    return [{"z_max": z_near[a:b].max()} for a, b in spans]


def run_group(table: pd.DataFrame, group: str, clean_dict: dict, artifact_dict: dict, processes: int | None = None,
              position: int = 0, leave: bool = True) -> pd.DataFrame:
    """
    Features of a group of the candidates of the mixtures of table (controls excluded), in parallel, with the current
    constants of remove_noise (copied to the workers by evaluation.init_worker).

    Parameters
    ----------
    table: pd.DataFrame
        Rows of the table of the candidates of C1 (get_features.py), e.g. the training mixtures of a fold
    group: str
        Key of feature_score.FEATURE_GROUPS
    clean_dict, artifact_dict: dict
        Clean signals (clean_id -> {"signal", "id", "label"}) and artifacts (art_id -> signal)
    processes: int or None
        Number of worker processes (None -> as many as CPU cores)
    position, leave: int, bool
        Line of the progress bar of the mixtures and whether it is kept when it ends

    Returns
    -------
    table: pd.DataFrame
        The candidates of the mixtures of table (sorted by mixture and candidate), with the features of the group
        recomputed
    """
    mix = table[table["art_id"].astype(str) != "Control"].sort_values(["id", "cand"]).reset_index(drop=True)
    tasks = [(rows.iloc[0].to_dict(), list(zip(rows["start"].astype(int), rows["end"].astype(int))), group)
             for _, rows in mix.groupby("id", sort=False)]
    params = {"remove_noise": ev.constants(rn), "evaluation": ev.constants(ev)}
    with Pool(processes, initializer=ev.init_worker, initargs=(clean_dict, artifact_dict, params)) as pool:
        results = list(tqdm(pool.imap(group_values, tasks, chunksize=CHUNKSIZE[group]), total=len(tasks),
                            desc="Mixtures", position=position, leave=leave))
    values = pd.DataFrame([v for mixture in results for v in mixture])
    mix[values.columns] = values.to_numpy()
    return mix


# ====================================================================================================================
# Helpers
# ====================================================================================================================

def create_folder_structure(path: Path) -> None:
    """
    Creates the following folders:
    - denoising/results/optimization/features
    - denoising/results/optimization/features/search
    - denoising/results/optimization/features/search/{group}
    """
    FEATURES_PATH.mkdir(parents=True, exist_ok=True)
    SEARCH_PATH.mkdir(exist_ok=True)
    path.mkdir(exist_ok=True)


class TqdmHandler(logging.Handler):
    """Logging handler that writes with tqdm.write, so the messages do not break the progress bars"""

    def emit(self, record):
        tqdm.write(self.format(record))


def get_logger(path: Path, name: str) -> logging.Logger:
    """Logger of the execution: the same messages, with the same format, to the terminal and to path/<name>.log"""
    logger = logging.getLogger(f"search_features.{path.name}.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(path / f"{name}.log", encoding="utf-8"), TqdmHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def n_mixtures(table: pd.DataFrame) -> str:
    """Number of mixtures with candidates, candidates, clean signals and artifacts of a set, as text"""
    mix = table[table["art_id"].astype(str) != "Control"]
    return (f"{mix['id'].nunique()} mixtures with candidates, {len(mix)} candidates "
            f"({mix['clean_id'].nunique()} signals x {mix['art_id'].nunique()} artifacts)")


def format_metrics(metrics: dict, reference: dict | None = None) -> str:
    """Metric and sAUC of each feature as text; with reference, also the difference of the metric with it"""
    text = f"metric = {metrics['metric']:.4f}"
    saucs = [f"{name.removeprefix('sauc_')} {value:.4f}" for name, value in metrics.items() if name.startswith("sauc_")]
    if len(saucs) > 1:
        text += f" ({', '.join(saucs)})"
    if reference is not None:
        text += f"  (Δ vs baseline {metrics['metric'] - reference['metric']:+.4f})"
    return text


def make_evaluate(table: pd.DataFrame, group: str, c1: dict, clean_dict: dict, artifact_dict: dict,
                  processes: int | None):
    """evaluate(params) -> metrics of the group of the candidates of table with the constants of C1 and params"""
    def evaluate(params):
        with cs.remove_noise_params({**c1, **params}):
            values = run_group(table, group, clean_dict, artifact_dict, processes, position=2, leave=False)
        return fsc.score_features(values, group)
    return evaluate


def write_summary(path: Path, group: str) -> pd.DataFrame:
    """summary.csv: one row per finished run (from the .json of the runs)"""
    rows = []
    for run in ["fold_" + r for r in RUNS[:-1]] + ["all"]:
        file = path / f"{run}.json"
        if not file.exists():
            continue
        s = json.loads(file.read_text(encoding="utf-8"))
        row = {"run": run, "best_trial": s["best"]["trial"], **s["best"]["params"]}
        for part in ("train", "validation"):
            for who in ("best", "baseline"):
                metrics = s[part][who] if s.get(part) else {}
                row.update({f"{part}_{who}_{name}": value for name, value in metrics.items()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(path / "summary.csv", index=False)
    return summary


def log_cv_summary(summary: pd.DataFrame, group: str, logger: logging.Logger) -> None:
    """Validation of the folds: best configuration of each fold and mean ± SD of its validation metric"""
    if summary.empty or "validation_best_metric" not in summary:
        return
    folds = summary[summary["run"].str.startswith("fold_")].dropna(subset=["validation_best_metric"])
    if folds.empty:
        return
    logger.info(f"=== Cross-validation: {len(folds)} of {N_CV_FOLDS} folds finished ===")
    params = folds.set_index("run")[list(fs.SEARCH_SPACE[group])].copy()
    for name in fs.TIME_PARAMS:
        if name in params:
            params[name] = (1000*params[name]).round(1)  # ms
    logger.info("Best configuration of each fold (times in ms):\n" + params.round(2).to_string())
    for part in ("train", "validation"):
        best, base = folds[f"{part}_best_metric"], folds[f"{part}_baseline_metric"]
        logger.info(f"{part.capitalize()} metric: best {best.mean():.4f} ± {best.std():.4f}, baseline "
                    f"{base.mean():.4f} ± {base.std():.4f}, Δ {(best - base).mean():+.4f} ± {(best - base).std():.4f} "
                    "(mean ± SD over the folds)")


# ====================================================================================================================
# One run
# ====================================================================================================================

def run_one(run: str, group: str, train: pd.DataFrame, validation: pd.DataFrame | None, c1: dict, clean_dict: dict,
            artifact_dict: dict, args, path: Path, runs_bar: tqdm, logger: logging.Logger) -> None:
    """Search, choice and validation of one run (a fold or the whole tuning set); saves <run>_trials.csv and <run>.json"""
    t0 = time.time()
    logger.info(f"=== Run {run} ===")
    logger.info(f"Training: {n_mixtures(train)}")
    if validation is not None:
        logger.info(f"Validation: {n_mixtures(validation)}")
    baseline_train = fsc.score_features(train, group)  # Default parameters: features of the table of C1
    logger.info(f"Baseline (training): {format_metrics(baseline_train)}")

    n_trials = args.trials or fs.N_TRIALS[group]
    bar = tqdm(total=n_trials, desc="Configurations", position=1, leave=False)
    state = {"best": None}

    def on_trial(study, trial):
        m = trial.user_attrs
        seconds = trial.duration.total_seconds() if trial.duration else float("nan")
        label = "baseline" if m.get("baseline") else fs.format_params(fs.trial_params(trial, group))
        logger.info(f"[{run} {trial.number + 1:3d}] {format_metrics(m)} | {label} | {seconds:.0f} s")
        if state["best"] is None or m["metric"] > state["best"]:
            state["best"] = m["metric"]
        bar.set_postfix_str(f"best metric {state['best']:.4f}")
        bar.update(1)

    evaluate = make_evaluate(train, group, c1, clean_dict, artifact_dict, args.processes)
    storage = f"sqlite:///{(path / f'{run}.db').as_posix()}"
    bar.update(0)
    result = fs.run_search(evaluate, storage, group, n_trials=n_trials, seed=args.seed, on_trial=on_trial)
    bar.close()

    trials = fs.trials_table(result["trials"], group)
    trials.to_csv(path / f"{run}_trials.csv", index=False)
    best, baseline = result["best"], result["baseline"]
    logger.info(f"Evaluated configurations: {len(trials)}")
    if abs(baseline.user_attrs["metric"] - baseline_train["metric"]) > 1e-9:
        logger.info(f"WARNING: the default configuration recomputed ({baseline.user_attrs['metric']:.6f}) differs from "
                    f"the table of C1 ({baseline_train['metric']:.6f})")

    # Best configuration: metrics on the training mixtures and, in the folds, on the validation mixtures
    params = fs.trial_params(best, group)
    best_train = {name: value for name, value in best.user_attrs.items() if name != "baseline"}
    summary = {"run": run, "group": group, "n_trials": len(trials),
               "best": {"trial": best.number, "baseline": bool(best.user_attrs.get("baseline", False)),
                        "params": params},
               "train": {"best": best_train, "baseline": baseline_train}, "validation": None}
    logger.info(f"Best configuration (trial {best.number + 1}): {fs.format_params(params)}")
    logger.info(f"Best (training): {format_metrics(best_train, baseline_train)}")
    if validation is not None:
        runs_bar.set_postfix_str(f"{run}, validation")
        baseline_validation = fsc.score_features(validation, group)
        best_validation = make_evaluate(validation, group, c1, clean_dict, artifact_dict, args.processes)(params)
        summary["validation"] = {"best": best_validation, "baseline": baseline_validation}
        logger.info(f"Baseline (validation): {format_metrics(baseline_validation)}")
        logger.info(f"Best (validation): {format_metrics(best_validation, baseline_validation)}")
    logger.info(f"Run {run} finished in {(time.time() - t0)/60:.0f} min")

    (path / f"{run}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ====================================================================================================================
# Main
# ====================================================================================================================

def main():
    parser = argparse.ArgumentParser(description="Bayesian search of the parameters of a group of features (phase 2), "
                                                 "with a nested cross-validation")
    parser.add_argument("-g", "--group", choices=GROUPS, required=True,
                        help="Group of features: width, isolation or impulsiveness")
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=RUNS,
                        help="Folds (1-5) and/or all (whole tuning set) (default: all of them)")
    parser.add_argument("--trials", type=int, default=None,
                        help="Configurations evaluated per run, the default one included (default: "
                             + ", ".join(f"{fs.N_TRIALS[g]} {o}" for o, g in GROUPS.items()) + ")")
    parser.add_argument("-p", "--processes", type=int, default=None,
                        help="Number of worker processes (default: as many as CPU cores)")
    parser.add_argument("--seed", type=int, default=fs.RANDOM_STATE, help="Seed of the sampler (default: 42)")
    args = parser.parse_args()
    group = GROUPS[args.group]
    runs = ["all" if r == "all" else f"fold_{r}" for r in args.runs]
    path = SEARCH_PATH / args.group

    create_folder_structure(path)
    logger = get_logger(path, "_".join(runs))
    logger.info(f"=== Bayesian search of the parameters of the features: group {group}, runs {', '.join(runs)} ===")
    logger.info(f"Search space: " + ", ".join(f"{name} [{low}, {high}]{' (log)' if log else ''}"
                                              for name, (low, high, log, _) in fs.SEARCH_SPACE[group].items()))
    logger.info(f"Default configuration (baseline): {fs.format_params(fs.default_params(group))}")
    logger.info(f"{args.trials or fs.N_TRIALS[group]} configurations per run; seed {args.seed}")

    if not C1_FEATURES_FILE.exists():
        raise FileNotFoundError(f"{C1_FEATURES_FILE} not found (run get_features.py -c C1 -s tune)")
    c1 = configuration("C1")
    logger.info(f"C1: {cs.format_params(c1)}")
    logger.info(f"Reading the candidates of C1 ({C1_FEATURES_FILE.name}) and the folds...")
    table = pd.read_csv(C1_FEATURES_FILE, low_memory=False)
    clean_folds, artifact_folds = load_cv_folds()
    logger.info(f"Tuning set: {n_mixtures(table)}")

    logger.info("Loading signals...")
    clean_dict = load_clean_signals()
    artifact_dict = load_artifacts()
    logger.info(f"{len(clean_dict)} clean signals, {len(artifact_dict)} artifacts")
    logger.info(f"Baseline (whole tuning set): {format_metrics(fsc.score_features(table, group))}")
    logger.info(f"Results in {path.relative_to(Path(__file__).resolve().parent.parent)}")

    # Progress bar of the runs (1st line); the configurations (2nd) and the mixtures (3rd) below it
    runs_bar = tqdm(runs, desc="Cross-validation", position=0)
    for run in runs_bar:
        runs_bar.set_postfix_str(run)
        if (path / f"{run}.json").exists():
            logger.info(f"Run {run} already finished ({run}.json): skipped")
            continue
        if run == "all":
            train, validation = table, None
        else:
            train, validation = cs.fold_sets(table, clean_folds, artifact_folds, int(run.removeprefix("fold_")))
        run_one(run, group, train, validation, c1, clean_dict, artifact_dict, args, path, runs_bar, logger)
        log_cv_summary(write_summary(path, group), group, logger)
    runs_bar.close()

    for handler in logger.handlers:
        handler.close()


if __name__ == "__main__":  # Required by multiprocessing (spawn) on Windows
    main()
