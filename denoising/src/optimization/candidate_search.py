"""
candidate_search.py

Protocol of the Bayesian search of the candidate generation (phase 1 of parameter_optimization.ipynb, sections 4.4 and
4.5; PARAMETER_OPTIMIZATION.md): search space, constraints, the two searches (a) and (b) with Optuna (TPE), the
safeguard, the choice of C1 and the training/validation mixtures of each fold of the nested cross-validation (the
folds are made by get_splits.py). Used by search_candidates.py, which runs it on each fold and on the whole tuning set.

The candidates are generated outside this module (run_candidates of get_candidates.py, a script): the searches receive
a function evaluate(params) -> metrics.

Protocol (one run = one fold, or the whole tuning set):
    (a) Maximum recall: maximize the candidate recall -> recall_max and its standard error SE (bootstrap grouped by
        signal and by artifact)
    (b) Minimum load: minimize the load subject to recall >= recall_max - SE and median w_i >= median w_i of the
        baseline. Warm start: the trials of (a) are added to (b) without evaluating them again
    Safeguard: if no evaluated configuration meets the constraints of (b), recall_max and SE are taken from the
        configurations of (a) with median w_i >= median w_i of the baseline, and (b) is run again (b2)
    Choice: among the evaluated configurations (a, b and b2) that meet the constraints of (b), the one with the minimum
        load -> C1
Both searches are subject to REF_INNER < REF_OUTER and REF_OUTER - REF_INNER >= MAX_EVENT, met by construction (see
ref_outer).
"""

import warnings
from contextlib import contextmanager
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.samplers._base import _CONSTRAINTS_KEY  # Key of the constraint values of a trial (Optuna 4.x)
from optuna.trial import FrozenTrial, TrialState

from ..algorithm import remove_noise as rn
from .candidate_score import candidate_score, quality


# ====================================================================================================================
# Constants
# ====================================================================================================================

RANDOM_STATE = 42  # Seed of the samplers and of the bootstrap

# Search space (section "Parameters of the search" of 4.3)
WAVELETS = ["db6", "db8", "db10", "db12", "db16", "db20", "coif4", "coif6", "coif8", "coif10", "coif12", "coif16"]
K_RANGE = (3, 7)                    # Times the reference (linear)
ENV_SMOOTH_RANGE = (0.002, 0.016)   # s (log)
MASK_DILATION_RANGE = (0, 0.03)     # s (linear)
REF_INNER_RANGE = (0.01, 0.07)      # s (log)
REF_OUTER_RANGE = (0.15, 0.4)       # s (linear, from max(0.15, REF_INNER + MAX_EVENT); see ref_outer)
MAX_EVENT = 0.14                    # s. Fixed (not searched)
SEARCH_PARAMS = ["WAVELET", "K", "ENV_SMOOTH", "MASK_DILATION", "REF_INNER", "REF_OUTER", "MAX_EVENT"]

# Searches
N_TRIALS_A = 100            # Configurations evaluated in (a)
N_TRIALS_B = 100            # Configurations evaluated in (b) (and in b2, if the safeguard is needed)
N_STARTUP_TRIALS = 20       # Random configurations before TPE starts
N_BOOT = 1000               # Bootstrap resamples of the SE of the recall

# Metrics saved for each configuration (user attributes of the trials)
METRICS = ["recall", "recall_se", "load", "w_med", "recall_unweighted", "recall_w", "overflow_ms_med", "load_mix"]


# ====================================================================================================================
# Folds of the nested cross-validation
# ====================================================================================================================

def fold_sets(df: pd.DataFrame, clean_folds: pd.DataFrame, artifact_folds: pd.DataFrame,
              fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Training and validation mixtures of a fold, crossing the folds of the signals and of the artifacts (as the
    tuning/test split): training = signals and artifacts of the other folds, validation = signals and artifacts of
    the fold. The mixtures that cross both (signal of the fold and artifact of another one, or the reverse) are in
    neither. Each part keeps the controls of its clean signals.

    Parameters
    ----------
    df: pd.DataFrame
        Tuning mixtures (+ controls): rows of experiment_1.csv or of a table of candidates
    clean_folds, artifact_folds: pd.DataFrame
        Folds of the clean signals and of the artifacts (load_cv_folds of splits.py, made by get_splits.py)
    fold: int
        Fold (1..N_CV_FOLDS)

    Returns
    -------
    train, validation: pd.DataFrame
        Rows of df
    """
    art_id = df["art_id"].astype(str)
    is_control = art_id == "Control"
    clean_fold = df["clean_id"].map(clean_folds.set_index("clean_id")["fold"])
    art_fold = art_id.map(artifact_folds.set_index(artifact_folds["art_id"].astype(str))["fold"])
    in_clean = clean_fold == fold
    in_art = art_fold == fold
    return df[~in_clean & (~in_art | is_control)], df[in_clean & (in_art | is_control)]


# ====================================================================================================================
# Search space
# ====================================================================================================================

def ref_outer(ref_inner: float, u: float) -> float:
    """
    REF_OUTER from REF_INNER and u in [0, 1]: from max(REF_OUTER_RANGE[0], REF_INNER + MAX_EVENT) (u = 0) to
    REF_OUTER_RANGE[1] (u = 1). So REF_OUTER - REF_INNER >= MAX_EVENT (and REF_INNER < REF_OUTER) always holds: an
    event longer than REF_OUTER is only detected at its center (2 REF_OUTER - L), which must not be shorter than
    MAX_EVENT, or the event would lose the protection of long events (see 4.3).
    """
    lo = max(REF_OUTER_RANGE[0], ref_inner + MAX_EVENT)
    return lo + u*(REF_OUTER_RANGE[1] - lo)


def suggest_params(trial: optuna.Trial) -> dict:
    """
    Configuration of a trial: values of the constants of remove_noise (SEARCH_PARAMS). REF_OUTER is suggested as
    REF_OUTER_U in [0, 1] (see ref_outer); MAX_EVENT is fixed.
    """
    params = {
        "WAVELET": trial.suggest_categorical("WAVELET", WAVELETS),
        "K": trial.suggest_float("K", *K_RANGE),
        "ENV_SMOOTH": trial.suggest_float("ENV_SMOOTH", *ENV_SMOOTH_RANGE, log=True),
        "MASK_DILATION": trial.suggest_float("MASK_DILATION", *MASK_DILATION_RANGE),
        "REF_INNER": trial.suggest_float("REF_INNER", *REF_INNER_RANGE, log=True),
    }
    params["REF_OUTER"] = ref_outer(params["REF_INNER"], trial.suggest_float("REF_OUTER_U", 0, 1))
    params["MAX_EVENT"] = MAX_EVENT
    return params


def trial_params(trial: FrozenTrial) -> dict:
    """Configuration (SEARCH_PARAMS) of an evaluated trial"""
    params = {name: trial.params[name] for name in ("WAVELET", "K", "ENV_SMOOTH", "MASK_DILATION", "REF_INNER")}
    params["REF_OUTER"] = ref_outer(params["REF_INNER"], trial.params["REF_OUTER_U"])
    params["MAX_EVENT"] = MAX_EVENT
    return params


@contextmanager
def remove_noise_params(params: dict):
    """
    Sets the constants of remove_noise to params (run_candidates copies them to the worker processes) and restores
    their values at the end, also if an error is raised
    """
    saved = {name: getattr(rn, name) for name in params}
    for name, value in params.items():
        setattr(rn, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(rn, name, value)


def format_params(params: dict) -> str:
    """Configuration as text (times in ms), for the logs"""
    times = ("ENV_SMOOTH", "MASK_DILATION", "REF_INNER", "REF_OUTER", "MAX_EVENT", "ISO_WIN", "Z_WIN")
    return ", ".join(f"{name} = {1000*value:.1f} ms" if name in times else
                     f"{name} = {value:.2f}" if isinstance(value, float) else f"{name} = {value}"
                     for name, value in params.items())


# ====================================================================================================================
# Metrics
# ====================================================================================================================

def recall_se(cand_df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = RANDOM_STATE) -> float:
    """
    Standard error of the candidate recall (candidate_score): bootstrap grouped by artifact and by clean signal (both
    resampled), with the same averaging (positions -> clean signals -> artifacts, cost-weighted over the SNR).

    Parameters
    ----------
    cand_df: pd.DataFrame
        Per-mixture results of the candidates (run_candidates), mixtures + controls
    n_boot: int
        Bootstrap resamples
    seed: int
        Seed of the resamples

    Returns
    -------
    se: float
        Standard deviation of the recall over the resamples
    """
    mix = cand_df[cand_df["art_id"].astype(str) != "Control"]
    mix = mix.assign(snr_db=pd.to_numeric(mix["snr_db"]), has_candidate=mix["has_candidate"].astype(bool).astype(float))
    cells = mix.groupby(["snr_db", "art_id", "clean_id"])["has_candidate"].mean()     # mean over the positions
    snr = np.sort(mix["snr_db"].unique())
    arts, cleans = np.sort(mix["art_id"].unique()), np.sort(mix["clean_id"].unique())
    R = (cells.unstack("clean_id").reindex(pd.MultiIndex.from_product([snr, arts])).reindex(columns=cleans)
         .to_numpy(dtype=float).reshape(len(snr), len(arts), len(cleans)))            # SNR x artifact x clean signal
    cost = 1 - quality(snr)

    def recall(r):
        recall_snr = np.nanmean(np.nanmean(r, axis=2), axis=1)                        # clean signals -> artifacts
        return (cost*recall_snr).sum() / cost.sum()

    rng = np.random.default_rng(seed)
    boot = [recall(R[:, rng.integers(0, len(arts), len(arts))][:, :, rng.integers(0, len(cleans), len(cleans))])
            for _ in range(n_boot)]
    return float(np.std(boot, ddof=1))


def score_candidates(cand_df: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Metrics of a configuration (METRICS): those of candidate_score, its scalar diagnostics and the SE of the recall"""
    metrics, diagnostics = candidate_score(cand_df)
    return {**{k: float(v) for k, v in metrics.items()},
            "recall_se": recall_se(cand_df, n_boot),
            **{k: float(diagnostics[k]) for k in ("recall_unweighted", "recall_w", "overflow_ms_med", "load_mix")}}


# ====================================================================================================================
# Searches
# ====================================================================================================================

def is_feasible(trial: FrozenTrial, recall_min: float, w_min: float) -> bool:
    """Whether an evaluated configuration meets the constraints of (b): recall >= recall_min, median w_i >= w_min"""
    return trial.user_attrs["recall"] >= recall_min and trial.user_attrs["w_med"] >= w_min


def own_trials(study: optuna.Study) -> list[FrozenTrial]:
    """Configurations evaluated by a study (without those added by the warm start)"""
    return [t for t in study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            if t.user_attrs.get("search") == study.study_name]


def _sampler(seed: int, constraints_func=None) -> TPESampler:
    with warnings.catch_warnings():  # multivariate TPE: ExperimentalWarning
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        return TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True, seed=seed,
                          constraints_func=constraints_func)


def _optimize(study: optuna.Study, evaluate: Callable[[dict], dict], objective_metric: str, n_trials: int,
              on_trial: Callable | None) -> None:
    """Evaluates configurations in study until it has n_trials of its own (a resumed study continues)"""
    def objective(trial):
        params = suggest_params(trial)
        metrics = evaluate(params)
        trial.set_user_attr("search", study.study_name)
        trial.set_user_attr("REF_OUTER", params["REF_OUTER"])
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return metrics[objective_metric]

    remaining = n_trials - len(own_trials(study))
    if remaining > 0:
        callbacks = [lambda s, t: on_trial(s, t)] if on_trial else None
        study.optimize(objective, n_trials=remaining, callbacks=callbacks)


def _warm_start(study: optuna.Study, trials: list[FrozenTrial], constraints: Callable) -> None:
    """Adds already evaluated configurations to study (objective: load), with their constraint values"""
    for t in trials:
        study.add_trial(optuna.trial.create_trial(
            params=t.params, distributions=t.distributions, value=t.user_attrs["load"], user_attrs=t.user_attrs,
            system_attrs={_CONSTRAINTS_KEY: tuple(constraints(t))}))


def run_search(evaluate: Callable[[dict], dict], storage: str, w_base: float, n_trials_a: int = N_TRIALS_A,
               n_trials_b: int = N_TRIALS_B, seed: int = RANDOM_STATE, on_trial: Callable | None = None,
               on_stage: Callable | None = None) -> dict:
    """
    Searches (a) and (b) of one run (a fold or the whole tuning set), the safeguard (b2) if needed and the choice of
    C1. The studies are saved in storage (studies "a", "b", "b2"), so an interrupted run continues where it stopped.

    Parameters
    ----------
    evaluate: callable
        evaluate(params) -> dict with METRICS: generates the candidates of the training mixtures of the run with the
        constants of remove_noise params and computes their metrics (score_candidates)
    storage: str
        Optuna storage, e.g. "sqlite:///.../fold_1.db"
    w_base: float
        Median w_i of the baseline on the training mixtures (constraint of (b))
    n_trials_a, n_trials_b: int
        Configurations evaluated in (a) and in (b) (also in b2)
    seed: int
        Seed of the samplers
    on_trial: callable or None
        on_trial(study, trial), called after each evaluated configuration (progress bar, log)
    on_stage: callable or None
        on_stage(search: str, info: dict), called when a search ("a", "b", "b2") starts: info = n_trials, done
        (configurations already evaluated, if the study is resumed) and, for b and b2, recall_max, se, recall_min and
        w_min

    Returns
    -------
    result: dict
        - recall_max, se, recall_min: maximum recall, its SE and the recall constraint of (b) finally used
        - w_min: median w_i of the baseline (constraint of (b))
        - safeguard: whether b2 was needed
        - c1: FrozenTrial of C1 (None if no configuration meets the constraints of (b), not even after b2)
        - trials: every evaluated configuration (a, b and b2), list of FrozenTrial
    """
    stage = on_stage or (lambda *args: None)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # (a) Maximum recall
    study_a = optuna.create_study(study_name="a", storage=storage, direction="maximize", sampler=_sampler(seed),
                                  load_if_exists=True)
    stage("a", {"n_trials": n_trials_a, "done": len(own_trials(study_a))})
    _optimize(study_a, evaluate, "recall", n_trials_a, on_trial)
    trials_a = own_trials(study_a)
    best = max(trials_a, key=lambda t: t.user_attrs["recall"])
    recall_max, se = best.user_attrs["recall"], best.user_attrs["recall_se"]

    def search_b(name, recall_min, previous):
        """(b) with the recall constraint recall_min, warm start with the configurations of previous"""
        constraints = lambda t: (recall_min - t.user_attrs["recall"], w_base - t.user_attrs["w_med"])  # <= 0: met
        study = optuna.create_study(study_name=name, storage=storage, direction="minimize",
                                    sampler=_sampler(seed + 1, constraints), load_if_exists=True)
        if not study.get_trials(deepcopy=False):
            _warm_start(study, previous, constraints)
        stage(name, {"n_trials": n_trials_b, "done": len(own_trials(study)), "recall_max": recall_max, "se": se,
                     "recall_min": recall_min, "w_min": w_base})
        _optimize(study, evaluate, "load", n_trials_b, on_trial)
        return own_trials(study)

    # (b) Minimum load
    recall_min = recall_max - se
    trials = trials_a + search_b("b", recall_min, trials_a)
    feasible = [t for t in trials if is_feasible(t, recall_min, w_base)]

    # Safeguard: recall_max from the configurations of (a) with median w_i >= that of the baseline, and (b) again
    safeguard = not feasible
    if safeguard:
        candidates = [t for t in trials_a if t.user_attrs["w_med"] >= w_base]
        if candidates:
            best = max(candidates, key=lambda t: t.user_attrs["recall"])
            recall_max, se = best.user_attrs["recall"], best.user_attrs["recall_se"]
            recall_min = recall_max - se
            trials = trials + search_b("b2", recall_min, trials)
            feasible = [t for t in trials if is_feasible(t, recall_min, w_base)]

    # Choice: minimum load among the configurations that meet the constraints of (b)
    c1 = min(feasible, key=lambda t: t.user_attrs["load"]) if feasible else None
    return dict(recall_max=recall_max, se=se, recall_min=recall_min, w_min=w_base, safeguard=safeguard, c1=c1,
                trials=trials)


def trials_table(trials: list[FrozenTrial], recall_min: float, w_min: float) -> pd.DataFrame:
    """
    One row per evaluated configuration: search (a, b, b2), number of the trial in its search, configuration
    (SEARCH_PARAMS), METRICS and whether it meets the constraints of (b) (feasible)
    """
    return pd.DataFrame([{"search": t.user_attrs["search"], "trial": t.number, **trial_params(t),
                          **{m: t.user_attrs[m] for m in METRICS}, "feasible": is_feasible(t, recall_min, w_min)}
                         for t in trials])
