"""
feature_search.py

Protocol of the Bayesian search of the parameters of the features (phase 2 of parameter_optimization.ipynb, section
5.x.2; PARAMETER_OPTIMIZATION.md): search space of each group of features, the search with Optuna (TPE) and the choice
of the best configuration. Used by search_features.py, which runs it on each fold of the nested cross-validation and on
the whole tuning set (the folds are those of phase 1, fold_sets of candidate_search.py).

The features are computed outside this module (search_features.py, a script): the search receives a function
evaluate(params) -> metrics.

Protocol (one run = one group of features and one fold, or the whole tuning set):
    - The candidates are those of C1 and do not change; only the parameters of the group are searched (the groups are
      independent: the metric of a group only depends on its own features)
    - The first configuration is the default one (baseline of phase 2), then TPE (multivariate) maximizes the metric of
      the group (mean standardized partial ROC-AUC of its features, see feature_score.py) on the training mixtures
    - Choice: the configuration with the highest metric -> C2' of the group (C2' = the choices of the three groups)
"""

import warnings
from typing import Callable

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState

from ..algorithm import remove_noise as rn


# ====================================================================================================================
# Constants
# ====================================================================================================================

RANDOM_STATE = 42  # Seed of the sampler

# Search space of each group: parameter -> (low, high, log scale, integer)
SEARCH_SPACE = {
    "Width": {"BG_FACTOR": (1, 8, True, False),          # Times the reference
              "WIDTH_FLOOR_DB": (6, 40, False, False),   # dB below the peak
              "EXTEND": (0, 4, False, False)},           # FWHM of the band at each side
    "Isolation": {"ISO_WIN": (0.02, 0.5, True, False)},  # s
    "Impulsiveness": {"Z_WIN": (0.02, 0.5, True, False),  # s
                      "AR_ORDER": (1, 40, False, True)},
}
# Default value of each parameter (baseline of phase 2), taken when the module is imported
DEFAULTS = {name: getattr(rn, name) for space in SEARCH_SPACE.values() for name in space}
TIME_PARAMS = ["ISO_WIN", "Z_WIN"]  # In s, shown in ms

# Configurations evaluated in each run (the default one included) and random ones before TPE starts
N_TRIALS = {"Width": 60, "Isolation": 20, "Impulsiveness": 40}
N_STARTUP_TRIALS = {"Width": 15, "Isolation": 5, "Impulsiveness": 10}


# ====================================================================================================================
# Search space
# ====================================================================================================================

def suggest_params(trial: optuna.Trial, group: str) -> dict:
    """Configuration of a trial: values of the constants of remove_noise of the group"""
    params = {}
    for name, (low, high, log, integer) in SEARCH_SPACE[group].items():
        if integer:
            params[name] = trial.suggest_int(name, low, high, log=log)
        else:
            params[name] = trial.suggest_float(name, low, high, log=log)
    return params


def trial_params(trial: FrozenTrial, group: str) -> dict:
    """Configuration of an evaluated trial"""
    return {name: trial.params[name] for name in SEARCH_SPACE[group]}


def default_params(group: str) -> dict:
    """Default configuration of a group (baseline of phase 2)"""
    return {name: DEFAULTS[name] for name in SEARCH_SPACE[group]}


def format_params(params: dict) -> str:
    """Configuration as text (times in ms), for the logs"""
    return ", ".join(f"{name} = {1000*value:.1f} ms" if name in TIME_PARAMS else
                     f"{name} = {value:.2f}" if isinstance(value, float) else f"{name} = {value}"
                     for name, value in params.items())


# ====================================================================================================================
# Search
# ====================================================================================================================

def completed_trials(study: optuna.Study) -> list[FrozenTrial]:
    """Evaluated configurations of a study"""
    return study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


def run_search(evaluate: Callable[[dict], dict], storage: str, group: str, n_trials: int | None = None,
               seed: int = RANDOM_STATE, on_trial: Callable | None = None) -> dict:
    """
    Search of one run (a group and a fold, or the whole tuning set) and choice of the best configuration. The study is
    saved in storage (study "search"), so an interrupted run continues where it stopped.

    Parameters
    ----------
    evaluate: callable
        evaluate(params) -> dict with the metrics of feature_score.score_features (metric, sauc_<feature>,
        auc_<feature>): computes the features of the group of the training mixtures of the run with the constants of
        remove_noise params (and those of C1) and their metrics
    storage: str
        Optuna storage, e.g. "sqlite:///.../fold_1.db"
    group: str
        Key of SEARCH_SPACE
    n_trials: int or None
        Configurations evaluated (the default one included). None -> N_TRIALS[group]
    seed: int
        Seed of the sampler
    on_trial: callable or None
        on_trial(study, trial), called after each evaluated configuration (progress bar, log)

    Returns
    -------
    result: dict
        - best: FrozenTrial of the configuration with the highest metric (C2' of the group in this run)
        - baseline: FrozenTrial of the default configuration
        - trials: every evaluated configuration, list of FrozenTrial
    """
    n_trials = N_TRIALS[group] if n_trials is None else n_trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():  # multivariate TPE: ExperimentalWarning
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS[group], multivariate=True, seed=seed)
    study = optuna.create_study(study_name="search", storage=storage, direction="maximize", sampler=sampler,
                                load_if_exists=True)
    if not study.get_trials(deepcopy=False):  # First configuration: the default one (baseline)
        study.enqueue_trial(default_params(group), user_attrs={"baseline": True})

    def objective(trial):
        metrics = evaluate(suggest_params(trial, group))
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return metrics["metric"]

    remaining = n_trials - len(completed_trials(study))
    if remaining > 0:
        callbacks = [lambda s, t: on_trial(s, t)] if on_trial else None
        study.optimize(objective, n_trials=remaining, callbacks=callbacks)

    trials = completed_trials(study)
    baseline = next(t for t in trials if t.user_attrs.get("baseline"))
    best = max(trials, key=lambda t: (t.value, -t.number))  # Ties: the first one
    return dict(best=best, baseline=baseline, trials=trials)


def trials_table(trials: list[FrozenTrial], group: str) -> pd.DataFrame:
    """
    One row per evaluated configuration, in the order of evaluation: number of the trial, whether it is the default
    configuration (baseline), configuration (parameters of the group) and metrics
    """
    return pd.DataFrame([{"trial": t.number, "baseline": bool(t.user_attrs.get("baseline", False)),
                          **trial_params(t, group),
                          **{name: value for name, value in t.user_attrs.items() if name != "baseline"}}
                         for t in trials])
