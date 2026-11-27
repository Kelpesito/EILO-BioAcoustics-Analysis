"""
get_candidates.py

CANDIDATES OF THE TRANSIENT REMOVAL FILTER (remove_noise)
---------------------------------------------------------
Phase 1 of parameter_optimization.ipynb (candidate generation): rebuilds the mixtures of experiment 1 (and its controls),
generates their candidates with the current parameters of remove_noise (generate_candidates of
src/algorithm/remove_noise.py) and computes the per-mixture metrics of the candidates (section 4.1), in parallel (one
worker process per CPU core by default).
Candidates: connected time spans of the union over the bands of the detections after the protection of the long
events.

Requires:
    denoising/results/experiments/baseline/experiment_1.csv
        Mixtures of experiment 1 (run_experiment.py -e 1, configuration baseline)
    denoising/audio/{id}.wav
        The 38 clean signals and the 32 fragments the artifacts are cut from
    denoising/splits/{clean, artifact}_split.csv
        Tuning/test split (get_splits.py), only if a set is given

Generates:
    denoising/results/optimization/candidates/{name}/{set}.csv
        One row per mixture (and per control): identification of the mixture and metrics of its candidates
        (set = all if no set is given). Columns with tuples are saved as JSON lists (read them with
        src.evaluation.evaluation.read_results)

Execution:
    python denoising/get_candidates.py [-n <name>] [-s <set>] [-p <processes>]
        -n, --name: name of the output folder, e.g. the configuration of the algorithm (default: baseline)
        -s, --set: tune, test, new_signals or new_artifacts (default: all the mixtures)
        -p, --processes: number of worker processes (default: as many as CPU cores)
"""


import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from tqdm.auto import tqdm

from run_experiment import load_artifacts, load_clean_signals
from src.algorithm import remove_noise as rn
from src.evaluation import evaluation as ev
from src.optimization.splits import SETS, split_results


# Paths
OPTIMIZATION_PATH = Path(__file__).resolve().parent / "results" / "optimization"
CANDIDATES_PATH = OPTIMIZATION_PATH / "candidates"  # + /{name}
EXPERIMENT_FILE = "baseline/experiment_1.csv"  # Mixtures of experiment 1, in denoising/results/experiments

# Columns of experiment_1.csv kept in the output (identification of the mixture)
ID_COLUMNS = ["id", "clean_id", "clean_file", "label", "art_id", "rep", "snr_db", "k0", "k1", "art_duration_ms"]


def create_folder_structure(name: str) -> None:
    """
    Creates the following folders:
    - denoising/results/optimization
    - denoising/results/optimization/candidates
    - denoising/results/optimization/candidates/{name}

    Parameters
    ----------
    name: str
        Name of the output folder (e.g. baseline)
    """
    OPTIMIZATION_PATH.mkdir(exist_ok=True)
    CANDIDATES_PATH.mkdir(exist_ok=True)
    (CANDIDATES_PATH / name).mkdir(exist_ok=True)


def candidate_metrics(span: np.ndarray, k0: int | None = None, k1: int | None = None) -> dict:
    """
    Per-mixture metrics of the candidates (section 4.1 of parameter_optimization.ipynb). Zone of the artifact
    Z = [k0 - TOL, k1 + TOL), as in evaluation.metrics. k0 = k1 = None -> control (no artifact): only the candidates.

    Parameters
    ----------
    span: np.ndarray
        Samples covered by some candidate (candidate_span of generate_candidates)
    k0, k1: int or None
        First and last + 1 sample of the artifact. None -> control

    Returns
    -------
    metrics: dict
        - n_candidates: number of candidates of the signal
        - n_candidates_out: candidates that do not touch Z (control: all of them)
        - has_candidate: some candidate touches Z (control: NaN)
        - zone_ms: |Z| (ms; control: NaN)
        - overflow_ms: time of the candidates that touch Z outside Z (ms; 0 without candidates; control: NaN)
        - cand_start, cand_end: first and last + 1 sample of each candidate (tuples)
    """
    N = len(span)
    lab, n = ndi.label(span)
    objects = ndi.find_objects(lab)
    spans = dict(cand_start=tuple(int(s[0].start) for s in objects), cand_end=tuple(int(s[0].stop) for s in objects))
    if k0 is None:
        return dict(n_candidates=n, n_candidates_out=n, has_candidate=np.nan, zone_ms=np.nan, overflow_ms=np.nan,
                    **spans)

    zone = np.zeros(N, bool)
    zone[max(0, k0 - ev.TOL):min(N, k1 + ev.TOL)] = True
    touching = np.setdiff1d(np.unique(lab[zone]), [0])     # Candidates that touch Z
    overflow = (np.isin(lab, touching) & ~zone).sum()      # |spans \ Z|

    return dict(
        n_candidates=n,
        n_candidates_out=n - len(touching),
        has_candidate=len(touching) > 0,
        zone_ms=zone.sum() / ev.FS * 1000,
        overflow_ms=overflow / ev.FS * 1000,
        **spans,
    )


def process_mixture_candidates(row: dict) -> dict:
    """
    Rebuilds a mixture of experiment 1 (row of experiment_1.csv; art_id = "Control" -> clean signal without
    artifact), generates its candidates and returns its identification + the metrics of its candidates. The signals
    are those stored by evaluation.init_worker.
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
    stages = rn.generate_candidates(mixed, ev.FS, max_event=rn.MAX_EVENT, split_levels=rn.SPLIT_LEVELS)

    return {**row, **candidate_metrics(stages["candidate_span"], k0, k1)}


def run_candidates(exp_df: pd.DataFrame, clean_dict: dict, artifact_dict: dict,
                   processes: int | None = None, position: int = 0, leave: bool = True) -> pd.DataFrame:
    """
    Candidates of every mixture (and control) of exp_df, in parallel, with the current constants of remove_noise
    (copied to the workers by evaluation.init_worker).

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
    position, leave: int, bool
        Line of the progress bar of the mixtures and whether it is kept when it ends (nested bars, see
        screen_candidates.py)

    Returns
    -------
    results: pd.DataFrame
        ID_COLUMNS of exp_df + the metrics of candidate_metrics, one row per mixture (in the order of exp_df)
    """
    rows = exp_df[ID_COLUMNS].to_dict("records")
    params = {"remove_noise": ev.constants(rn), "evaluation": ev.constants(ev)}
    with Pool(processes, initializer=ev.init_worker, initargs=(clean_dict, artifact_dict, params)) as pool:
        results = list(tqdm(pool.imap(process_mixture_candidates, rows, chunksize=ev.CHUNKSIZE), total=len(rows),
                            desc="Mixtures", position=position, leave=leave))

    return pd.DataFrame(results)


def main():
    print()
    parser = argparse.ArgumentParser(description="Candidates of the transient removal filter in the mixtures of "
                                                 "experiment 1")
    parser.add_argument("-n", "--name", default="baseline",
                        help="Name of the output folder, in results/optimization/candidates (default: baseline)")
    parser.add_argument("-s", "--set", choices=SETS, default=None,
                        help="Set of mixtures: tune, test, new_signals or new_artifacts (default: all)")
    parser.add_argument("-p", "--processes", type=int, default=None,
                        help="Number of worker processes (default: as many as CPU cores)")
    args = parser.parse_args()

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

    cand_df = run_candidates(exp_df, clean_dict, artifact_dict, processes=args.processes)

    create_folder_structure(args.name)
    path = CANDIDATES_PATH / args.name / f"{set_name}.csv"
    ev._tuples_to_json(cand_df).to_csv(path, index=False)
    print(f"\nResults saved in {path.relative_to(Path(__file__).resolve().parent.parent)}")


if __name__ == "__main__":  # Required by multiprocessing (spawn) on Windows
    main()
