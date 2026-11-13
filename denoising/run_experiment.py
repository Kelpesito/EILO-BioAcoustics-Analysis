"""
run_experiments.py

EVALUATION OF THE TRANSIENT REMOVAL FILTER (remove_noise)
---------------------------------------------------------
Runs the evaluation experiments of evaluation_denoise.ipynb from the terminal: loads the clean signals and the
artifacts, generates the mixtures (get_mixes_1 / get_mixes_2, same code and constants as the notebook) and processes
them in parallel with run_experiment (src/evaluation.py, one worker process per CPU core by default).
    - Experiment 1: insertion of a single artifact (+ controls without artifact)
    - Experiment 2: insertion of several artifacts (paired subsets, equal / unequal SNR) + idempotence (+ controls)

Requires:
    denoising/audio/{id}.wav
        The 38 clean signals and the 40 fragments the artifacts are cut from

Generates:
    denoising/results/experiment_{1, 2}.csv
        Metrics of each mixture (same file as the notebook, wherever the script is run from)

Execution:
    python denoising/run_experiments.py -e <experiment> [-p <processes>]
        - experiment: 1 or 2
        -p, --processes: number of worker processes (default: as many as CPU cores)
"""


import argparse
import random
from itertools import combinations
from pathlib import Path

import pandas as pd
import soundfile as sf
from scipy.signal import detrend
from scipy.signal.windows import tukey

from src.evaluation import FS, insert_artifact, insert_artifacts, process_mixture, process_mixture_multi, run_experiment


# Paths
AUDIO_PATH = Path(__file__).resolve().parent / "audio"  # denoising/audio (independent of the cwd)

# Signals (section 4 of evaluation_denoise.ipynb)
# Clean signals: 1 fragment per patient, no clicks, scrapes or impacts audible or visible in the spectrogram
CLEAN_SIGNALS = {
    "Normal": [19, 42, 86, 113, 129, 278, 334, 348, 382, 403],
    "Wheeze": [192, 309, 859, 1109, 1339, 1729, 1964, 2233, 2601, 2654],
    "Rhonchi": [2, 4272, 4622, 4855, 5372, 7392, 7830, 8094, 8499, 9828],
    "Stridor": [4108, 9423, 9735, 14033, 14505, 17812, 23218, 15817],
}
# Artifacts: fragment id and time span (s); at most 1 per patient, patients not in the clean set
ARTIFACTS = [
    {"id": 22260, "t_ini": 1.15, "t_fin": 1.19},
    {"id": 22115, "t_ini": 1.01, "t_fin": 1.08},
    {"id": 5085,  "t_ini": 0.18, "t_fin": 0.24},
    {"id": 4999,  "t_ini": 1.53, "t_fin": 1.58},
    {"id": 16489, "t_ini": 0.54, "t_fin": 0.57},
    {"id": 2183,  "t_ini": 0.95, "t_fin": 1.00},
    {"id": 4303,  "t_ini": 0.51, "t_fin": 0.61},
    {"id": 15159, "t_ini": 1.27, "t_fin": 1.34},
    {"id": 13234, "t_ini": 0.30, "t_fin": 0.33},
    {"id": 984,   "t_ini": 0.96, "t_fin": 0.99},
    {"id": 22005, "t_ini": 0.38, "t_fin": 0.45},
    {"id": 924,   "t_ini": 0.74, "t_fin": 0.80},
    {"id": 16375, "t_ini": 1.48, "t_fin": 1.55},
    {"id": 7882,  "t_ini": 0.39, "t_fin": 0.44},
    {"id": 1673,  "t_ini": 0.11, "t_fin": 0.16},
    {"id": 22951, "t_ini": 0.15, "t_fin": 0.19},
    {"id": 21671, "t_ini": 1.65, "t_fin": 1.75},
    {"id": 20260, "t_ini": 1.60, "t_fin": 1.65},
    {"id": 5132,  "t_ini": 1.22, "t_fin": 1.38},
    {"id": 852,   "t_ini": 0.75, "t_fin": 0.86},
    {"id": 16814, "t_ini": 0.06, "t_fin": 0.11},
    {"id": 9135,  "t_ini": 1.05, "t_fin": 1.09},
    {"id": 17690, "t_ini": 1.24, "t_fin": 1.29},
    {"id": 23533, "t_ini": 1.35, "t_fin": 1.47},
    {"id": 2512,  "t_ini": 1.67, "t_fin": 1.74},
    {"id": 22895, "t_ini": 1.01, "t_fin": 1.08},
    {"id": 5963,  "t_ini": 0.06, "t_fin": 0.40},
    {"id": 9009,  "t_ini": 0.26, "t_fin": 0.32},
    {"id": 7577,  "t_ini": 0.50, "t_fin": 0.56},
    {"id": 14131, "t_ini": 0.36, "t_fin": 0.44},
    {"id": 23570, "t_ini": 0.80, "t_fin": 0.87},
    {"id": 20493, "t_ini": 1.04, "t_fin": 1.22},
    {"id": 15420, "t_ini": 0.10, "t_fin": 0.15},
    {"id": 7458,  "t_ini": 1.44, "t_fin": 1.48},
    {"id": 9793,  "t_ini": 0.79, "t_fin": 0.87},
    {"id": 22175, "t_ini": 1.33, "t_fin": 1.36},
    {"id": 22497, "t_ini": 0.66, "t_fin": 0.76},
    {"id": 14269, "t_ini": 0.24, "t_fin": 0.31},
    {"id": 14166, "t_ini": 0.76, "t_fin": 0.79},
    {"id": 22699, "t_ini": 1.00, "t_fin": 1.10},
]

# Common constants
EDGE_DIST = int(0.1 * FS)  # Minimum distance of the artifact (exp. 1) / layout (exp. 2) to the edges (samples)
SEED = 42                  # Random seed for reproducibility

# Experiment 1 (section 5.1 of evaluation_denoise.ipynb)
SNR_LIST_1 = [-30, -20, -10, -5, 0, 5, 10, 20]  # SNR list to test (locals, in dB)
N_REPS = 5                                       # Number of random positions per pair

# Experiment 2 (section 6.1 of evaluation_denoise.ipynb)
GAP_LIST = [0.025, 0.05, 0.1, 0.3, 0.5]        # Edge-to-edge gaps between artifacts (s)
SNR_LIST_2 = [-20, -10, -5, 0, 5, 10, 20]      # Equal condition: SNR of every artifact (local, dB)
SNR_STRONG = -20                               # Unequal condition: SNR of the strong artifact (dB)
SNR_WEAK_LIST = [-10, -5, 0]                   # Unequal condition: SNR of the weak artifacts (dB)
N_GROUPS = 20                                  # Artifact triplets per clean signal

# Output files (in src.evaluation.RESULTS_DIR = denoising/results)
CSV_FILES = {1: "experiment_1.csv", 2: "experiment_2.csv"}


def load_clean_signals() -> dict:
    """
    Loads the clean signals (section 4.1 of evaluation_denoise.ipynb).

    Returns
    -------
    clean_dict: dict
        clean_id -> {"signal", "id", "label"} (clean_id: 0, 1, ... in the order of CLEAN_SIGNALS)
    """
    clean_dict = {}
    count = 0
    for label, ids in CLEAN_SIGNALS.items():
        for id in ids:
            clean_dict[count] = {}
            clean_dict[count]["signal"], _ = sf.read(AUDIO_PATH / f"{id}.wav")
            clean_dict[count]["id"] = id
            clean_dict[count]["label"] = label

            count += 1

    return clean_dict


def load_artifacts() -> dict:
    """
    Cuts the artifacts (section 4.2 of evaluation_denoise.ipynb): span [t_ini - 5 ms, t_fin + 5 ms] of the fragment,
    mean removed and Tukey window with 5 ms ramps.

    Returns
    -------
    artifact_dict: dict
        art_id -> signal (art_id: 0, 1, ... in the order of ARTIFACTS)
    """
    artifact_dict = {}
    for i, artifact in enumerate(ARTIFACTS):
        full_signal, fs = sf.read(AUDIO_PATH / f"{artifact['id']}.wav")
        j_ini = int(round((artifact["t_ini"]-0.005) * fs))
        j_fin = int(round((artifact["t_fin"]+0.005) * fs))

        artifact_x = full_signal[j_ini:j_fin]
        artifact_pre = detrend(artifact_x, type="constant")

        alpha_tukey = 0.01*fs / len(artifact_pre)
        win = tukey(len(artifact_pre), alpha=alpha_tukey)
        artifact_pre = artifact_pre * win

        artifact_dict[i] = artifact_pre

    return artifact_dict


def get_mixes_1(clean_dict: dict, artifact_dict: dict) -> pd.DataFrame:
    """
    Mixtures of experiment 1 (get_mixes of section 5.1 of evaluation_denoise.ipynb): each clean signal x each
    artifact (that fits in it), at N_REPS random positions (at least EDGE_DIST from the borders) and at each local SNR
    of SNR_LIST_1. If the same position is drawn twice, the repeated mixture is dropped. Control: each clean signal
    without any artifact (art_id = snr_db = "Control").

    Parameters
    ----------
    clean_dict: dict
        Clean signals (see load_clean_signals)
    artifact_dict: dict
        Artifacts (see load_artifacts)

    Returns
    -------
    exp_df: pd.DataFrame
        One row per mixture (and per control)
    """
    random.seed(SEED)

    mixes_info = []
    count = 0
    for i, clean in clean_dict.items():
        clean_signal = clean["signal"]
        for j, artifact in artifact_dict.items():
            fit_margin = len(clean_signal) - len(artifact) - 2*EDGE_DIST
            if fit_margin > 0:
                for k in range(N_REPS):
                    k0 = EDGE_DIST + random.randint(0, fit_margin)
                    for snr in SNR_LIST_1:
                        _, info = insert_artifact(clean_signal, artifact, k0, snr)
                        mixes_info.append({
                            "id": count,
                            "clean_id": i,
                            "clean_file": clean["id"],
                            "label": clean["label"],
                            "art_id": j,
                            "rep": k,
                            "snr_db": snr,
                            **info
                        })
                        count += 1

    # Control: each clean signal without any artifact (no position, no mixture information)
    for i, clean in clean_dict.items():
        mixes_info.append({
            "id": count,
            "clean_id": i,
            "clean_file": clean["id"],
            "label": clean["label"],
            "art_id": "Control",
            "snr_db": "Control",
        })
        count += 1

    exp_df = pd.DataFrame(mixes_info)
    exp_df = exp_df.astype({"rep": "Int64", "k0": "Int64", "k1": "Int64"})  # NaN in the control rows
    exp_df = exp_df.drop_duplicates(["clean_id", "art_id", "k0", "snr_db"])  # Same position drawn twice -> same mixture

    return exp_df


def get_mixes_2(clean_dict: dict, artifact_dict: dict) -> pd.DataFrame:
    """
    Mixtures of experiment 2 (get_mixes of section 6.1 of evaluation_denoise.ipynb): for each clean signal, N_GROUPS
    random triplets of artifacts; for each triplet and gap of GAP_LIST, a layout (random order, same gap, random
    position at least EDGE_DIST from the borders; skipped if it does not fit). From each layout, the 7 paired subsets
    (singles, pairs, triplet) at the same positions, for each SNR condition: equal (SNR_LIST_2) and unequal (one
    strong artifact at SNR_STRONG, the others at each SNR of SNR_WEAK_LIST). Repeated mixtures are kept once. Control:
    each clean signal without any artifact (n_art = 0, art_ids = snr_db = "Control").

    Parameters
    ----------
    clean_dict: dict
        Clean signals (see load_clean_signals)
    artifact_dict: dict
        Artifacts (see load_artifacts)

    Returns
    -------
    exp_df: pd.DataFrame
        One row per mixture (and per control). Per-artifact columns are tuples
    """
    random.seed(SEED)

    art_ids = list(artifact_dict)
    triplets = list(combinations(art_ids, 3))           # all possible artifact triplets

    mixes_info = []
    count, group = 0, 0
    for i, clean in clean_dict.items():
        clean_signal = clean["signal"]

        for triplet in random.sample(triplets, N_GROUPS):
            for gap in GAP_LIST:
                gap_n = int(round(gap * FS))

                # Layout: random order, same gap between consecutive artifacts, random position
                order = random.sample(triplet, 3)
                lens = [len(artifact_dict[a]) for a in order]
                fit_margin = len(clean_signal) - sum(lens) - 2 * gap_n - 2 * EDGE_DIST
                if fit_margin <= 0:
                    continue
                start = EDGE_DIST + random.randint(0, fit_margin)
                k0s = [start,
                       start + lens[0] + gap_n,
                       start + lens[0] + lens[1] + 2 * gap_n]
                strong = random.randrange(3)            # strong artifact of the unequal condition

                # SNR conditions: (condition, reference SNR, SNR of each layout position)
                conds = ([("equal", s, [s] * 3) for s in SNR_LIST_2] +
                         [("unequal", w, [SNR_STRONG if p == strong else w for p in range(3)])
                          for w in SNR_WEAK_LIST])

                for cond, snr_ref, snrs_layout in conds:
                    # Paired subsets: singles, pairs and the triplet, at the same positions
                    for r in (1, 2, 3):
                        for subset in combinations(range(3), r):
                            arts = [order[p] for p in subset]
                            k0_sub = [k0s[p] for p in subset]
                            snr_sub = [snrs_layout[p] for p in subset]
                            _, infos = insert_artifacts(clean_signal, [artifact_dict[a] for a in arts],
                                                        k0_sub, snr_sub)
                            mixes_info.append({
                                "id": count,
                                "group_id": group,
                                "clean_id": i,
                                "clean_file": clean["id"],
                                "label": clean["label"],
                                "gap_s": gap,
                                "condition": cond,
                                "snr_db": snr_ref,                              # equal: SNR · unequal: weak SNR
                                "n_art": r,
                                "subset": "s" + "".join(map(str, subset)),      # layout positions present
                                "art_ids": tuple(arts),
                                "snrs_db": tuple(snr_sub),
                                "is_strong": tuple(cond == "unequal" and p == strong for p in subset),
                                # Per-artifact info from insert_artifact (k0, k1, ...), as tuples
                                **{key: tuple(info[key] for info in infos) for key in infos[0]},
                            })
                            count += 1
                group += 1

    # Control: each clean signal without any artifact
    for i, clean in clean_dict.items():
        mixes_info.append({
            "id": count,
            "clean_id": i,
            "clean_file": clean["id"],
            "label": clean["label"],
            "condition": "Control",
            "n_art": 0,
            "art_ids": "Control",
            "snr_db": "Control",
        })
        count += 1

    exp_df = pd.DataFrame(mixes_info)
    exp_df = exp_df.astype({"group_id": "Int64", "n_art": "Int64"})               # NaN in the control rows
    # Same artifacts at the same positions and SNRs -> same mixture (e.g. a single artifact of the
    # unequal condition is identical to the one of the equal condition)
    exp_df = exp_df.drop_duplicates(["clean_id", "art_ids", "k0", "snrs_db"])

    return exp_df


def main():
    print()
    parser = argparse.ArgumentParser(description="Run an evaluation experiment of the transient removal filter")
    parser.add_argument("-e", "--experiment", required=True, type=int, choices=[1, 2],
                        help="Experiment to run: 1 (single artifact) or 2 (several artifacts + idempotence)")
    parser.add_argument("-p", "--processes", type=int, default=None,
                        help="Number of worker processes (default: as many as CPU cores)")
    args = parser.parse_args()

    print("Loading signals...")
    clean_dict = load_clean_signals()
    artifact_dict = load_artifacts()
    print(f"{len(clean_dict)} clean signals, {len(artifact_dict)} artifacts\n")

    print(f"Generating the mixtures of experiment {args.experiment}...")
    if args.experiment == 1:
        exp_df = get_mixes_1(clean_dict, artifact_dict)
        worker = process_mixture
    else:
        exp_df = get_mixes_2(clean_dict, artifact_dict)
        worker = process_mixture_multi
    print(f"{len(exp_df)} mixtures (controls included)\n")

    csv_file = CSV_FILES[args.experiment]
    run_experiment(exp_df, csv_file, clean_dict, artifact_dict, processes=args.processes, worker=worker)
    print(f"\nResults saved in denoising/results/{csv_file}")


if __name__ == "__main__":  # Required by multiprocessing (spawn) on Windows
    main()
