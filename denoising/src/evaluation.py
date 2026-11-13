"""
evaluation.py

Experiment functions of evaluation_denoise.ipynb: insertion of artifacts into clean signals, metrics of the
transient removal (remove_noise) and parallel execution of the experiment (multiprocessing, one worker process per
CPU core by default). They are in a module (and not in the notebook) because multiprocessing needs importable
functions on Windows (spawn).

Experiment 1: one artifact per mixture       -> process_mixture / metrics
Experiment 2: one to three artifacts per mixture + idempotence (second pass)
                                             -> process_mixture_multi / metrics_multi
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import scipy.ndimage as ndi
from scipy.ndimage import uniform_filter1d
from tqdm.auto import tqdm

from . import remove_noise as rn


FS = 4000                     # Sampling frequency (Hz)
SNR_MARGIN = int(0.025 * FS)  # Ampliation of the window to compute local SNR (samples)
TOL = int(0.025 * FS)         # Tolerance around the artifact (delays between bands), in samples
ATEN_DB = 3                   # Minimum ΔSNR for considering the artifact "attenuated"
RMS_WIN = int(0.03 * FS)      # 30 ms: Duration of window for RMS calculation
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"  # denoising/results (independent of the cwd)
CHUNKSIZE = 1                 # Mixtures sent to a worker at a time

# Data of the worker processes (see init_worker)
_clean_dict = {}
_artifact_dict = {}


def insert_artifact(clean, artifact, k0, snr):
    N = len(clean)
    L = len(artifact)

    # Local relative power calculation (clean)
    a = k0 - SNR_MARGIN
    b = k0 + L + SNR_MARGIN

    p_local = np.mean(clean[a:b]**2)
    p_total = np.mean(clean**2)

    # Artifact SNR calculation
    p_art = np.mean(artifact**2)
    g = np.sqrt(p_local / (p_art * 10**(snr/10)))  # Gain to obtain local SNR

    # Mixed signal
    mixed = clean.copy()
    mixed[k0:k0+L] += g * artifact

    # Mixture information
    info = {
        "k0": k0,
        "k1": k0 + L,
        "t0_s": k0 / FS,
        "art_duration_ms": L / FS * 1000,
        "g": g,
        "p_local_rel_db": 10 * np.log10(p_local / p_total),
        "snr_global_db": 10 * np.log10(p_total * N / (g**2 * p_art * L))
    }

    return mixed, info


def insert_artifacts(clean_signal, artifacts, k0s, snrs):
    """
    Insert several artifacts in the same clean signal.

    Each artifact is scaled with respect to the CLEAN signal (same local SNR definition as in
    experiment 1), so the scaling of one artifact does not depend on the others. The mixture is
    the clean signal plus the contribution of every artifact.

    Returns the mixture and the list of `insert_artifact` infos (one per artifact, same order).
    """
    clean_signal = np.asarray(clean_signal, dtype=float)
    mix = clean_signal.copy()
    infos = []
    for art, k0, snr in zip(artifacts, k0s, snrs):
        mix_i, info = insert_artifact(clean_signal, art, k0, snr)
        mix += mix_i - clean_signal                     # contribution of this artifact alone
        infos.append(info)
    return mix, infos


def hp15(x):
    """Signal > 15 Hz: SWT reconstruction with the detail coefficients only (as remove_noise)"""
    x = x - x.mean()
    N = len(x)
    n_levels = round(np.log2(FS / rn.F_APPROX)) - 1
    n_tot = int(np.ceil(N / 2**n_levels)) * 2**n_levels
    n_pad = n_tot - N
    padded = np.concatenate((x, x[N - n_pad:][::-1]))
    coeffs = pywt.swt(padded, rn.WAVELET, level=n_levels)
    detail = np.array([cD for _, cD in coeffs[::-1]])
    return rn.iswt_details(detail)[:N]


# ============================================================================ experiment 1 (one artifact)
def metrics(ref, mix, out, k0, k1, click_span):
    # k0 = k1 = None -> control (no artifact): the whole signal is "out of the artifact zone" and the metrics of the
    # artifact zone are NaN
    N = len(ref)
    control = k0 is None
    art = mix - ref
    err = out - ref

    # Artifact zone
    zone = np.zeros(N, bool)
    if not control:
        L = k1 - k0
        a = max(0, k0 - TOL)
        b = min(N, k1 + TOL)
        zone[a:b] = True

    # Out of artifact zone
    out_zone = ~zone

    # Power calculations
    p_ref_out_zone = np.mean(ref[out_zone]**2)
    p_err_out_zone = np.mean(err[out_zone]**2)
    if not control:
        p_ref_in_zone = np.mean(ref[zone]**2)
        p_art = np.sum(art[zone]**2) / L
        p_err_in_zone = np.sum(err[zone]**2) / L

    # A. Detection
    lab, n = ndi.label(click_span)
    clicks_in_zone = set(np.unique(lab[zone])) - {0}
    detected = len(clicks_in_zone) > 0 if not control else np.nan  # 1 if there are detected clicks in W; 0 if not
    n_false = n - len(clicks_in_zone)   # Number of false positive clicks detected (control: all the clicks)

    # B. Elimination
    if not control:
        snr_in = 10*np.log10(p_ref_in_zone / p_art)           # Amount of artifact before processing
        snr_out = 10*np.log10(p_ref_in_zone / p_err_in_zone)  # Amount of artifact after processing
        dsnr = snr_out - snr_in                               # dB of improve after processing
        attenuated = dsnr >= ATEN_DB                          # 1 if the attenuation is sufficient (3 dB); 0 if not
    else:
        snr_in = snr_out = dsnr = attenuated = np.nan

    # C. Conservation
    nmse_out_zone = 10*np.log10(p_err_out_zone / p_ref_out_zone)  # How the signal out of artifact has changed

    # D. Amplitude (moving RMS of RMS_WIN samples)
    rms_ref = np.sqrt(uniform_filter1d(ref**2, RMS_WIN))
    rms_mix = np.sqrt(uniform_filter1d(mix**2, RMS_WIN))
    rms_out = np.sqrt(uniform_filter1d(out**2, RMS_WIN))

    # Local (artifact zone; NaN in the control)
    if control:
        err_rms_max_loc_mix = err_rms_max_loc_out = err_rms_mean_loc_mix = err_rms_mean_loc_out = np.nan
    else:
        err_rms_max_loc_mix = (rms_mix[zone].max() - rms_ref[zone].max()) / rms_ref[zone].max() * 100       # % error of the RMS peak in the artifact zone, before processing (how much the artifact inflates the local peak)
        err_rms_max_loc_out = (rms_out[zone].max() - rms_ref[zone].max()) / rms_ref[zone].max() * 100       # % error of the RMS peak in the artifact zone, after processing (ideal ~0; <0 = over-attenuation)
        err_rms_mean_loc_mix = (rms_mix[zone].mean() - rms_ref[zone].mean()) / rms_ref[zone].mean() * 100   # % error of the mean RMS in the artifact zone, before processing
        err_rms_mean_loc_out = (rms_out[zone].mean() - rms_ref[zone].mean()) / rms_ref[zone].mean() * 100   # % error of the mean RMS in the artifact zone, after processing (ideal ~0)

    # Global (whole fragment)
    err_rms_max_glob_mix = (rms_mix.max() - rms_ref.max()) / rms_ref.max() * 100       # % error of the RMS peak of the whole fragment, before processing (~0 if the artifact is weaker than the signal peaks)
    err_rms_max_glob_out = (rms_out.max() - rms_ref.max()) / rms_ref.max() * 100       # % error of the RMS peak of the whole fragment, after processing (also reveals changes to the signal's own peaks)
    err_rms_mean_glob_mix = (rms_mix.mean() - rms_ref.mean()) / rms_ref.mean() * 100   # % error of the mean RMS of the whole fragment, before processing (effect on a global amplitude parameter)
    err_rms_mean_glob_out = (rms_out.mean() - rms_ref.mean()) / rms_ref.mean() * 100   # % error of the mean RMS of the whole fragment, after processing (<0 = signal attenuated somewhere)

    return {
        "detected": detected,
        "n_false": n_false,
        "snr_in_db": snr_in,
        "snr_out_db": snr_out,
        "dsnr_db": dsnr,
        "attenuated": attenuated,
        "nmse_out_zone_db": nmse_out_zone,
        "err_rms_max_loc_mix": err_rms_max_loc_mix,
        "err_rms_max_loc_out": err_rms_max_loc_out,
        "err_rms_mean_loc_mix": err_rms_mean_loc_mix,
        "err_rms_mean_loc_out": err_rms_mean_loc_out,
        "err_rms_max_glob_mix": err_rms_max_glob_mix,
        "err_rms_max_glob_out": err_rms_max_glob_out,
        "err_rms_mean_glob_mix": err_rms_mean_glob_mix,
        "err_rms_mean_glob_out": err_rms_mean_glob_out,
    }


# ============================================================================ experiment 2 (several artifacts)
def artifact_zones(N, k0s, k1s):
    """
    Zone of each artifact: [k0 - TOL, k1 + TOL), as in experiment 1, but cut at the midpoint between two
    consecutive artifacts when their zones would overlap (small gaps), so every sample belongs to one zone at most.
    Returns a list of boolean masks, in the order of k0s.
    """
    order = np.argsort(k0s)
    for p, n in zip(order[:-1], order[1:]):             # the design never overlaps two artifacts
        assert k1s[p] <= k0s[n], f"Overlapping artifacts: [{k0s[p]}, {k1s[p]}) and [{k0s[n]}, {k1s[n]})"
    zones = [None] * len(k0s)
    for pos, i in enumerate(order):
        a = max(0, k0s[i] - TOL)
        b = min(N, k1s[i] + TOL)
        if pos > 0:                                     # previous artifact: cut at the midpoint of the gap
            prev = order[pos - 1]
            a = max(a, (k1s[prev] + k0s[i]) // 2)
        if pos < len(order) - 1:                        # next artifact: cut at the midpoint of the gap
            nxt = order[pos + 1]
            b = min(b, (k1s[i] + k0s[nxt]) // 2)
        zone = np.zeros(N, bool)
        zone[a:b] = True
        zones[i] = zone
    return zones


def metrics_multi(ref, mix, out, arts, k0s, k1s, click_span):
    """
    Metrics of a mixture with several artifacts (k0s = [] -> control).

    Per-artifact metrics (tuples, in the order of k0s): detection, merging, elimination (ΔSNR, attenuated) and local
    amplitude, each one in its own zone. arts[i] is the contribution of artifact i alone to the mixture (> 15 Hz),
    so its power does not include the other artifacts.
    Per-mixture metrics: number of detected artifacts, false positives, conservation out of all the zones and global
    amplitude.
    """
    N = len(ref)
    n_art = len(k0s)
    control = n_art == 0
    err = out - ref

    zones = artifact_zones(N, k0s, k1s)
    union = np.logical_or.reduce(zones) if not control else np.zeros(N, bool)

    rms_ref = np.sqrt(uniform_filter1d(ref**2, RMS_WIN))
    rms_mix = np.sqrt(uniform_filter1d(mix**2, RMS_WIN))
    rms_out = np.sqrt(uniform_filter1d(out**2, RMS_WIN))

    # A. Detection
    lab, n = ndi.label(click_span)
    clicks = [set(np.unique(lab[z])) - {0} for z in zones]          # click spans touching each zone
    detected = [len(c) > 0 for c in clicks]
    # Merged: a click span of this artifact also touches the zone of another artifact
    merged = [any(c & clicks[j] for j in range(n_art) if j != i) for i, c in enumerate(clicks)]
    clicks_in_zones = set().union(*clicks) if not control else set()
    n_false = n - len(clicks_in_zones)          # Click spans that touch no artifact zone (control: all the clicks)

    # B. Elimination and local amplitude, per artifact
    per_art = {key: [] for key in ("snr_in_db", "snr_out_db", "dsnr_db", "attenuated",
                                   "err_rms_max_loc_mix", "err_rms_max_loc_out",
                                   "err_rms_mean_loc_mix", "err_rms_mean_loc_out")}
    for art, z, k0, k1 in zip(arts, zones, k0s, k1s):
        L = k1 - k0
        p_ref_in_zone = np.mean(ref[z]**2)
        p_art = np.sum(art[z]**2) / L
        p_err_in_zone = np.sum(err[z]**2) / L
        snr_in = 10*np.log10(p_ref_in_zone / p_art)
        snr_out = 10*np.log10(p_ref_in_zone / p_err_in_zone)
        per_art["snr_in_db"].append(float(snr_in))
        per_art["snr_out_db"].append(float(snr_out))
        per_art["dsnr_db"].append(float(snr_out - snr_in))
        per_art["attenuated"].append(bool(snr_out - snr_in >= ATEN_DB))
        per_art["err_rms_max_loc_mix"].append(float((rms_mix[z].max() - rms_ref[z].max()) / rms_ref[z].max() * 100))
        per_art["err_rms_max_loc_out"].append(float((rms_out[z].max() - rms_ref[z].max()) / rms_ref[z].max() * 100))
        per_art["err_rms_mean_loc_mix"].append(float((rms_mix[z].mean() - rms_ref[z].mean()) / rms_ref[z].mean() * 100))
        per_art["err_rms_mean_loc_out"].append(float((rms_out[z].mean() - rms_ref[z].mean()) / rms_ref[z].mean() * 100))

    # C. Conservation (out of all the artifact zones)
    out_zone = ~union
    nmse_out_zone = 10*np.log10(np.mean(err[out_zone]**2) / np.mean(ref[out_zone]**2))

    # Signal between consecutive artifacts (from the end of one to the start of the next, in order of position):
    # how much a merged span changes the signal between the clicks. One value per pair of neighbours
    order = np.argsort(k0s)
    nmse_between = []
    for p, q in zip(order[:-1], order[1:]):
        gap = slice(k1s[p], k0s[q])
        nmse_between.append(float(10*np.log10(np.mean(err[gap]**2) / np.mean(ref[gap]**2))))

    return {
        # Per artifact (tuples)
        "detected": tuple(detected),
        "merged": tuple(merged),
        **{key: tuple(values) for key, values in per_art.items()},
        # Per mixture
        "n_detected": sum(detected) if not control else np.nan,
        "all_detected": all(detected) if not control else np.nan,
        "n_false": n_false,
        "nmse_out_zone_db": nmse_out_zone,
        "nmse_between_db": tuple(nmse_between),      # n_art - 1 values, neighbours in order of position
        "err_rms_max_glob_mix": (rms_mix.max() - rms_ref.max()) / rms_ref.max() * 100,
        "err_rms_max_glob_out": (rms_out.max() - rms_ref.max()) / rms_ref.max() * 100,
        "err_rms_mean_glob_mix": (rms_mix.mean() - rms_ref.mean()) / rms_ref.mean() * 100,
        "err_rms_mean_glob_out": (rms_out.mean() - rms_ref.mean()) / rms_ref.mean() * 100,
        # Internal: union of the zones (used for the idempotence metrics; removed before saving)
        "_union": union,
    }


# ============================================================================ parallel execution
def constants(module) -> dict:
    """Constants (UPPERCASE names) of a module"""
    return {name: value for name, value in vars(module).items() if name.isupper()}


def init_worker(clean_dict: dict, artifact_dict: dict, params: dict) -> None:
    """
    Initializer of each worker process: stores the signals and sets the constants of remove_noise and of this
    module to the values of the main process (a worker imports the modules again, with their default values).
    """
    global _clean_dict, _artifact_dict
    _clean_dict, _artifact_dict = clean_dict, artifact_dict
    for module, values in ((rn, params["remove_noise"]), (sys.modules[__name__], params["evaluation"])):
        for name, value in values.items():
            setattr(module, name, value)


def process_mixture(row: dict) -> dict:
    """
    Experiment 1. Rebuilds a mixture (row of exp_df; art_id = "Control" -> clean signal without artifact),
    processes it with remove_noise and returns its information + its metrics.
    """
    clean = _clean_dict[row["clean_id"]]["signal"]
    if row["art_id"] == "Control":  # No artifact
        mixed, k0, k1 = clean.copy(), None, None
    else:
        art = _artifact_dict[row["art_id"]]
        k0 = int(row["k0"])
        k1 = k0 + len(art)
        mixed, _ = insert_artifact(clean, art, k0, row["snr_db"])  # Rebuild the mixture

    out, det = rn.remove_noise(mixed, FS, return_details=True)  # Apply the algorithm
    m = metrics(hp15(clean), hp15(mixed), out, k0, k1, det["click_span"])  # References > 15 Hz

    return {**row, **m}


def process_mixture_multi(row: dict) -> dict:
    """
    Experiment 2. Rebuilds a mixture with one to three artifacts (row of exp_df; art_ids = "Control" -> clean
    signal without artifacts), processes it with remove_noise TWICE (idempotence) and returns its information + its
    metrics.
    """
    clean = np.asarray(_clean_dict[row["clean_id"]]["signal"], dtype=float)
    ref = hp15(clean)                                               # Reference > 15 Hz

    if isinstance(row["art_ids"], str) and row["art_ids"] == "Control":   # No artifacts
        mixed, k0s, k1s, arts = clean.copy(), [], [], []
    else:
        artifacts = [_artifact_dict[a] for a in row["art_ids"]]
        k0s = [int(k) for k in row["k0"]]
        k1s = [k0 + len(art) for k0, art in zip(k0s, artifacts)]
        # Rebuild the mixture and keep the contribution of each artifact alone (> 15 Hz; hp15 is linear)
        mixed, arts = clean.copy(), []
        for art, k0, snr in zip(artifacts, k0s, row["snrs_db"]):
            mixed_i, _ = insert_artifact(clean, art, k0, snr)
            mixed += mixed_i - clean
            arts.append(hp15(mixed_i) - ref)

    # First pass
    out, det = rn.remove_noise(mixed, FS, return_details=True)
    m = metrics_multi(ref, hp15(mixed), out, arts, k0s, k1s, det["click_span"])
    union = m.pop("_union")

    # Global SNR of the whole mixture (all the artifacts together; raw signals, as in insert_artifact)
    m["snr_global_mix_db"] = (10*np.log10(np.mean(clean**2) / np.mean((mixed - clean)**2)) if k0s else np.nan)

    # Second pass (idempotence): the output is processed again
    out2, det2 = rn.remove_noise(out, FS, return_details=True)
    lab2, n2 = ndi.label(det2["click_span"])
    m["n_clicks_pass2"] = n2                                                  # Ideal: 0
    m["n_clicks_pass2_in_zones"] = len(set(np.unique(lab2[union])) - {0})    # Of them, in the artifact zones
    m["idem_nmse_db"] = 10*np.log10(np.mean((out2 - out)**2) / np.mean(out**2))   # Change of the 2nd pass (ideal: -inf)

    return {**row, **m}


def _json_default(o):
    """numpy scalars -> Python scalars (for json.dumps)"""
    return o.item() if hasattr(o, "item") else str(o)


def _tuples_to_json(df: pd.DataFrame) -> pd.DataFrame:
    """Columns with tuples/lists -> JSON strings, so they survive the .csv (see read_results)"""
    df = df.copy()
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (tuple, list))).any():
            df[col] = df[col].map(lambda v: json.dumps(list(v), default=_json_default)
                                  if isinstance(v, (tuple, list)) else v)
    return df


def read_results(csv_file: str) -> pd.DataFrame:
    """Reads a results .csv of RESULTS_DIR, converting the JSON lists back to tuples"""
    df = pd.read_csv(Path(RESULTS_DIR) / csv_file, low_memory=False)   # whole file at once: one type per column
    for col in df.select_dtypes(include=["object", "string"]).columns:    # text columns (pandas 3: dtype str)
        is_list = df[col].map(lambda v: isinstance(v, str) and v.startswith("["))
        if is_list.any():
            df[col] = df[col].map(lambda v: tuple(json.loads(v)) if isinstance(v, str) and v.startswith("[") else v)
    return df


def run_experiment(exp_df: pd.DataFrame, csv_file: str, clean_dict: dict, artifact_dict: dict,
                   processes: int | None = None, worker=process_mixture) -> pd.DataFrame:
    """
    Runs the experiment in parallel: each mixture of exp_df (and each control) is rebuilt, processed with
    remove_noise and evaluated. The results (one row per mixture: its information from exp_df + its metrics, in
    the order of exp_df) are saved in RESULTS_DIR/csv_file. Columns with tuples (experiment 2) are saved as JSON
    lists; read them back with read_results.

    Parameters
    ----------
    exp_df: pd.DataFrame
        Mixtures to evaluate (see get_mixes). Control rows: clean signal without artifacts
    csv_file: str
        Name of the .csv file of the results (in RESULTS_DIR)
    clean_dict: dict
        Clean signals: clean_id -> {"signal", "id", "label"}
    artifact_dict: dict
        Artifacts: art_id -> signal
    processes: int or None
        Number of worker processes (None -> as many as CPU cores)
    worker: callable
        process_mixture (experiment 1) or process_mixture_multi (experiment 2)

    Returns
    -------
    results: pd.DataFrame
        exp_df with the metrics of each mixture
    """
    rows = [f.to_dict() for _, f in exp_df.iterrows()]
    params = {"remove_noise": constants(rn), "evaluation": constants(sys.modules[__name__])}
    with Pool(processes, initializer=init_worker, initargs=(clean_dict, artifact_dict, params)) as pool:
        results = list(tqdm(pool.imap(worker, rows, chunksize=CHUNKSIZE), total=len(rows), desc="Mixtures"))

    results = pd.DataFrame(results)
    path = Path(RESULTS_DIR) / csv_file
    path.parent.mkdir(parents=True, exist_ok=True)
    _tuples_to_json(results).to_csv(path, index=False)

    return results
