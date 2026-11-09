"""
evaluation.py

Experiment functions of evaluation_denoise.ipynb: insertion of artifacts into clean signals, metrics of the
transient removal (remove_noise) and parallel execution of the experiment (multiprocessing, one worker process per
CPU core by default). They are in a module (and not in the notebook) because multiprocessing needs importable
functions on Windows (spawn).
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import scipy.ndimage as ndi
from scipy.ndimage import uniform_filter1d
from tqdm.auto import tqdm

import remove_noise as rn


FS = 4000                     # Sampling frequency (Hz)
SNR_MARGIN = int(0.025 * FS)  # Ampliation of the window to compute local SNR (samples)
TOL = int(0.025 * FS)         # Tolerance around the artifact (delays between bands), in samples
ATEN_DB = 3                   # Minimum ΔSNR for considering the artifact "attenuated"
RMS_WIN = int(0.03 * FS)      # 30 ms: Duration of window for RMS calculation
RESULTS_DIR = "results"       # Folder of the results (.csv)
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
    Rebuilds a mixture (row of exp_df; art_id = "Control" -> clean signal without artifact), processes it with
    remove_noise and returns its information + its metrics.
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


def run_experiment(exp_df: pd.DataFrame, csv_file: str, clean_dict: dict, artifact_dict: dict,
                   processes: int | None = None) -> pd.DataFrame:
    """
    Runs the experiment in parallel: each mixture of exp_df (and each control) is rebuilt, processed with
    remove_noise and evaluated with metrics. The results (one row per mixture: its information from exp_df + its
    metrics, in the order of exp_df) are saved in RESULTS_DIR/csv_file.

    Parameters
    ----------
    exp_df: pd.DataFrame
        Mixtures to evaluate (see get_mixes). Rows with art_id = "Control": clean signal without artifact
    csv_file: str
        Name of the .csv file of the results (in RESULTS_DIR)
    clean_dict: dict
        Clean signals: clean_id -> {"signal", "id", "label"}
    artifact_dict: dict
        Artifacts: art_id -> signal
    processes: int or None
        Number of worker processes (None -> as many as CPU cores)

    Returns
    -------
    results: pd.DataFrame
        exp_df with the metrics of each mixture
    """
    rows = [f.to_dict() for _, f in exp_df.iterrows()]
    params = {"remove_noise": constants(rn), "evaluation": constants(sys.modules[__name__])}
    with Pool(processes, initializer=init_worker, initargs=(clean_dict, artifact_dict, params)) as pool:
        results = list(tqdm(pool.imap(process_mixture, rows, chunksize=CHUNKSIZE), total=len(rows),
                            desc="Mixtures"))

    results = pd.DataFrame(results)
    path = Path(RESULTS_DIR) / csv_file
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)

    return results
