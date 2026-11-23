"""
candidate_score.py

Metrics of the candidate generation (phase 1 of parameter_optimization.ipynb), as a module so that the scripts can use
them (used by screen_candidates.py). Copy of quality (section 2) and candidate_score (section 4.1) of the notebook: if
one of them is changed in the notebook, it must also be changed here.
"""

import numpy as np
import pandas as pd


F_DB = -30  # dB. Per-artifact quality: q = 0 at SNR_out <= F_DB ...
C_DB = 0    # dB. ... and q = 1 at SNR_out >= C_DB
SNR_MAX_CANDIDATES = 0  # dB. Overflow and unweighted recall: artifacts with SNR <= SNR_MAX_CANDIDATES


def quality(snr_out_db, f=F_DB, c=C_DB):
    """
    Per-artifact quality: q = (clip(SNR_out, F, C) - F) / (C - F).

    Parameters
    ----------
    snr_out_db: array-like
        SNR in the artifact zone after the filter (dB)
    f, c: float
        Floor and ceiling of the SNR (dB): q = 0 at SNR_out <= f, q = 1 at SNR_out >= c

    Returns
    -------
    q: array-like
        Quality of each artifact, in [0, 1]
    """
    return (np.clip(snr_out_db, f, c) - f) / (c - f)


def candidate_score(cand_df, snr_max=SNR_MAX_CANDIDATES, cost=None):
    """
    Metrics of the candidate generation (see 4.1) and their diagnostics, from the per-mixture results of the
    candidates. The recall at each SNR is averaged as Q (see 2.2): mean over the positions of each cell (s, a, c) ->
    the clean signals -> the artifacts. The candidate recall (the metric that is optimized) is the mean over all the
    SNR of the recall at each SNR, weighted by the cost of missing an artifact at that SNR.

    Parameters
    ----------
    cand_df: pd.DataFrame
        Per-mixture results of the candidates (mixtures + controls, art_id = snr_db = "Control"), already restricted
        to the set to evaluate. Columns:
        - clean_id, label, art_id, snr_db: identification of the mixture
        - n_candidates: number of candidates of the signal (used in the controls)
        - n_candidates_out: candidates that do not touch the zone Z_i of the artifact (mixtures)
        - has_candidate: some candidate touches Z_i (mixtures)
        - zone_ms: |Z_i| (ms) (mixtures)
        - overflow_ms: time of the candidates that touch Z_i outside Z_i (ms; 0 without candidates) (mixtures)
    snr_max: float
        Only the mixtures with SNR <= snr_max (dB) are used for the overflow, the unweighted recall, the weighted
        recall (w_i) and the load in the mixtures. The candidate recall and the per-SNR table use all the SNR
    cost: pd.Series or None
        Cost of missing an artifact at each SNR (index: SNR in dB), e.g. the measured dq of score (see 3.2).
        None -> ideal elimination: 1 - q(s) (= -s/30 between -30 and 0 dB, 0 above 0 dB). Negative costs -> 0

    Returns
    -------
    metrics: dict
        - recall: candidate recall, weighted by the cost of missing an artifact at each SNR (all the SNR)
        - load: mean number of candidates per control signal
        - w_med: median of w_i over the artifacts that have some candidate
    diagnostics: dict
        - snr_max: the argument snr_max
        - recall_unweighted: proportion of artifacts with SNR <= snr_max that have some candidate
        - recall_w: weighted candidate recall (w_i = 0 without candidates)
        - overflow_ms_med: median overflow (ms) over the artifacts that have some candidate
        - load_mix: candidates out of Z_i per mixture (averaged as FP, see 2.3)
        - per_snr: pd.DataFrame, one row per SNR (all the SNR): number of mixtures, cost of missing an artifact,
          recall, weighted recall, median w_i and median overflow (ms) of the artifacts with some candidate, and mean
          candidates out of Z_i per mixture
        - load_label: pd.DataFrame, per label of the clean signal: candidates per control, number of controls and
          candidates out of Z_i per mixture (as load_mix)
        - recall_art: pd.DataFrame, candidate recall of each artifact (recall at each SNR = mean over the clean
          signals): cost-weighted over all the SNR (recall) and unweighted over SNR <= snr_max (recall_unweighted)
    """
    df = cand_df.assign(snr_db=pd.to_numeric(cand_df["snr_db"], errors="coerce"))  # "Control" -> NaN
    is_control = df["art_id"].astype(str) == "Control"
    controls = df[is_control]
    mix_all = df[~is_control]
    has_candidate = mix_all["has_candidate"].astype(bool)
    w = mix_all["zone_ms"] / (mix_all["zone_ms"] + mix_all["overflow_ms"])         # w_i = |Z_i| / (|Z_i| + overflow_i)
    mix_all = mix_all.assign(has_candidate=has_candidate, w=w.where(has_candidate, 0.0))  # w_i = 0 without candidates
    mix = mix_all[mix_all["snr_db"] <= snr_max]                                     # Mixtures with SNR <= snr_max
    with_cand_all = mix_all[mix_all["has_candidate"]]
    with_cand = mix[mix["has_candidate"]]

    cells = ["snr_db", "art_id", "clean_id"]

    def per_art_snr_mean(column):
        """Mean of column per artifact and SNR: positions -> clean signals (rows: artifacts, columns: SNR)."""
        return (mix_all.groupby(cells)[column].mean()
                .groupby(level=["art_id", "snr_db"]).mean()
                .unstack("snr_db"))

    recall_art_snr = per_art_snr_mean("has_candidate")
    recall_snr = recall_art_snr.mean()                                    # Recall_s: mean over the artifacts
    recall_w_snr = per_art_snr_mean("w").mean()
    levels = recall_snr.index
    in_snr_max = levels <= snr_max

    # Cost of missing an artifact at each SNR: c_s (ideal elimination: 1 - q(s))
    if cost is None:
        cost = pd.Series(1 - quality(levels.to_numpy()), index=levels)
    cost = pd.Series(cost).reindex(levels).fillna(0).clip(lower=0)

    def cost_weighted(recall):
        """Mean over the SNR of the recall at each SNR (Series indexed by SNR), weighted by c_s."""
        return (cost * recall).sum() / cost.sum()

    # ============================================================================ metrics
    metrics = dict(
        recall=cost_weighted(recall_snr),
        load=controls["n_candidates"].mean(),
        w_med=with_cand["w"].median(),
    )

    # ============================================================================ diagnostics
    load_clean = (mix.groupby(["clean_id", "snr_db", "art_id"])["n_candidates_out"].mean()   # mean over the positions
                  .groupby(level=["clean_id", "snr_db"]).mean()                              # mean over the artifacts
                  .groupby(level="clean_id").mean())                                         # mean over the SNR

    per_snr = pd.DataFrame({
        "n_mixtures": mix_all.groupby("snr_db").size(),
        "cost": cost,
        "recall": recall_snr,
        "recall_w": recall_w_snr,
        "w_med": with_cand_all.groupby("snr_db")["w"].median(),
        "overflow_ms_med": with_cand_all.groupby("snr_db")["overflow_ms"].median(),
        "n_candidates_out": mix_all.groupby("snr_db")["n_candidates_out"].mean(),
    })
    per_snr.index.name = "snr_db"

    label = df.groupby("clean_id")["label"].first()
    load_label = pd.DataFrame({
        "load": controls.groupby("label")["n_candidates"].mean(),
        "n_controls": controls.groupby("label").size(),
        "load_mix": load_clean.groupby(label).mean(),
    })
    load_label.index.name = "label"

    recall_art = pd.DataFrame({
        "recall": recall_art_snr.apply(cost_weighted, axis=1),
        "recall_unweighted": recall_art_snr.loc[:, in_snr_max].mean(axis=1),
    })

    diagnostics = dict(
        snr_max=snr_max,
        recall_unweighted=recall_snr[in_snr_max].mean(),
        recall_w=recall_w_snr[in_snr_max].mean(),
        overflow_ms_med=with_cand["overflow_ms"].median(),
        load_mix=load_clean.mean(),
        per_snr=per_snr,
        load_label=load_label,
        recall_art=recall_art,
    )

    return metrics, diagnostics
