"""
feature_score.py

Metric of the feature definition (phase 2 of parameter_optimization.ipynb), as a module so that the scripts can use it
(used by search_features.py). Copy of the constants and of feature_samples, partial_auc, feature_auc and feature_metric
of section 5.1 of the notebook: if one of them is changed in the notebook, it must also be changed here.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .candidate_score import quality


MAX_FPR = 0.1  # Partial ROC-AUC of the selectors: false positive rate <= MAX_FPR
TPR_MIN = 0.9  # Partial ROC-AUC of the vetoes: true positive rate >= TPR_MIN
# Sign of each feature: +1 -> the artifacts have higher values, -1 -> lower values (the widths)
FEATURE_SIGN = {"width_med": -1, "width_max": -1, "isolation": 1, "z_max": 1}
# Region of the ROC curve of each feature: "fpr" -> FPR <= MAX_FPR (selectors, conditions >=), "tpr" -> TPR >= TPR_MIN
# (vetoes, conditions <=)
FEATURE_REGION = {"width_med": "tpr", "width_max": "tpr", "isolation": "fpr", "z_max": "fpr"}
# Features of each group (parameters of 5.2-5.4)
FEATURE_GROUPS = {"Width": ["width_med", "width_max"], "Isolation": ["isolation"], "Impulsiveness": ["z_max"]}


def feature_samples(feat_df, feature, cost=None):
    """
    Samples of the weighted ROC curve of a feature (see 5.1), from the candidates of the mixtures (controls excluded):
    - Artifacts: one per mixture with some candidate in its zone Z_i, with the most artifact-like value of those
      candidates (maximum of sign * feature) and weight c_s / N_{a,c}
    - Rest: the candidates of the mixtures that do not touch Z_i, with weight 1 / N_{a,c}
    The samples with weight 0 (artifacts with SNR >= 0 dB) are dropped.

    Parameters
    ----------
    feat_df: pd.DataFrame
        Output of load_features
    feature: str
        Name of the feature (key of FEATURE_SIGN)
    cost: pd.Series or None
        Cost of missing an artifact at each SNR (index: SNR in dB). None -> ideal elimination: 1 - q(s)

    Returns
    -------
    y: np.ndarray
        1 -> artifact, 0 -> rest
    x: np.ndarray
        sign * feature (higher -> more artifact-like)
    v: np.ndarray
        Weight of each sample
    """
    df = feat_df[feat_df["art_id"].astype(str) != "Control"]
    snr = pd.to_numeric(df["snr_db"])
    c = 1 - quality(snr) if cost is None else snr.map(pd.Series(cost)).fillna(0).clip(lower=0)
    x = FEATURE_SIGN[feature] * df[feature].astype(float)
    touches = df["touches_zone"].astype(str) == "True"

    artifacts = (pd.DataFrame({"id": df["id"], "x": x, "v": c / df["n_positions"]})[touches]
                 .groupby("id").agg(x=("x", "max"), v=("v", "first")))
    rest = pd.DataFrame({"x": x, "v": 1 / df["n_positions"]})[~touches]

    y = np.r_[np.ones(len(artifacts)), np.zeros(len(rest))]
    x = np.r_[artifacts["x"].to_numpy(), rest["x"].to_numpy()]
    v = np.r_[artifacts["v"].to_numpy(), rest["v"].to_numpy()]
    keep = v > 0
    return y[keep], x[keep], v[keep]


def partial_auc(y, x, v, region, max_fpr=MAX_FPR, min_tpr=TPR_MIN):
    """
    Standardized partial ROC-AUC (McClish) in a region of the ROC curve (see 5.1):
    - "fpr": FPR <= max_fpr
    - "tpr": TPR >= min_tpr. With the classes swapped and the sign of x changed, FPR' = 1 - TPR and TPR' = 1 - FPR, so
      this region is FPR' <= 1 - min_tpr
    """
    if region == "fpr":
        return roc_auc_score(y, x, sample_weight=v, max_fpr=max_fpr)
    return roc_auc_score(1 - y, -x, sample_weight=v, max_fpr=1 - min_tpr)


def feature_auc(feat_df, feature, max_fpr=MAX_FPR, min_tpr=TPR_MIN, cost=None):
    """
    Standardized partial ROC-AUC of a feature in its region (sauc, see partial_auc and FEATURE_REGION), its full ROC-AUC
    (auc) and the number of artifacts and of rest candidates.
    """
    y, x, v = feature_samples(feat_df, feature, cost)
    region = FEATURE_REGION[feature]
    return dict(region=f"FPR <= {max_fpr:g}" if region == "fpr" else f"TPR >= {min_tpr:g}",
                sauc=partial_auc(y, x, v, region, max_fpr, min_tpr), auc=roc_auc_score(y, x, sample_weight=v),
                n_artifacts=int(y.sum()), n_rest=int(len(y) - y.sum()))


def feature_metric(feat_df, group, max_fpr=MAX_FPR, min_tpr=TPR_MIN, cost=None):
    """
    Metric of a group of features (see 5.1).

    Returns
    -------
    table: pd.DataFrame
        One row per feature of the group: region, sauc, auc, n_artifacts, n_rest
    metric: float
        Mean sauc over the features of the group
    """
    table = pd.DataFrame({f: feature_auc(feat_df, f, max_fpr, min_tpr, cost) for f in FEATURE_GROUPS[group]}).T
    table = table.astype({"sauc": float, "auc": float, "n_artifacts": int, "n_rest": int})
    table.index.name = "feature"
    return table, table["sauc"].mean()


def score_features(feat_df: pd.DataFrame, group: str, max_fpr: float = MAX_FPR, min_tpr: float = TPR_MIN) -> dict:
    """
    Metrics of a group of features (used by the search of phase 2): metric (mean sauc over the features of the group,
    the one that is optimized) and, for each feature, sauc_<feature> and auc_<feature> (see feature_metric)
    """
    table, metric = feature_metric(feat_df, group, max_fpr, min_tpr)
    return {"metric": float(metric),
            **{f"sauc_{f}": float(table.loc[f, "sauc"]) for f in table.index},
            **{f"auc_{f}": float(table.loc[f, "auc"]) for f in table.index}}
