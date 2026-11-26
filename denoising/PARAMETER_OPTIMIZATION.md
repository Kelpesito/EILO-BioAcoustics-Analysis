# 🎯 Denoising — Parameter optimization of the click removal algorithm

This document describes the **parameter optimization plan** of the transient (click) removal filter of [🧹 Denoising](README.md). Its goal is to replace the current thresholds, **selected by eye** (see [⚙️ Parameters](README.md#️-parameters)), with a phased optimization over mixtures of clean signals and artifacts, like those of [evaluation_denoise.ipynb](README.md#evaluation_denoiseipynb--evaluation-of-the-algorithm).

> 🚧 **In progress:** see [📋 Status](#-status).

---

## Table of Contents

- [📐 Score definition](#-score-definition)
    - [Per-artifact quality](#per-artifact-quality)
    - [Global quality](#global-quality)
    - [Number of false positives](#number-of-false-positives)
    - [Score](#score)
    - [Diagnostics](#diagnostics)
- [🔁 Parameter optimization](#-parameter-optimization)
    - [0️⃣ Phase 0 — Preparation](#0️⃣-phase-0--preparation)
    - [1️⃣ Phase 1 — Candidate generation](#1️⃣-phase-1--candidate-generation)
    - [2️⃣ Phase 2 — Feature definition](#2️⃣-phase-2--feature-definition)
    - [3️⃣ Phase 3 — Rule selection](#3️⃣-phase-3--rule-selection)
    - [4️⃣ Phase 4 — Attenuation](#4️⃣-phase-4--attenuation)
    - [5️⃣ Phase 5 — Validation](#5️⃣-phase-5--validation)
- [📋 Status](#-status)

---

## 📐 Score definition

### Per-artifact quality

$$
q_i = \frac{\text{clip}(\text{SNR}_i^{\text{out}}, F, C) - F}{C-F}
$$

where $F=-30$ dB and $C=0$ dB.

### Global quality

$$
Q = \frac{1}{|\mathcal{S}|}\sum_{s\in\mathcal{S}} Q_s, \qquad
Q_s = \frac{1}{|\mathcal{A}| |\mathcal{C}|}\sum_{a\in\mathcal{A}}\sum_{c\in\mathcal{C}} \bar{q}_{s,a,c}
$$

where:

- $Q_s$ is the quality at SNR $s$.
- $\mathcal{S}$ is the set of SNR values.
- $\mathcal{A}$ is the set of artifacts.
- $\mathcal{C}$ is the set of clean signals (every artifact is inserted into every clean signal).
- $\bar{q}_{s,a,c}$ is the mean quality of artifact $a$, for clean signal $c$, at SNR $s$ (mean over the different positions of the artifact):

$$
\bar{q}_{s,a,c} = \frac{1}{N_{a,c}}\sum_{i=1}^{N_{a,c}} q_{s,a,c}^{(i)}
$$

where $N_{a,c}$ is the number of mixtures in which artifact $a$ is inserted into clean signal $c$.

> Each artifact and each clean signal weighs the same, whatever its number of positions $N_{a,c}$.

### Number of false positives

$$
FP = \frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}}\bar{n}_c^{\text{false}}
$$

where $\bar{n}_c^{\text{false}}$ is the mean number of false positives of the mixtures of clean signal $c$, averaged as $Q$ (first over the positions):

$$
\bar{n}_c^{\text{false}} = \frac{1}{|\mathcal{S}| |\mathcal{A}|}\sum_{s\in\mathcal{S}}\sum_{a\in\mathcal{A}} \bar{n}_{s,a,c}^{\text{false}}, \qquad
\bar{n}_{s,a,c}^{\text{false}} = \frac{1}{N_{a,c}}\sum_{i=1}^{N_{a,c}} n_{s,a,c}^{\text{false},(i)}
$$

Each artifact weighs the same, whatever its number of positions $N_{a,c}$.

### Score

$$
S = Q - \lambda\cdot FP
$$

where $\lambda$ is a parameter that penalizes the score according to the detected false positives (see [0.2](#02-penalty-λ)):

| $\lambda$ | One false positive is worth… |
|:---------:|:-----------------------------|
| 1    | Missing a −30 dB artifact  |
| 0.5  | Missing a −15 dB artifact  |
| 0.25 | Missing a −7.5 dB artifact |

### Diagnostics

- Detection per SNR.
- SNR at 50 % detection.
- Conservation.
- False positives per label.
- Local RMS error of the detected artifacts.

---

## 🔁 Parameter optimization

The optimization is split into 6 phases. Each phase fixes its parameters before moving on to the next one:

| Phase | What it optimizes | Parameters | Metric | Output |
|:-----:|:------------------|:-----------|:-------|:-------|
| **0** | Preparation | — | Score, Q, FP | Tuning/test split, $\lambda$, baseline |
| **1** | Candidate generation | `WAVELET`, `K`, `ENV_SMOOTH`, `MASK_DILATION`, `REF_INNER`, `REF_OUTER` (`MAX_EVENT` fixed) | Candidate recall, load, overflow | C1 |
| **2** | Feature definition | `BG_FACTOR`, `WIDTH_FLOOR_DB`, `EXTEND`, `ISO_WIN`, `Z_WIN`, `AR_ORDER` | Standardized partial ROC-AUC | C2, candidate table and scores |
| **3** | Rule selection | Rule structure and thresholds | $\hat{S}$ (estimated score) | Final rules and thresholds |
| **4** | Attenuation | `G_FAC`, `GAIN_SMOOTH`, `LOCAL_BG` | Score | Optimized version |
| **5** | Validation | — | S, Q, FP | Comparison and generalization |

---

### 0️⃣ Phase 0 — Preparation

#### 0.1. Tuning/test split

75 % tuning / 25 % test (first of 4 folds as test, [get_splits.py](get_splits.py)), grouped by signal and artifact:

- **Signals:** stratified by label.
- **Artifacts:** stratified by duration.

| Set | Tuning | Test | Total |
|:----|:------:|:----:|:-----:|
| **Signals**   | 29 | 9 | 38 |
| **Artifacts** | 23 | 9 | 32 |

Signal × artifact combinations of each set:

| Set | Signals | Artifacts | Combinations |
|:----|:-------:|:---------:|:------------:|
| **Tuning**        | 29 (tuning) | 23 (tuning) | 667 |
| **Test**          | 9 (test)   | 9 (test)   | 81 |
| **New signals**   | 9 (test)   | 23 (tuning) | 207 |
| **New artifacts** | 29 (tuning) | 9 (test)   | 261 |

#### 0.2. Penalty $\lambda$

It is set from an external criterion: **a missed artifact is acceptable if it does not mask the signal, and when in doubt the signal is kept**. So one false positive is worth missing an artifact right at the masking limit $s^*$.

**Masking SNR $s^*$:** the SNR at which an artifact starts to distort the input of the classifier.

- **Masking:** the artifact raises the global RMS peak (moving RMS, 30 ms) by more than 10 % (`err_rms_max_glob_mix` > 10 %).
- $s^*$ = SNR at which 50 % of the mixtures are masked (linear interpolation between the SNR of `SNR_LIST`).

**Cost of missing an artifact at $s^*$**, computed in two ways:

1. **Approximation (ideal elimination):** if the detection left $q = 1$, missing an artifact at SNR $s$ would cost $1 - q(s) = -s/30$ (with $F=-30$ dB and $C=0$ dB), so:

$$
\lambda \approx -\frac{s^*}{30}
$$

2. **Measured:** the detection does not leave $q = 1$ ($q^{\text{det}} < 1$), so the real cost of missing an artifact at SNR $s$ is:

$$
\Delta q_s = \bar{q}^{\text{det}}_s - \bar{q}^{\text{undet}}_s, \qquad \lambda = \Delta q_{s^*}
$$

where:

- $\bar{q}^{\text{det}}_s$ and $\bar{q}^{\text{undet}}_s$ are the mean $q$ of the detected and the undetected artifacts at SNR $s$, averaged as in $Q$ (weighted by the detection of each cell, see [Decomposition of the quality](#decomposition-of-the-quality) in 3.2).
- $\Delta q_{s^*}$ is interpolated linearly between the SNR of `SNR_LIST`.

If $\bar{q}^{\text{det}}_{s^*}$ is close to 1, both values of $\lambda$ are almost equal and the approximation can be kept.

Its effect on the rule selection is studied in [5.1](#51-sensitivity-to-λ).

#### 0.3. Baseline

Compute the Score, Q and FP on the tuning set, with the $\lambda$ of [0.2](#02-penalty-λ).

---

### 1️⃣ Phase 1 — Candidate generation

- **Goal:** almost every artifact that matters should generate some candidate, without an excessive load on the clean signals and without overflowing the artifact. If an artifact does not become a candidate, no rule will be able to detect it later; if its span overflows the artifact, it merges with neighbouring sounds, which degrades its features ([Phase 2](#2️⃣-phase-2--feature-definition)).
- **Optimization of:** `WAVELET`, `K`, `ENV_SMOOTH`, `MASK_DILATION`, `REF_INNER`, `REF_OUTER`. `MAX_EVENT` is screened, but fixed at 140 ms (see the procedure).
- **Metrics:** missing a candidate and overflowing the artifact have very different costs (a missed artifact is lost for good; a wider span only degrades its features, which the rules of [Phase 3](#3️⃣-phase-3--rule-selection) can partly compensate), so they are measured apart and with priority:
    1. **Candidate recall:** recall at each SNR weighted by the cost of missing an artifact at that SNR, over every SNR.
    2. **Load:** candidates per control signal.
    3. **Overflow:** median of $w_i$ over the artifacts with SNR ≤ 0 dB that have some candidate.

**Candidate recall:**

$$
\text{Recall} = \frac{\sum\limits_{s\in\mathcal{S}} c_s\,\text{Recall}_s}{\sum\limits_{s\in\mathcal{S}} c_s}, \qquad c_s = 1 - q(s)
$$

where:

- $\text{Recall}_s$ is the proportion of artifacts with SNR $s$ that have some candidate in their zone $Z_i$ (artifact ± 25 ms).
- $c_s$ is the cost of missing an artifact at SNR $s$, with ideal elimination ([Penalty λ](#02-penalty-λ)): $-s/30$ between $-30$ and $0$ dB, $0$ above. The measured $\Delta q_s$ can be used instead.

> Missing an artifact at SNR ≥ 0 dB costs nothing, and there most of the candidates in $Z_i$ are background candidates that touch it by chance (recall ≈ 0.54 at 20 dB in the baseline).

**Overflow:**

$$
w_i = \frac{|Z_i|}{|Z_i \cup \text{spans}_i|} = \frac{|Z_i|}{|Z_i| + \text{overflow}_i}
$$

where:

- $\text{spans}_i$ is the union of the candidate spans that touch $Z_i$.
- $\text{overflow}_i = |\text{spans}_i \setminus Z_i|$ is the time of those spans outside the zone.

> $w_i = 1$ if the spans do not leave the zone, $0.5$ if the overflow equals the length of the zone. A span shorter than the zone is not penalized. An isolated click already overflows the zone a little (time resolution of the low bands, `MASK_DILATION`), so $w_i < 1$ even in the best case; this bias is similar for every configuration.

- **Diagnostics:**
    - Median overflow (ms).
    - Weighted candidate recall ($\text{Recall}_w = \frac{1}{N}\sum_i w_i$ over the $N$ artifacts with SNR ≤ 0 dB, with $w_i = 0$ if there is no candidate).
    - Unweighted candidate recall: proportion of artifacts with SNR ≤ 0 dB that have some candidate in $Z_i$.
    - Candidate recall at every SNR, $\text{Recall}_s$, and of each artifact (cost-weighted and unweighted).
    - **Load in the mixtures:** candidates that do not touch $Z_i$ per mixture, averaged as $FP$ ([Number of false positives](#number-of-false-positives)). The false positives of [Phase 3](#3️⃣-phase-3--rule-selection) come from these candidates; a difference with the load shows that the artifact changes the candidates of the rest of the signal.

> `K` moves recall and overflow in opposite directions (a low `K` gives more candidates, but wider spans), and `MASK_DILATION` moves load and overflow in opposite directions (less widening gives more candidates, but narrower ones; the recall barely changes): a single combined number would hide these conflicts.

**Procedure:**

1. **Screening:** each parameter is varied alone. If the metrics do not change, it is fixed.
    - **`MAX_EVENT` is fixed at 140 ms**, although it changes the metrics: it is the limit of the protection of the adventitious sounds, which phase 1 does not measure, and it caps the width of the candidates (a long detection is removed only in its band, while the span of a candidate is the union over the bands), so a search would use it to lower the overflow. It is swept later with the Score.
2. **Search:** two Bayesian optimizations, one after the other, both subject to `REF_INNER` < `REF_OUTER` and `REF_OUTER` − `REF_INNER` ≥ `MAX_EVENT`:
    - **(a) Maximum recall:** maximize the candidate recall (cost-weighted) → recall<sub>max</sub> and its standard error $SE$ (bootstrap grouped by signal and by artifact).
    - **(b) Minimum load:** minimize the load subject to recall ≥ recall<sub>max</sub> − $SE$ and median $w_i$ ≥ median $w_i$ of the baseline.
    - **Safeguard:** if (b) finds no configuration that meets both constraints, recall<sub>max</sub> and $SE$ are recomputed from the configurations of (a) with median $w_i$ ≥ median $w_i$ of the baseline, and (b) is run again.
    - The recall, load and overflow of every evaluated configuration (of every search) are saved.
    - **Nested cross-validation:** the tuning clean signals and artifacts are split into 5 folds (grouped and stratified as the tuning/test split); in fold *k*, the searches and the choice use the signals and artifacts of the other folds, and the C1 of the fold is evaluated (with the baseline) on the signals and artifacts of fold *k* → 5 C1 (stability of the parameters) + 5 validation metrics (optimism). The final C1 comes from the same protocol on the whole tuning set.
3. **Choice:** among all the evaluated configurations (a, b and, with the safeguard, b2) that meet the constraints of (b), the one with the minimum load → **C1**. The load–overflow Pareto front of those configurations is used to check it (e.g. whether a little more load would give much less overflow).

> **Tolerance $SE$:** the recall of the baseline has $SE$ = 0.0041 (2000 resamples of signals and artifacts), so differences of recall smaller than $SE$ cannot be told apart from the sampling of signals and artifacts (1-SE rule, as in the [backward pruning](#34-backward-pruning)). In terms of the Score, giving up 0.004 of recall costs at most $0.004 \cdot \sum_s c_s / |\mathcal{S}| \approx 0.001$ of $Q$.
>
> **Overflow only in (b):** recall<sub>max</sub> is the maximum recall, whatever the overflow and the load, so the recall keeps its priority. If that maximum is only reached with wide spans, no configuration with little overflow comes close to it and (b) has no solution, hence the safeguard.
>
> **Range and constraints:** the search ranges are those of each parameter alone; the constraints apply to the whole configuration. A value that alone does not meet a constraint (e.g. `ENV_SMOOTH` = 2 ms: median $w_i$ = 0.849) can meet it together with other values, so it stays in the range.

**`REF_OUTER` − `REF_INNER` ≥ `MAX_EVENT`:** the reference is the median of two windows of `REF_OUTER` − `REF_INNER`, one at each side, so in an event longer than `REF_OUTER` the reference rises at the edges and only the center is detected (2·`REF_OUTER` − L for a flat event of length L < `REF_INNER` + `REF_OUTER`; longer events are not detected at all). With windows shorter than `MAX_EVENT`, that center can be shorter than `MAX_EVENT`: the event loses the protection of long events and becomes a short candidate.

**Outputs:**

- Screening plots.
- History of the searches of each run: recall, load and overflow of each configuration in the order of evaluation, with the best feasible configuration so far and the best recall of (a).
- Recall–load and load–overflow Pareto fronts, with the constraints of (b) and the chosen point.
- Nested cross-validation: C1 − baseline on the training and validation mixtures of each fold, optimism, and mean ± SD over the folds; stability of the parameters of C1 across the folds.
- C1.

**C1** (whole tuning set; search (b2), trial 289):

| Parameter | Baseline | C1 |
|:----------|:--------:|:--:|
| `WAVELET`       | db4    | db8      |
| `K`             | 5      | 5.50     |
| `ENV_SMOOTH`    | 4 ms   | 15.8 ms  |
| `MASK_DILATION` | 15 ms  | 8.8 ms   |
| `REF_INNER`     | 100 ms | 36.1 ms  |
| `REF_OUTER`     | 400 ms | 233.1 ms |
| `MAX_EVENT`     | 140 ms | 140 ms (fixed) |

| Metric | Baseline | C1 | Validation: C1 − baseline (mean ± SD over the 5 folds) |
|:-------|:--------:|:--:|:------------------------------------------------------:|
| Candidate recall                | 0.986   | 0.993   | +0.010 ± 0.011 |
| Load (candidates per control)   | 7.4     | 3.3     | −1.8 ± 2.7     |
| Median $w_i$                    | 0.860   | 0.876   | +0.005 ± 0.025 |
| Unweighted recall (SNR ≤ 0 dB)  | 0.943   | 0.962   | +0.025 ± 0.017 |
| Median overflow                 | 15.3 ms | 13.3 ms | −0.6 ± 2.8 ms  |

---

### 2️⃣ Phase 2 — Feature definition

- **Goal:** measure each feature so that it separates the artifacts from the rest as well as possible. The candidates are those of C1 and do not change.
- **Optimization of:**

| Feature | Parameters |
|:--------|:-----------|
| **Width** (`width_med`, `width_max`) | `BG_FACTOR`, `WIDTH_FLOOR_DB`, `EXTEND` |
| **Isolation** (`isolation`) | `ISO_WIN` |
| **Impulsiveness** (`z_max`) | `Z_WIN`, `AR_ORDER` |

- **Metric:** standardized partial ROC-AUC with a false positive rate ≤ 0.1, over all the candidates of the tuning mixtures, between:
    - **Artifacts (positives):** each artifact $i$ with some candidate in its zone $Z_i$ (artifact ± 25 ms). Its value is the most artifact-like among those candidates (the narrowest width, the highest isolation or $z_{\max}$): with a threshold, the artifact is detected if any of its candidates is.
    - **Rest (negatives):** the candidates of the mixtures that do not touch $Z_i$ (those that give the false positives of $FP$).

**Weights** (averaged as $Q$ and $FP$):

$$
v_i = \frac{c_s}{N_{a,c}} \ \text{(artifact)}, \qquad v_j = \frac{1}{N_{a,c}} \ \text{(rest)}, \qquad c_s = 1 - q(s)
$$

where $c_s$ is the cost of missing an artifact at SNR $s$, as in the candidate recall ([Phase 1](#1️⃣-phase-1--candidate-generation)): the artifacts with SNR ≥ 0 dB weigh 0.

**Weighted ROC curve:** for a threshold $t$ on the feature $x$ (with its sign: $-x$ for the widths):

$$
\text{TPR}(t) = \frac{\sum\limits_{i\in\mathcal{P}} v_i\,\mathbb{1}[x_i \geq t]}{\sum\limits_{i\in\mathcal{P}} v_i}, \qquad
\text{FPR}(t) = \frac{\sum\limits_{j\in\mathcal{N}} v_j\,\mathbb{1}[x_j \geq t]}{\sum\limits_{j\in\mathcal{N}} v_j}
$$

where $\mathcal{P}$ are the artifacts and $\mathcal{N}$ the rest.

**Standardized partial AUC** (McClish, 1989), with $\text{FPR}_{\max} = 0.1$:

$$
\text{pAUC} = \int_0^{\text{FPR}_{\max}} \text{TPR}\; d\,\text{FPR}, \qquad
\text{sAUC} = \frac{1}{2}\left(1 + \frac{\text{pAUC} - \text{FPR}_{\max}^2/2}{\text{FPR}_{\max} - \text{FPR}_{\max}^2/2}\right)
$$

sAUC = 0.5: no better than chance; sAUC = 1: perfect separation at FPR ≤ 0.1. **Metric of each group:** mean sAUC of its features.

**Baselines:** the default parameters of the features, with the candidates of the baseline and with those of C1.

**Procedure:**

1. **Search:** for each group and combination of parameters, compute the features.
2. **Choice:** pick the combination with the best mean partial AUC → **C2'**. C1 together with C2' → **C2**.
3. **Full run:** run of the filter on the tuning set with C2.

**Outputs:**

- AUC vs value (for each parameter).
- C2.
- Candidate table: features of every candidate.
- Scores: $q$, detection and false positives of every mixture, needed for $\hat{S}$ in [Phase 3](#3️⃣-phase-3--rule-selection).

---

### 3️⃣ Phase 3 — Rule selection

- **Goal:** define the rule matrix: 4 cases × 8 features (see [6️⃣ Click rules](README.md#6️⃣-click-rules)).

#### 3.0. Descriptive analysis

- Distributions of the features per class.
- Pairwise scatter plots per class.
- Correlation matrix.

#### 3.1. Univariate analysis

For each case:

**1. Filter with the main condition of the case:** among the candidates, keep only those that meet the main condition of each case. For example, for the impulsive case, keep only the candidates with `z_max` ≥ `Z_CLICK`.

**2. Minimum condition:** there must be enough candidates of both classes to compute the AUC reliably. The standard error of the AUC is:

$$
SE = \sqrt{\frac{n_1 + n_0 + 1}{12n_1n_0}}
$$

So, to distinguish the AUC by 0.1 ($|AUC - 0.5|\geq 0.1$) at 99.7 %, it must hold that:

$$
3\cdot SE \leq 0.1 \Longrightarrow SE \leq \frac{0.1}{3} \approx 0.033 \approx 0.03
$$

- The samples used to compute the AUC are not independent, so the standard error is computed with a **bootstrap grouped by signal and by artifact**.
- If $SE < 0.03$ is not met, relax the filter to cover one more percentile of the data, and repeat.

> **Adaptive margin:** relaxing the filter adds candidates, but not different signals or artifacts, so the SE of the grouped bootstrap has a floor that depends on how many signals and artifacts there are. First of all, compute the SE with all the candidates (no filter): if it is already $> 0.03$, the criterion can never be met.
>
> If $SE < 0.03$ is still not met at the relaxation limit (the most permissive value of the search range):
> - $|AUC - 0.5| \geq \max(0.1,\ 3 \cdot SE)$ is required: with less information, stronger evidence is required, and a useless feature still has a ~0.3 % probability of passing.
> - New conditions that do not pass the margin are not added; those already in v4 are kept (the pruning will decide whether they are redundant).

**3. Prefilter:** for each of the other 7 features, compute the AUC. Depending on the AUC:

| AUC | Interpretation | Condition |
|:----|:---------------|:---------:|
| $\geq 0.5$ | Artifacts have higher values | ≥ |
| $\leq 0.5$ | Artifacts have lower values | ≤ |
| $\|AUC - 0.5\| \leq 0.1$ | The distributions overlap and the feature does not help in this case | Not added |

#### 3.2. Provisional optimization

- **Goal:** estimate the Score of a rule set from its detections on the candidate table, without running the filter. Same mean as $Q$ ([Global quality](#global-quality)), with the quality of each cell $k = (s, a, c)$ split into its detected and undetected parts:

$$
\hat{S} = \frac{1}{|\mathcal{S}| |\mathcal{A}| |\mathcal{C}|}\sum_{s\in\mathcal{S}}\sum_{a\in\mathcal{A}}\sum_{c\in\mathcal{C}}\left[\widehat{\text{Det}}_{s,a,c}\cdot \hat{q}^{\text{det}}_{s,a,c} + \left(1-\widehat{\text{Det}}_{s,a,c}\right)\cdot \hat{q}^{\text{undet}}_{s,a,c}\right] - \lambda \cdot \widehat{FP}
$$

where:

- $\widehat{\text{Det}}_{s,a,c}$ is the fraction of the $N_{a,c}$ positions of the cell detected by the rules.
- $\hat{q}^{\text{det}}_{s,a,c}$ and $\hat{q}^{\text{undet}}_{s,a,c}$ are the quality of the detected and the undetected positions of the cell used by $\hat{S}$: the values $\bar{q}_k^{\text{det}}$ and $\bar{q}_k^{\text{undet}}$ measured in the full run of [Phase 2](#2️⃣-phase-2--feature-definition) ([Decomposition of the quality](#decomposition-of-the-quality)), or imputed when missing ([Imputation](#imputation)). They do not depend on the rules: only $\widehat{\text{Det}}_{s,a,c}$ and $\widehat{FP}$ change.
- $\widehat{FP}$ is the number of false positives with the rules, averaged as $FP$.

##### Decomposition of the quality

$\hat{S}$ only uses the cell level; the aggregated means are used for $\lambda$ ([0.2](#02-penalty-λ)) and the [imputation](#imputation).

**Cell:** if $p_k$ is the fraction of the positions of cell $k$ detected in the full run, and $\bar{q}_k^{\text{det}}$ and $\bar{q}_k^{\text{undet}}$ are the mean $q$ of its detected and undetected positions:

$$
\bar{q}_k = p_k\cdot \bar{q}_k^{\text{det}} + (1-p_k)\cdot \bar{q}_k^{\text{undet}}
$$

The term in brackets of $\hat{S}$ is this decomposition with $p_k$ replaced by $\widehat{\text{Det}}_{s,a,c}$, keeping $\bar{q}_k^{\text{det}}$ and $\bar{q}_k^{\text{undet}}$ from the full run. With the detections of the full run, $\hat{S} = S$ exactly. With other rules, $\hat{S}$ is an approximation: it assumes that only the fraction of detected positions changes, but the real $q$ of the cell can also change (e.g. the positions that become detected may be harder to attenuate than those already detected). The real $S$ is computed in [Phase 4](#4️⃣-phase-4--attenuation).

**Aggregated means** (per artifact $\bar{q}_{s,a}$, per SNR $\bar{q}_s$): weighted mean of the detected ($\mathcal{D}$) or undetected ($\mathcal{U}$) mixtures, each one with its weight in $Q$:

$$
w_i = \frac{1}{N_{a,c}}, \qquad
\bar{q}^{\text{det}} = \frac{\sum\limits_{i\in\mathcal{D}} w_i\, q_i}{\sum\limits_{i\in\mathcal{D}} w_i}, \qquad
\bar{q}^{\text{undet}} = \frac{\sum\limits_{i\in\mathcal{U}} w_i\, q_i}{\sum\limits_{i\in\mathcal{U}} w_i}
$$

With these weights, the decomposition also holds per SNR, with $\text{Det}_s$ the mean over $a, c$ of $p_k$:

$$
Q_s = \text{Det}_s\cdot \bar{q}^{\text{det}}_s + \left(1-\text{Det}_s\right)\cdot \bar{q}^{\text{undet}}_s
$$

##### Imputation

A value is missing when an artifact is never detected (or never missed) in a cell in the full run, but the new rules do detect (or miss) it. Each value used by $\hat{S}$ is taken from the closest level with data:

- **$\hat{q}^{\text{undet}}_{s,a,c}$:** $\bar{q}_k^{\text{undet}}$; if missing, the filter does not touch the zone of the artifact, so $\text{SNR}^{\text{out}} \approx s$ and $\hat{q}^{\text{undet}}_{s,a,c} \approx q(s)$.
- **$\hat{q}^{\text{det}}_{s,a,c}$:** $\bar{q}_k^{\text{det}}$; if missing, $\bar{q}^{\text{det}}_{s,a}$ (same artifact in the other clean signals); if the artifact is not detected in any clean signal at that SNR, $\bar{q}^{\text{det}}_s$.

> If the full run gives $q^{\text{det}}$ of every mixture (forcing the attenuation of its candidate), no imputation is needed.

**Procedure:**

1. **Search:** Bayesian optimization over the conditions that passed the prefilter, maximizing the estimated score.
2. **Cross-validation:** 5-fold cross-validation → 5 sets of thresholds + 5 values of $\hat{S}$.

#### 3.3. Importance of each condition

| Measure | Definition |
|:--------|:-----------|
| **Unique rejections** | Number of candidates that meet every condition except this one and that this condition rejects |
| **Ablation** | $\Delta\hat{S}$ when it is disabled |
| **Stability** | Variability of the threshold across folds |
| **Unique detections per case** | Number of artifacts detected by that case and by no other |

#### 3.4. Backward pruning

1. **Pruning:** remove the condition with the highest $\Delta\hat{S}$.
2. **Optimization:** re-optimize the remaining thresholds.
3. **Repeat** until reaching a rule with a single condition.
4. **1-SE rule:** choose the simplest rule set whose $\hat{S}$ is within 1 SE of the best.
5. **Final thresholds:** optimize the thresholds of the remaining conditions on the whole tuning table.

**Outputs:**

- AUC matrix (case × feature).
- Importance table (initial).
- Pruning curve.
- Importance table (final).
- Final rule structure.
- Final thresholds.

---

### 4️⃣ Phase 4 — Attenuation

- **Optimization of:** `G_FAC`, `GAIN_SMOOTH`, `LOCAL_BG`.
- **Metric:** Score.

**Output:** the optimized version.

---

### 5️⃣ Phase 5 — Validation

#### 5.1. Sensitivity to $\lambda$

- Study the rule selection as a function of the value of $\lambda$.

#### 5.2. Comparison

- S (Q, FP) for different values of $\lambda$.
- Wilcoxon signed-rank test, 95 % bootstrap CI of ΔS and number of signals that improve.

#### 5.3. Generalization

- The optimism is the ΔS of the tuning set minus the ΔS of the test sets (test, new signals and new artifacts).

#### 5.4. Evaluation experiments

- Generate the figures of experiment 1 and experiment 2 (see [evaluation_denoise.ipynb](README.md#evaluation_denoiseipynb--evaluation-of-the-algorithm)).

---

## 📋 Status

- [x] **Phase 0** — Preparation
- [x] **Phase 1** — Candidate generation
- [ ] **Phase 2** — Feature definition
- [ ] **Phase 3** — Rule selection
- [ ] **Phase 4** — Attenuation
- [ ] **Phase 5** — Validation
