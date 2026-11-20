# 🎯 Denoising — Parameter optimization of the click removal algorithm

This document describes the **parameter optimization plan** of the transient (click) removal filter of [🧹 Denoising](README.md). Its goal is to replace the current thresholds, **selected by eye** (see [⚙️ Parameters](README.md#️-parameters)), with a phased optimization over mixtures of clean signals and artifacts, like those of [evaluation_denoise.ipynb](README.md#evaluation_denoiseipynb--evaluation-of-the-algorithm).

> 🚧 **Plan:** the phases described here are not implemented yet (see [📋 Status](#-status)).

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
Q = \frac{1}{|\mathcal{S}| |\mathcal{A}| |\mathcal{C}|}\sum_{s\in\mathcal{S}}\sum_{a\in\mathcal{A}}\sum_{c\in\mathcal{C}} \bar{q}_{s,a,c}
$$

where:

- $\mathcal{S}$ is the set of SNR values.
- $\mathcal{A}$ is the set of artifacts.
- $\mathcal{C}$ is the set of clean signals.
- $\bar{q}_{s,a,c}$ is the mean quality of artifact $a$, for clean signal $c$, at SNR $s$ (mean over the different positions of the artifact):

$$
\bar{q}_{s,a,c} = \frac{1}{N_{a,c}}\sum_{i=1}^{N_{a,c}} q_{s,a,c}^{(i)}
$$

where $N_{a,c}$ is the number of mixtures in which artifact $a$ is inserted into clean signal $c$.

### Number of false positives

$$
FP = \frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}}\bar{n}_c^{\text{false}}
$$

where $\bar{n}_c^{\text{false}}$ is the mean number of false positives of clean signal $c$:

$$
\bar{n}_c^{\text{false}} = \frac{1}{N_c}\sum_{i=1}^{N_c}n_i^{\text{false}}
$$

where $N_c$ is the number of mixtures whose clean signal is $c$.

### Score

$$
S = Q - \lambda\cdot FP
$$

where $\lambda$ is a parameter that penalizes the score according to the detected false positives (value chosen in [0.3](#03-penalty-λ)):

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
| **0** | Preparation | — | Score, Q, FP | Tuning/test split, baseline, $\lambda$ |
| **1** | Candidate generation | `WAVELET`, `K`, `ENV_SMOOTH`, `MASK_DILATION`, `REF_INNER`, `REF_OUTER`, `MAX_EVENT` | Candidate recall, load, overflow | C1 |
| **2** | Feature definition | `BG_FACTOR`, `WIDTH_FLOOR_DB`, `EXTEND`, `ISO_WIN`, `Z_WIN`, `AR_ORDER` | Standardized partial ROC-AUC | Candidate table |
| **3** | Rule selection | Rule structure and thresholds | $\hat{S}$ (estimated score) | Final rules and thresholds |
| **4** | Attenuation | `G_FAC`, `GAIN_SMOOTH`, `LOCAL_BG` | Score | Optimized version |
| **5** | Validation | — | S, Q, FP | Comparison and generalization |

---

### 0️⃣ Phase 0 — Preparation

#### 0.1. Tuning/test split

70 % tuning / 30 % test, grouped by signal and artifact, stratified by label:

| Set | Tuning | Test | Total |
|:----|:------:|:----:|:-----:|
| **Signals**   | 27 | 11 | 38 |
| **Artifacts** | 22 | 10 | 32 |

Signal × artifact combinations of each set:

| Set | Signals | Artifacts | Combinations |
|:----|:-------:|:---------:|:------------:|
| **Tuning**        | 27 (tuning) | 22 (tuning) | 594 |
| **Test**          | 11 (test)   | 10 (test)   | 110 |
| **New signals**   | 11 (test)   | 22 (tuning) | 242 |
| **New artifacts** | 27 (tuning) | 10 (test)   | 270 |

#### 0.2. Baseline

Compute the Score, Q and FP on the tuning set.

#### 0.3. Penalty $\lambda$

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
\Delta q_s = \hat{q}^{\text{det}}_s - \hat{q}^{\text{undet}}_s, \qquad \lambda = \Delta q_{s^*}
$$

where $\hat{q}^{\text{det}}_s$ and $\hat{q}^{\text{undet}}_s$ are the mean $q$ of the detected and the undetected artifacts at SNR $s$, and $\Delta q_{s^*}$ is interpolated linearly between the SNR of `SNR_LIST`. If $\hat{q}^{\text{det}}_{s^*}$ is close to 1, both values are almost equal and the approximation can be kept.

Its effect on the rule selection is studied in [5.1](#51-sensitivity-to-λ).

---

### 1️⃣ Phase 1 — Candidate generation

- **Goal:** almost every artifact that matters should generate some candidate, without an excessive load on the clean signals and without overflowing the artifact. If an artifact does not become a candidate, no rule will be able to detect it later; if its span overflows the artifact, it merges with neighbouring sounds, which degrades its features ([Phase 2](#2️⃣-phase-2--feature-definition)).
- **Optimization of:** `WAVELET`, `K`, `ENV_SMOOTH`, `MASK_DILATION`, `REF_INNER`, `REF_OUTER`, `MAX_EVENT`.
- **Metrics:** missing a candidate and overflowing the artifact have very different costs (a missed artifact is lost for good; a wider span only degrades its features, which the rules of [Phase 3](#3️⃣-phase-3--rule-selection) can partly compensate), so they are measured apart and with priority:
    1. **Candidate recall:** proportion of artifacts with SNR ≤ 0 dB that have some candidate in their zone $Z_i$ (artifact ± 25 ms).
    2. **Load:** candidates per control signal.
    3. **Overflow:** median of $w_i$ over the artifacts that have some candidate:

$$
w_i = \frac{|Z_i|}{|Z_i \cup \text{spans}_i|} = \frac{|Z_i|}{|Z_i| + \text{overflow}_i}
$$

where:

- $\text{spans}_i$ is the union of the candidate spans that touch $Z_i$.
- $\text{overflow}_i = |\text{spans}_i \setminus Z_i|$ is the time of those spans outside the zone.

> $w_i = 1$ if the spans do not leave the zone, $0.5$ if the overflow equals the length of the zone. A span shorter than the zone is not penalized. An isolated click already overflows the zone a little (time resolution of the low bands, `MASK_DILATION`), so $w_i < 1$ even in the best case; this bias is similar for every configuration.

- **Diagnostics:** median overflow (ms) and weighted candidate recall ($\text{Recall}_w = \frac{1}{N}\sum_i w_i$ over the $N$ artifacts with SNR ≤ 0 dB, with $w_i = 0$ if there is no candidate).

> `K` and `MASK_DILATION` move recall and overflow in opposite directions (a low `K` gives more candidates, but wider spans): a single combined number would hide that conflict.

**Procedure:**

1. **Screening:** each parameter is varied alone. If the metrics do not change, it is fixed.
2. **Search:** two Bayesian optimizations, one after the other:
    - **(a) Maximum recall:** maximize the recall subject to load < 2 × baseline → recall<sub>max</sub>.
    - **(b) Minimum load:** minimize the load subject to recall ≥ recall<sub>max</sub> − 0.005 and median $w_i$ ≥ median $w_i$ of the baseline.
    - The recall, load and overflow of every evaluated configuration (of both searches) are saved.
3. **Choice:** among all the evaluated configurations (a and b) that meet the two constraints of (b), the one with the minimum load → **C1**. The load–overflow Pareto front of the configurations with recall ≥ recall<sub>max</sub> − 0.005 is used to check it (e.g. whether a little more load would give much less overflow).
4. **Full run:** computation of the estimates needed for the Score in [Phase 3](#3️⃣-phase-3--rule-selection).

**Outputs:**

- Screening plots.
- Recall–load curve (a), load–overflow Pareto front (b), with the chosen point.
- C1.

---

### 2️⃣ Phase 2 — Feature definition

- **Goal:** measure each feature so that it separates the artifacts from the rest as well as possible. The candidates are those of C1 and do not change.
- **Optimization of:**

| Feature | Parameters |
|:--------|:-----------|
| **Width** (`width_med`, `width_max`) | `BG_FACTOR`, `WIDTH_FLOOR_DB`, `EXTEND` |
| **Isolation** (`isolation`) | `ISO_WIN` |
| **Impulsiveness** (`z_max`) | `Z_WIN`, `AR_ORDER` |

- **Metric:** standardized partial ROC-AUC with a false positive rate ≤ 0.1, within the cases where each feature is used.

**Procedure:**

1. **Search:** for each group and combination of parameters, compute the features.
2. **Choice:** pick the combination with the best mean partial AUC.
3. **Run:** computation of the features.

**Outputs:**

- AUC vs value (for each parameter).
- Candidate table.

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

- **Goal:** estimate the Score:

$$
\hat{S} = \frac{1}{|\mathcal{S}|}\sum_{s\in\mathcal{S}}\left[\left(1-\widehat{\text{Det}}_s\right)\cdot\hat{q}_s^{\text{undet}} + \hat{q}_s^{\text{det}}\cdot\widehat{\text{Det}}_s\right] - \lambda \cdot \widehat{FP}
$$

where:

- $\hat{q}_s^{\text{undet}}$ is the mean q when the artifact is not detected (equivalent to $SNR_s$) for input SNR $s$.
- $\hat{q}_s^{\text{det}}$ is the mean q when the artifact is detected for input SNR $s$.
- $\widehat{\text{Det}}_s$ is the artifact detection rate at SNR $s$.

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

- [ ] **Phase 0** — Preparation
- [ ] **Phase 1** — Candidate generation
- [ ] **Phase 2** — Feature definition
- [ ] **Phase 3** — Rule selection
- [ ] **Phase 4** — Attenuation
- [ ] **Phase 5** — Validation
