# 🧠 Classification — Respiratory Sound Classification

This folder contains the **classification pipeline** for the **EILO-BioAcoustics-Analysis** project. It takes the fragment-level audio and time–frequency representations prepared by the [📊 Dataset pipeline](../dataset/README.md) and trains models to classify pediatric respiratory sounds.

The pipeline is currently **a work in progress**. Only the train/val/test split stage is in place; the ablation study and the hierarchical-classification experiments are planned but not yet implemented (see [📋 Status](#-status) for a per-stage checklist).

---

## Table of Contents

- [📂 Repository layout](#-repository-layout)
- [⚙️ Quickstart](#️-quickstart)
- [🔁 Pipeline overview](#-pipeline-overview)
- [1️⃣ Build patient-level data splits — get_splits.py](#1️⃣-build-patient-level-data-splits--get_splitspy)
- [🧪 Ablation study — Selection of model architecture](#-ablation-study--selection-of-model-architecture)
    - [🎨 Select 2D representations (stage 1)](#-select-2d-representations-stage-1)
    - [🎵 Select 1D representation (stage 2)](#-select-1d-representation-stage-2)
    - [🔗 Select input fusion (stage 3)](#-select-input-fusion-stage-3)
    - [🏗️ Modify the CNN (stage 4)](#️-modify-the-cnn-stage-4)
    - [🧩 Late fusion with attention (stage 5)](#-late-fusion-with-attention-stage-5)
    - [🧠 Classifier choice (stage 6)](#-classifier-choice-stage-6)
    - [🪜 Classification type (stage 7)](#-classification-type-stage-7)
- [📋 Status](#-status)

---

## 📂 Repository layout

```
classification/
├── README.md                    ← this file
├── get_splits.py                ← step 1
├── data_splits.ipynb            ← interactive version of step 1
├── splits/
│   └── splits.csv               ← generated train/val/test partition (step 1)
└── src/                         ← planned (model definitions, training loops)
```

The folder currently ships with the split-generation script and its notebook counterpart. The remaining scripts and modules described below are planned but not yet present.

---

## ⚙️ Quickstart

From the project root, after running the [📊 Dataset pipeline](../dataset/README.md) end-to-end:

```bash
# 1. Build patient-level train/val/test splits
python classification/get_splits.py
```

The ablation stages below are not yet implemented. Each one will take the output of the previous one plus `fragments_metadata.csv` and is designed to be runnable independently for ablation purposes.

---

## 🔁 Pipeline overview

| Step | Script | Input | Output |
|:----:|:-------|:------|:-------|
| **1** | `get_splits.py` | `fragments_metadata.csv` | `splits/splits.csv` |
| **?** | *(planned)* | — | — |

Only the data-split step is implemented. The remaining stages of the pipeline are still being defined — see [🧪 Ablation study](#-ablation-study--selection-of-model-architecture) for the current design sketch.

---

## 1️⃣ Build patient-level data splits — `get_splits.py`

Partitions the fragments into disjoint **train / validation / test** sets at the **patient level**, so that all fragments from a given patient land in the same split. This prevents patient leakage between splits and is the only safeguard against overly optimistic evaluation numbers.

The split is written as a single CSV that assigns every fragment to one of the three sets:

```
classification/
└── splits/
    └── splits.csv    # columns: fragment_id, name (patient), split ∈ {train, val, test}
```

The split uses `StratifiedGroupKFold` with 5 folds to preserve the original label proportions, yielding roughly a **64 / 16 / 20** train / val / test partition.

**Run:**
```bash
python classification/get_splits.py
```

**Notebook equivalent:** `data_splits.ipynb` performs the same operation interactively, with the additional ability to inspect split statistics and patient-distribution histograms before saving.

> The splits produced here are the canonical input for every downstream training step.

---

## 🧪 Ablation study — Selection of model architecture

Once the data splits exist, the remaining seven steps form a single ablation study that walks through every design decision the model depends on — from input representation all the way to the formulation of the classification problem. Each stage takes the winning configuration of the previous one and sweeps a small set of alternatives, picking the best before moving on.

### 🎨 Select 2D representations (stage 1)

Evaluates which combination of **2D time–frequency representations** carries the most discriminative information for the seven respiratory-sound classes. Each option is trained on the same backbone and the same splits; only the input differs.

**Options:**

1. **STFT** (spectrogram)
2. **CWT** (scalogram)
3. **WSST** (synchrosqueezed scalogram)
4. **STFT + CWT**
5. **STFT + WSST**
6. **CWT + WSST**
7. **STFT + CWT + WSST**

The winning configuration is carried into stage 3.

> All three RTFs are produced by the [📊 Dataset pipeline](../dataset/README.md) (`STFT` is already implemented; `SCALOGRAM` and `WSST` are still planned there).

### 🎵 Select 1D representation (stage 2)

Evaluates which **1D signal representation** — applied directly to the time-domain waveform — works best as an alternative or complement to the 2D representations from stage 1.

**Options:**

1. **Raw signal** — 1D CNN
2. **IMF** (Intrinsic Mode Functions, EMD/EMD-like decomposition) — 1D CNN with one input channel per IMF
3. **IMF reconstruction** — 1D CNN with a single reconstructed channel
4. **IMF with late fusion** — 1D CNN on each IMF, fused at the decision stage
5. **RNN** (e.g. GRU) over the raw signal

The winning configuration is carried into stage 3.

### 🔗 Select input fusion (stage 3)

Determines whether the model benefits from combining the best 2D and 1D representations, or whether either modality alone is sufficient.

**Options:**

1. **2D only** — best configuration from stage 1
2. **1D only** — best configuration from stage 2
3. **2D + 1D** — both modalities fused at the input or feature level

The winning configuration is carried into stage 4.

### 🏗️ Modify the CNN (stage 4)

Replaces the baseline CNN backbone with more expressive architectures and measures how much each architectural change helps on the winning input from stage 3.

**Options:**

1. **Baseline** — the unmodified CNN used in stages 1–3
2. **Residual** — ResNet-style skip connections
3. **Dense** — DenseNet-style dense connectivity
4. **SE / CBAM** — channel- or spatial-attention modules (Attention Convolutional Neural Network)

The winning architecture is carried into stage 5.

### 🧩 Late fusion with attention (stage 5)

Tests whether adding an **attention-based late fusion** on top of the winning backbone from stage 4 improves over a plain aggregation of per-fragment / per-modality predictions.

**Options:**

1. **No attention** — the unmodified model from stage 4
2. **With attention** — learned attention weights over the fused representations

The winning strategy is carried into stage 6.

### 🧠 Classifier choice (stage 6)

Swaps the **classifier head** on top of the winning backbone + fusion from stages 4–5, to determine whether a richer head helps or whether a simple linear model suffices given a strong backbone.

**Options:**

1. **Multilayer perceptron (MLP)** — the default head used in stages 1–5
2. **Logistic Regression (LR)**
3. **Support Vector Machine (SVM)**
4. **LightGBM (LGB)** — gradient-boosted decision trees on the learned features

The winning classifier is carried into stage 7.

### 🪜 Classification type (stage 7)

Finally, compares two formulations of the classification problem itself.

**Options:**

1. **Multiclass** — direct prediction over the seven labels (`Normal`, `Rhonchi`, `Wheeze`, `Stridor`, `Coarse Crackle`, `Fine Crackle`, `Wheeze+Crackle`)
2. **Hierarchical** — first `Normal` vs. `Adventitious`, then the adventitious subtype within the predicted branch

The hierarchical formulation leverages the binary `category` field already present in `fragments_metadata.csv`.

---

## 📋 Status

This folder is a **work in progress**. The checklist below tracks the pipeline steps and the ablation stages.

**Pipeline steps:**

- [x] **Step 1 — Data splits** (`get_splits.py` + `data_splits.ipynb`)
- [ ] **Step 2 — Ablation study**

**Ablation study — per stage:**

- [ ] **Stage 1** — 2D representations (STFT, CWT, WSST, and all combinations)
- [ ] **Stage 2** — 1D representations (raw signal, IMF, IMF reconstruction, IMF late fusion, RNN)
- [ ] **Stage 3** — Input fusion (2D only / 1D only / 2D + 1D)
- [ ] **Stage 4** — CNN modifications (baseline, Residual, Dense, SE/CBAM)
- [ ] **Stage 5** — Late fusion with attention (no attention / with attention)
- [ ] **Stage 6** — Classifier choice (MLP, LR, SVM, LGB)
- [ ] **Stage 7** — Classification type (multiclass, hierarchical)
