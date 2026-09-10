# 🧠 Classification — Respiratory Sound Classification

This folder contains the **classification pipeline** for the **EILO-BioAcoustics-Analysis** project. It takes the fragment-level audio and time–frequency representations prepared by the [📊 Dataset pipeline](../dataset/README.md) and trains models to classify pediatric respiratory sounds.

The pipeline is currently **a work in progress**. Only the train/val/test split stage is in place; the ablation study and the hierarchical-classification experiments are planned but not yet implemented (see [📋 Status](#-status) for a per-stage checklist).

---

## Table of Contents

- [📂 Repository layout](#-repository-layout)
- [⚙️ Quickstart](#️-quickstart)
- [🔁 Pipeline overview](#-pipeline-overview)
- [1️⃣ Build patient-level data splits — get_splits.py](#1️⃣-build-patient-level-data-splits--get_splitspy)
- [🏋️ Training and hyperparameter tuning](#️-training-and-hyperparameter-tuning)
    - [🧱 CNN architecture](#-cnn-architecture--cnn2dpy)
    - [📉 Learning rate scheduler](#-learning-rate-scheduler--build_schedulerpy)
    - [🎛️ Hyperparameter search space](#️-hyperparameter-search-space--tunepy)
- [🧪 Ablation study — Selection of model architecture](#-ablation-study--selection-of-model-architecture)
    - [🎨 Select 2D representations (stage 1)](#-select-2d-representations-stage-1)
    - [🎵 Select 1D representation (stage 2)](#-select-1d-representation-stage-2)
    - [🔗 Select input fusion (stage 3)](#-select-input-fusion-stage-3)
    - [🏗️ Modify the CNN (stage 4)](#️-modify-the-cnn-stage-4)
    - [🧩 Late fusion with attention (stage 5)](#-late-fusion-with-attention-stage-5)
    - [🧠 Classifier choice (stage 6)](#-classifier-choice-stage-6)
    - [🪜 Classification type (stage 7)](#-classification-type-stage-7)
- [📁 Final folder layout](#-final-folder-layout)
- [📓 Notebooks](#-notebooks)
- [📋 Status](#-status)

---

## 📂 Repository layout

When you clone the repository, the `classification/` folder ships with the split-generation script/notebook, the `training.ipynb` notebook, and the full `src/` package (model, data, training and tuning code) — but no generated splits, results, or trained models. After running the steps below, see [📁 Final folder layout](#-final-folder-layout) for what ends up in this directory.

```
classification/
├── README.md                    ← this file
├── get_splits.py                ← step 1
├── data_splits.ipynb            ← interactive version of step 1
├── training.ipynb               ← hyperparameter tuning + cross-validation (work in progress)
└── src/                         ← model definitions, data pipeline, training & tuning logic
    ├── data/
    │   ├── class_to_idx.py      ← label ↔ index dictionaries (multiclass / hierarchical)
    │   ├── dataset.py           ← ImageDataset: reads a fragment's .tiff RTF + label
    │   ├── dataloaders.py       ← train/val DataLoaders, per-fold mean/std, WeightedRandomSampler
    │   └── transforms.py        ← SpecAugment-style augmentations (shift, time/freq mask, noise)
    ├── models/
    │   └── cnn2d.py             ← configurable 2D-CNN classifier (CNNClassifier_MultiClass)
    ├── training/
    │   ├── train.py             ← train/eval loop; single-fold fit() and 5-fold train_cv()
    │   ├── early_stopping.py    ← early stopping on validation MCC
    │   └── losses.py            ← FocalLoss
    ├── tuning/
    │   └── tune.py              ← Optuna hyperparameter search (model + training params)
    └── utils/
        ├── build_loss.py        ← loss factory (cross-entropy / focal)
        ├── build_optimizer.py   ← optimizer factory (Adam / AdamW / SGD)
        ├── build_scheduler.py   ← LR scheduler (linear warmup + cosine annealing)
        └── seed.py              ← reproducibility seeding
```

The ablation-study scripts themselves (one per stage, see below) are still planned and not yet present.

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

## 🏋️ Training and hyperparameter tuning

The `src/` package implements the model, data pipeline, and training/tuning logic shared by every stage of the ablation study below — only the input representation, backbone, fusion strategy, or classifier head changes between stages; the training loop, metrics, and tuning procedure stay the same.

| Module | File | Role |
|:-------|:-----|:-----|
| `data` | `class_to_idx.py` | Label ↔ index dictionaries: `MULTICLASS_IDX` (7 classes) and `BINARY_IDX` (`Normal` / `Adventitious`, for the hierarchical setup) |
| `data` | `dataset.py` | `ImageDataset` — reads a fragment's `.tiff` RTF and its encoded label |
| `data` | `dataloaders.py` | Builds train/val `DataLoader`s for a given CV fold: computes normalization mean/std from the training fold only, and oversamples via `WeightedRandomSampler` to counter the [class imbalance](../dataset/README.md#-results) |
| `data` | `transforms.py` | SpecAugment-style augmentation for the training set only: random temporal shift, time/frequency masking, Gaussian noise |
| `models` | `cnn2d.py` | `CNNClassifier_MultiClass` — configurable 2D-CNN baseline (stacked `Conv→BatchNorm→LeakyReLU→Dropout` blocks, global-average-pooled embedding, MLP head); serves as the "Baseline" architecture in [stage 4](#️-modify-the-cnn-stage-4) |
| `training` | `train.py` | Shared train/eval loop — `fit()` (single fold) and `train_cv()` (5-fold CV); see below |
| `training` | `early_stopping.py` | Early stopping on validation MCC |
| `training` | `losses.py` | `FocalLoss`, for the class-imbalanced label distribution |
| `tuning` | `tune.py` | Optuna hyperparameter search (`tune()`); see below |
| `utils` | `build_loss.py` / `build_optimizer.py` / `build_scheduler.py` | Factories building the loss (CE / focal), optimizer (Adam / AdamW / SGD), and LR scheduler (linear warmup + cosine annealing) from a plain hyperparameter dict |
| `utils` | `seed.py` | Fixes all random seeds for reproducibility |

The three entry points that tie these modules together:

| Function | Scope | What it does | Output |
|:---------|:------|:--------------|:-------|
| `fit()` | 1 CV fold | Trains one model with a linear-warmup + cosine-annealing LR schedule and early stopping on validation MCC (patience 10); logs loss, balanced accuracy, macro/weighted F1, MCC, macro PR-AUC and per-class F1/support every epoch | `(history, final_metrics)` |
| `train_cv()` | 5 CV folds | Repeats `fit()` over every fold defined in `splits.csv` | Per-fold weights + `cv_results.csv` / `cv_history.csv` under `classification/{models,results}/cv/<model_name>/` |
| `tune()` | 1 fold (default) | Wraps `fit()` in an [Optuna](https://optuna.org/) study (`objective()`) that searches model hyperparameters (depth, filters, embedding/hidden size, dropout, LeakyReLU slope) and training hyperparameters (batch size, LR schedule, weight decay, optimizer, loss + focal gamma), using a TPE sampler and a Hyperband pruner, maximizing validation MCC | `Study` (+ SQLite storage under `classification/results/optuna/<study_name>/` when `save=True`) |

### 🧱 CNN architecture — `cnn2d.py`

`CNNClassifier_MultiClass` is a plain encoder → embedding → classifier stack. `depth` controls how many `ConvBlock`s are chained, and the filter count doubles at every block starting from `base_filters`:

```
Input  (in_channels × 224 × 224)
   │
   ▼
┌───────────────────────────────────────┐
│ ConvBlock × depth                     │   filters: base_filters, ×2, ×4, … (per block)
│   Conv2D 3×3 → BatchNorm → LeakyReLU  │
│   → Dropout2d                         │   (repeated twice per block)
│   Conv2D 3×3 → BatchNorm → LeakyReLU  │
│   → Dropout2d                         │
│   → MaxPool2d(2)                      │
└───────────────────────────────────────┘
   │
   ▼
AdaptiveAvgPool2d(1) → Flatten
   │
   ▼
Linear → BatchNorm1d                       (embedding_dim)      ← feature embedding
   │
   ▼
Linear → BatchNorm1d → LeakyReLU → Dropout (hidden_dim)
   │
   ▼
Linear → logits                            (num_classes = 7)
```

`forward_features()` exposes the embedding on its own (before the classifier head), which the later ablation stages (fusion, attention, alternative classifier heads) build on top of.

### 📉 Learning rate scheduler — `build_scheduler.py`

Training uses a `SequentialLR` combining three phases, driven by `lr_max`, `lr_min_ratio` (with `lr_min = lr_min_ratio · lr_max`) and `warmup_epochs` (default 5):

| Phase | Epochs | Behavior |
|:------|:-------|:---------|
| **Warmup** (`LinearLR`) | `1` → `warmup_epochs` | LR rises linearly from `lr_min` to `lr_max` |
| **Cosine annealing** (`CosineAnnealingLR`) | `warmup_epochs` → `num_epochs` | LR decays from `lr_max` back down to `lr_min` following a cosine curve |
| **Constant** (`ConstantLR`) | `num_epochs` → end of training | LR held at `lr_min` for any remaining epochs (training can run up to `MAX_EPOCHS = 120`, subject to early stopping) |

### 🎛️ Hyperparameter search space — `tune.py`

`objective()` samples the following hyperparameters for every Optuna trial:

**Model hyperparameters**

| Hyperparameter | Search space | Notes |
|:----------------|:--------------|:------|
| `depth` | `[2, 4]` (int) | Number of `ConvBlock`s |
| `base_filters` | `{32, 64, 128, 256}` | Filters in the first `ConvBlock`; doubles every block |
| `alpha_leaky_relu` | `[0.001, 0.3]` (log) | LeakyReLU negative slope |
| `embedding_dim` | `{32, 64, 128, 256, 512}` | Size of the pooled feature embedding |
| `hidden_dim` | `{4, 8, 16, 32, 64}` | Size of the classifier's hidden layer |
| `dropout_cnn` | `[0.0, 0.5]` | `Dropout2d` inside the `ConvBlock`s |
| `dropout_fc` | `[0.0, 0.5]` | Dropout inside the classifier head |

**Training hyperparameters**

| Hyperparameter | Search space | Notes |
|:----------------|:--------------|:------|
| `batch_size` | `{16, 32, 64}` | |
| `lr_max` | `[1e-4, 1e-2]` (log) | Peak learning rate, reached after warmup |
| `lr_min_ratio` | `[3e-3, 1e-1]` (log) | `lr_min / lr_max`; sets both the warmup start and the cosine floor |
| `num_epochs` | `[20, 100]` (int) | Warmup + cosine-annealing horizon (see scheduler above) |
| `weight_decay` | `[1e-4, 1e-2]` (log) | |
| `optimizer` | `{adam, adamw, sgd}` | |
| `loss` | `{ce, fl}` | Cross-entropy vs. `FocalLoss` |
| `gamma_focal` | `[2.0, 5.0]` | Only sampled when `loss == "fl"` |

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

## 📁 Final folder layout

After running the data-split step and, optionally, training/tuning, the relevant outputs are (compare with [📂 Repository layout](#-repository-layout) to see what ships with the repo vs. what the pipeline produces):

```
classification/
├── splits/                 ← step 1 (get_splits.py / data_splits.ipynb)
│   └── splits.csv                 # id, name (patient), label, category, fold ∈ {-1, 1…5}
├── results/                ← training / tuning (src/training/train.py, src/tuning/tune.py)
│   ├── cv/
│   │   └── <model_name>/
│   │       ├── cv_results.csv     # per-fold final validation metrics
│   │       └── cv_history.csv     # per-fold, per-epoch metrics
│   └── optuna/
│       └── <study_name>/
│           └── <study_name>.db    # Optuna study storage (when save=True)
└── models/
    └── cv/
        └── <model_name>/
            └── <model_name>_fold_{1..5}.pt   # trained weights, one per CV fold
```

`splits.csv` is fold `-1` for the held-out test set and `1`–`5` for the train/val cross-validation folds (see [step 1](#1️⃣-build-patient-level-data-splits--get_splitspy)). `results/` and `models/` are only created once `train_cv()` or `tune()` is run, and `<model_name>`/`<study_name>` are chosen per experiment (e.g. one per ablation-study configuration).

---

## 📓 Notebooks

### [data_splits.ipynb](data_splits.ipynb) — Interactive data splitting

Interactive counterpart to [`get_splits.py`](#1️⃣-build-patient-level-data-splits--get_splitspy): builds the same patient-level, stratified train/val/test partition, with the added ability to inspect split statistics and patient-distribution histograms before saving `splits/splits.csv`.

### [training.ipynb](training.ipynb) — Hyperparameter tuning and cross-validation *(work in progress)*

Drives the [training and hyperparameter tuning](#️-training-and-hyperparameter-tuning) infrastructure in `src/`: loads `splits/splits.csv`, then runs an Optuna study via `tune()` over a chosen input representation (e.g. `STFT` spectrograms). It is **not yet finished** — the hyperparameter-tuning and cross-validation (`train_cv()`) stages it is meant to host have not been tested end-to-end. Once validated, it will be converted into a runnable Python script, following the same pattern as `get_splits.py`/`data_splits.ipynb`.

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
