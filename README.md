# EILO-BioAcoustics-Analysis

> A research-oriented Python project for respiratory sound analysis, focused on acoustic signal processing and time–frequency representations as a step toward non-invasive assessment of Exercise-Induced Laryngeal Obstruction (EILO).

---

## Table of Contents

- [🫁 Overview](#-overview)
- [⚙️ Installation](#️-installation)
- [📦 Dependencies](#-dependencies)
- [🗂️ Repository Structure](#️-repository-structure)
- [🧭 Directory Overview](#-directory-overview)
- [📝 Changelog](#-changelog)

---

## 🫁 Overview

**EILO-BioAcoustics-Analysis** is a research project developed within a PhD framework carried out under the project *"Exercise-induced laryngeal obstruction (EILO) in children and adolescents with exercise-induced dyspnea: from prevalence and characterization to improving diagnosis through AI-powered acoustic respiratory analysis"*, funded by **Fundació La Marató de TV3**.

EILO is a prevalent condition in children and adolescents, with a significant impact on their quality of life and sport performance. The diagnosis of EILO currently requires visualisation of the larynx during exercise, with the reference test being **continuous exercise laryngoscopy (CLE)** — an invasive procedure that demands trained personnel and sophisticated equipment. The overarching aim of the project is to develop and evaluate an **AI-based acoustic respiratory analysis tool** as a **non-invasive alternative for the screening of EILO**.

As project-specific respiratory sound recordings are not yet available for this project, the initial development and training of the models will be carried out using **[SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)**, a publicly available database of annotated respiratory sounds from **pediatric patients**. The dataset contains a variety of **normal and adventitious respiratory sounds** and provides a suitable starting point for developing and evaluating the **signal-processing and machine-learning pipeline**. A second public database, **[ICBHI 2017](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)**, is used as an auxiliary reference to characterize respiratory-cycle durations in a pediatric population, since it (unlike SPRSound) segments its recordings into full respiratory cycles.

The project will progressively explore different approaches for **respiratory sound representation, analysis, and classification**, providing the basis for **future work with project-specific recordings**.

---

## ⚙️ Installation

The project is developed in **Python 3.14** and uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and project dependencies. The repository specifies the Python version in `.python-version` and uses `uv.lock` to ensure reproducible dependency versions.

### 1. Clone the repository

```bash
git clone https://github.com/Kelpesito/EILO-BioAcoustics-Analysis.git
cd EILO-BioAcoustics-Analysis
```

### 2. Synchronize the environment with `uv`

`uv` will automatically pick up the Python version from `.python-version`, create a local virtual environment in `.venv/`, and install the locked dependencies:

```bash
uv sync
```

Activate the virtual environment

```bash
.venv\Scripts\activate    # Windows (PowerShell / cmd)
# source .venv/bin/activate   # Linux / macOS
```

### 3. External data source (optional)

The data preparation pipeline expects a clone of the [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) repository. This is only required if you intend to regenerate the dataset locally. The pipeline scripts and notebooks can be inspected and reviewed without it.

```bash
git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git
```

The auxiliary [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) dataset used in `ICBHI/` is not git-clonable — it must be downloaded manually from the challenge website, with every file placed directly under `ICBHI/ICBHI/data/` (non-data files such as `filename_differences.txt`, `filename_format.txt`, … removed). See [ICBHI/README.md](ICBHI/README.md) for details.

---

## 📦 Dependencies

The project declares the following direct dependencies in `pyproject.toml`:

| Dependency        | Purpose                                                                          |
| ----------------- | -------------------------------------------------------------------------------- |
| `ipywidgets`      | Interactive widgets for Jupyter notebooks (e.g. progress bars, interactive controls). |
| `jupyter`         | Notebook environment used to run the project's `.ipynb` notebooks.               |
| `kaleido`         | Static image export engine for `Plotly` figures.                                   |
| `matplotlib`      | Static plotting for exploratory analysis and visualisation notebooks.             |
| `nbformat`        | Parsing and validating Jupyter notebook structure (`.ipynb`).                    |
| `optuna`          | Hyperparameter optimization for the classification pipeline (TPE sampler + Hyperband pruner). |
| `pandas`          | Tabular data handling and metadata CSV processing.                               |
| `plotly`          | Interactive plotting in the visualization notebooks.                             |
| `scikit-image`    | Image processing utilities (used for RTF image export and resizing).             |
| `scikit-learn`    | Machine-learning utilities (e.g. `StratifiedGroupKFold` for patient-level data splits in the classification pipeline). |
| `scipy`           | Numerical computing, signal processing, and the STFT implementation.             |
| `soundfile`       | Reading and writing audio files (`.wav`).                                        |
| `torch`           | Deep-learning framework used to build and train the classification CNN models.   |
| `torchvision`     | PyTorch computer-vision utilities (image transforms/augmentation) for the classification data pipeline. |
| `tqdm`            | Progress bars for long-running batch operations in the pipeline scripts.         |

---

## 🗂️ Repository Structure

The repository, as cloned, contains the following files and directories:

```text
EILO-BioAcoustics-Analysis/
├── .python-version
├── README.md
├── pyproject.toml
├── uv.lock
├── dataset/
│   └── README.md
├── classification/
│   └── README.md
└── ICBHI/
    └── README.md
```

---

## 🧭 Directory Overview

| Directory / File       | Description                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `dataset/`             | Data preparation pipeline: ingestion of SPRSound, metadata curation, quality filtering, fragment extraction, RTFs, and exploratory notebooks. See [dataset/README.md](dataset/README.md) for the full pipeline description. |
| `classification/`      | Respiratory sound classification pipeline (work in progress): patient-level data splits and a planned ablation study for model-architecture selection. See [classification/README.md](classification/README.md) for the full pipeline description. |
| `ICBHI/`               | Auxiliary analysis on the ICBHI 2017 Challenge dataset: extracts per-cycle fragments and studies respiratory-cycle duration distributions in a pediatric population, as a sanity check for whether SPRSound's fragments need duration-based filtering. Not part of the main dataset/classification pipeline. See [ICBHI/README.md](ICBHI/README.md) for the full description. |
| `README.md`            | This file.                                                                                                                |

---

## 📝 Changelog
- **v1.3.0**
    - Generated splits from filtered csv.
    - Normalization is now applied per image (before per training set).
    - Added circularity to `TimeShift` in `classification/src/data/transforms.py`.
    - Changed learning rate scheduler: `ReduceLROnPlateau` for Linear Warm-Up + Cosine Annealing
    - Created new submodule `classification/src/training/callbacks` including custom callbacks: `early_stopping.py` (before in `classification/src/training`) and `lwu_ca.py`.
    - Now `classification/src/utils/build_scheduler.py` does not only builds the `lwu_ca`, now can initialize the `ReduceLROnPlateau`. 

- **v1.2.4**
    - Modification of RTF extraction pipeline:
        - Normalized the segment durations to 4 seconds:
            - If < 4 seconds, cyclic padding
            - If > 4 seconds, choose the most energetic window
        - Signal normalization before RTF (standardization)
    - Modification of STFT parameters:
        - Spectrogram normalization (max = 0 dB)
        - Clip frequencies < 1050 Hz

- **v1.2.3**
    - Filtered SPRSound database by label (dropped crackles)
    - Updated `ICBHI/README.md`.

- **v1.2.2**
    - Updated `ICBHI/README.md`.

- **v1.2.1**
    - Added label distribution to `ICBHI/EDA.ipynb`.

- **v1.2.0**
    - Study of ICBHI 2017 database: auxiliary analysis for duration-based filtering.
    - Optimized `dataset/get_metadata.py` script: Now the images are not fully read, just extract the metadata info with `soundfile.info`.
    - Removed .csv files from Git tracking.

- **v1.1.1**
    - Fixed images visualization in `dataset/README.md`.

- **v1.1.0**
    - Implemented the classification training & hyperparameter tuning pipeline (`classification/src/`):
        - Data loading / Data augmentation / Dataloaders
        - 2D-CNN model
        - Training loop
        - Optuna based hyperparameter tuning search space
    - Added `classification/training.ipynb` to drive hyperparameter tuning and cross-validation tests (*work in progress*).
    - Expanded Exploratory Data Analysis (EDA) in `dataset/EDA.ipynb`.

- **v1.0.0**
    - Started `classification` repository.
    - Split data in Train/Validation/Test sets by patient.
    - Defined ablation study to select the architecture model.

---

- **v0.1.2**
    - Created folder `dataset/src`.
    - Moved file `dataset/calculate_rtf.py` to `dataset/src/calculate_rtf.py`.

- **v0.1.1**
    - Updated repository link

- **v0.1.0**
    - Added the initial repository structure with `pyproject.toml`, `uv.lock`, and Python 3.14 environment pinning.
    - Added the dataset preparation pipeline under `dataset/`, comprising five ordered scripts.
    - Added the fragment extraction step with bandpass filtering (70–1900 Hz) and resampling (8 kHz → 4 kHz).
    - Added STFT-based RTF computation, exporting 224×224 log-frequency, dB-scale spectrogram images.
    - Added three exploratory notebooks: `visualization_signal.ipynb`, `visualization_fragment.ipynb`, and `EDA.ipynb`.
    - Added detailed `dataset/README.md` documenting the full pipeline and exploratory results.
