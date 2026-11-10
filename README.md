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

#### Alternative: Install with `pip`

If you prefer not to use `uv`, the dependencies are also exported to `requirements.txt` (generated with `uv export --format requirements-txt --no-hashes`). Create and activate a virtual environment with Python 3.14, then install them with `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows (PowerShell / cmd)
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
```

The `--extra-index-url` points `pip` to the PyTorch CUDA 12.6 wheels, matching the index configured for `uv` in `pyproject.toml`.

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
| `matplotlib`      | Static plotting for exploratory analysis and visualisation notebooks |
| `nbformat`        | Parsing and validating Jupyter notebook structure (`.ipynb`).                    |
| `optuna`          | Hyperparameter optimization for the classification pipeline (TPE sampler + Hyperband pruner). |
| `pandas`          | Tabular data handling and metadata CSV processing.                               |
| `plotly`          | Interactive plotting in the visualization notebooks and in the summary figures of the transient removal filter (`denoising/summary_plots.py`). |
| `pywavelets`      | Stationary Wavelet Transform (SWT) used by the transient removal filter (dataset pipeline and `denoising/`). |
| `scikit-image`    | Image processing utilities (used for RTF image export and resizing).             |
| `scikit-learn`    | Machine-learning utilities (e.g. `StratifiedGroupKFold` for patient-level data splits in the classification pipeline). |
| `scipy`           | Numerical computing, signal processing (STFT, Welch PSD, AR filtering, frequency response) and `scipy.ndimage` labelling/filters of the transient removal filter. |
| `seaborn`         | Statistical plotting (built on `matplotlib`) for the results figures of the transient removal filter evaluation (`denoising/evaluation_denoise.ipynb`). |
| `soundfile`       | Reading and writing audio files (`.wav`).                                        |
| `tensorboard`     | Logging and visualization of training curves and Optuna trial hyperparameters in the classification pipeline. |
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
├── requirements.txt
├── uv.lock
├── dataset/
│   └── README.md
├── denoising/
│   ├── __init__.py
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
| `dataset/`             | Data preparation pipeline: ingestion of SPRSound, metadata curation, quality filtering, fragment extraction, RTFs, label and duration filtering, and exploratory notebooks. See [dataset/README.md](dataset/README.md) for the full pipeline description. |
| `denoising/`           | Development of the transient (click) removal filter used in the fragment pre-processing: algorithm development notebooks (`denoise_soundwaveletSWT.ipynb`, `denoise_summary_plots.ipynb`) and interactive summary figures (`summary_plots.py`). Not yet ported to the dataset pipeline. See [denoising/README.md](denoising/README.md). |
| `classification/`      | Respiratory sound classification pipeline (work in progress): patient-level data splits, training and hyperparameter tuning (Optuna) of a baseline 2D-CNN, and a planned ablation study for model-architecture selection. See [classification/README.md](classification/README.md) for the full pipeline description. |
| `ICBHI/`               | Auxiliary analysis on the ICBHI 2017 Challenge dataset: extracts per-cycle fragments and studies respiratory-cycle duration distributions in a pediatric population, as a sanity check for whether SPRSound's fragments need duration-based filtering. Not part of the main dataset/classification pipeline. See [ICBHI/README.md](ICBHI/README.md) for the full description. |
| `README.md`            | This file.                                                                                                                |

---

## 📝 Changelog
- **v2.1.3**
    - Updated `README.md`.
    - Updated `denoising/README.md`.

- **v2.1.2**
    - Minor changes:
        - Training metrics computed without Data Augmentation: new `train_eval_loader` in `classification/src/data/dataloaders.py`. The training loss of `train_one_epoch` is unchanged.
        - Fixed the docstring of `classification/get_splits.py`: it requires `dataset/fragments_metadata_filtered.csv`.
    - Rebuilt the experiment in `denoising/evaluation_denoise.ipynb`:
        - Ampliation of `SNR_LIST` to `[-30, -20, -10, -5, 0, 5, 10, 20]`, before `[-20, -10, -5, 0, 5]`.
    - Added two figures to evaluation denoise algorithm in `denoising/evaluation_denoise.ipynb`:
        - `plot_elimination`: SNR out vs SNR in of the detected artifacts, ΔSNR distribution per SNR level, proportion of attenuated artifacts (ΔSNR ≥ 3 dB) and detection rate vs SNR, net SNR improvement vs SNR, and SNR improvement of the detected artifacts by class of the clean signal.
        - `plot_conservation`: NMSE out of the artifact zone vs SNR, NMSE out of the artifact zone vs number of detections out of the zone for mixtures and control signals, the same two per class of the clean signal, and cumulative distribution of the NMSE out of the artifact zone.

- **v2.1.1**
    - First result figures of experiment 1 in `denoising/evaluation_denoise.ipynb` (section 5.4.1):
        - `plot_detection`: detection rate vs SNR per artifact and global, detection rate vs SNR by class of the clean signal, false positives vs SNR, and SNR at 50 % detection per artifact with its 95 % CI (bootstrap over the clean signals).
    - `SNR_LIST` extended.

- **v2.1.0**
    - Towards the evaluation of the transient removal algorithm:
        - New module `denoising/remove_noise.py`: the algorithm as an importable module.
        - New module `denoising/evaluation.py`: artifact insertion at a given local SNR, evaluation metrics and `run_experiment`, which runs the experiment in parallel with `multiprocessing` (one worker process per CPU core) and saves the metrics as a `.csv`.
        - New notebook `denoising/evaluation_denoise.ipynb`: selection of clean signals and artifacts, and experiment 1 (insertion of a single artifact + control signals without any artifact). The figures of the results are still missing.
        - Results of experiment 1 saved in `denoising/results/experiment_1.csv` (tracked by git, the rest of the `.csv` are still ignored).

- **v2.0.1**
    - Updated `denoising/README.md`.
    - Updated `dataset/README.md`.
    - Updated `denoising/denoise_soundwaveletSWT.ipynb` with the new algorithm version.

- **v2.0.0**
    - Modified the transient removal algorithm:
        - Decision rules based on the effective width of the events and on the alignment of the band peaks.
        - New *aligned transient* rule.
        - Not implemented in `dataset/src/remove_heart_sounds.py` and `dataset/denoise_soundwaveletSWT.ipynb` yet. 

    - New subpackage `denoising/` for the transient removal algorithm:
        - Moved `dataset/denoise_soundwaveletSWT.ipynb` → `denoising/denoise_soundwaveletSWT.ipynb`.
        - New notebook `denoising/denoise_summary_plots.ipynb`: New version of the algorithm and interactive (`Plotly`) versions of the summary figures.
        - New script `denoising/summary_plots.py`: the visualization functions of the notebook to show all the figures one after the other.

---

- **v1.5.2**
    - Heart sound removal → transient removal: the goal of the filter is now to attenuate clicks, crackles and other short broadband transients that mask the respiratory sound, without damaging the respiratory, adventitious or heart sounds (heart sounds are no longer the main target).
    - Renamed `dataset/cardio_soundwaveletSWT.ipynb` → `dataset/denoise_soundwaveletSWT.ipynb`, with the new algorithm (`remove_noise`, before `remove_heart_sounds`):
        - Click-only mode (`CLICK_ONLY = True`): after the detection per band and the long-event protection, each candidate is classified with 4 features (bands and intensity, peak width, impulsiveness `z` from a robust AR(20) model, isolation before/after) and only the transients are attenuated.
        - Spectrograms of the plots in linear power with new custom colormap.
    - Not yet ported to the pipeline: `dataset/src/remove_heart_sounds.py` (used by `dataset/get_rtf.py`) and `dataset/visualization_fragment.ipynb` still use the previous version.

- **v1.5.1**
    - New duration normalization applied.

- **v1.5.0**
    - New script `classification/hyperparameter_tuning.py` (step 2 of the classification pipeline): hyperparameter tuning of the model with Optuna.
    - Segment duration (`dataset/get_rtf.py`): fragments are now normalized to 2 seconds (before, 4 seconds):
        - If < 2 seconds, cyclic padding.
        - If >= 2 seconds, choose the most energetic window.

- **v1.4.6**
    - Updated `classification/data_splits.ipynb`.

- **v1.4.5**
    - Duration-based filered database: 
        - Rejected fragments outside [0.5, 3] seconds of duration.
    - Removed `Duration-based filtering` section from `dataset/EDA_filtered.ipynb`.

- **v1.4.4**
    - Heart sound removal (`dataset/src/remove_heart_sounds.py`):
        - Long-event protection per band: a long event (> `MAX_EVENT`) only protects its band and the `PROTECT_LEVELS` adjacent ones (now `0`), before all levels.
        - Band splitting (stationary wavelet packet): the levels in `SPLIT_LEVELS` (now 3, 4 and 5, 62–500 Hz) are split into 2 half-bands → 10 bands
        - Border protection disabled (`EDGE = 0`, before 0.02 s) and no widening of the protected zones (`PROTECT_MARGIN = 0`, before 0.05 s).
    - STFT normalization (`dataset/src/calculate_rtf.py`): the spectrogram is now normalized to its maximum within 50–1050 Hz (before, over the whole frequency range).

- **v1.4.3**
    - Towards duration-based filtering of the fragments: new section *Duration-based filtering* in `dataset/EDA_filtered.ipynb`:
        - Fragments kept (globally and by label) as a function of the minimum / maximum duration thresholds.
        - Table of fragments kept for each combination of thresholds and detail for some concrete thresholds (`duration_filter_summary`).
        - Duration distribution by label after the filter.

- **v1.4.2**
    - Allocation of precomputed tensors in memory:
        - New `CachedImageDataset` in `classification/src/data/dataset.py`: images are loaded into memory once with the deterministic transforms already applied; only the augmentations run per item (~3× faster data loading per epoch).
        - Split `classification/src/data/transforms.py` into deterministic (`get_deterministic_transforms`) and random (`get_augmentation_transforms`) transforms.
        - New `cache` argument in `get_dataloaders` (default `True`). With cache, the validation `DataLoader` runs without workers.

- **v1.4.1**
    - Changed support `n` to prevalence `p` in PR-AUC per class to epoch log during training.

- **v1.4.0**
    - Added PR-AUC per class to epoch log during training.
    - Improved hyperparameter tuning logs in TensorBoard.

- **v1.3.12**
    - New pre-processing step in `dataset/get_rtf.py`: heart sound (S1/S2) removal before the duration normalization:
        - New module `dataset/src/remove_heart_sounds.py`: adaptive wavelet filter (`soundwaveletSWT`, SWT `db4`) with a long-event protection.
        - Local reference computed every 5 ms and interpolated: ×17 faster.
    - New notebook `dataset/cardio_soundwaveletSWT.ipynb` (development and tests of the filter).
    - Added the heart sound removal to `dataset/visualization_fragment.ipynb`.

- **v1.3.11**
    - Tested parallelization: Works well
    - Changed parallelization's chunck size to 1, before 32.

- **v1.3.10**
    - Parallelized `dataset/get_rtf.py` with `multiprocessing` (one worker process per CPU core).

- **v1.3.9**
    - Towards the parallelization of the file `dataset/get_rtf.py`:
        - Reproducibility in cyclic padding.
        - Moved the per-fragment processing into a new function `process_file`.

- **v1.3.8**
    - Added `requirements.txt`.

- **v1.3.7**
    - Added examples to `EDA_filtered.ipynb`.

- **v1.3.6**
    - Implemented tensorboard for logging training processes.

- **v1.3.5**
    - Modified STFT parameter: new window length `N = 400` samples.
    - Changed some hyperparameter values.
    - New monitored score: Exponential Moving Average (EMA) of the validation macro PR-AUC:
        - New callback `classification/src/training/callbacks/ema.py`.

- **v1.3.4**
    - Changed some hyperparameter values
    - Tested hyperparameter tuning code: model trains correctly, not optimal

- **v1.3.3**
    - Added progress bars to training loops.

- **v1.3.2**
    - Fixed bug in `classification/src/training/train.py`.

- **v1.3.1**
    - Updated `classification/README.md`.

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
