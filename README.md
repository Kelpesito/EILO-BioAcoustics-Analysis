# EILO-BioAcoustics-Analysis

> A research-oriented Python project for respiratory sound analysis, focused on acoustic signal processing and time–frequency representations as a step toward non-invasive assessment of Exercise-Induced Laryngeal Obstruction (EILO).

---

## Table of Contents

- [🌬️ Overview](#-overview)
- [⚙️ Installation](#️-installation)
- [📦 Dependencies](#-dependencies)
- [🗂️ Repository Structure](#️-repository-structure)
- [🧭 Directory Overview](#-directory-overview)
- [📝 Changelog](#-changelog)

---

## 🫁 Overview

**EILO-BioAcoustics-Analysis** is a research project developed within a PhD framework carried out under the project *"Exercise-induced laryngeal obstruction (EILO) in children and adolescents with exercise-induced dyspnea: from prevalence and characterization to improving diagnosis through AI-powered acoustic respiratory analysis"*, funded by **Fundació La Marató de TV3**.

EILO is a prevalent condition in children and adolescents, with a significant impact on their quality of life and sport performance. The diagnosis of EILO currently requires visualisation of the larynx during exercise, with the reference test being **continuous exercise laryngoscopy (CLE)** — an invasive procedure that demands trained personnel and sophisticated equipment. The overarching aim of the project is to develop and evaluate an **AI-based acoustic respiratory analysis tool** as a **non-invasive alternative for the screening of EILO**.

As project-specific respiratory sound recordings are not yet available for this project, the initial development and training of the models will be carried out using **[SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)**, a publicly available database of annotated respiratory sounds from **pediatric patients**. The dataset contains a variety of **normal and adventitious respiratory sounds** and provides a suitable starting point for developing and evaluating the **signal-processing and machine-learning pipeline**.

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

---

## 📦 Dependencies

The project declares the following direct dependencies in `pyproject.toml`:

| Dependency        | Purpose                                                                          |
| ----------------- | -------------------------------------------------------------------------------- |
| `matplotlib`      | Static plotting for exploratory analysis and visualisation notebooks.             |
| `nbformat`        | Parsing and validating Jupyter notebook structure (`.ipynb`).                    |
| `pandas`          | Tabular data handling and metadata CSV processing.                               |
| `plotly`          | Interactive plotting in the visualization notebooks.                             |
| `scikit-image`    | Image processing utilities (used for RTF image export and resizing).             |
| `scipy`           | Numerical computing, signal processing, and the STFT implementation.             |
| `soundfile`       | Reading and writing audio files (`.wav`).                                        |
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
│
└── dataset/
    └── README.md
```

---

## 🧭 Directory Overview

| Directory / File       | Description                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `dataset/`             | Data preparation pipeline: ingestion of SPRSound, metadata curation, quality filtering, fragment extraction, RTFs, and exploratory notebooks. See [`dataset/README.md`](dataset/README.md) for the full pipeline description. |
| `README.md`            | This file.                                                                                                                |

---

## 📝 Changelog

- **v0.1.1**
    - Updated repository link

- **v0.1.0**

    - Added the initial repository structure with `pyproject.toml`, `uv.lock`, and Python 3.14 environment pinning.
    - Added the dataset preparation pipeline under `dataset/`, comprising five ordered scripts.
    - Added the fragment extraction step with bandpass filtering (70–1900 Hz) and resampling (8 kHz → 4 kHz).
    - Added STFT-based RTF computation, exporting 224×224 log-frequency, dB-scale spectrogram images.
    - Added three exploratory notebooks: `visualization_signal.ipynb`, `visualization_fragment.ipynb`, and `EDA.ipynb`.
    - Added detailed `dataset/README.md` documenting the full pipeline and exploratory results.
