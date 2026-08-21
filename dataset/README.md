# 📊 Dataset — Respiratory Sound Preparation Pipeline

This folder contains the **data preparation pipeline** for the **EILO-BioAcoustics-Analysis** project. It transforms the raw [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) dataset of pediatric respiratory sound recordings into clean, fragment-level audio and time–frequency representations ready for modeling.

The pipeline is implemented as **5 ordered, independent scripts** that build on each other's outputs.

---

## Table of Contents

- [📂 Repository layout](#-repository-layout)
- [⚙️ Quickstart (full pipeline)](#️-quickstart-full-pipeline)
- [🔁 Pipeline overview](#-pipeline-overview)
    - [1️⃣ Extract the raw SPRSound records — get_sprsound_dataset.py](#1️⃣-extract-the-raw-sprsound-records--get_sprsound_datasetpy)
    - [2️⃣ Build the metadata table — get_metadata.py](#2️⃣-build-the-metadata-table--get_metadatapy)
    - [3️⃣ Split records by quality — get_poor_quality_records.py](#3️⃣-split-records-by-quality--get_poor_quality_recordspy)
    - [4️⃣ Extract annotated respiratory fragments — get_fragments.py](#4️⃣-extract-annotated-respiratory-fragments--get_fragmentspy)
    - [5️⃣ Compute time–frequency representations — get_rtf.py]()
        - [⚙️ RTF configurarion](#rtf-configurarion-see-calculate_rtf)
- [📁 Final folder layout](#-final-folder-layout)
- [📓 Notebooks](#-notebooks)
    - [visualization_signal.ipynb — Raw record + preprocessing](#visualizationsignalipynb--raw-record--preprocessing)
    - [visualization_fragment.ipynb — Single fragment + STFT](#visualizationfragmentipynb--single-fragment--stft)
    - [EDA.ipynb — Exploratory Data Analysis](#edaipynb--exploratory-data-analysis)
        - [📝 Results](#-results)
    - [▶️ How to run](#-how-to-run)

---

## 📂 Repository layout

When you clone the repository, the `dataset/` folder ships with the pipeline scripts and three exploratory notebooks — but no data. After running the full pipeline, see [📂 Final folder layout](#-final-folder-layout) for what ends up in this directory.

```
dataset/
├── README.md                    ← this file
├── get_sprsound_dataset.py      ← step 1
├── get_metadata.py              ← step 2
├── get_poor_quality_records.py  ← step 3
├── get_fragments.py             ← step 4
├── get_rtf.py                   ← step 5
├── src/
│   └── calculate_rtf.py         ← RTF functions used by get_rtf.py
├── EDA.ipynb                    ← exploratory analysis of fragments_metadata.csv
├── visualization_signal.ipynb   ← raw record + preprocessing visualization
└── visualization_fragment.ipynb ← single-fragment + STFT visualization
```

---

## ⚙️ Quickstart (full pipeline)

From the project root:

```bash
# 0. Clone SPRSound (one-time)
git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git

# 1–5. Run the pipeline in order
python dataset/get_sprsound_dataset.py
python dataset/get_metadata.py
python dataset/get_poor_quality_records.py
python dataset/get_fragments.py
python dataset/get_rtf.py -t STFT
```

The resulting `dataset/dataset/spectrogram/*.tiff` files together with `dataset/fragments_metadata.csv` are the canonical input for downstream training.

---

## 🔁 Pipeline overview

| Step | Script | Input | Output |
|:----:|:-------|:------|:-------|
| **1/5** | `get_sprsound_dataset.py` | `../SPRSound/` (cloned repo) | `full_dataset/wav/`, `full_dataset/json/` |
| **2/5** | `get_metadata.py` | `full_dataset/wav/`, `full_dataset/json/` | `metadata.csv` |
| **3/5** | `get_poor_quality_records.py` | `full_dataset/`, `metadata.csv` | `good_quality/`, `poor_quality/` |
| **4/5** | `get_fragments.py` | `good_quality/` | `dataset/audio/`, `fragments_metadata.csv` |
| **5/5** | `get_rtf.py` | `dataset/audio/`, `fragments_metadata.csv` | `dataset/{stft,scalogram,wsst}/*.tiff` |

Each step depends on the outputs of the previous ones, so they must be run in order.

---

### 1️⃣ Extract the raw SPRSound records — [get_sprsound_dataset.py](get_sprsound_dataset.py)

Collects all `.wav` and `.json` files from the four SPRSound sub-repositories (BioCAS2022, BioCAS2023, BioCAS2024, BioCAS2025) and copies them into a single flat structure.

**Requires:** a clone of the SPRSound repository at `../SPRSound/`
```bash
git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git
```

**Generates:**
```
dataset/
├── full_dataset/
│   ├── wav/   # {patient}_{age}_{gender}_{position}_{record_id}.wav
│   └── json/  # {patient}_{age}_{gender}_{position}_{record_id}.json
```

**Run:**
```bash
python dataset/get_sprsound_dataset.py
```

---

### 2️⃣ Build the metadata table — [get_metadata.py](get_metadata.py)

Walks every `.json` annotation file and produces a single `metadata.csv` describing each record.

**Per-record fields:**

| Field | Description |
|:------|:------------|
| `record_id` | SPRSound record id |
| `patient_id` | SPRSound patient id |
| `age` | Patient's age |
| `gender` | `"0"` = male, `"1"` = female |
| `position` | Recording position (e.g. chest, back, neck) |
| `json_path` | Relative path to the `.json` annotation |
| `wav_path` | Relative path to the `.wav` file |
| `Poor Quality` | `1` if SPRSound flags the record as *Poor Quality*, else `0` |

**Run:**
```bash
python dataset/get_metadata.py
```

**Generates:** `dataset/metadata.csv`

---

### 3️⃣ Split records by quality — [get_poor_quality_records.py](get_poor_quality_records.py)

Reads `metadata.csv` and copies each record into one of two folders based on SPRSound's `record_annotation` flag.

**Generates:**
```
dataset/
├── good_quality/
│   ├── wav/
│   └── json/
└── poor_quality/
    ├── wav/
    └── json/
```

Only `good_quality/` records are used downstream for fragment extraction.

**Run:**
```bash
python dataset/get_poor_quality_records.py
```

---

### 4️⃣ Extract annotated respiratory fragments — [get_fragments.py](get_fragments.py)

For every `good_quality` record, this script:

1. Loads the `.wav` signal and the event annotations from the `.json`.
2. **Pre-processes** the signal:
   - 8th-order Butterworth **bandpass filter** at **70–1900 Hz**
   - **Resamples** from 8 kHz to 4 kHz
3. Crops the signal at each event's `[start, end]` timestamp.
4. Renames each fragment to a sequential id (`1.wav`, `2.wav`, …).
5. Builds a `fragments_metadata.csv` with per-fragment labels and patient info.

**Per-fragment fields:**

| Field | Description |
|:------|:------------|
| `id` | Sequential fragment id (matches the `.wav` filename) |
| `name` | Anonymized patient id (`P1`, `P2`, …) |
| `age` | Patient's age |
| `gender` | `"0"` = male, `"1"` = female |
| `position` | Recording position |
| `record_id` | SPRSound source record id |
| `segment` | Index of this segment within its record (0-based) |
| `label` | One of: `Normal`, `Rhonchi`, `Wheeze`, `Stridor`, `Coarse Crackle`, `Fine Crackle`, `Wheeze+Crackle` |
| `category` | `"Normal"` if `label == "Normal"`, else `"Adventitious"` |
| `duration` | Fragment duration in seconds |

**Generates:**
```
dataset/dataset/
├── audio/                       # {id}.wav  (resampled, filtered fragments)
└── fragments_metadata.csv
```

**Run:**
```bash
python dataset/get_fragments.py
```

---

### 5️⃣ Compute time–frequency representations — [get_rtf.py](get_rtf.py)

For each fragment `.wav`, this step computes one or more **time–frequency representations (RTFs)** and stores each as a 224×224 `.tiff` image suitable for CNN-based models. Each RTF is written to its own subfolder so they can be used independently for ablations or multi-view training.

**Available RTFs:**

| RTF | Status | Library | Description |
|:----|:------:|:--------|:------------|
| `STFT` | ✅ implemented | `scipy.signal.ShortTimeFFT` | Log-frequency, dB-scale spectrogram (see config below) |
| `SCALOGRAM` | 🟡 planned |  |  |
| `WSST` | 🟡 planned |  |  |

#### RTF configurarion (see [calculate_rtf](src/calculate_rtf.py)):

**STFT configuration**

| Parameter | Value |
|:----------|:------|
| Window | Hann, **179 samples** |
| Hop size | **10 samples** |
| FFT length | **2048** points |
| Frequency range | **50–2000 Hz** (cropped) |
| Frequency scale | **Logarithmic** (`np.geomspace`) |
| Amplitude scale | **Decibels** (`10·log10`) |
| Output size | **224 × 224** pixels (resized, vertically flipped) |

> All RTFs share the same output contract: 224×224 float32 array written as `.tiff`. 

**Run (one RTF at a time):**
```bash
python dataset/get_rtf.py -t STFT
# python dataset/get_rtf.py -t SCALOGRAM   # once implemented
# python dataset/get_rtf.py -t WSST        # once implemented
```

**Generates:**
```
dataset/dataset/
├── spectrogram/        # STFT
│   └── {id}.tiff
├── scalogram/          # CWT (planned)
│   └── {id}.tiff
└── wsst/               # Wavelet Synchrosqueezed STFT (planned)
    └── {id}.tiff
```

> The `-t` flag selects the representation.
> New RTFs can be added by implementing a `calculate_<name>(signal, fs)` function in `src/calculate_rtf.py` and registering it in the `CALCULATE_RTF` dict.
> The rest of the pipeline (folder creation, `.tiff` export, resize) is handled generically.

---

## 📁 Final folder layout

After running the full pipeline, the relevant outputs are (compare with
[📂 Repository layout](#-repository-layout) to see what ships with the repo vs. what the pipeline produces):

```
dataset/
├── full_dataset/          ← step 1
│   ├── wav/                       
│   └── json/                        
├── good_quality/          ← step 3
│   ├── wav/                       
│   └── json/                        
├── poor_quality/          ← step 3
│   ├── wav/                       
│   └── json/                        
├── dataset/               ← steps 4 & 5
│   ├── audio/             # filtered, resampled fragments
│   ├── spectrogram/       # 224×224 STFT .tiff images
│   ├── scalogram/         # 224×224 CWT .tiff images (planned)
│   └── wsst/              # 224×224 WSST .tiff images (planned)
├── metadata.csv           ← step 2
└── fragments_metadata.csv ← step 4
```

---

## 📓 Notebooks

The dataset ships with three exploratory / visualization notebooks alongside the pipeline scripts (see [📂 Repository layout](#-repository-layout)). They are **read-only documentation and exploration** — running them does not modify the pipeline outputs. 

### [visualization_signal.ipynb](visualization_signal.ipynb) — Raw record + preprocessing

Visualizes one full record end-to-end: loads a `.wav` and its companion `.json` from `good_quality/`, applies the pipeline's preprocessing, and plots the time-domain signal (with event annotations) and the PSD before vs. after preprocessing. Useful for sanity-checking the filter and resampling steps on a single record.

### [visualization_fragment.ipynb](visualization_fragment.ipynb) — Single fragment + STFT

Visualizes a single fragment from `dataset/audio/`: plots the raw waveform and the STFT spectrogram that the pipeline writes to `spectrogram/<id>.tiff`, both as an interactive heatmap and as the final 224×224 preview.

### [EDA.ipynb](EDA.ipynb) — Exploratory Data Analysis

Exploratory analysis of the prepared fragment dataset using `fragments_metadata.csv`. The notebook examines four aspects of the dataset:
- the **distribution of respiratory-sound labels**, 
- fragment **durations**,
- the **number of fragments** contributed by each patient,
- and the **number of distinct labels** observed per patient.

Each section includes visualizations and descriptive statistics to characterize the dataset and identify relevant patterns.

#### 📝 Results

- **Label distribution:**

|     Category     | Proportion | Relative fraction |
|:----------------:|:----------:|:----------:|
| **Normal**       |            |            |
|      Normal      |    76.4%   |    100%    |
| **Adventitious** |            |            |
|   Fine Crackle   |    14.4%   |    60.8%   |
|      Wheeze      |    6.1%    |    25.9%   |
|  Wheeze+Crackle  |    1.2%    |    5.22%   |
|      Ronchi      |    0.9%    |    3.74%   |
|  Coarse Crackle  |    0.7%    |    3.05%   |
|      Stridor     |    0.3%    |    1.27%   |

- **Fragment duration:**

| Metric | Duration (s) |
|:------:|:------------:|
|  Mean  |     1.74     |
|   std  |     0.75     |
|   min  |     0.13     |
|   Q1   |     1.21     |
| Median |     1.68     |
|   Q3   |     2.18     |
|   max  |     9.27     |

- **Fragments per patient:** 

| Metric | Fragments per patient |
|:------:|:---------------------:|
|  Mean  |          25.7         |
|   std  |          30.1         |
|   min  |           1           |
|   Q1   |           8           |
| Median |           17          |
|   Q3   |           32          |
|   max  |          298          |

- **Labels per patient:**

| Number of labels per patient | Count | Proportion (%) |
|:----------------------------:|:-----:|:--------------:|
|               1              |  650  |      67.85     |
|               2              |  212  |      22.13     |
|               3              |   68  |      7.10      |
|               4              |   21  |      2.19      |
|               5              |   5   |      0.52      |
|               6              |   2   |      0.21      |

| Metric | Number of labels per patient |
|:------:|:----------------------------:|
|  Mean  |              1.4             |
|   std  |              0.8             |
|   min  |               1              |
|   Q1   |               1              |
| Median |               1              |
|   Q3   |               2              |
|   max  |               6              |

### ▶️ How to run

From the `dataset/` folder, launch Jupyter / VS Code and open the notebook of interest, then run all cells in order.
