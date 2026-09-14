# Industrial Bottle Inspection System (Binary)

Automated visual inspection pipeline for high-speed industrial packaging lines, developed for the [1st Krones Vision AI Challenge](https://www.kaggle.com/competitions/1st-krones-vision-ai-challenge) on Kaggle. The system classifies bottle images as **reusable** or **non-reusable** using region-of-interest (ROI) crops from conveyor-line imagery. That mirrors real quality-control workflows where only the product region matters and decisions must be fast, accurate, and deployable.

This repository is **code and notebooks only**. Competition images, labels, annotations, checkpoints, and submission CSVs are gitignored and are not shipped here.

---

## Table of contents

1. [Overview](#overview)
2. [Problem statement](#problem-statement)
3. [Pipeline](#pipeline)
4. [Dataset expectations](#dataset-expectations)
5. [Models and training](#models-and-training)
6. [Inference and submission](#inference-and-submission)
7. [Metrics and calibration](#metrics-and-calibration)
8. [Repository structure](#repository-structure)
9. [Tech stack](#tech-stack)
10. [Setup and installation](#setup-and-installation)
11. [How to run](#how-to-run)
12. [Kaggle notebooks](#kaggle-notebooks)
13. [Results](#results)
14. [Limitations and unknowns](#limitations-and-unknowns)
15. [License and data notice](#license-and-data-notice)

---

## Overview

The project implements a **binary ROI image classifier** for industrial packaging inspection:

| Aspect | Detail |
|--------|--------|
| Task | Binary classification: reusable vs non-reusable containers |
| Input | RGB images plus optional COCO-style ROI boxes |
| Cropping | Bottle ROI from annotations (category **22** in training notebooks); missing ROI falls back to the full image |
| Labels (main scripts) | `target` in `train.csv`: **1 = reusable**, **0 = not_reusable** (see `prepare_data.py`, `calculate_f1.py`, and softmax index `[:, 1]` in training/inference) |
| Primary metric | **F1-score** on thresholded probabilities (not a fixed 0.5 cutoff) |
| Current local line | **v18**: dual-backbone training (`convnext_large` + `tf_efficientnetv2_m`), OOF threshold file, multi-checkpoint TTA ensemble inference |
| Deployment path (notebooks) | PyTorch checkpoints exported to **ONNX** for the competition evaluation notebook |

The codebase evolved through multiple experiment generations (`train_model_v7.py` through `train_model_v18.py`, plus matching inference scripts). Older scripts remain for reference (including YOLO-assisted fusion in `inference_v16.py`). The active dual-model path is **v18**.

---

## Problem statement

Industrial returnable-bottle lines need automated visual checks that decide whether a container is reusable. Full-frame conveyor images contain background and line structure that are irrelevant to that decision. The competition (and this codebase) therefore:

1. Load each image.
2. Crop to a provided bottle ROI when available.
3. Classify the crop as reusable or not.
4. Produce a competition `submission.csv` with columns `[image_id, target]`.

Notebooks also stress a **12-hour GPU runtime budget** on Kaggle (train + infer + ONNX), so configuration trades model size, epochs, and TTA depth against wall-clock time.

---

## Pipeline

High-level flow used by the local v18 scripts:

```
train.csv + train_images + train_annotations.json
        |
        v
   ROI crop (COCO bbox) --> augment / normalize --> timm 2-class head
        |
        +-- Stratified K-fold training (default 3 folds)
        |     EMA weights, MixUp/CutMix, soft-target focal loss
        |     F1-optimal threshold per fold --> OOF probs
        |
        +-- Optional full-data fine-tune (init from averaged fold weights)
        |
        +-- Optional pseudo-label rounds (--use-pseudo)
        |
        v
ensemble_threshold_v18.txt + fold/full checkpoints
        |
        v
test_images + test_annotations_roi_only.json
        |
        v
multi-checkpoint TTA --> average P(reusable) --> threshold --> submission_v18.csv
```

Supporting side paths:

- **YOLO classification layout**: `prepare_data.py` builds `yolo_dataset/{train,val}/{reusable,not_reusable}/`; `calculate_f1.py` scores an Ultralytics YOLO-cls checkpoint.
- **Pseudo labels**: `generate_pseudo_labels.py` (older v6-fold ensemble) and v18's built-in `run_pseudo_inference` write `pseudo_labels.csv`; high/low confidence rows can be mixed into training.
- **Sweep**: `sweep_v18_submissions.py` rebuilds or reuses test probabilities and writes multiple threshold/variant CSVs without full retraining.

---

## Dataset expectations

Competition data is **not in this repo** (see `.gitignore`). Local scripts assume a Windows project root hardcoded as:

```text
BASE_DIR = D:\CV
```

Expected layout under that root (as referenced by the Python scripts):

```text
D:\CV\
├── images\
│   ├── train_images\              # training RGB images
│   ├── test_images\               # test RGB images
│   ├── train.csv                  # columns include image_id, target
│   ├── sample_submission.csv      # image_id (+ target placeholder)
│   ├── train_annotations.json     # COCO-style ROI annotations
│   ├── test_annotations_roi_only.json
│   └── bottletypes.csv            # optional; used in eval notebook diagnostics
├── yolo_dataset\                  # produced by prepare_data.py (gitignored)
├── runs\classify\krones_challenge\  # checkpoints (gitignored)
├── pseudo_labels.csv              # optional (gitignored)
└── venv312\                       # local CUDA venv (not in git)
```

Kaggle notebooks use paths under `/kaggle/input/...` (for example `data_dir`: `/kaggle/input/krones-data/images` in `krones_eval_notebook.ipynb`). Update notebook `CONFIG` if your dataset slug differs.

### ROI notes

- Training notebooks load bottle ROIs with **COCO category_id == 22**.
- Local `train_model_v18.py` / `inference_v18.py` `load_rois()` merge bboxes from `train_annotations.json` and `test_annotations_roi_only.json` without re-filtering category in that helper (they trust the provided annotation files).
- `KronesDataset` / test datasets crop `[x, y, w, h]` when present; otherwise the full image is used.
- Collate helpers skip missing files during test inference instead of crashing.

### Class balance hint in code

`sweep_v18_submissions.py` defines `TRAIN_POS_RATE = 0.583` for a train-rate-matched threshold option. Treat that as a calibration constant present in code, not as a published dataset card.

---

## Models and training

### Current line: v18 (`train_model_v18.py`)

Defaults:

| Setting | Default |
|---------|---------|
| Model 1 | `convnext_large` (timm), image size **320** |
| Model 2 | `tf_efficientnetv2_m` (timm), image size **384** |
| Folds | 3 (`StratifiedKFold`, `random_state=42`) |
| Fold epochs | 40 |
| Full-data epochs | 15 |
| Optimizer | AdamW (`lr=3e-5`, `weight_decay=0.08` on folds) |
| Schedule | Linear warmup then cosine annealing |
| Regularization | `drop_rate` / `drop_path_rate`, MixUp/CutMix, soft-target focal loss |
| EMA | `timm.utils.ModelEmaV2` |
| Precision | CUDA AMP + GradScaler |
| Batch / accum | Large models: batch 4, accum 4; `tf_efficientnetv2_m`: batch 8, accum 2 |
| Early stop | Patience 12 on folds; early advance if fold F1 reaches `TARGET_F1 = 0.9735` |
| Pause/resume | `pause_training.flag` + `last.pth` / `last_opt.pth` |

Augmentations (train): resize oversized, random crop, H/V flips, rotation, color jitter, perspective, autocontrast, sharpness, random erasing, ImageNet normalization.

Validation transforms: resize, center crop, ImageNet normalization.

CLI highlights:

```bash
python train_model_v18.py
python train_model_v18.py --model convnext_large --model2 tf_efficientnetv2_m --folds 3 --epochs 40
python train_model_v18.py --use-pseudo --pseudo-rounds 2
python train_model_v18.py --full-only
python train_model_v18.py --no-full --no-infer
```

Checkpoints land under:

```text
runs/classify/krones_challenge/{model}_v18_fold_{k}/best.pth
runs/classify/krones_challenge/{model}_v18_full/best_full.pth
```

OOF probabilities and a global ensemble threshold are written as `oof_probs_v18.npz` and `ensemble_threshold_v18.txt` (gitignored).

### Predecessor: v17

`train_model_v17.py` / `inference_v17.py` are the single-model ConvNeXt-Large oriented predecessors (same versioning pattern with `_v17_` run dirs). Batch helpers: `train_model_v17.bat`, `inference_v17.bat`.

### Earlier generations (legacy, still in repo)

| Script | Role (from code) |
|--------|------------------|
| `train_model_v7.py` / `v8.py` | Train `convnext_base` |
| `train_model_v9.py` | Multi-arch: `convnext_base`, `convnext_large`, `tf_efficientnetv2_m` |
| `inference_v15.py` | Weighted discovery/fusion across v6/v7/v9 fold runs (includes `repvit_m1_0` weight map) |
| `inference_v16.py` | timm + optional **Ultralytics YOLO** score fusion (`--no-yolo` to disable) |
| `generate_pseudo_labels.py` | Pseudo labels from older `{model}_v6_fold_*` checkpoints |
| `prepare_data.py` | Build YOLO-cls folder dataset (80/20 stratified split) |
| `calculate_f1.py` | Val F1/precision/recall/accuracy for a YOLO-cls `best.pt` |

### Starter baseline notebook

`starter-training-notebook.ipynb` is the official-style **MobileNetV3-Small** baseline with ROI cropping, BCE-with-logits training, validation metrics (accuracy, AUC, F1, precision, recall), and **ONNX export**. Note: its markdown describes a single binary logit label convention that may differ from the two-class `target` mapping used in the YOLO prep scripts and timm softmax `[:, 1]` path. Prefer the label convention of whichever training entrypoint you actually run.

---

## Inference and submission

### v18 ensemble (`inference_v18.py`)

1. Discover fold `best.pth` and optional `best_full.pth` for `--model` and `--model2`.
2. Load test ROI map; crop when available.
3. For each checkpoint, run TTA (default **12** views: identity, flips, small rotations, 90/180/270, zoom crop), average view probabilities, accumulate across checkpoints.
4. Apply threshold from `ensemble_threshold_v18.txt` if present, else **0.5**.
5. Write `submission_v18.csv` (or `--output`).

```bash
python inference_v18.py
python inference_v18.py --model convnext_large --model2 convnext_large   # ConvNeXt-only
python inference_v18.py --tta 4 --no-full
```

`train_model_v18.py` can also auto-run inference after training unless `--no-infer` is set (`--infer-tta`, `--infer-output`).

### Threshold / variant sweep (`sweep_v18_submissions.py`)

```bash
python sweep_v18_submissions.py              # infer, save probs, write CSVs
python sweep_v18_submissions.py --from-npz   # reuse test_probs_v18.npz
```

Writes multiple `submission_v18_*_*.csv` variants (including ConvNeXt-only and OOF-best threshold choices). Default TTA for the sweep is 12.

### YOLO-era helpers

```bash
python prepare_data.py      # build yolo_dataset from train.csv
python calculate_f1.py      # score YOLO-cls val folders
```

Paths inside these helpers also assume `D:\CV`.

---

## Metrics and calibration

What the code optimizes and reports:

- **Binary F1** via `sklearn.metrics.f1_score` after searching a probability threshold (`find_best_threshold`, typically 199 steps).
- Validation during training prints loss and F1 at the F1-optimal threshold for that epoch.
- OOF fold F1 and thresholds are saved under each fold run (`optimal_threshold.txt`).
- Ensemble threshold file aggregates calibration for inference.
- `calculate_f1.py` additionally prints accuracy, precision, and recall for YOLO-cls validation folders.
- Kaggle notebooks score both **performance (F1)** and **efficiency (wall-clock)** under a max runtime guard (`max_runtime_hours: 12`, `reserve_infer_hours: 1.5` in notebook CONFIG).

**Important:** This repository does **not** contain logged leaderboard scores, final private/public F1 numbers, or committed experiment tables. `TARGET_F1 = 0.9735` is an in-training early-advance target in `train_model_v18.py`, not a claim that the system achieved that score on the hidden test set. Do not treat it as a published result.

---

## Repository structure

```text
.
├── README.md                          # this file
├── .gitignore                         # excludes data, weights, CSVs, venvs, runs
├── requirements.txt                   # pinned local environment snapshot
│
├── train_model_v18.py                 # dual-model k-fold + full fine-tune + optional pseudo
├── inference_v18.py                   # multi-checkpoint TTA ensemble inference
├── sweep_v18_submissions.py           # threshold / architecture submission sweeps
├── train_model_v17.py                 # v17 single-model line
├── inference_v17.py
├── train_model_v17.bat
├── inference_v17.bat
├── train_model_v9.py                  # earlier multi-arch training
├── train_model_v8.py
├── train_model_v7.py
├── inference_v16.py                   # timm + optional YOLO fusion
├── inference_v15.py                   # weighted multi-version fold fusion
│
├── prepare_data.py                    # YOLO-cls folder dataset builder
├── calculate_f1.py                    # YOLO-cls validation metrics
├── generate_pseudo_labels.py          # older pseudo-label generator
├── env_check.py                       # require venv312 + optional CUDA
├── keep_awake.py                      # Windows sleep prevention helper
│
├── activate.ps1                       # activate venv312
├── run.ps1                            # run a script with venv312 Python
├── pause_training.ps1                 # create pause_training.flag
├── resume_training.ps1                # clear pause flag
├── scripts/
│   └── terminal_init.ps1              # auto-activate venv in new terminals
│
├── krones_train_infer_notebook.ipynb  # Kaggle: ConvNeXt-Large train + infer + ONNX
├── krones_eval_notebook.ipynb         # Kaggle: train/eval with runtime budget + bottle-type diag
└── starter-training-notebook.ipynb    # MobileNetV3-Small starter baseline + ONNX
```

Not present in git (by design): images, `train.csv` / submission CSVs, JSON annotation payloads used locally, `*.pth` / `*.pt` / ONNX weights, `runs/`, `yolo_dataset/`, virtualenvs.

---

## Tech stack

From `requirements.txt` and imports in the training/inference code:

| Area | Packages / tools |
|------|------------------|
| Deep learning | `torch`, `torchvision`, `torchaudio` (CUDA nightly pins in requirements), `timm` |
| Detection/cls helper | `ultralytics` (YOLO path) |
| Vision / data | `opencv-python`, `pillow`, `numpy`, `pandas`, `polars`, `scikit-learn`, `scipy` |
| Training utilities | MixUp/CutMix and EMA via timm; AMP via PyTorch |
| Plotting / UX | `matplotlib`, `tqdm`, `rich` |
| Hub / export related | `huggingface_hub`, `safetensors`; notebooks use `onnxscript` / `torch.onnx.export` |
| Orchestration | PowerShell helpers for Windows (`activate.ps1`, `run.ps1`, pause/resume) |
| Language | Python (scripts target a local **Python 3.12** `venv312` on Windows) |

`env_check.py` expects `...\venv312\Scripts\python.exe` and can require CUDA.

---

## Setup and installation

### Prerequisites

- Python 3.12 recommended (matches `venv312` helpers).
- NVIDIA GPU with enough VRAM for ConvNeXt-Large at 320px (notebooks use batch 4 with grad accum 4 as a reference; local large-model default is the same pattern).
- Competition dataset placed locally (or attached on Kaggle). Paths in scripts currently point at `D:\CV`.

### Create environment

```bash
python -m venv venv312
# Windows PowerShell
.\activate.ps1
# or
.\venv312\Scripts\Activate.ps1

pip install -r requirements.txt
```

`requirements.txt` pins a specific CUDA-enabled PyTorch build (`torch==2.12.0.dev20260408+cu128` and matching torchvision/torchaudio). If those wheels are unavailable on your platform, install a compatible torch/vision pair for your CUDA/OS first, then install the remaining packages.

Minimal conceptual set used by the main classifiers:

```bash
pip install torch torchvision timm pandas scikit-learn pillow tqdm
# YOLO legacy path
pip install ultralytics
# Kaggle ONNX export (notebooks)
pip install onnxscript
```

Verify:

```bash
python env_check.py
# or
.\run.ps1 env_check.py
```

### Path adaptation

Before training on a non-`D:\CV` machine, update `BASE_DIR` in the scripts you run (`train_model_v18.py`, `inference_v18.py`, `sweep_v18_submissions.py`, `prepare_data.py`, `calculate_f1.py`, `generate_pseudo_labels.py`, and related helpers), or symlink/copy data into the expected layout.

---

## How to run

### 1. Prepare data (optional YOLO path)

```bash
python prepare_data.py
```

### 2. Train v18

```bash
# PowerShell helper (uses venv312)
.\run.ps1 train_model_v18.py

# or directly after activate
python train_model_v18.py --folds 3 --epochs 40 --full-epochs 15
```

Graceful pause (end of epoch):

```powershell
.\pause_training.ps1
```

Clear pause and restart the same script to resume from `last.pth`:

```powershell
.\resume_training.ps1
python train_model_v18.py
```

`keep_awake.py` can prevent Windows sleep during long runs. Training itself also uses an in-process sleep-prevention context on Windows.

### 3. Inference

```bash
python inference_v18.py --tta 12
```

### 4. Sweep submissions

```bash
python sweep_v18_submissions.py
```

### 5. YOLO baseline metrics (if you trained Ultralytics cls)

```bash
python calculate_f1.py
```

---

## Kaggle notebooks

| Notebook | Purpose |
|----------|---------|
| `starter-training-notebook.ipynb` | MobileNetV3-Small baseline: ROI crop, train/val metrics, ONNX export |
| `krones_train_infer_notebook.ipynb` | ConvNeXt-Large end-to-end: ROI (cat 22), EMA/MixUp/focal, dual-checkpoint TTA, ONNX, `submission.csv` |
| `krones_eval_notebook.ipynb` | Similar ConvNeXt train/infer with explicit 12h budget controls and optional `bottletypes.csv` diagnostics |

Example notebook CONFIG defaults (`krones_train_infer_notebook.ipynb`):

- `model_name`: `convnext_large`
- `imgsz`: 320
- `batch_size`: 4, `grad_accum`: 4
- `max_epochs`: 12, `patience`: 4, `val_fraction`: 0.08
- `lr`: 3e-5, `weight_decay`: 0.08
- `tta_n`: 4
- `max_runtime_hours`: 12, `reserve_infer_hours`: 1.5

On Kaggle, enable **GPU** and **Internet** when you need timm pretrained weights and ONNX-related installs.

ONNX export in the train+infer notebook wraps the model so the exported graph exposes a reusable-class logit (`input` / `logits`, opset 18, dynamic batch axis).

---

## Results

### What is reproducible from this repo alone

- Training and inference **code paths** for v7 through v18.
- Notebook pipelines that, given competition data and a GPU, produce checkpoints, optional ONNX, and `submission.csv`.
- Local utilities for pause/resume, env checks, pseudo labels, and threshold sweeps.

### What is not in this repo

- Trained weights and ONNX files.
- Raw or processed competition images and annotation JSON payloads.
- Any committed public/private leaderboard score, confusion matrix, or final F1 table.

If you run training locally or on Kaggle, record validation F1 (and the threshold used) from the script/notebook logs yourself. Until those numbers are measured on your run, treat performance as **unknown** rather than implied by `TARGET_F1` or training-positive-rate constants.

---

## Limitations and unknowns

1. **Hardcoded Windows paths**: Most scripts use `BASE_DIR = r"D:\CV"` and Windows `venv312\Scripts\python.exe`. Linux/macOS users must edit paths and env helpers.
2. **No data in git**: You cannot train or evaluate from a fresh clone without separately obtaining competition (or licensed) data.
3. **No shipped metrics**: README intentionally does not invent leaderboard or holdout scores.
4. **Label-convention subtlety**: Main timm/YOLO-prep path treats class **1** as reusable; the MobileNet starter uses a single logit with its own documented 0/1 wording. Align preprocessing and metric code with the entrypoint you choose.
5. **Legacy scripts**: v15/v16/v7-v9 and YOLO fusion are retained for experimentation; defaults and weight maps may assume checkpoint directory names from older runs that you may not have.
6. **Hardware assumptions**: ConvNeXt-Large + EfficientNetV2-M ensembles and 12-view TTA are GPU-heavy; consumer GPUs may need lower TTA, `--no-full`, or smaller backbones (`convnext_base` is mentioned in eval-notebook comments as a faster alternative).
7. **Proprietary competition constraints**: Dataset usage is subject to Kaggle competition rules; do not redistribute withheld data or artifacts.

---

## License and data notice

Code in this repository is provided for educational and portfolio purposes. Competition data, weights trained on proprietary challenge images, and submission files are intentionally excluded via `.gitignore`.

The dataset for this project is proprietary and subject to Kaggle competition confidentiality rules. Raw data and source images are completely omitted from this repository.

---

## Quick reference

```bash
# env
.\activate.ps1
python env_check.py

# train + infer (v18)
python train_model_v18.py
python inference_v18.py

# sweeps / legacy
python sweep_v18_submissions.py
python inference_v16.py --no-yolo
```

For Kaggle: open `krones_train_infer_notebook.ipynb` or `krones_eval_notebook.ipynb`, attach the competition dataset, enable GPU (+ Internet as needed), and run all cells through ONNX export / submission write.
