# Industrial Bottle Inspection System (Binary)

Code and notebooks for a binary bottle classifier (reusable vs not reusable) built for the [1st Krones Vision AI Challenge](https://www.kaggle.com/competitions/1st-krones-vision-ai-challenge) on Kaggle. Images are cropped to a bottle ROI when annotations are available, then classified with timm models.

This repo ships **code and notebooks only**. Competition images, labels, annotations, checkpoints, and submission CSVs are gitignored.

## Table of contents

1. [Overview](#overview)
2. [Problem and pipeline](#problem-and-pipeline)
3. [Dataset layout](#dataset-layout)
4. [Training (v18)](#training-v18)
5. [Inference and submission](#inference-and-submission)
6. [Earlier scripts](#earlier-scripts)
7. [Repository structure](#repository-structure)
8. [Setup](#setup)
9. [How to run](#how-to-run)
10. [Kaggle notebooks](#kaggle-notebooks)
11. [Metrics note](#metrics-note)
12. [Limitations](#limitations)
13. [License and data](#license-and-data)

## Overview

- **Task**: binary classification (reusable vs non-reusable)
- **Input**: RGB images, optional COCO-style ROI boxes
- **Cropping**: bottle ROI from annotations (category **22** in training notebooks); if missing, use the full image
- **Labels** (main scripts): `target` in `train.csv`, **1 = reusable**, **0 = not_reusable** (see `prepare_data.py`, `calculate_f1.py`, and softmax index `[:, 1]`)
- **Metric**: F1 on thresholded probabilities (threshold is searched, not fixed at 0.5)
- **Active local path**: **v18**, dual-backbone (`convnext_large` + `tf_efficientnetv2_m`), OOF threshold file, multi-checkpoint TTA ensemble
- **Notebooks**: PyTorch checkpoints can be exported to ONNX for the competition eval flow

Older scripts (`train_model_v7.py` through `v17`, plus `inference_v15`/`v16`) stay in the tree for reference. Prefer **v18** for new runs.

## Problem and pipeline

Returnable-bottle lines need a visual check for reusability. Full conveyor frames include background that is not useful for that decision, so the flow is:

1. Load the image
2. Crop to the bottle ROI when available
3. Classify the crop
4. Write `submission.csv` with columns `[image_id, target]`

Kaggle notebooks also keep a **12-hour GPU** budget in mind (train + infer + ONNX), so model size, epochs, and TTA depth are tradeoffs against runtime.

v18 local flow:

```
train.csv + train_images + train_annotations.json
        |
        v
   ROI crop --> augment / normalize --> timm 2-class head
        |
        +-- Stratified K-fold (default 3 folds)
        |     EMA, MixUp/CutMix, soft-target focal loss
        |     F1-optimal threshold per fold --> OOF probs
        |
        +-- Optional full-data fine-tune
        +-- Optional pseudo-label rounds (--use-pseudo)
        |
        v
ensemble_threshold_v18.txt + fold/full checkpoints
        |
        v
test images + ROI annotations
        |
        v
multi-checkpoint TTA --> avg P(reusable) --> threshold --> submission_v18.csv
```

Other paths in the repo:

- **YOLO-cls layout**: `prepare_data.py` builds `yolo_dataset/{train,val}/{reusable,not_reusable}/`; `calculate_f1.py` scores an Ultralytics checkpoint
- **Pseudo labels**: `generate_pseudo_labels.py` (older folds) and v18's `run_pseudo_inference` write `pseudo_labels.csv`
- **Sweep**: `sweep_v18_submissions.py` reuses or rebuilds test probabilities and writes several threshold/variant CSVs

## Dataset layout

Data is **not** in this repo. Local scripts assume:

```text
BASE_DIR = D:\CV
```

Expected layout:

```text
D:\CV\
├── images\
│   ├── train_images\
│   ├── test_images\
│   ├── train.csv
│   ├── sample_submission.csv
│   ├── train_annotations.json
│   ├── test_annotations_roi_only.json
│   └── bottletypes.csv            # optional; eval notebook diagnostics
├── yolo_dataset\                  # from prepare_data.py (gitignored)
├── runs\classify\krones_challenge\
├── pseudo_labels.csv              # optional
└── venv312\
```

Kaggle notebooks use `/kaggle/input/...` (for example `data_dir`: `/kaggle/input/krones-data/images` in `krones_eval_notebook.ipynb`). Change notebook `CONFIG` if your dataset slug differs.

**ROI notes**

- Training notebooks use COCO **category_id == 22** for bottle ROIs.
- Local `train_model_v18.py` / `inference_v18.py` `load_rois()` merge boxes from the train and test annotation files without re-filtering category in that helper.
- Datasets crop `[x, y, w, h]` when present; otherwise they use the full image.
- Collate helpers skip missing test files instead of crashing.

`sweep_v18_submissions.py` sets `TRAIN_POS_RATE = 0.583` for a train-rate-matched threshold option. That is a calibration constant in code, not a published dataset card.

## Training (v18)

Script: `train_model_v18.py`

Defaults worth knowing:

- Models: `convnext_large` @ **320**, `tf_efficientnetv2_m` @ **384** (timm)
- 3-fold `StratifiedKFold` (`random_state=42`), 40 fold epochs, 15 full-data epochs
- AdamW (`lr=3e-5`, `weight_decay=0.08` on folds), linear warmup then cosine
- MixUp/CutMix, soft-target focal loss, EMA (`ModelEmaV2`), CUDA AMP
- Large models: batch 4, accum 4; EfficientNetV2-M: batch 8, accum 2
- Early stop patience 12 on folds; early advance if fold F1 reaches `TARGET_F1 = 0.9735` (an in-training target, not a reported test score)
- Pause/resume via `pause_training.flag` and `last.pth` / `last_opt.pth`

Train augments include resize, random crop, flips, rotation, color jitter, perspective, autocontrast, sharpness, random erasing, and ImageNet normalization. Val uses resize, center crop, and ImageNet normalization.

```bash
python train_model_v18.py
python train_model_v18.py --model convnext_large --model2 tf_efficientnetv2_m --folds 3 --epochs 40
python train_model_v18.py --use-pseudo --pseudo-rounds 2
python train_model_v18.py --full-only
python train_model_v18.py --no-full --no-infer
```

Checkpoints:

```text
runs/classify/krones_challenge/{model}_v18_fold_{k}/best.pth
runs/classify/krones_challenge/{model}_v18_full/best_full.pth
```

OOF probs and the ensemble threshold go to `oof_probs_v18.npz` and `ensemble_threshold_v18.txt` (gitignored).

**v17**: `train_model_v17.py` / `inference_v17.py` are the single-model ConvNeXt-Large predecessors (plus `.bat` helpers).

## Inference and submission

### v18 (`inference_v18.py`)

1. Find fold `best.pth` and optional `best_full.pth` for `--model` and `--model2`
2. Load test ROIs and crop when available
3. Per checkpoint, run TTA (default **12** views), average, then average across checkpoints
4. Threshold from `ensemble_threshold_v18.txt` if present, else **0.5**
5. Write `submission_v18.csv` (or `--output`)

```bash
python inference_v18.py
python inference_v18.py --model convnext_large --model2 convnext_large
python inference_v18.py --tta 4 --no-full
```

Training can auto-run inference unless `--no-infer` is set.

### Sweep (`sweep_v18_submissions.py`)

```bash
python sweep_v18_submissions.py
python sweep_v18_submissions.py --from-npz
```

Writes several `submission_v18_*_*.csv` variants. Default TTA is 12.

### YOLO helpers

```bash
python prepare_data.py
python calculate_f1.py
```

These also assume `D:\CV`.

## Earlier scripts

| Script | Role |
|--------|------|
| `train_model_v7.py` / `v8.py` | Train `convnext_base` |
| `train_model_v9.py` | Multi-arch: `convnext_base`, `convnext_large`, `tf_efficientnetv2_m` |
| `inference_v15.py` | Weighted fusion across older fold runs |
| `inference_v16.py` | timm + optional Ultralytics YOLO fusion (`--no-yolo` to disable) |
| `generate_pseudo_labels.py` | Pseudo labels from older `{model}_v6_fold_*` checkpoints |
| `prepare_data.py` | YOLO-cls folder dataset (80/20 stratified split) |
| `calculate_f1.py` | Val F1 / precision / recall / accuracy for a YOLO-cls `best.pt` |

`starter-training-notebook.ipynb` is a MobileNetV3-Small baseline with ROI crop, BCE-with-logits training, val metrics, and ONNX export. Its single-logit label wording may differ from the two-class `target` / softmax `[:, 1]` path used by the main scripts. Follow the convention of the entrypoint you run.

## Repository structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── train_model_v18.py
├── inference_v18.py
├── sweep_v18_submissions.py
├── train_model_v17.py / inference_v17.py (+ .bat)
├── train_model_v9.py / v8.py / v7.py
├── inference_v16.py / inference_v15.py
├── prepare_data.py / calculate_f1.py / generate_pseudo_labels.py
├── env_check.py / keep_awake.py
├── activate.ps1 / run.ps1 / pause_training.ps1 / resume_training.ps1
├── scripts/terminal_init.ps1
├── krones_train_infer_notebook.ipynb
├── krones_eval_notebook.ipynb
└── starter-training-notebook.ipynb
```

Not in git: images, CSVs, annotation payloads, weights, `runs/`, `yolo_dataset/`, virtualenvs.

**Stack** (from `requirements.txt` and imports): `torch` / `torchvision` / `timm`, optional `ultralytics`, `opencv-python`, `pillow`, `numpy`, `pandas`, `polars`, `scikit-learn`, `scipy`, `matplotlib`, `tqdm`, `rich`. Notebooks use ONNX export (`onnxscript` / `torch.onnx.export`). Local helpers target Python **3.12** (`venv312`) on Windows. `env_check.py` expects that venv and can require CUDA.

## Setup

- Python 3.12 recommended
- NVIDIA GPU with enough VRAM for ConvNeXt-Large at 320px (batch 4 + grad accum 4 is the usual pattern)
- Competition data placed locally, or attached on Kaggle
- Edit `BASE_DIR` if you are not on `D:\CV`

```bash
python -m venv venv312
.\activate.ps1
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA PyTorch nightly (`torch==2.12.0.dev20260408+cu128` and matching vision/audio). If those wheels are unavailable, install a compatible torch pair first, then the rest.

Minimal set for the main classifiers:

```bash
pip install torch torchvision timm pandas scikit-learn pillow tqdm
pip install ultralytics   # YOLO path
pip install onnxscript    # Kaggle ONNX export
```

```bash
python env_check.py
# or
.\run.ps1 env_check.py
```

Update `BASE_DIR` in the scripts you run (`train_model_v18.py`, `inference_v18.py`, `sweep_v18_submissions.py`, `prepare_data.py`, `calculate_f1.py`, `generate_pseudo_labels.py`, and related helpers) before training elsewhere.

## How to run

```bash
# optional YOLO layout
python prepare_data.py

# train (PowerShell helper uses venv312)
.\run.ps1 train_model_v18.py
# or
python train_model_v18.py --folds 3 --epochs 40 --full-epochs 15

# pause at end of epoch / resume
.\pause_training.ps1
.\resume_training.ps1
python train_model_v18.py

# infer / sweep
python inference_v18.py --tta 12
python sweep_v18_submissions.py

# YOLO val metrics (if you trained Ultralytics cls)
python calculate_f1.py
```

`keep_awake.py` can keep Windows from sleeping on long runs. Training also uses an in-process sleep-prevention context on Windows.

## Kaggle notebooks

| Notebook | Purpose |
|----------|---------|
| `starter-training-notebook.ipynb` | MobileNetV3-Small baseline: ROI, metrics, ONNX |
| `krones_train_infer_notebook.ipynb` | ConvNeXt-Large train/infer, TTA, ONNX, `submission.csv` |
| `krones_eval_notebook.ipynb` | Similar train/infer with 12h budget controls and optional `bottletypes.csv` diagnostics |

Example defaults from `krones_train_infer_notebook.ipynb`: `convnext_large`, imgsz 320, batch 4 / accum 4, max_epochs 12, patience 4, val_fraction 0.08, lr `3e-5`, weight_decay `0.08`, tta_n 4, max_runtime_hours 12, reserve_infer_hours 1.5.

Enable **GPU** (and **Internet** if you need pretrained weights or ONNX installs). The train+infer notebook exports a reusable-class logit (`input` / `logits`, opset 18, dynamic batch).

## Metrics note

The code reports binary F1 via `sklearn.metrics.f1_score` after a probability threshold search (`find_best_threshold`). Training logs loss and F1 at the F1-optimal threshold; fold thresholds land in `optimal_threshold.txt`; the ensemble threshold file is used at inference. `calculate_f1.py` also prints accuracy, precision, and recall for YOLO-cls val folders. Notebooks track F1 and wall-clock under the runtime guard above.

This repository does **not** include leaderboard scores, private/public F1 numbers, or committed experiment tables. `TARGET_F1 = 0.9735` is only an early-advance target in training. Record validation F1 from your own logs if you need numbers.

## Limitations

1. Most scripts hardcode `BASE_DIR = r"D:\CV"` and Windows `venv312\Scripts\python.exe`.
2. No data in git: you need competition (or licensed) data separately.
3. No shipped metrics or weights.
4. Label conventions differ slightly between the MobileNet starter (single logit) and the main timm/YOLO path (class 1 = reusable). Match the entrypoint you use.
5. Legacy v15/v16/v7-v9 paths may expect older checkpoint directory names.
6. ConvNeXt-Large + EfficientNetV2-M with 12-view TTA is GPU-heavy; lower TTA, `--no-full`, or a smaller backbone may be needed.
7. Dataset use is subject to Kaggle competition rules; do not redistribute withheld data.

## License and data

Code here is for educational and portfolio use. Competition data, weights trained on challenge images, and submission files are excluded via `.gitignore`. The dataset is proprietary under Kaggle confidentiality rules and is omitted from this repository.

## Quick reference

```bash
.\activate.ps1
python env_check.py
python train_model_v18.py
python inference_v18.py
python sweep_v18_submissions.py
```

On Kaggle: open `krones_train_infer_notebook.ipynb` or `krones_eval_notebook.ipynb`, attach the dataset, enable GPU, and run through ONNX / submission write.
