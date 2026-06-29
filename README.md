# Industrial Computer Vision Quality Control

Automated visual inspection pipeline for high-speed industrial packaging lines, developed in the context of the [1st Krones Vision AI Challenge](https://www.kaggle.com/competitions/1st-krones-vision-ai-challenge) on Kaggle. The system classifies bottle images as **reusable** or **non-reusable** using region-of-interest (ROI) crops from conveyor-line imagery—mirroring real-world quality-control workflows where only the product region matters and decisions must be fast, accurate, and deployable.

---

## Overview

This repository contains training, inference, and Kaggle evaluation notebooks for a **binary ROI image classifier** aimed at industrial packaging inspection. Full-frame images are paired with COCO-style bounding boxes; each sample is cropped to the bottle region before classification. The primary metric is **F1-score** on thresholded predictions.

The codebase evolved through multiple experiment generations (v7–v18), from multi-model timm ensembles toward a streamlined **dual-backbone** pipeline (ConvNeXt-Large + EfficientNetV2-M) with out-of-fold threshold calibration, test-time augmentation (TTA), and ONNX export for competition evaluation.

**Problem framing (industrial QC):**

| Aspect | Detail |
|--------|--------|
| Task | Binary classification: reusable vs non-reusable containers |
| Input | RGB images + optional ROI boxes (category 22 in COCO annotations) |
| Output | Class label (0/1) and/or probability of reusable |
| Deployment path | PyTorch checkpoints → ONNX for evaluation notebooks |

---

## Architectures

### ConvNeXt (primary classifier)

**ConvNeXt-Large** (`convnext_large` via [timm](https://github.com/huggingface/pytorch-image-models)) is the main backbone in the current **v18** pipeline:

- Image size **320×320**, 3-fold cross-validation + full-data fine-tune
- ImageNet-pretrained weights, 2-class head
- Paired with **tf_efficientnetv2_m** (384px) in dual-model ensemble inference
- Training recipe: EMA (`ModelEmaV2`), MixUp/CutMix, soft-target focal loss, AdamW + cosine schedule, mixed precision (AMP)

Earlier generations also explored **ConvNeXt-V2-Tiny** and multi-fold ensembles with EfficientNet-V2-S for OOF calibration and weighted fusion.

### YOLO (detection-assisted ensemble)

**YOLO** (Ultralytics) appears in the **v16** inference stack as a complementary scorer:

- timm classifiers and YOLO detectors produce per-image scores
- **Weighted rank fusion** combines heterogeneous model outputs
- Optional logistic calibration and quantile-based thresholds
- ROI cropping is applied before both timm and YOLO forward passes

The v18 line focuses on pure classification; YOLO remains in legacy inference scripts as a reference for detection + classification fusion patterns common in line-speed QC systems.

### Supporting components

- **MobileNetV3-Small** — baseline in the official Kaggle starter training notebook (ONNX export pattern)
- **Multi-checkpoint ensembling** — fold models + full-data checkpoints, probability averaging
- **Dual-checkpoint inference** — average best-validation and last-epoch weights (Kaggle train+infer notebook)

---

## Engineering Highlights

### Data pipeline

- **ROI loading** from COCO JSON (`train_annotations.json`, `test_annotations_roi_only.json`); category 22 = bottle region
- **PyTorch `Dataset`** classes crop to `[x, y, w, h]` before transforms; missing ROI falls back to full image
- **Stratified splits** for validation and k-fold training; `bottletypes.csv` supports per-type diagnostics (not shipped in this repo)
- **Collate functions** skip missing files gracefully during test inference

### Augmentations

- **Training:** resize → random crop, flips, rotation, color jitter, perspective, autocontrast, sharpness, random erasing (torchvision v2)
- **Validation / inference:** resize → center crop, ImageNet normalization
- **TTA:** 4-view (flips), 8-view (+ small rotations), or 12-view (+ 90°/180°/270° + zoom crop)

### Training loop

- Gradient accumulation for effective batch size on consumer GPUs
- Early stopping on validation F1 with **F1-optimal threshold search** (not fixed 0.5)
- Resume-safe checkpoints (`last.pth`, `best.pth`), pause/resume via flag file
- GPU warmup and power-state checks for laptop training stability
- **12-hour runtime budget** guard for Kaggle GPU sessions (train + infer + ONNX export)

### Inference & submission

- Checkpoint discovery across fold and full-model directories
- Ensemble probability averaging → global or OOF-calibrated threshold → `submission.csv`
- **ONNX export** wrapper for competition evaluation notebooks
- Sweep utilities for threshold and ConvNeXt-only ablations without retraining

### Project layout (code only)

```
.
├── train_model_v18.py          # Dual-model k-fold + full fine-tune
├── inference_v18.py            # 8-checkpoint TTA ensemble inference
├── train_model_v17.py          # Single-model ConvNeXt predecessor
├── inference_v15.py / v16.py   # Legacy ensemble + YOLO fusion
├── sweep_v18_submissions.py    # Threshold / architecture sweeps
├── env_check.py                # CUDA venv guard
├── krones_train_infer_notebook.ipynb   # Kaggle starter-format train+infer+ONNX
├── krones_eval_notebook.ipynb          # Standalone eval notebook
└── scripts/                    # Shell helpers (activate, pause, resume)
```

---

## Setup

### Requirements

- Python 3.12 (CUDA-enabled PyTorch recommended)
- NVIDIA GPU with ≥8 GB VRAM for ConvNeXt-Large training
- Core packages: `torch`, `torchvision`, `timm`, `pandas`, `scikit-learn`, `Pillow`, `tqdm`

```bash
python -m venv venv312
# Windows
venv312\Scripts\activate
# Linux/macOS
source venv312/bin/activate

pip install torch torchvision timm pandas scikit-learn pillow tqdm
# Optional (YOLO legacy inference)
pip install ultralytics
# Optional (Kaggle ONNX export)
pip install onnxscript
```

### Data layout (local — not included in repo)

Place competition data locally in a generic structure:

```
data/
├── train/
│   ├── images/              # training images
│   ├── labels.csv           # columns: image_id, target
│   └── annotations.json     # COCO ROI annotations
└── test/
    ├── images/
    ├── sample_submission.csv
    └── annotations_roi.json
```

Update paths in training/inference scripts or notebook `CONFIG` to match your layout, e.g. `data/train/images/`.

---

## Usage

### Train (v18 dual-model)

```bash
python train_model_v18.py
# Options: --model convnext_large --model2 tf_efficientnetv2_m --folds 3 --epochs 40
```

Checkpoints are written under `runs/classify/krones_challenge/` (gitignored).

### Inference

```bash
python inference_v18.py
# ConvNeXt-only: --model convnext_large --model2 convnext_large
# Faster TTA: --tta 4
```

Output: `submission_v18.csv` (gitignored).

### Threshold sweep (no retrain)

```bash
python sweep_v18_submissions.py
```

### Kaggle notebooks

- **`krones_train_infer_notebook.ipynb`** — full train + TTA infer + ONNX export (starter format)
- **`krones_eval_notebook.ipynb`** — evaluation-focused notebook with runtime budget controls

Enable **GPU** and **Internet** on Kaggle for timm pretrained weights and ONNX export.

### Environment guard

```bash
python env_check.py   # ensures CUDA venv312 is active
```

---

## License

Code in this repository is provided for educational and portfolio purposes. Competition data, weights trained on proprietary data, and submission files are intentionally excluded.

---

Note: The dataset utilized for this project is proprietary and subject to Kaggle competition confidentiality rules; raw data and source images are completely omitted from this repository.
