"""
v16 ensemble: timm (no RepViT) + YOLO + rank fusion + optional logistic calibration.
Run: python evaluate_v16.py  then  python inference_v16.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from env_check import require_venv312
from ensemble_lib import (
    BASE_DIR,
    CALIBRATOR_FILE,
    ImageDataset,
    apply_logistic_calibrator,
    collate_pil,
    discover_timm_models,
    discover_yolo_models,
    load_calibrator,
    load_rois,
    make_transform,
    predict_timm,
    predict_yolo,
    quantile_threshold,
    train_positive_ratio,
    weighted_rank_fuse,
)
import timm

PROBS_FILE = os.path.join(BASE_DIR, "ensemble_v16_probs.npz")


def fuse_mean(scores, weights):
    wsum = sum(weights)
    out = np.zeros_like(scores[0])
    for s, w in zip(scores, weights):
        out += s * w
    return out / wsum


def collect_scores(df, images_dir, img_name_to_bbox, device, n_tta, use_yolo: bool):
    n = len(df)
    all_scores, all_weights = [], []

    for cfg in discover_timm_models(include_repvit=False):
        print(f"\ntimm {cfg['arch']} {cfg['version']} fold {cfg['fold']} (w={cfg['weight']})")
        model = timm.create_model(cfg["arch"], pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(cfg["path"], map_location=device, weights_only=True))
        model.to(device)
        out = np.zeros(n, dtype=np.float64)
        ds = ImageDataset(df, images_dir, img_name_to_bbox)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_pil)
        predict_timm(
            model, loader, make_transform(cfg["version"], cfg["arch"], cfg["imgsz"]),
            device, n_tta, cfg["thresh"], out, cfg["weight"], True,
        )
        all_scores.append(out.copy())
        all_weights.append(cfg["weight"])
        del model
        torch.cuda.empty_cache()

    if use_yolo:
        from ultralytics import YOLO
        for cfg in discover_yolo_models():
            print(f"\nyolo {cfg['run']} fold {cfg['fold']} (w={cfg['weight']})")
            model = YOLO(cfg["path"])
            out = np.zeros(n, dtype=np.float64)
            predict_yolo(model, df, images_dir, img_name_to_bbox, 0, out, cfg["weight"], min(4, n_tta))
            all_scores.append(out.copy())
            all_weights.append(cfg["weight"])
            del model
            torch.cuda.empty_cache()

    return all_scores, all_weights


def finalize_probs(all_scores, all_weights, cal: dict) -> np.ndarray:
    use_rank = "rank" in cal.get("mode", "rank")
    probs = weighted_rank_fuse(all_scores, all_weights) if use_rank else fuse_mean(all_scores, all_weights)
    if "logistic" in cal.get("mode", ""):
        probs = apply_logistic_calibrator(probs, cal.get("coef", 1.0), cal.get("intercept", 0.0))
    return probs


def main():
    require_venv312(require_cuda=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", type=int, default=8)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "submission_v16.csv"))
    parser.add_argument("--from-probs", action="store_true")
    parser.add_argument("--probs-only", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(BASE_DIR, "images", "sample_submission.csv"))
    test_dir = os.path.join(BASE_DIR, "images", "test_images")
    roi = load_rois("test_annotations_roi_only.json")

    if os.path.exists(CALIBRATOR_FILE):
        cal = load_calibrator(CALIBRATOR_FILE)
        print(f"Calibrator: {cal}")
    else:
        cal = {"mode": "rank_grid", "threshold": 0.5, "coef": 0.0, "intercept": 0.0}
        print("No ensemble_calibrator.json — run evaluate_v16.py first (using rank + grid threshold)")

    if args.from_probs:
        data = np.load(PROBS_FILE)
        use_rank = "rank" in cal.get("mode", "rank")
        probs = data["rank_probs"] if use_rank else data["mean_probs"]
        if "logistic" in cal.get("mode", ""):
            probs = apply_logistic_calibrator(probs, cal.get("coef", 1.0), cal.get("intercept", 0.0))
    else:
        require_venv312(require_cuda=True)
        device = torch.device("cuda")
        print(f"Device: {device} | TTA={args.tta} | YOLO={not args.no_yolo}")
        all_scores, all_weights = collect_scores(
            df, test_dir, roi, device, args.tta, use_yolo=not args.no_yolo,
        )
        mean_p = fuse_mean(all_scores, all_weights)
        rank_p = weighted_rank_fuse(all_scores, all_weights)
        np.savez(PROBS_FILE, mean_probs=mean_p, rank_probs=rank_p, image_id=df["image_id"].values)
        print(f"Saved {PROBS_FILE}")
        probs = finalize_probs(all_scores, all_weights, cal)

    if args.probs_only:
        return

    if "quantile" in cal.get("mode", ""):
        t = quantile_threshold(probs, train_positive_ratio())
    else:
        t = float(cal.get("threshold", 0.5))

    df["target"] = (probs >= t).astype(int)
    df.to_csv(args.output, index=False)
    print(f"\nThreshold: {t:.4f} | reusable: {df['target'].mean()*100:.2f}%")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
