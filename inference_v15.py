"""
Test inference v15 — fixes LB gap vs validation:
  - Per-model threshold calibration before ensemble
  - 8x TTA (default)
  - Quantile threshold matched to train class ratio (not val-tuned 0.4802)
  - Saves raw probs for fast threshold retuning without re-running models
"""
import os
import argparse
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import timm

from env_check import require_venv312

BASE_DIR = r"D:\CV"
PROBS_FILE = os.path.join(BASE_DIR, "ensemble_test_probs.npz")
LARGE_MODELS = {"convnext_large", "tf_efficientnetv2_m"}


def load_train_positive_ratio() -> float:
    train_csv = os.path.join(BASE_DIR, "images", "train.csv")
    return float(pd.read_csv(train_csv)["target"].mean())


def calibrate_prob(p: float, t: float) -> float:
    if t <= 0 or t >= 1:
        return p
    if p < t:
        return p / (2.0 * t)
    return 0.5 + (p - t) / (2.0 * (1.0 - t))


def load_rois(roi_file: str) -> dict:
    img_name_to_bbox = {}
    if not os.path.exists(roi_file):
        return img_name_to_bbox
    with open(roi_file, "r") as f:
        roi_data = json.load(f)
    id_to_name = {img["id"]: img["file_name"] for img in roi_data.get("images", [])}
    for ann in roi_data.get("annotations", []):
        img_name_to_bbox[id_to_name[ann["image_id"]]] = ann["bbox"]
    return img_name_to_bbox


def make_transform(version: str, arch: str, imgsz: int) -> transforms.Compose:
    if version == "v9" or arch in LARGE_MODELS:
        oversize = int(imgsz * 1.05)
        return transforms.Compose([
            transforms.Resize((oversize, oversize)),
            transforms.CenterCrop(imgsz),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_imgsz(arch: str, version: str) -> int:
    if arch in LARGE_MODELS:
        return 256 if version == "v9" else 320
    return 448


def tta_views(img: Image.Image, n_tta: int):
    views = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.rotate(5, resample=Image.BICUBIC),
        img.rotate(-5, resample=Image.BICUBIC),
    ]
    if n_tta <= 4:
        return views[:n_tta]
    views.extend([
        img.rotate(10, resample=Image.BICUBIC),
        img.rotate(-10, resample=Image.BICUBIC),
        img.transpose(Image.FLIP_LEFT_RIGHT).rotate(5, resample=Image.BICUBIC),
        img.transpose(Image.FLIP_LEFT_RIGHT).rotate(-5, resample=Image.BICUBIC),
    ])
    return views[:n_tta]


def discover_models():
    root = os.path.join(BASE_DIR, "runs", "classify", "krones_challenge")
    configs = []
    weights = {
        ("convnext_base", "v7"): 3.0,
        ("convnext_base", "v6"): 2.0,
        ("convnext_base", "v9"): 2.5,
        ("convnext_large", "v9"): 2.5,
        ("tf_efficientnetv2_m", "v9"): 2.5,
        ("repvit_m1_0", "v6"): 1.0,
    }
    for name in sorted(os.listdir(root)):
        if "_fold_" not in name:
            continue
        parts = name.rsplit("_fold_", 1)
        if len(parts) != 2:
            continue
        prefix, fold_s = parts
        fold = int(fold_s)
        for version in ("v9", "v7", "v6"):
            suffix = f"_{version}"
            if prefix.endswith(suffix):
                arch = prefix[: -len(suffix)]
                break
        else:
            continue
        model_path = os.path.join(root, name, "best.pth")
        if not os.path.exists(model_path):
            continue
        thresh_path = os.path.join(root, name, "optimal_threshold.txt")
        thresh = 0.5
        if os.path.exists(thresh_path):
            with open(thresh_path) as f:
                thresh = float(f.read().strip())
        w = weights.get((arch, version), 1.0)
        imgsz = get_imgsz(arch, version)
        configs.append((arch, version, fold, imgsz, w, thresh, model_path))
    return configs


class TestImageDataset(Dataset):
    def __init__(self, df, test_images_dir, img_name_to_bbox):
        self.df = df.reset_index(drop=True)
        self.test_images_dir = test_images_dir
        self.img_name_to_bbox = img_name_to_bbox

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.test_images_dir, row["image_id"])
        if not os.path.exists(path):
            return None, idx
        img = Image.open(path).convert("RGB")
        if row["image_id"] in self.img_name_to_bbox:
            x, y, w, h = self.img_name_to_bbox[row["image_id"]]
            img = img.crop((x, y, x + w, y + h))
        return img, idx


def collate_pil(batch):
    imgs, indices = [], []
    for img, idx in batch:
        if img is not None:
            imgs.append(img)
            indices.append(idx)
    return imgs, indices


@torch.inference_mode()
def run_model_probs(model, loader, transform, device, n_tta, model_thresh, prob_sum, weight,
                    use_calibration: bool):
    model.eval()
    use_amp = device.type == "cuda"
    for pil_imgs, indices in tqdm(loader, leave=False):
        batch_tensors = []
        batch_map = []
        for j, img in enumerate(pil_imgs):
            for v in tta_views(img, n_tta):
                batch_tensors.append(transform(v))
                batch_map.append(j)
        if not batch_tensors:
            continue
        x = torch.stack(batch_tensors).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        per_img = [[] for _ in range(len(pil_imgs))]
        for p, j in zip(probs, batch_map):
            if use_calibration:
                p = calibrate_prob(float(p), model_thresh)
            per_img[j].append(p)
        for j, idx in enumerate(indices):
            prob_sum[idx] += float(np.mean(per_img[j])) * weight


def quantile_threshold(probs: np.ndarray, positive_ratio: float) -> float:
    n = len(probs)
    k = int(np.round(n * positive_ratio))
    k = max(1, min(k, n))
    sorted_probs = np.sort(probs)[::-1]
    return float(sorted_probs[k - 1])


def weighted_model_threshold(configs) -> float:
    wsum = sum(c[4] for c in configs)
    return sum(c[4] * c[5] for c in configs) / wsum


def apply_threshold(df: pd.DataFrame, probs: np.ndarray, mode: str, configs,
                    fixed_t: float | None) -> tuple[np.ndarray, float]:
    train_ratio = load_train_positive_ratio()

    if mode == "quantile":
        t = quantile_threshold(probs, train_ratio)
    elif mode == "weighted":
        t = weighted_model_threshold(configs)
    elif mode == "fixed":
        t = fixed_t if fixed_t is not None else 0.5
    else:
        raise ValueError(f"Unknown mode: {mode}")

    preds = (probs >= t).astype(int)
    return preds, t


def main():
    require_venv312(require_cuda=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tta", type=int, default=8, choices=[1, 4, 8])
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Disable per-model threshold calibration")
    parser.add_argument("--mode", default="quantile",
                        choices=["quantile", "weighted", "fixed"],
                        help="quantile=match train %% reusable (recommended for LB)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Only for --mode fixed")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "submission_v15.csv"))
    parser.add_argument("--probs-only", action="store_true",
                        help="Only save ensemble_test_probs.npz (skip CSV)")
    parser.add_argument("--from-probs", action="store_true",
                        help="Skip inference; rebuild CSV from saved probs")
    args = parser.parse_args()

    sample_sub = os.path.join(BASE_DIR, "images", "sample_submission.csv")
    df = pd.read_csv(sample_sub)
    configs = discover_models()

    if args.from_probs:
        if not os.path.exists(PROBS_FILE):
            raise FileNotFoundError(f"Missing {PROBS_FILE}. Run inference first.")
        data = np.load(PROBS_FILE)
        probs = data["probs"]
        print(f"Loaded probs from {PROBS_FILE}")
    else:
        require_venv312(require_cuda=True)
        device = torch.device("cuda")
        test_images_dir = os.path.join(BASE_DIR, "images", "test_images")
        roi_file = os.path.join(BASE_DIR, "images", "test_annotations_roi_only.json")
        img_name_to_bbox = load_rois(roi_file)

        print(f"Device: {device} | TTA={args.tta} | calibrate={not args.no_calibrate}")
        print(f"Models: {len(configs)} | Test images: {len(df)}")

        n = len(df)
        prob_sum = np.zeros(n)
        dataset = TestImageDataset(df, test_images_dir, img_name_to_bbox)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pil,
            pin_memory=device.type == "cuda",
        )

        for arch, version, fold, imgsz, weight, model_thresh, model_path in configs:
            print(f"\n{arch} {version} fold {fold} (w={weight}, t={model_thresh:.3f}, sz={imgsz})")
            model = timm.create_model(arch, pretrained=False, num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            transform = make_transform(version, arch, imgsz)
            run_model_probs(
                model, loader, transform, device, args.tta, model_thresh,
                prob_sum, weight, use_calibration=not args.no_calibrate,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        weight_sum = sum(c[4] for c in configs)
        probs = prob_sum / weight_sum
        np.savez(
            PROBS_FILE,
            probs=probs,
            image_id=df["image_id"].values,
        )
        print(f"\nSaved probabilities: {PROBS_FILE}")

    if args.probs_only:
        return

    preds, t = apply_threshold(df, probs, args.mode, configs, args.threshold)
    df["target"] = preds
    df.to_csv(args.output, index=False)

    n_pos = int(preds.sum())
    print(f"\nMode: {args.mode} | threshold: {t:.4f}")
    print(f"Reusable: {n_pos}/{len(df)} ({100 * n_pos / len(df):.2f}%)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
