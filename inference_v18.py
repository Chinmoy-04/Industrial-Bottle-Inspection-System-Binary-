"""
v18 inference — dual-model average across folds + full models.
"""
from env_check import relaunch_with_venv312

relaunch_with_venv312()

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
import timm

from env_check import require_venv312

BASE_DIR = r"D:\CV"
VERSION = "v18"
GLOBAL_THRESH_FILE = os.path.join(BASE_DIR, f"ensemble_threshold_{VERSION}.txt")
MODEL_IMGSZ = {"convnext_large": 320, "tf_efficientnetv2_m": 384}


def full_model_path(model_name: str) -> str:
    return os.path.join(BASE_DIR, "runs", "classify", "krones_challenge", f"{model_name}_{VERSION}_full", "best_full.pth")


def load_rois():
    img_name_to_bbox = {}
    path = os.path.join(BASE_DIR, "images", "test_annotations_roi_only.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        id_to_name = {i["id"]: i["file_name"] for i in data.get("images", [])}
        for ann in data.get("annotations", []):
            img_name_to_bbox[id_to_name[ann["image_id"]]] = ann["bbox"]
    return img_name_to_bbox


def tta_views(img, n=12):
    w, h = img.size
    c = int(min(w, h) * 0.85)
    left = (w - c) // 2
    top = (h - c) // 2
    zoom = img.crop((left, top, left + c, top + c)).resize((w, h), Image.BICUBIC)
    views = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
        img.rotate(5, Image.BICUBIC),
        img.rotate(-5, Image.BICUBIC),
        img.rotate(10, Image.BICUBIC),
        img.rotate(-10, Image.BICUBIC),
        img.rotate(90, Image.BICUBIC),
        img.rotate(180, Image.BICUBIC),
        img.rotate(270, Image.BICUBIC),
        zoom,
    ]
    return views[:n]


class TestDS(Dataset):
    def __init__(self, df, img_dir, rois):
        self.df, self.img_dir, self.rois = df.reset_index(drop=True), img_dir, rois

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        p = os.path.join(self.img_dir, row["image_id"])
        if not os.path.exists(p):
            return None, i
        img = Image.open(p).convert("RGB")
        if row["image_id"] in self.rois:
            x, y, w, h = self.rois[row["image_id"]]
            img = img.crop((x, y, x + w, y + h))
        return img, i


def collate(batch):
    imgs, idx = [], []
    for a, b in batch:
        if a is not None:
            imgs.append(a)
            idx.append(b)
    return imgs, idx


def make_val_tf(imgsz: int):
    return transforms.Compose([
        transforms.Resize((int(imgsz * 1.08), int(imgsz * 1.08))),
        transforms.CenterCrop(imgsz),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def discover_checkpoints(model_name: str, n_folds: int, use_full: bool):
    root = os.path.join(BASE_DIR, "runs", "classify", "krones_challenge")
    paths = []
    for f in range(1, n_folds + 1):
        p = os.path.join(root, f"{model_name}_{VERSION}_fold_{f}", "best.pth")
        if os.path.exists(p):
            paths.append(("fold", p, model_name))
    if use_full:
        fp = full_model_path(model_name)
        if os.path.exists(fp):
            paths.append(("full", fp, model_name))
    return paths


@torch.inference_mode()
def run_checkpoint(ckpt_path, model_name, loader, device, n_tta, prob_sum, imgsz):
    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
    val_tf = make_val_tf(imgsz)
    for imgs, indices in tqdm(loader, leave=False):
        batch_t, batch_map = [], []
        for j, img in enumerate(imgs):
            for v in tta_views(img, n_tta):
                batch_t.append(val_tf(v))
                batch_map.append(j)
        x = torch.stack(batch_t).to(device)
        with torch.amp.autocast("cuda"):
            p = torch.softmax(model(x), 1)[:, 1].cpu().numpy()
        per = [[] for _ in range(len(imgs))]
        for prob, j in zip(p, batch_map):
            per[j].append(prob)
        for j, idx in enumerate(indices):
            prob_sum[idx] += float(np.mean(per[j]))
    del model
    torch.cuda.empty_cache()


def get_imgsz(model_name: str, global_imgsz: int) -> int:
    return global_imgsz or MODEL_IMGSZ.get(model_name, 320)


def main():
    require_venv312(require_cuda=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convnext_large")
    parser.add_argument("--model2", default="tf_efficientnetv2_m")
    parser.add_argument("--imgsz", type=int, default=0, help="override both model image sizes")
    parser.add_argument("--tta", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--no-full", action="store_true", help="Use folds only (no best_full.pth)")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "submission_v18.csv"))
    args = parser.parse_args()

    thresh = 0.5
    if os.path.exists(GLOBAL_THRESH_FILE):
        with open(GLOBAL_THRESH_FILE) as f:
            thresh = float(f.read().strip())

    models = [args.model] + ([args.model2] if args.model2 != args.model else [])
    ckpts = []
    for m in models:
        ckpts.extend(discover_checkpoints(m, args.folds, use_full=not args.no_full))
    if not ckpts:
        raise FileNotFoundError("No v18 checkpoints. Run train_model_v18.py first.")

    by_model = {m: sum(1 for _, _, mm in ckpts if mm == m) for m in models}
    print(f"Models: {models} | threshold={thresh:.4f}", flush=True)
    print(f"Checkpoints by model: {by_model}", flush=True)
    print(f"TTA={args.tta} | imgsz map: { {m: get_imgsz(m, args.imgsz) for m in models} }", flush=True)

    df = pd.read_csv(os.path.join(BASE_DIR, "images", "sample_submission.csv"))
    test_dir = os.path.join(BASE_DIR, "images", "test_images")
    rois = load_rois()
    n = len(df)
    prob_sum = np.zeros(n, dtype=np.float64)
    device = torch.device("cuda")
    loader = DataLoader(TestDS(df, test_dir, rois), batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)

    for i, (kind, path, model_name) in enumerate(ckpts, 1):
        imgsz = get_imgsz(model_name, args.imgsz)
        print(
            f"\n[{i}/{len(ckpts)}] {model_name} {kind}: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} | imgsz={imgsz}",
            flush=True,
        )
        run_checkpoint(path, model_name, loader, device, args.tta, prob_sum, imgsz)

    probs = prob_sum / len(ckpts)
    preds = (probs >= thresh).astype(int)
    df["target"] = preds
    df.to_csv(args.output, index=False)
    print(f"\nReusable: {preds.sum()}/{n} ({100*preds.mean():.2f}%)", flush=True)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
