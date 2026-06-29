"""
v17 inference — average 5 CV folds + full-data model (6 checkpoints).
Threshold from OOF: ensemble_threshold_v17.txt

  python train_model_v17.py
  python inference_v17.py
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
VERSION = "v17"
GLOBAL_THRESH_FILE = os.path.join(BASE_DIR, f"ensemble_threshold_{VERSION}.txt")


def full_model_path(model_name: str) -> str:
    return os.path.join(
        BASE_DIR, "runs", "classify", "krones_challenge",
        f"{model_name}_{VERSION}_full", "best_full.pth",
    )


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


def tta_views(img, n=8):
    views = [
        img, img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
        img.rotate(5, Image.BICUBIC), img.rotate(-5, Image.BICUBIC),
        img.rotate(10, Image.BICUBIC), img.rotate(-10, Image.BICUBIC),
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


def discover_checkpoints(model_name: str, n_folds: int, use_full: bool):
    root = os.path.join(BASE_DIR, "runs", "classify", "krones_challenge")
    paths = []
    for f in range(1, n_folds + 1):
        p = os.path.join(root, f"{model_name}_{VERSION}_fold_{f}", "best.pth")
        if os.path.exists(p):
            paths.append(("fold", p))
    if use_full:
        fp = full_model_path(model_name)
        if os.path.exists(fp):
            paths.append(("full", fp))
    return paths


@torch.inference_mode()
def run_checkpoint(ckpt_path, model_name, loader, val_tf, device, n_tta, prob_sum):
    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
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


def main():
    require_venv312(require_cuda=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convnext_large")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--tta", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-full", action="store_true", help="Use folds only (no best_full.pth)")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "submission_v17.csv"))
    args = parser.parse_args()

    thresh = 0.5
    if os.path.exists(GLOBAL_THRESH_FILE):
        with open(GLOBAL_THRESH_FILE) as f:
            thresh = float(f.read().strip())

    ckpts = discover_checkpoints(args.model, args.folds, use_full=not args.no_full)
    if not ckpts:
        raise FileNotFoundError("No v17 checkpoints. Run train_model_v17.py first.")

    n_folds = sum(1 for t, _ in ckpts if t == "fold")
    has_full = any(t == "full" for t, _ in ckpts)
    print(f"Model: {args.model} | imgsz={args.imgsz} | threshold={thresh:.4f}")
    print(f"Checkpoints: {n_folds} folds" + (" + full-data model" if has_full else ""))

    df = pd.read_csv(os.path.join(BASE_DIR, "images", "sample_submission.csv"))
    test_dir = os.path.join(BASE_DIR, "images", "test_images")
    rois = load_rois()
    n = len(df)
    prob_sum = np.zeros(n)
    device = torch.device("cuda")

    val_tf = transforms.Compose([
        transforms.Resize((int(args.imgsz * 1.08), int(args.imgsz * 1.08))),
        transforms.CenterCrop(args.imgsz),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(TestDS(df, test_dir, rois), batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)

    for i, (kind, path) in enumerate(ckpts, 1):
        print(f"\n[{i}/{len(ckpts)}] {kind}: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}")
        run_checkpoint(path, args.model, loader, val_tf, device, args.tta, prob_sum)

    probs = prob_sum / len(ckpts)
    preds = (probs >= thresh).astype(int)
    df["target"] = preds
    df.to_csv(args.output, index=False)
    print(f"\nReusable: {preds.sum()}/{n} ({100*preds.mean():.2f}%)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
