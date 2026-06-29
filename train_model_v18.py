"""
Krones v18 — dual-model training + pseudo-label rounds.
"""
from __future__ import annotations

from env_check import relaunch_with_venv312

relaunch_with_venv312()

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
import timm
from timm.data import Mixup
from timm.utils import ModelEmaV2

from env_check import require_venv312

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

BASE_DIR = r"D:\CV"
NUM_WORKERS = 2 if os.name == "nt" else 4
MIN_GPU_IT_PER_S = 2.0
PAUSE_CHECK_EVERY = 50
LOG_FIRST = 50
LOG_EVERY = 400
TARGET_F1 = 0.9735
VERSION = "v18"
PAUSE_FLAG = os.path.join(BASE_DIR, "pause_training.flag")
OOF_FILE = os.path.join(BASE_DIR, f"oof_probs_{VERSION}.npz")
GLOBAL_THRESH_FILE = os.path.join(BASE_DIR, f"ensemble_threshold_{VERSION}.txt")
MODEL_IMGSZ = {"convnext_large": 320, "tf_efficientnetv2_m": 384}
LARGE_MODELS = {"convnext_large", "convnext_xlarge", "tf_efficientnetv2_l", "tf_efficientnetv2_xl"}


def run_dir_fold(model_name: str, fold: int) -> str:
    return os.path.join(BASE_DIR, "runs", "classify", "krones_challenge", f"{model_name}_{VERSION}_fold_{fold}")


def run_dir_full(model_name: str) -> str:
    return os.path.join(BASE_DIR, "runs", "classify", "krones_challenge", f"{model_name}_{VERSION}_full")


def fold_best_path(model_name: str, fold: int) -> str:
    return os.path.join(run_dir_fold(model_name, fold), "best.pth")


def full_best_path(model_name: str) -> str:
    return os.path.join(run_dir_full(model_name), "best_full.pth")


def make_train_transforms(imgsz: int) -> transforms.Compose:
    oversize = int(imgsz * 1.15)
    return transforms.Compose([
        transforms.Resize((oversize, oversize)),
        transforms.RandomCrop(imgsz),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.08),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.3),
        transforms.RandomApply([transforms.RandomAutocontrast(p=1.0)], p=0.3),
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0)], p=0.3),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])


def make_val_transforms(imgsz: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((int(imgsz * 1.08), int(imgsz * 1.08))),
        transforms.CenterCrop(imgsz),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class PreventSleep:
    _FLAGS = 0x80000000 | 0x00000001 | 0x00000002

    def __enter__(self):
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(self._FLAGS)
            try_boost_gpu_power()
        return self

    def __exit__(self, *_exc):
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def try_boost_gpu_power() -> None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.max_limit", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if r.returncode != 0:
            return
        pl = int(float(r.stdout.strip().splitlines()[0]))
        subprocess.run(
            ["nvidia-smi", "-pl", str(pl)],
            capture_output=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        print(f"GPU power limit -> {pl} W", flush=True)
    except Exception:
        pass


def wake_gpu(device: torch.device, model: nn.Module, batch_size: int, imgsz: int, steps: int = 30) -> float:
    model.train()
    x = torch.randn(batch_size, 3, imgsz, imgsz, device=device)
    for _ in range(5):
        with torch.amp.autocast("cuda"):
            model(x).sum().backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.amp.autocast("cuda"):
            model(x).sum().backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    return steps / (time.perf_counter() - t0)


def report_gpu_warmup(it_per_s: float) -> None:
    pwr = ""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,pstate", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if r.returncode == 0:
            pwr = f" ({r.stdout.strip()})"
    except Exception:
        pass
    if it_per_s >= MIN_GPU_IT_PER_S:
        print(f"GPU warmup OK: {it_per_s:.2f} it/s{pwr}", flush=True)
        return
    print(
        f"\n*** GPU TOO SLOW: {it_per_s:.2f} it/s (need >={MIN_GPU_IT_PER_S}){pwr} ***\n"
        "  Common after pause: laptop GPU stuck in P4.\n"
        "  - Plug in AC, Windows -> Best performance, NVIDIA -> Max performance\n"
        "  - Task Manager -> end ALL python.exe, reboot if needed\n"
        f"  - Then: .\\train_model_{VERSION}.bat\n",
        flush=True,
    )


def make_loaders(train_ds, val_ds, batch_size: int) -> tuple[DataLoader, DataLoader]:
    kw: dict = {"num_workers": NUM_WORKERS, "pin_memory": True}
    if NUM_WORKERS > 0:
        if os.name != "nt":
            kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **kw)
    return train_loader, val_loader


def log_step_progress(phase: str, epoch_disp: int, total_epochs: int, step: int, total: int, t0: float) -> None:
    if step != total and step != LOG_FIRST and step % LOG_EVERY != 0:
        return
    elapsed = time.time() - t0
    rate = step / elapsed if elapsed > 0 else 0.0
    eta_min = (total - step) / rate / 60.0 if rate > 0 else 0.0
    print(
        f"  Ep {epoch_disp}/{total_epochs} {phase} {step}/{total} | {rate:.2f} it/s | ~{eta_min:.0f}m left",
        flush=True,
    )


class TrainingPaused(Exception):
    pass


class PauseHandler:
    def __init__(self, flag_file: str):
        self.flag_file = flag_file
        self._interrupted = False

    def requested(self) -> bool:
        return os.path.isfile(self.flag_file)

    def clear(self):
        if os.path.isfile(self.flag_file):
            os.remove(self.flag_file)

    def mark_interrupted(self):
        self._interrupted = True

    def should_stop(self) -> bool:
        return self.requested() or self._interrupted

    def handle_stop(self, phase: str):
        if self.requested():
            self.clear()
        print(f"\n[PAUSE] Stopped during: {phase}", flush=True)
        print("  Checkpoints written (last.pth / last_full.pth).", flush=True)
        print("  Resume with the same command, e.g.:", flush=True)
        print(f"    python train_model_{VERSION}.py", flush=True)
        raise TrainingPaused()


def save_fold_checkpoint(last_path, last_opt_path, model, ema, optimizer, scheduler, scaler, epoch, best_f1, patience_counter):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_ema_state_dict": ema.module.state_dict(),
        "best_f1": best_f1,
        "patience_counter": patience_counter,
    }
    tmp = last_path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, last_path)
    opt = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    tmp_opt = last_opt_path + ".tmp"
    torch.save(opt, tmp_opt)
    os.replace(tmp_opt, last_opt_path)


class GpuTempMonitor:
    def __init__(self):
        self.running = False
        self.temps = []
        self.thread = None

    def _monitor(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                    encoding="utf-8",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                self.temps.append(float(out.strip()))
            except Exception:
                pass
            time.sleep(5)

    def start(self):
        self.running = True
        self.temps = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.running = False
        if self.thread:
            self.thread.join()
        return sum(self.temps) / len(self.temps) if self.temps else 0.0


def load_rois() -> dict:
    img_name_to_bbox = {}
    for fname in ("train_annotations.json", "test_annotations_roi_only.json"):
        path = os.path.join(BASE_DIR, "images", fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            roi_data = json.load(f)
        id_to_name = {img["id"]: img["file_name"] for img in roi_data.get("images", [])}
        for ann in roi_data.get("annotations", []):
            img_name_to_bbox[id_to_name[ann["image_id"]]] = ann["bbox"]
    return img_name_to_bbox


class KronesDataset(Dataset):
    def __init__(self, df, img_name_to_bbox, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_name_to_bbox = img_name_to_bbox
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        if row["image_id"] in self.img_name_to_bbox:
            x, y, w, h = self.img_name_to_bbox[row["image_id"]]
            img = img.crop((x, y, x + w, y + h))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row["target"], dtype=torch.long)


class SoftTargetFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets):
        log_probs = self.log_softmax(inputs)
        probs = torch.exp(log_probs)
        return (-torch.sum(targets * (1 - probs) ** self.gamma * log_probs, dim=-1)).mean()


def find_best_threshold(y_true, y_probs, steps: int = 199):
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, steps):
        score = f1_score(y_true, (y_probs >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t, best_f1


def load_pseudo_labels() -> pd.DataFrame | None:
    path = os.path.join(BASE_DIR, "pseudo_labels.csv")
    if not os.path.exists(path):
        return None
    pdf = pd.read_csv(path)
    hi = pdf[pdf["prob_reusable"] > 0.95].copy()
    lo = pdf[pdf["prob_reusable"] < 0.05].copy()
    hi["target"] = 1
    lo["target"] = 0
    pseudo = pd.concat([hi, lo], ignore_index=True)
    pseudo["img_path"] = pseudo["image_id"].apply(lambda x: os.path.join(BASE_DIR, "images", "test_images", x))
    return pseudo[["image_id", "target", "img_path"]]


def average_fold_weights(model_name: str, n_folds: int) -> dict[str, torch.Tensor]:
    avg = None
    n = 0
    for fold in range(1, n_folds + 1):
        path = fold_best_path(model_name, fold)
        if not os.path.exists(path):
            continue
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if avg is None:
            avg = {k: v.float().clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                avg[k] += v.float()
        n += 1
    if avg is None or n == 0:
        raise FileNotFoundError(f"No fold checkpoints found for {model_name} ({VERSION})")
    for k in avg:
        avg[k] /= n
    print(f"Averaged weights from {n} fold checkpoints for {model_name}.", flush=True)
    return avg


def get_imgsz(model_name: str, global_imgsz: int) -> int:
    return global_imgsz or MODEL_IMGSZ.get(model_name, 448)


def get_batch_accum(model_name: str) -> tuple[int, int]:
    if model_name == "tf_efficientnetv2_m":
        return 8, 2
    is_large = model_name in LARGE_MODELS or "large" in model_name or "_l" in model_name
    return (4, 4) if is_large else (8, 2)


@torch.inference_mode()
def _predict_val_oof(model_name, fold, val_df, img_name_to_bbox, imgsz, best_path):
    if not os.path.exists(best_path):
        return None
    device = torch.device("cuda")
    model = timm.create_model(model_name, pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    val_ds = KronesDataset(val_df, img_name_to_bbox, make_val_transforms(imgsz))
    kw = {"num_workers": NUM_WORKERS, "pin_memory": True}
    if NUM_WORKERS > 0 and os.name != "nt":
        kw["persistent_workers"] = True
    loader = DataLoader(val_ds, batch_size=16, shuffle=False, **kw)
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            p = torch.softmax(model(x), 1)[:, 1]
        probs.extend(p.cpu().numpy())
        targets.extend(y.numpy())
    thresh, f1 = find_best_threshold(targets, probs)
    with open(os.path.join(run_dir_fold(model_name, fold), "optimal_threshold.txt"), "w") as f:
        f.write(str(thresh))
    print(f"{model_name} fold {fold} OOF F1={f1:.4f} @ {thresh:.4f}", flush=True)
    return val_df["image_id"].values, np.array(probs), np.array(targets)


def train_fold(model_name: str, fold: int, n_folds: int, train_df: pd.DataFrame, val_df: pd.DataFrame, img_name_to_bbox: dict,
               pause_handler: PauseHandler, imgsz: int, epochs: int):
    run_dir = run_dir_fold(model_name, fold)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best.pth")
    last_path = os.path.join(run_dir, "last.pth")
    last_opt_path = os.path.join(run_dir, "last_opt.pth")
    thresh_path = os.path.join(run_dir, "optimal_threshold.txt")
    if os.path.exists(thresh_path) and os.path.exists(best_path):
        print(f"{model_name} fold {fold} complete — loading for OOF.", flush=True)
        return _predict_val_oof(model_name, fold, val_df, img_name_to_bbox, imgsz, best_path), False

    batch_size, accum = get_batch_accum(model_name)
    warmup_epochs = 5
    patience = 12
    train_loader, val_loader = make_loaders(KronesDataset(train_df, img_name_to_bbox, make_train_transforms(imgsz)),
                                            KronesDataset(val_df, img_name_to_bbox, make_val_transforms(imgsz)), batch_size)
    n_train_steps = len(train_loader)
    print(f"  {model_name} steps/epoch: {n_train_steps} | batch={batch_size} | workers={NUM_WORKERS}", flush=True)
    device = torch.device("cuda")
    resuming = os.path.exists(last_path)
    model = timm.create_model(model_name, pretrained=not resuming, num_classes=2, drop_rate=0.3, drop_path_rate=0.2).to(device)
    ema = ModelEmaV2(model, decay=0.9998, device=device)
    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.9, switch_prob=0.5, mode="batch", label_smoothing=0.1, num_classes=2)
    criterion = SoftTargetFocalLoss(gamma=2.0)
    val_crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.08)
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
                                         CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=5e-8)],
                             milestones=[warmup_epochs])
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_f1, patience_counter = 0, 0.0, 0
    if resuming:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "model_ema_state_dict" in ckpt:
            ema.module.load_state_dict(ckpt["model_ema_state_dict"])
        start_epoch = ckpt["epoch"]
        best_f1 = ckpt["best_f1"]
        patience_counter = ckpt.get("patience_counter", 0)
        if os.path.exists(last_opt_path):
            opt = torch.load(last_opt_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(opt["optimizer_state_dict"])
            scheduler.load_state_dict(opt["scheduler_state_dict"])
            scaler.load_state_dict(opt["scaler_state_dict"])
        print(f"Resumed {model_name} fold {fold} @ epoch {start_epoch}, best F1={best_f1:.4f}", flush=True)

    monitor = GpuTempMonitor()
    monitor.start()
    print(f"\n{'='*60}\n{model_name} fold {fold}/{n_folds} | train={len(train_df)} val={len(val_df)} | {imgsz}px\n{'='*60}", flush=True)
    report_gpu_warmup(wake_gpu(device, model, batch_size, imgsz))
    paused = False
    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            if pause_handler.should_stop():
                save_fold_checkpoint(last_path, last_opt_path, model, ema, optimizer, scheduler, scaler, epoch, best_f1, patience_counter)
                pause_handler.handle_stop(f"{model_name} fold {fold}/{n_folds} (before epoch {epoch+1})")
            model.train()
            train_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            t_train = time.time()
            for i, (x, y) in enumerate(train_loader):
                if i % PAUSE_CHECK_EVERY == 0 and pause_handler.requested():
                    save_fold_checkpoint(last_path, last_opt_path, model, ema, optimizer, scheduler, scaler, epoch, best_f1, patience_counter)
                    pause_handler.handle_stop(f"{model_name} fold {fold}/{n_folds} (mid-epoch {epoch+1})")
                log_step_progress("train", epoch + 1, epochs, i + 1, n_train_steps, t_train)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                x, y = mixup_fn(x, y)
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(x), y) / accum
                scaler.scale(loss).backward()
                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                train_loss += loss.item() * accum
            train_loss /= len(train_loader)
            ema.module.eval()
            y_true, y_probs, val_loss = [], [], 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda"):
                        out = ema.module(x)
                        val_loss += val_crit(out, y).item()
                    y_probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
                    y_true.extend(y.cpu().numpy())
            val_loss /= len(val_loader)
            _, ep_f1 = find_best_threshold(y_true, y_probs)
            scheduler.step()
            print(f"{model_name} ep {epoch+1} | loss {train_loss:.4f}/{val_loss:.4f} | F1 {ep_f1:.4f}", flush=True)
            if ep_f1 > best_f1:
                best_f1 = ep_f1
                patience_counter = 0
                torch.save(ema.module.state_dict(), best_path)
            else:
                patience_counter += 1
            if ep_f1 >= TARGET_F1:
                torch.save(ema.module.state_dict(), best_path)
                print(f"TARGET F1 {ep_f1:.4f} reached — advancing to next fold", flush=True)
                save_fold_checkpoint(last_path, last_opt_path, model, ema, optimizer, scheduler, scaler, epoch + 1, best_f1, patience_counter)
                break
            if patience_counter >= patience:
                print(f"{model_name} early stop @ epoch {epoch+1}", flush=True)
                break
            save_fold_checkpoint(last_path, last_opt_path, model, ema, optimizer, scheduler, scaler, epoch + 1, best_f1, patience_counter)
            if pause_handler.should_stop():
                pause_handler.handle_stop(f"{model_name} fold {fold}/{n_folds} (after epoch {epoch+1})")
    except TrainingPaused:
        paused = True
    except KeyboardInterrupt:
        pause_handler.mark_interrupted()
        save_fold_checkpoint(last_path, last_opt_path, model, ema, optimizer, scheduler, scaler, current_epoch + 1, best_f1, patience_counter)
        print("\n[PAUSE] Ctrl+C — checkpoint saved.", flush=True)
        paused = True
    finally:
        print(f"Avg GPU temp: {monitor.stop():.1f}C", flush=True)
    if paused:
        return None, True
    return _predict_val_oof(model_name, fold, val_df, img_name_to_bbox, imgsz, best_path), False


def train_full(model_name: str, full_train_df: pd.DataFrame, val_df: pd.DataFrame, img_name_to_bbox: dict,
               pause_handler: PauseHandler, imgsz: int, epochs: int, n_folds: int, init_from_folds: bool = True) -> bool:
    run_dir = run_dir_full(model_name)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_full.pth")
    last_path = os.path.join(run_dir, "last_full.pth")
    last_opt_path = os.path.join(run_dir, "last_full_opt.pth")
    if os.path.exists(best_path) and os.path.exists(os.path.join(run_dir, "full_done.flag")):
        print(f"{model_name} full already done.", flush=True)
        return False
    batch_size, accum = get_batch_accum(model_name)
    warmup_epochs, patience = 2, 8
    train_loader, val_loader = make_loaders(KronesDataset(full_train_df, img_name_to_bbox, make_train_transforms(imgsz)),
                                            KronesDataset(val_df, img_name_to_bbox, make_val_transforms(imgsz)), batch_size)
    n_train_steps = len(train_loader)
    device = torch.device("cuda")
    resuming_full = os.path.exists(last_path)
    if resuming_full:
        model = timm.create_model(model_name, pretrained=False, num_classes=2, drop_rate=0.25, drop_path_rate=0.15).to(device)
    elif init_from_folds:
        model = timm.create_model(model_name, pretrained=False, num_classes=2, drop_rate=0.25, drop_path_rate=0.15).to(device)
        model.load_state_dict(average_fold_weights(model_name, n_folds))
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=2, drop_rate=0.25, drop_path_rate=0.15).to(device)
    ema = ModelEmaV2(model, decay=0.9999, device=device)
    mixup_fn = Mixup(mixup_alpha=0.6, cutmix_alpha=0.8, prob=0.7, switch_prob=0.5, mode="batch", label_smoothing=0.05, num_classes=2)
    criterion = SoftTargetFocalLoss(gamma=1.5)
    val_crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                                         CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-8)],
                             milestones=[warmup_epochs])
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_f1, patience_counter = 0, 0.0, 0
    if resuming_full:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema.module.load_state_dict(ckpt["model_ema_state_dict"])
        start_epoch = ckpt["epoch"]
        best_f1 = ckpt["best_f1"]
        patience_counter = ckpt.get("patience_counter", 0)
        if os.path.exists(last_opt_path):
            opt = torch.load(last_opt_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(opt["optimizer_state_dict"])
            scheduler.load_state_dict(opt["scheduler_state_dict"])
            scaler.load_state_dict(opt["scaler_state_dict"])
    monitor = GpuTempMonitor()
    monitor.start()
    print(f"\n{'='*60}\nFULL DATA | {model_name} | train={len(full_train_df)} monitor_val={len(val_df)} | {imgsz}px | {epochs} epochs\n{'='*60}", flush=True)
    report_gpu_warmup(wake_gpu(device, model, batch_size, imgsz))
    paused = False
    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            if pause_handler.should_stop():
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "model_ema_state_dict": ema.module.state_dict(),
                            "best_f1": best_f1, "patience_counter": patience_counter}, last_path)
                torch.save({"optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict()}, last_opt_path)
                pause_handler.handle_stop(f"{model_name} full-data")
            model.train()
            train_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            t_train = time.time()
            for i, (x, y) in enumerate(train_loader):
                if i % PAUSE_CHECK_EVERY == 0 and pause_handler.requested():
                    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "model_ema_state_dict": ema.module.state_dict(),
                                "best_f1": best_f1, "patience_counter": patience_counter}, last_path)
                    torch.save({"optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                                "scaler_state_dict": scaler.state_dict()}, last_opt_path)
                    pause_handler.handle_stop(f"{model_name} full-data (mid-epoch {epoch+1})")
                log_step_progress("train", epoch + 1, epochs, i + 1, n_train_steps, t_train)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                x, y = mixup_fn(x, y)
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(x), y) / accum
                scaler.scale(loss).backward()
                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                train_loss += loss.item() * accum
            train_loss /= len(train_loader)
            ema.module.eval()
            y_true, y_probs, val_loss = [], [], 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda"):
                        out = ema.module(x)
                        val_loss += val_crit(out, y).item()
                    y_probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
                    y_true.extend(y.cpu().numpy())
            val_loss /= len(val_loader)
            _, ep_f1 = find_best_threshold(y_true, y_probs)
            scheduler.step()
            print(f"{model_name} full ep {epoch+1} | loss {train_loss:.4f}/{val_loss:.4f} | monitor F1 {ep_f1:.4f}", flush=True)
            if ep_f1 > best_f1:
                best_f1 = ep_f1
                patience_counter = 0
                torch.save(ema.module.state_dict(), best_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "model_ema_state_dict": ema.module.state_dict(),
                        "best_f1": best_f1, "patience_counter": patience_counter}, last_path)
            torch.save({"optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict()}, last_opt_path)
            if pause_handler.should_stop():
                pause_handler.handle_stop(f"{model_name} full-data (after epoch {epoch+1})")
    except TrainingPaused:
        paused = True
    except KeyboardInterrupt:
        pause_handler.mark_interrupted()
        torch.save({"epoch": current_epoch + 1, "model_state_dict": model.state_dict(), "model_ema_state_dict": ema.module.state_dict(),
                    "best_f1": best_f1, "patience_counter": patience_counter}, last_path)
        torch.save({"optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict()}, last_opt_path)
        paused = True
    finally:
        print(f"Avg GPU temp (full): {monitor.stop():.1f}C", flush=True)
    if paused:
        return True
    if os.path.exists(best_path):
        open(os.path.join(run_dir, "full_done.flag"), "w").close()
    return False


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


def tta_views(img):
    w, h = img.size
    c = int(min(w, h) * 0.85)
    left = (w - c) // 2
    top = (h - c) // 2
    zoom = img.crop((left, top, left + c, top + c)).resize((w, h), Image.BICUBIC)
    return [
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


@torch.inference_mode()
def run_pseudo_inference(model_names: list[str], folds: int, global_imgsz: int) -> pd.DataFrame:
    sample = pd.read_csv(os.path.join(BASE_DIR, "images", "sample_submission.csv"))
    test_dir = os.path.join(BASE_DIR, "images", "test_images")
    rois = load_rois()
    loader = DataLoader(TestDS(sample, test_dir, rois), batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)
    device = torch.device("cuda")
    prob_sum = np.zeros(len(sample), dtype=np.float32)
    ckpts: list[tuple[str, str, int]] = []
    for m in model_names:
        imgsz = get_imgsz(m, global_imgsz)
        for fold in range(1, folds + 1):
            p = fold_best_path(m, fold)
            if os.path.exists(p):
                ckpts.append((m, p, imgsz))
    if not ckpts:
        raise FileNotFoundError("No fold checkpoints found for pseudo inference.")
    for m, p, imgsz in ckpts:
        tf = make_val_transforms(imgsz)
        model = timm.create_model(m, pretrained=False, num_classes=2).to(device)
        model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        model.eval()
        for imgs, indices in loader:
            batch_t, batch_map = [], []
            for j, img in enumerate(imgs):
                for v in tta_views(img):
                    batch_t.append(tf(v))
                    batch_map.append(j)
            x = torch.stack(batch_t).to(device)
            with torch.amp.autocast("cuda"):
                pr = torch.softmax(model(x), 1)[:, 1].cpu().numpy()
            per = [[] for _ in range(len(imgs))]
            for prob, j in zip(pr, batch_map):
                per[j].append(prob)
            for j, idx in enumerate(indices):
                prob_sum[idx] += float(np.mean(per[j]))
        del model
        torch.cuda.empty_cache()
    probs = prob_sum / len(ckpts)
    out = pd.DataFrame({"image_id": sample["image_id"], "prob_reusable": probs})
    out.to_csv(os.path.join(BASE_DIR, "pseudo_labels.csv"), index=False)
    return out


def reset_model_folds(model_name: str, folds: int):
    for fold in range(1, folds + 1):
        rd = run_dir_fold(model_name, fold)
        if not os.path.isdir(rd):
            continue
        for fn in ("best.pth", "last.pth", "last_opt.pth", "optimal_threshold.txt"):
            fp = os.path.join(rd, fn)
            if os.path.exists(fp):
                os.remove(fp)


def run_auto_inference(args) -> None:
    """Run inference_v18.py after training completes (same model/fold settings)."""
    script = os.path.join(BASE_DIR, f"inference_{VERSION}.py")
    cmd = [
        sys.executable,
        script,
        "--model", args.model,
        "--model2", args.model2,
        "--folds", str(args.folds),
        "--tta", str(args.infer_tta),
        "--output", args.infer_output,
    ]
    if args.imgsz:
        cmd.extend(["--imgsz", str(args.imgsz)])
    if args.no_full:
        cmd.append("--no-full")
    print(f"\n{'='*60}", flush=True)
    print("Training complete — starting inference automatically...", flush=True)
    print(f"  {' '.join(cmd)}", flush=True)
    print(f"{'='*60}\n", flush=True)
    subprocess.run(cmd, check=True)


def main():
    require_venv312(require_cuda=True)
    parser = argparse.ArgumentParser(description="v18 dual-model fine-tune")
    parser.add_argument("--model", default="convnext_large")
    parser.add_argument("--model2", default="tf_efficientnetv2_m")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=0, help="override both model image sizes")
    parser.add_argument("--use-pseudo", action="store_true")
    parser.add_argument("--pseudo-rounds", type=int, default=2)
    parser.add_argument("--full-epochs", type=int, default=15)
    parser.add_argument("--no-full", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--skip-folds", action="store_true")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--no-infer", action="store_true", help="Skip automatic inference after training")
    parser.add_argument("--infer-tta", type=int, default=12, help="TTA views for auto inference")
    parser.add_argument(
        "--infer-output",
        default=os.path.join(BASE_DIR, "submission_v18.csv"),
        help="Output CSV for auto inference",
    )
    args = parser.parse_args()

    global NUM_WORKERS
    if args.workers >= 0:
        NUM_WORKERS = args.workers

    models = [args.model] + ([args.model2] if args.model2 != args.model else [])
    print(
        f"v18 | models={models} | folds={args.folds} | epochs={args.epochs} | full_epochs={args.full_epochs} "
        f"| workers={NUM_WORKERS} | pseudo_rounds={args.pseudo_rounds}",
        flush=True,
    )
    print("MODEL_IMGSZ:", {m: get_imgsz(m, args.imgsz) for m in models}, flush=True)
    images_dir = os.path.join(BASE_DIR, "images", "train_images")
    df = pd.read_csv(os.path.join(BASE_DIR, "images", "train.csv"))
    df["img_path"] = df["image_id"].apply(lambda x: os.path.join(images_dir, x))
    roi = load_rois()
    pause = PauseHandler(PAUSE_FLAG)
    print("\nPAUSE: .\\pause_training.ps1  (waits for epoch end — do not Ctrl+C mid-epoch)", flush=True)
    print("       Sleep prevention is ON while this process runs\n", flush=True)

    training_paused = False
    pseudo = load_pseudo_labels() if args.use_pseudo else None
    with PreventSleep():
        try:
            rounds = max(args.pseudo_rounds, 1) if args.use_pseudo else 1
            for r in range(1, rounds + 1):
                if r > 1:
                    for m in models:
                        reset_model_folds(m, args.folds)
                if pseudo is not None:
                    print(f"Pseudo-label round {r}/{rounds}: {len(pseudo)} images added", flush=True)
                if not args.full_only:
                    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
                    per_model_oof: dict[str, dict[str, list]] = {m: {"id": [], "prob": [], "target": []} for m in models}
                    if not args.skip_folds:
                        for fold, (tr_idx, va_idx) in enumerate(skf.split(df, df["target"]), 1):
                            train_df = df.iloc[tr_idx].copy()
                            val_df = df.iloc[va_idx].copy()
                            if pseudo is not None:
                                train_df = pd.concat([train_df, pseudo], ignore_index=True)
                            for m in models:
                                oof, was_paused = train_fold(
                                    m, fold, args.folds, train_df, val_df, roi, pause, get_imgsz(m, args.imgsz), args.epochs
                                )
                                if was_paused:
                                    training_paused = True
                                    break
                                if oof is None:
                                    continue
                                ids, probs, targets = oof
                                per_model_oof[m]["id"].extend(ids)
                                per_model_oof[m]["prob"].extend(probs)
                                per_model_oof[m]["target"].extend(targets)
                            if training_paused:
                                break
                    else:
                        for fold, (_, va_idx) in enumerate(skf.split(df, df["target"]), 1):
                            val_df = df.iloc[va_idx].copy()
                            for m in models:
                                p = fold_best_path(m, fold)
                                if not os.path.exists(p):
                                    continue
                                result = _predict_val_oof(m, fold, val_df, roi, get_imgsz(m, args.imgsz), p)
                                if result:
                                    ids, probs, targets = result
                                    per_model_oof[m]["id"].extend(ids)
                                    per_model_oof[m]["prob"].extend(probs)
                                    per_model_oof[m]["target"].extend(targets)
                    if training_paused:
                        break
                    common_ids = np.array(per_model_oof[models[0]]["id"])
                    common_targets = np.array(per_model_oof[models[0]]["target"])
                    avg_probs = np.zeros(len(common_ids), dtype=np.float64)
                    for m in models:
                        avg_probs += np.array(per_model_oof[m]["prob"], dtype=np.float64)
                    avg_probs /= len(models)
                    np.savez(OOF_FILE, image_id=common_ids, prob=avg_probs, target=common_targets)
                    global_t, global_f1 = find_best_threshold(common_targets, avg_probs)
                    with open(GLOBAL_THRESH_FILE, "w") as f:
                        f.write(f"{global_t:.6f}\n")
                    print(f"OOF saved: {OOF_FILE}", flush=True)
                    print(f"Global OOF F1: {global_f1:.4f} @ threshold {global_t:.4f}", flush=True)
                    print(f"Threshold file: {GLOBAL_THRESH_FILE}", flush=True)

                if args.use_pseudo and r < rounds:
                    run_pseudo_inference(models, args.folds, args.imgsz)
                    pseudo = load_pseudo_labels()
                    n_pseudo = 0 if pseudo is None else len(pseudo)
                    print(f"Pseudo-label round {r+1}/{rounds}: {n_pseudo} images added", flush=True)

            if not args.no_full and not training_paused:
                train_part, monitor_val = train_test_split(df, test_size=0.08, random_state=42, stratify=df["target"])
                if pseudo is not None:
                    train_part = pd.concat([train_part, pseudo], ignore_index=True)
                for m in models:
                    if train_full(m, train_part, monitor_val, roi, pause, get_imgsz(m, args.imgsz), args.full_epochs, args.folds):
                        training_paused = True
                        break
        except TrainingPaused:
            training_paused = True

    if training_paused:
        print("\nTraining paused. Run the same command again to continue.", flush=True)
        return
    if args.no_infer:
        print(f"\nNext: python inference_{VERSION}.py --model {args.model} --model2 {args.model2}", flush=True)
        return
    run_auto_inference(args)


if __name__ == "__main__":
    main()
