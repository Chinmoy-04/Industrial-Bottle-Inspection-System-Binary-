import os
import time
import json
import threading
import subprocess
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Prevent Windows WDDM PCIe swapping by limiting PyTorch's VRAM usage explicitly
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import timm
from timm.data import Mixup
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# ──────────────────────────────────────────────────────────────
# PAUSE / RESUME
# ──────────────────────────────────────────────────────────────
BASE_DIR   = r"D:\CV"
PAUSE_FLAG = os.path.join(BASE_DIR, "pause_training.flag")


class PauseHandler:
    def __init__(self, flag_file: str):
        self.flag_file = flag_file

    def requested(self) -> bool:
        if os.path.exists(self.flag_file):
            print(f"\n[PAUSE] Flag file found — finishing epoch then saving …")
            return True
        return False

    def clear(self):
        if os.path.exists(self.flag_file):
            os.remove(self.flag_file)


# ──────────────────────────────────────────────────────────────
# GPU TEMPERATURE MONITOR
# ──────────────────────────────────────────────────────────────
class GpuTempMonitor:
    def __init__(self):
        self.running = False
        self.temps   = []
        self.thread  = None

    def _monitor(self):
        while self.running:
            try:
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                    encoding='utf-8',
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                self.temps.append(float(result.strip()))
            except Exception:
                pass
            time.sleep(5)

    def start(self):
        self.running = True
        self.temps   = []
        self.thread  = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.running = False
        if self.thread:
            self.thread.join()
        return sum(self.temps) / len(self.temps) if self.temps else 0.0


# ──────────────────────────────────────────────────────────────
# ROI LOADING
# ──────────────────────────────────────────────────────────────
def load_rois(filepaths: list) -> dict:
    img_name_to_bbox = {}
    for filepath in filepaths:
        if not os.path.exists(filepath):
            continue
        print(f"Loading ROIs from {filepath} …")
        with open(filepath, 'r') as f:
            roi_data = json.load(f)
        id_to_name = {img['id']: img['file_name'] for img in roi_data.get('images', [])}
        for ann in roi_data.get('annotations', []):
            img_name_to_bbox[id_to_name[ann['image_id']]] = ann['bbox']
    return img_name_to_bbox


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class KronesDataset(Dataset):
    def __init__(self, df, img_name_to_bbox, transform=None):
        self.df               = df.reset_index(drop=True)
        self.img_name_to_bbox = img_name_to_bbox
        self.transform        = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        target = row['target']

        img = Image.open(row['img_path']).convert('RGB')
        img_name = row['image_id']
        if img_name in self.img_name_to_bbox:
            x, y, w, h = self.img_name_to_bbox[img_name]
            img = img.crop((x, y, x + w, y + h))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(target, dtype=torch.long)


# ──────────────────────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────────────────────
class SoftTargetFocalLoss(nn.Module):
    """Focal loss that accepts soft (MixUp) targets."""

    def __init__(self, gamma: float = 1.5):
        super().__init__()
        self.gamma       = gamma
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets):
        log_probs = self.log_softmax(inputs)
        probs     = torch.exp(log_probs)
        loss      = -torch.sum(targets * (1 - probs) ** self.gamma * log_probs, dim=-1)
        return loss.mean()


# ──────────────────────────────────────────────────────────────
# THRESHOLD SEARCH (vectorised)
# ──────────────────────────────────────────────────────────────
def find_best_threshold(y_true, y_probs, n_steps: int = 99):
    """Return (best_thresh, best_f1) over a fine grid."""
    y_true  = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds, average='binary', zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, t
    return best_thresh, best_f1


# ──────────────────────────────────────────────────────────────
# FOLD TRAINING
# ──────────────────────────────────────────────────────────────
def train_fold(model_name, fold, train_df, val_df, base_dir,
               img_name_to_bbox, pause_handler: PauseHandler):

    run_dir = os.path.join(base_dir, "runs", "classify", "krones_challenge",
                           f"{model_name}_v9_fold_{fold}")
    os.makedirs(run_dir, exist_ok=True)

    best_weights_path = os.path.join(run_dir, "best.pth")
    last_weights_path = os.path.join(run_dir, "last.pth")
    last_opt_path     = os.path.join(run_dir, "last_opt.pth")  # optimizer/scaler state (separate, large)
    thresh_file       = os.path.join(run_dir, "optimal_threshold.txt")

    if os.path.exists(thresh_file) and os.path.exists(best_weights_path):
        print(f"\nFold {fold} for {model_name} already completely trained — skipping.")
        return

    print(f"\nStarting training for {model_name} — Fold {fold}")
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

    # ── Hyper-parameters ─────────────────────────────────────
    epochs        = 45
    warmup_epochs = 3
    patience      = 8

    LARGE_MODELS = {'convnext_large', 'efficientnetv2_m'}
    if model_name in LARGE_MODELS:
        batch_size         = 4
        accumulation_steps = 4
        imgsz              = 256
        print(f"[INFO] Large model detected — batch_size=4, imgsz=256")
    else:
        batch_size         = 4
        accumulation_steps = 4
        imgsz              = 448

    # ── Transforms ───────────────────────────────────────────
    oversize = int(imgsz * 1.1)

    train_transform = transforms.Compose([
        transforms.Resize((oversize, oversize)),
        transforms.RandomCrop(imgsz),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((int(imgsz * 1.05), int(imgsz * 1.05))),
        transforms.CenterCrop(imgsz),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Datasets & Loaders ───────────────────────────────────
    train_dataset = KronesDataset(train_df, img_name_to_bbox, transform=train_transform)
    val_dataset   = KronesDataset(val_df,   img_name_to_bbox, transform=val_transform)

    loader_kwargs = dict(num_workers=2, pin_memory=False, persistent_workers=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size * 2,
                              shuffle=False, **loader_kwargs)

    # ── Model ────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = timm.create_model(model_name, pretrained=True, num_classes=2).to(device)

    model_ema = ModelEmaV2(model, decay=0.9999, device=device)

    # ── Loss & Optimiser ─────────────────────────────────────
    mixup_fn      = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0,
                          switch_prob=0.5, mode='batch',
                          label_smoothing=0.1, num_classes=2)
    criterion     = SoftTargetFocalLoss(gamma=1.5)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)

    warmup  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                       total_iters=warmup_epochs)
    cosine  = CosineAnnealingLR(optimizer,
                                 T_max=max(epochs - warmup_epochs, 1),
                                 eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                              milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler('cuda')

    # ── Resume from checkpoint ───────────────────────────────
    start_epoch      = 0
    best_f1          = 0.0
    patience_counter = 0

    if os.path.exists(last_weights_path):
        print(f"Resuming {model_name} Fold {fold} from checkpoint …")
        ckpt = torch.load(last_weights_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'model_ema_state_dict' in ckpt:
            model_ema.module.load_state_dict(ckpt['model_ema_state_dict'])
        start_epoch      = ckpt['epoch']
        best_f1          = ckpt['best_f1']
        patience_counter = ckpt.get('patience_counter', 0)
        
        if os.path.exists(last_opt_path):
            opt_ckpt = torch.load(last_opt_path, map_location='cpu')
            optimizer.load_state_dict(opt_ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(opt_ckpt['scheduler_state_dict'])
            scaler.load_state_dict(opt_ckpt['scaler_state_dict'])
        print(f"  Resumed at epoch {start_epoch} | Best F1: {best_f1:.4f}")

    # ── Training loop ─────────────────────────────────────────
    monitor = GpuTempMonitor()
    monitor.start()
    paused = False

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            for i, (inputs, targets) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train")):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = mixup_fn(inputs, targets)

                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss    = criterion(outputs, targets) / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    model_ema.update(model)

                train_loss += loss.item() * accumulation_steps

            train_loss /= len(train_loader)

            # ── Validation (EMA model) ────────────────────────────
            model_ema.module.eval()
            val_loss = 0.0
            y_true, y_probs = [], []

            with torch.no_grad():
                for inputs, targets in tqdm(val_loader,
                                            desc=f"Epoch {epoch+1}/{epochs} Val",
                                            leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.amp.autocast('cuda'):
                        outputs = model_ema.module(inputs)
                        loss    = val_criterion(outputs, targets)
                    val_loss  += loss.item()
                    y_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    y_true.extend(targets.cpu().numpy())

            val_loss /= len(val_loader)
            scheduler.step()

            _, epoch_best_f1 = find_best_threshold(y_true, y_probs)
            current_lr = scheduler.get_last_lr()[0]

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"LR: {current_lr:.2e} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1 (EMA): {epoch_best_f1:.4f}")

            if epoch_best_f1 > best_f1:
                best_f1          = epoch_best_f1
                patience_counter = 0
                torch.save(model_ema.module.state_dict(), best_weights_path)
                print(f"  --> New best EMA model saved (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs.")
                    break

            # ── Save resumable checkpoint ───
            ckpt = {
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'model_ema_state_dict': model_ema.module.state_dict(),
                'best_f1':              best_f1,
                'patience_counter':     patience_counter,
            }
            tmp = last_weights_path + ".tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, last_weights_path)
            
            opt_ckpt = {
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict':    scaler.state_dict(),
            }
            tmp_opt = last_opt_path + ".tmp"
            torch.save(opt_ckpt, tmp_opt)
            os.replace(tmp_opt, last_opt_path)

            if pause_handler.requested():
                print(f"[PAUSE] Checkpoint saved at epoch {epoch+1}. "
                      f"Re-run the script to resume.")
                paused = True
                pause_handler.clear()
                break

    except KeyboardInterrupt:
        print(f"\n[PAUSE] Ctrl+C received — last completed checkpoint is already saved."
              f"\n        Re-run the script to resume from epoch {epoch+1}.")
        paused = True

    avg_temp = monitor.stop()
    print(f"\n[INFO] Avg GPU Temp — {model_name} Fold {fold}: {avg_temp:.2f}°C")

    if paused:
        return True  

    if os.path.exists(best_weights_path):
        model_ema.module.load_state_dict(
            torch.load(best_weights_path, map_location=device))
        model_ema.module.eval()
        y_true, y_probs = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader,
                                        desc="Threshold search", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model_ema.module(inputs)
                y_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                y_true.extend(targets.cpu().numpy())

        best_thresh, best_thresh_f1 = find_best_threshold(y_true, y_probs)
        with open(thresh_file, "w") as f:
            f.write(str(best_thresh))
        print(f"Optimal F1: {best_thresh_f1:.4f} @ threshold {best_thresh:.4f} "
              f"({model_name} Fold {fold})")
    return False 


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("PyTorch timm Pipeline V9 — ConvNeXt/EfficientNetV2 | "
          "Focal Loss + EMA + MixUp + 512 px | Pausable")
    print("=" * 70)
    print("HOW TO PAUSE: create 'pause_training.flag' in the CV folder,")
    print("              OR press Ctrl+C.  Re-run the script to resume.\n")

    images_dir = os.path.join(BASE_DIR, "images", "train_images")
    csv_file   = os.path.join(BASE_DIR, "images", "train.csv")

    roi_files = [
        os.path.join(BASE_DIR, "images", "train_annotations.json"),
        os.path.join(BASE_DIR, "images", "test_annotations_roi_only.json"),
    ]
    img_name_to_bbox = load_rois(roi_files)

    original_df = pd.read_csv(csv_file)
    original_df['img_path'] = original_df['image_id'].apply(
        lambda x: os.path.join(images_dir, x))

    # ── Pseudo-labels ─────────────────────────────────────────
    pseudo_df   = None
    pseudo_file = os.path.join(BASE_DIR, "pseudo_labels.csv")
    if os.path.exists(pseudo_file):
        print(f"Loading pseudo-labels from {pseudo_file} …")
        pdf = pd.read_csv(pseudo_file)

        high_conf_1 = pdf[pdf['prob_reusable'] > 0.98].copy()
        high_conf_1['target'] = 1

        high_conf_0 = pdf[pdf['prob_reusable'] < 0.02].copy()
        high_conf_0['target'] = 0

        pseudo_df = pd.concat([high_conf_1, high_conf_0])
        pseudo_df['img_path'] = pseudo_df['image_id'].apply(
            lambda x: os.path.join(BASE_DIR, "images", "test_images", x))
        pseudo_df = pseudo_df[['image_id', 'target', 'img_path']]
        print(f"  {len(pseudo_df)} high-confidence pseudo-labels "
              f"(out of {len(pdf)} test images).")

    # ── Models & folds ───────────────────────────────────────
    models_to_train = [
        'convnext_base',
        'convnext_large',
        'tf_efficientnetv2_m',
    ]

    pause_handler = PauseHandler(PAUSE_FLAG)

    for model_name in models_to_train:
        print(f"\n{'='*70}\nModel: {model_name}\n{'='*70}")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(
                skf.split(original_df['image_id'], original_df['target']), 1):

            if fold > 3:
                print(f"\nStopping at fold 3 for {model_name} as configured.")
                break

            # ────────────────────────────────────────────────────────
            # MANUAL SKIP FOR CONVNEXT_LARGE FOLD 3
            # ────────────────────────────────────────────────────────
            if model_name == 'convnext_large' and fold == 3:
                print(f"\n[SKIP] Skipping {model_name} Fold {fold} as requested.")
                continue

            print(f"\n{'-'*40}\n{model_name} | Fold {fold}/3\n{'-'*40}")

            train_fold_df = original_df.iloc[train_idx].copy()
            val_fold_df   = original_df.iloc[val_idx].copy()

            if pseudo_df is not None:
                train_fold_df = pd.concat([train_fold_df, pseudo_df]).reset_index(drop=True)

            was_paused = train_fold(model_name, fold, train_fold_df, val_fold_df,
                                     BASE_DIR, img_name_to_bbox, pause_handler)

            if was_paused:
                print("[PAUSE] Stopping fold loop. Re-run to continue.")
                return

    print("\n\nAll models and folds complete!")


if __name__ == "__main__":
    main()