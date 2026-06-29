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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import timm
from timm.data import Mixup
from timm.utils import ModelEmaV2

class GpuTempMonitor:
    def __init__(self):
        self.running = False
        self.temps = []
        self.thread = None

    def _monitor(self):
        while self.running:
            try:
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                    encoding='utf-8',
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                temp = float(result.strip())
                self.temps.append(temp)
            except Exception:
                pass
            time.sleep(5)

    def start(self):
        self.running = True
        self.temps = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.temps:
            return sum(self.temps) / len(self.temps)
        return 0.0

def load_rois(filepaths):
    img_name_to_bbox = {}
    for filepath in filepaths:
        if os.path.exists(filepath):
            print(f"Loading ROIs from {filepath}...")
            with open(filepath, 'r') as f:
                roi_data = json.load(f)
            img_id_to_name = {img['id']: img['file_name'] for img in roi_data.get('images', [])}
            for ann in roi_data.get('annotations', []):
                img_name = img_id_to_name[ann['image_id']]
                img_name_to_bbox[img_name] = ann['bbox']
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
        img_path = row['img_path']
        img_name = row['image_id']
        target = row['target']
        
        img = Image.open(img_path).convert('RGB')
        
        # Crop using ROI if available
        if img_name in self.img_name_to_bbox:
            x, y, w, h = self.img_name_to_bbox[img_name]
            img = img.crop((x, y, x + w, y + h))
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(target, dtype=torch.long)

class SoftTargetFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(SoftTargetFocalLoss, self).__init__()
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets):
        log_probs = self.log_softmax(inputs)
        probs = torch.exp(log_probs)
        loss = -torch.sum(targets * (1 - probs)**self.gamma * log_probs, dim=-1)
        return loss.mean()

def train_fold(model_name, fold, train_df, val_df, base_dir, img_name_to_bbox):
    run_dir = os.path.join(base_dir, "runs", "classify", "krones_challenge", f"{model_name}_v8_fold_{fold}")
    os.makedirs(run_dir, exist_ok=True)
    
    best_weights_path = os.path.join(run_dir, "best.pth")
    last_weights_path = os.path.join(run_dir, "last.pth")
    thresh_file = os.path.join(run_dir, "optimal_threshold.txt")
    
    if os.path.exists(thresh_file) and os.path.exists(best_weights_path):
        print(f"\nFold {fold} for {model_name} already completely trained! Skipping...")
        return
        
    print(f"\nStarting new training for {model_name} - Fold {fold}...")
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
    
    # Hyperparameters for V8
    epochs = 30
    batch_size = 8
    accumulation_steps = 2  # Effective batch size = 16
    imgsz = 512             # Increased from 448
    
    # Transforms with RandAugment
    train_transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = KronesDataset(train_df, img_name_to_bbox, transform=train_transform)
    val_dataset = KronesDataset(val_df, img_name_to_bbox, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name, pretrained=True, num_classes=2).to(device)
    model_ema = ModelEmaV2(model, decay=0.9998, device=device)
    
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=2
    )
    
    criterion = SoftTargetFocalLoss(gamma=2.0)
    val_criterion = nn.CrossEntropyLoss()
    
    # Lower LR for fine-tuning the 512x512 resolution
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    start_epoch = 0
    best_f1 = 0.0
    patience = 5
    patience_counter = 0

    if os.path.exists(last_weights_path):
        print(f"Resuming {model_name} Fold {fold} from {last_weights_path}...")
        checkpoint = torch.load(last_weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'model_ema_state_dict' in checkpoint:
            model_ema.module.load_state_dict(checkpoint['model_ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed at epoch {start_epoch} with Best F1: {best_f1:.4f}")
    
    monitor = GpuTempMonitor()
    monitor.start()
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train")):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = mixup_fn(inputs, targets)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps
                
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
        
        # Evaluate using the EMA model for more stability
        model_ema.module.eval()
        val_loss = 0.0
        y_true = []
        y_probs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Val", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model_ema.module(inputs)
                    loss = val_criterion(outputs, targets)
                    
                val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                y_probs.extend(probs)
                y_true.extend(targets.cpu().numpy())
                
        val_loss /= len(val_loader)
        scheduler.step()
        
        epoch_best_f1 = 0
        for thresh in np.linspace(0.1, 0.9, 9):
            preds = [1 if p >= thresh else 0 for p in y_probs]
            score = f1_score(y_true, preds, zero_division=0)
            if score > epoch_best_f1:
                epoch_best_f1 = score
                
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1 (EMA): {epoch_best_f1:.4f}")
        
        if epoch_best_f1 > best_f1:
            best_f1 = epoch_best_f1
            patience_counter = 0
            torch.save(model_ema.module.state_dict(), best_weights_path)
            print(f"--> Saved new best EMA model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'model_ema_state_dict': model_ema.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_f1': best_f1,
            'patience_counter': patience_counter
        }
        tmp_last = last_weights_path + ".tmp"
        torch.save(checkpoint, tmp_last)
        os.replace(tmp_last, last_weights_path)
                
    avg_temp = monitor.stop()
    print(f"\n[INFO] Average GPU Temperature for {model_name} Fold {fold}: {avg_temp:.2f}°C\n")
    
    if os.path.exists(best_weights_path):
        model_ema.module.load_state_dict(torch.load(best_weights_path))
        model_ema.module.eval()
        y_true = []
        y_probs = []
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Optimal Threshold Search", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model_ema.module(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                y_probs.extend(probs)
                y_true.extend(targets.cpu().numpy())
                
        best_thresh_f1 = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.01, 0.99, 99):
            preds = [1 if p >= thresh else 0 for p in y_probs]
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_thresh_f1:
                best_thresh_f1 = score
                best_thresh = thresh
                
        with open(thresh_file, "w") as f:
            f.write(str(best_thresh))
        print(f"Optimal F1 Score for {model_name} Fold {fold}: {best_thresh_f1:.4f} at threshold: {best_thresh:.2f}")

def main():
    print("Initializing PyTorch timm Pipeline (V8 - Focal Loss + EMA + MixUp + 512px)...")
    
    base_dir = r"D:\CV"
    images_dir = os.path.join(base_dir, "images", "train_images")
    csv_file = os.path.join(base_dir, "images", "train.csv")
    
    roi_files = [
        os.path.join(base_dir, "images", "train_annotations.json"),
        os.path.join(base_dir, "images", "test_annotations_roi_only.json")
    ]
    img_name_to_bbox = load_rois(roi_files)
    
    original_df = pd.read_csv(csv_file)
    original_df['img_path'] = original_df['image_id'].apply(lambda x: os.path.join(images_dir, x))
    
    pseudo_df = None
    pseudo_file = os.path.join(base_dir, "pseudo_labels.csv")
    if os.path.exists(pseudo_file):
        print(f"Loading pseudo-labels from {pseudo_file}...")
        pdf = pd.read_csv(pseudo_file)
        
        high_conf_1 = pdf[pdf['prob_reusable'] > 0.98].copy()
        high_conf_1['target'] = 1
        
        high_conf_0 = pdf[pdf['prob_reusable'] < 0.02].copy()
        high_conf_0['target'] = 0
        
        pseudo_df = pd.concat([high_conf_1, high_conf_0])
        pseudo_df['img_path'] = pseudo_df['image_id'].apply(lambda x: os.path.join(base_dir, "images", "test_images", x))
        pseudo_df = pseudo_df[['image_id', 'target', 'img_path']]
        print(f"Found {len(pseudo_df)} high-confidence pseudo-labels out of {len(pdf)} test images.")
    
    models_to_train = ['convnext_base']
    
    for model_name in models_to_train:
        print(f"\n\n{'='*60}\nStarting pipeline for model: {model_name}\n{'='*60}")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(original_df['image_id'], original_df['target']), 1):
            if fold > 3:
                print(f"\nStopping training loop for {model_name} after Fold 3 as requested.")
                break
                
            print(f"\n{'-'*30}\nModel: {model_name} | Fold {fold}/5\n{'-'*30}")
            
            train_fold_df = original_df.iloc[train_idx].copy()
            val_fold_df = original_df.iloc[val_idx].copy()
            
            # Inject pseudo labels ONLY into training data! No leakage into validation.
            if pseudo_df is not None:
                train_fold_df = pd.concat([train_fold_df, pseudo_df]).reset_index(drop=True)
                
            train_fold(model_name, fold, train_fold_df, val_fold_df, base_dir, img_name_to_bbox)

if __name__ == "__main__":
    main()
