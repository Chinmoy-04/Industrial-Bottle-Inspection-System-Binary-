import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.v2 as transforms
import timm

def main():
    print("Initializing PyTorch timm Inference Pipeline with TTA & Optimal Thresholds (V6)...")
    
    base_dir = r"D:\CV"
    test_images_dir = os.path.join(base_dir, "images", "test_images")
    sample_sub_file = os.path.join(base_dir, "images", "sample_submission.csv")
    output_file = os.path.join(base_dir, "pseudo_labels.csv")
    roi_file = os.path.join(base_dir, "images", "test_annotations_roi_only.json")
    
    # Load ROIs
    img_name_to_bbox = {}
    if os.path.exists(roi_file):
        print(f"Loading ROIs from {roi_file}...")
        with open(roi_file, 'r') as f:
            roi_data = json.load(f)
        
        img_id_to_name = {img['id']: img['file_name'] for img in roi_data.get('images', [])}
        for ann in roi_data.get('annotations', []):
            img_name = img_id_to_name[ann['image_id']]
            img_name_to_bbox[img_name] = ann['bbox']
    else:
        print(f"Warning: ROI file not found at {roi_file}")

    # Load Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    thresholds = []
    
    # Trying pure ConvNeXt ensemble without RepViT noise
    model_names = ['convnext_base']
    
    for model_name in model_names:
        # Give ConvNeXt higher weight since it scored ~0.968, and RepViT ~0.95
        model_weight = 2.5 if 'convnext' in model_name else 1.0 
        
        for fold in range(1, 4):
            run_dir = os.path.join(base_dir, "runs", "classify", "krones_challenge", f"{model_name}_v6_fold_{fold}")
            model_path = os.path.join(run_dir, "best.pth")
            thresh_path = os.path.join(run_dir, "optimal_threshold.txt")
            
            if os.path.exists(model_path):
                print(f"Loading {model_name} Fold {fold} (Weight: {model_weight})...")
                model = timm.create_model(model_name, pretrained=False, num_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                models.append((model, model_weight))
                
                if os.path.exists(thresh_path):
                    with open(thresh_path, "r") as f:
                        thresholds.append(float(f.read().strip()))
                else:
                    print(f"Warning: optimal_threshold.txt missing for {model_name} fold {fold}, defaulting to 0.5")
                    thresholds.append(0.5)
            else:
                print(f"Warning: Model {model_name} fold {fold} not found. Ensure training completed.")
                
    if not models:
        print("Error: No models found. Please run train_model_v6.py completely first.")
        return

    avg_threshold = sum(thresholds) / len(thresholds)
    print(f"\nSuccessfully loaded {len(models)} models.")
    print(f"Using average Optimal F1 Threshold for 'reusable': {avg_threshold:.4f}")
    
    df = pd.read_csv(sample_sub_file)
    results_list = []
    
    imgsz = 448
    val_transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Running inference on {len(df)} test images with Weighted Ensemble and 2x TTA...")
    
    for img_id in tqdm(df['image_id']):
        img_path = os.path.join(test_images_dir, img_id)
        if not os.path.exists(img_path):
            print(f"\nWarning: {img_path} not found. Defaulting target to 0.")
            results_list.append(0)
            continue
            
        try:
            # Load original image
            img = Image.open(img_path).convert('RGB')
            
            # Crop image if ROI is available
            if img_id in img_name_to_bbox:
                x, y, w, h = img_name_to_bbox[img_id]
                img = img.crop((x, y, x + w, y + h))
                
            # Create a horizontally flipped version for TTA
            img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Transform to tensors
            t_img = val_transform(img)
            t_img_flipped = val_transform(img_flipped)
            
            # Batch size of 2 for the two TTA images
            inputs = torch.stack([t_img, t_img_flipped]).to(device)
            
            total_probs = np.zeros(2)
            total_weight_sum = 0.0
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    for model, weight in models:
                        outputs = model(inputs)
                        # outputs is shape [2, 2] -> apply softmax to get probabilities
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        # Sum probabilities over the 2 TTA inputs, multiplied by the model's weight
                        total_probs += probs.sum(axis=0) * weight
                        total_weight_sum += weight * 2  # * 2 for the 2 TTA images
            
            # Average probabilities using the total weight
            avg_probs = total_probs / total_weight_sum
            
            # Index 1 corresponds to 'reusable'
            prob_reusable = avg_probs[1]
            results_list.append(prob_reusable)
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            results_list.append(0)
            
    df['prob_reusable'] = results_list
    df.to_csv(output_file, index=False)
    print(f"\nPseudo-labels complete! Probabilities are saved in: {output_file}")

if __name__ == "__main__":
    main()
