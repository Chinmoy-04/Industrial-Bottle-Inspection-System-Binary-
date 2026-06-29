import os
from ultralytics import YOLO
import glob
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

def main():
    base_dir = r"D:\CV"
    model_path = os.path.join(base_dir, "runs", "classify", "krones_challenge", "yolo_cls_run", "weights", "best.pt")
    val_dir = os.path.join(base_dir, "yolo_dataset", "val")
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    y_true = []
    y_pred = []
    
    classes = ['not_reusable', 'reusable']
    
    for cls_name in classes:
        cls_dir = os.path.join(val_dir, cls_name)
        images = glob.glob(os.path.join(cls_dir, "*.png")) + glob.glob(os.path.join(cls_dir, "*.jpg"))
        
        true_label = 1 if cls_name == 'reusable' else 0
        
        print(f"Running inference on {len(images)} images in {cls_name}...")
        for img_path in tqdm(images):
            # verbose=False to keep output clean
            results = model(img_path, verbose=False)
            top_class_idx = results[0].probs.top1
            pred_class_name = model.names[top_class_idx]
            
            pred_label = 1 if pred_class_name == 'reusable' else 0
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n--- Validation Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
