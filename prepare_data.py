import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    # Paths
    base_dir = r"D:\CV"
    images_dir = os.path.join(base_dir, "images", "train_images")
    csv_file = os.path.join(base_dir, "images", "train.csv")
    output_dir = os.path.join(base_dir, "yolo_dataset")
    
    # Target map (Assuming 1 is reusable, 0 is not reusable based on typical conventions)
    target_map = {1: "reusable", 0: "not_reusable"}
    
    print("Reading CSV...")
    df = pd.read_csv(csv_file)
    
    print(f"Total images in CSV: {len(df)}")
    
    # Split 80/20 train/val
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    # Create directory structure
    for split in ['train', 'val']:
        for label in target_map.values():
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)
            
    def process_split(split_df, split_name):
        print(f"Processing {split_name} split ({len(split_df)} images)...")
        # We use a simple loop, tqdm for progress
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            img_id = row['image_id']
            target = row['target']
            class_name = target_map[target]
            
            src_path = os.path.join(images_dir, img_id)
            dst_path = os.path.join(output_dir, split_name, class_name, img_id)
            
            if os.path.exists(src_path):
                # Using hardlink if possible to save disk space and time, fallback to copy
                try:
                    if not os.path.exists(dst_path):
                        os.link(src_path, dst_path)
                except OSError:
                    shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Image {src_path} not found.")

    process_split(train_df, 'train')
    process_split(val_df, 'val')
    print("Dataset preparation complete! Ready for YOLOv8 Classification.")

if __name__ == "__main__":
    main()
