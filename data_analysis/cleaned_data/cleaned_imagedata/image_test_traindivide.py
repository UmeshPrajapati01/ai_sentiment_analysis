
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, target_base_dir, train_ratio=0.8):
    """
    Splits the dataset into training and testing sets by copying files.
    
    Args:
        source_dir (str): Path to the source directory containing class subfolders.
        target_base_dir (str): Path to the base directory where train/test folders will be created.
        train_ratio (float): Ratio of data to use for training (default 0.8).
    """
    source_path = Path(source_dir)
    target_path = Path(target_base_dir)
    
    train_dir = target_path / "imagetraindata"
    test_dir = target_path / "imagetestdata" # User requested "image test data", normalizing to imagetestdata
    
    # Create target directories if they don't exist
    for dir_path in [train_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    classes = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(classes)} potential classes in {source_dir}")
    
    total_images = 0
    total_train = 0
    total_test = 0

    for class_dir in classes:
        class_name = class_dir.name
        
        # Skip "Master Folder" as it's likely not a class to split or empty based on analysis
        if class_name.lower() == "master folder":
            print(f"Skipping '{class_name}'...")
            continue
            
        print(f"Processing class: {class_name}")
        
        # Get all image files (assuming jpg based on analysis, but checking common exts)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(list(class_dir.glob(ext)))
            
        # Sort to ensure reproducibility before shuffle, though we shuffle next anyway
        images.sort()
        
        # Shuffle images
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create class subdirectories in target folders
        class_train_dir = train_dir / class_name
        class_test_dir = test_dir / class_name
        
        class_train_dir.mkdir(exist_ok=True)
        class_test_dir.mkdir(exist_ok=True)
        
        # Copy files
        for img in train_images:
            shutil.copy2(img, class_train_dir / img.name)
            
        for img in test_images:
            shutil.copy2(img, class_test_dir / img.name)
            
        print(f"  - Total: {len(images)}")
        print(f"  - Training: {len(train_images)}")
        print(f"  - Testing: {len(test_images)}")
        
        total_images += len(images)
        total_train += len(train_images)
        total_test += len(test_images)

    print("-" * 30)
    print("Split complete.")
    print(f"Total images processed: {total_images}")
    print(f"Total training images: {total_train}")
    print(f"Total testing images: {total_test}")
    print(f"Output directory: {target_base_dir}")

if __name__ == "__main__":
    # Define paths relative to the script location or absolute paths
    # Using absolute paths based on user context for robustness
    
    BASE_PROJECT_DIR = Path(r"c:\Users\ramak\OneDrive\Desktop\project_smartx\project_mew\Dev-of-an-AI-Based-Cat-Emotion-Recognition-System-Using-Facial-and-Vocal-Analysis_Feb_Batch-8_2026")
    SOURCE_DIR = BASE_PROJECT_DIR / "data_analysis/datasets/imagefiles/cat_classifieddataimage"
    TARGET_BASE_DIR = BASE_PROJECT_DIR / "data_analysis/test_traindeddata"
    
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_BASE_DIR}")
    
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory not found: {SOURCE_DIR}")
    else:
        split_dataset(SOURCE_DIR, TARGET_BASE_DIR)
