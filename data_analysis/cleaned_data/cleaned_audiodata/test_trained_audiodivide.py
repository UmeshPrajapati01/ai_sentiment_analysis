
import os
import shutil
import random
from pathlib import Path

def split_audio_dataset(source_dir, target_base_dir, train_ratio=0.8):
    """
    Splits the audio dataset into training and testing sets by copying files.
    
    Args:
        source_dir (str): Path to the source directory containing class subfolders.
        target_base_dir (str): Path to the base directory where train/test folders will be created.
        train_ratio (float): Ratio of data to use for training (default 0.8).
    """
    source_path = Path(source_dir)
    target_path = Path(target_base_dir)
    
    train_dir = target_path / "train_data"
    test_dir = target_path / "test_data"
    
    # Create target directories if they don't exist
    for dir_path in [train_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    classes = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(classes)} potential classes in {source_dir}")
    
    total_files = 0
    total_train = 0
    total_test = 0

    for class_dir in classes:
        class_name = class_dir.name
        print(f"Processing class: {class_name}")
        
        # Get all audio files (assuming mp3 based on analysis, but checking common exts)
        extensions = ['*.mp3', '*.wav', '*.flac']
        audio_files = []
        for ext in extensions:
            audio_files.extend(list(class_dir.glob(ext)))
            
        # Sort to ensure reproducibility before shuffle
        audio_files.sort()
        
        # Shuffle files
        random.shuffle(audio_files)
        
        split_idx = int(len(audio_files) * train_ratio)
        train_files = audio_files[:split_idx]
        test_files = audio_files[split_idx:]
        
        # CLEAR TARGET FIRST? No, that's dangerous if we do it per class loop.
        # But we should ensure we don't just add to existing.
        
        # Create class subdirectories in target folders
        class_train_dir = train_dir / class_name
        class_test_dir = test_dir / class_name
        
        # Clean existing class dirs to ensure fresh split
        if class_train_dir.exists():
            shutil.rmtree(class_train_dir)
        class_train_dir.mkdir(parents=True, exist_ok=True)
            
        if class_test_dir.exists():
            shutil.rmtree(class_test_dir)
        class_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for f in train_files:
            shutil.copy2(f, class_train_dir / f.name)
            
        for f in test_files:
            shutil.copy2(f, class_test_dir / f.name)
            
        print(f"  - Total: {len(audio_files)}")
        print(f"  - Training: {len(train_files)}")
        print(f"  - Testing: {len(test_files)}")
        
        total_files += len(audio_files)
        total_train += len(train_files)
        total_test += len(test_files)

    print("-" * 30)
    print("Audio Split complete.")
    print(f"Total files processed: {total_files}")
    print(f"Total training files: {total_train}")
    print(f"Total testing files: {total_test}")
    print(f"Output directory: {target_base_dir}")

if __name__ == "__main__":
    # Define paths relative to the script location
    # Script location: data_analysis/cleaned_data/cleaned_audiodata/test_trained_audiodivide.py
    # Project root is 3 levels up
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parents[2] 
    
    # Verify we are in the correct structure
    if not (PROJECT_ROOT / "data_analysis").exists():
        # Fallback: try to find the project root by looking for known folder
        print(f"Warning: Standard project structure check failed at {PROJECT_ROOT}")
        print("Attempting to locate project root...")
        current = SCRIPT_DIR
        while current.parent != current:
            if (current / "data_analysis").exists():
                PROJECT_ROOT = current
                break
            current = current.parent
            
    print(f"Project Root detected: {PROJECT_ROOT}")

    SOURCE_DIR = PROJECT_ROOT / "data_analysis/datasets/audiofiles/classified_audio"
    TARGET_BASE_DIR = PROJECT_ROOT / "data_analysis/test_traindeddata/audio_data"
    
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_BASE_DIR}")
    
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory not found: {SOURCE_DIR}")
        print("Please ensure you are running this script within the proper project structure.")
    else:
        split_audio_dataset(SOURCE_DIR, TARGET_BASE_DIR)
