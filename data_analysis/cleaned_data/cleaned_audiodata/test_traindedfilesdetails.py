
import os
from pathlib import Path

def generate_split_report():
    # Define paths relative to the script location
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

    AUDIO_DATA_ROOT = PROJECT_ROOT / "data_analysis/test_traindeddata/audio_data"
    OUTPUT_FILE = SCRIPT_DIR / "audio_split_details.txt"
    
    print(f"Scanning directory: {AUDIO_DATA_ROOT}")
    print(f"Output report: {OUTPUT_FILE}")
    
    if not AUDIO_DATA_ROOT.exists():
        print(f"Error: Directory not found: {AUDIO_DATA_ROOT}")
        return

    with open(OUTPUT_FILE, 'w') as f:
        f.write("Audio Data Split Details\n")
        f.write("========================\n\n")
        
        total_files_all = 0
        
        # Iterate over train_data and test_data
        for split_type in ["train_data", "test_data"]:
            split_dir = AUDIO_DATA_ROOT / split_type
            if not split_dir.exists():
                f.write(f"Directory not found: {split_type}\n\n")
                continue
                
            f.write(f"Folder: {split_type}\n")
            f.write("-" * 20 + "\n")
            
            classes = [d for d in split_dir.iterdir() if d.is_dir()]
            split_total = 0
            
            for class_dir in classes:
                class_name = class_dir.name
                files = [x for x in class_dir.iterdir() if x.is_file()]
                file_count = len(files)
                split_total += file_count
                
                f.write(f"  Class: {class_name} ({file_count} files)\n")
                if file_count > 0:
                     f.write(f"    Files: {', '.join([file.name for file in files])}\n")
                f.write("\n")
            
            f.write(f"Total in {split_type}: {split_total}\n")
            f.write("=" * 30 + "\n\n")
            total_files_all += split_total

        f.write(f"Grand Total Audio Files: {total_files_all}\n")
        
    print("Report generated successfully.")

if __name__ == "__main__":
    generate_split_report()
