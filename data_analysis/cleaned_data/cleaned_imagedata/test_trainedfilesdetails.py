
import os
from pathlib import Path

def generate_image_details_report():
    # Define paths relative to the script location
    # Script: data_analysis/cleaned_data/cleaned_imagedata/test_trainedfilesdetails.py
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parents[2] 
    
    # Verify we are in the correct structure
    if not (PROJECT_ROOT / "data_analysis").exists():
        print(f"Warning: Standard project structure check failed at {PROJECT_ROOT}")
        # Fallback logic could go here, but assuming standard structure
            
    print(f"Project Root detected: {PROJECT_ROOT}")

    # Input Directory: data_analysis/test_traindeddata/image_data
    # Contains: imagetraindata, imagetestdata
    IMAGE_DATA_ROOT = PROJECT_ROOT / "data_analysis/test_traindeddata/image_data"
    
    # Output File: Same directory as script
    OUTPUT_FILE = SCRIPT_DIR / "test_trained_files_details.txt"
    
    print(f"Scanning directory: {IMAGE_DATA_ROOT}")
    print(f"Output report: {OUTPUT_FILE}")
    
    if not IMAGE_DATA_ROOT.exists():
        print(f"Error: Directory not found: {IMAGE_DATA_ROOT}")
        return

    with open(OUTPUT_FILE, 'w') as f:
        f.write("Image Data Analysis Details\n")
        f.write("===========================\n\n")
        
        total_files_all = 0
        
        # Folder names as per previous steps
        target_folders = ["imagetraindata", "imagetestdata"]
        
        for folder_name in target_folders:
            folder_path = IMAGE_DATA_ROOT / folder_name
            
            if not folder_path.exists():
                f.write(f"Folder not found: {folder_name}\n\n")
                continue
                
            f.write(f"Main Folder: {folder_name}\n")
            f.write("-" * 30 + "\n")
            
            classes = [d for d in folder_path.iterdir() if d.is_dir()]
            folder_total = 0
            
            if not classes:
                f.write("  No class folders found.\n")
            
            for class_dir in classes:
                class_name = class_dir.name
                files = [x for x in class_dir.iterdir() if x.is_file()]
                file_count = len(files)
                folder_total += file_count
                
                f.write(f"  Class: {class_name}\n")
                f.write(f"    Count: {file_count}\n")
                
                # List first few files as sample
                if file_count > 0:
                    sample_size = 5
                    sample_files = [file.name for file in files[:sample_size]]
                    f.write(f"    Sample Files: {', '.join(sample_files)}")
                    if file_count > sample_size:
                        f.write(" ... and more")
                    f.write("\n")
                f.write("\n")
            
            f.write(f"Total files in {folder_name}: {folder_total}\n")
            f.write("=" * 30 + "\n\n")
            total_files_all += folder_total

        f.write(f"Grand Total Analysis: {total_files_all} files processed.\n")
        
    print("Report generated successfully.")

if __name__ == "__main__":
    generate_image_details_report()
