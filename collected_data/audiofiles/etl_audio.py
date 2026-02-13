import os

def analyze_audio_classes(base_path, output_file):
    """
    Analyzes the classified_audio folder to count classes, files per class, 
    total files, and audio file extensions. Outputs to a text file.
    """
    classified_path = os.path.join(base_path, 'colected_data', 'audiofiles', 'classified_audio')
    
    print(f"Debug: Base path: {base_path}")
    print(f"Debug: Classified path: {classified_path}")
    
    if not os.path.exists(classified_path):
        with open(output_file, 'w') as f:
            f.write(f"Error: Path {classified_path} does not exist.\n")
        return
    
    try:
        # Get list of class folders (subdirectories)
        classes = [d for d in os.listdir(classified_path) if os.path.isdir(os.path.join(classified_path, d))]
        total_classes = len(classes)
        total_files = 0
        all_extensions = set()  # For total extensions
        
        with open(output_file, 'w') as f:
            f.write(f"Total classes: {total_classes}\n")
            f.write(f"Classes: {', '.join(classes)}\n")
            f.write("\nClass details:\n")
            
            for cls in sorted(classes):  # Sort for consistent output
                cls_path = os.path.join(classified_path, cls)
                files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
                num_files = len(files)
                total_files += num_files
                
                # Collect extensions per class
                class_extensions = set()
                for file in files:
                    ext = os.path.splitext(file)[1]
                    if ext:  # Only add if there's an extension
                        class_extensions.add(ext)
                        all_extensions.add(ext)
                
                f.write(f"- {cls}: {num_files} audio files, extensions: {', '.join(sorted(class_extensions)) if class_extensions else 'None'}\n")
            
            f.write(f"\nTotal audio files across all classes: {total_files}\n")
            f.write(f"Audio file extensions (all): {', '.join(sorted(all_extensions)) if all_extensions else 'None'}\n")
    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"Error during analysis: {str(e)}\n")

if __name__ == "__main__":
    # Assuming the script is run from the project root
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_path, 'colected_data', 'cleaned_data', 'cleaned_audiodata')
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    output_file = os.path.join(output_dir, 'audio_analysis.txt')
    analyze_audio_classes(base_path, output_file)
    print(f"Analysis output saved to: {output_file}")