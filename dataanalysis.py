import os
import argparse

def analyze_audio_files(directory_path):
    """
    Recursively analyzes a directory to count .mp3, .wav, and other files.

    Args:
        directory_path (str): The path to the directory to analyze.
    """
    total_files = 0
    mp3_count = 0
    wav_count = 0
    other_count = 0

    if not os.path.exists(directory_path):
        print(f"Error: The directory {directory_path} does not exist.")
        return

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            total_files += 1
            if file.lower().endswith('.mp3'):
                mp3_count += 1
            elif file.lower().endswith('.wav'):
                wav_count += 1
            else:
                other_count += 1

    print(f"Analysis for directory: {directory_path}")
    print("-" * 30)
    print(f"Total files found: {total_files}")
    print(f"Number of .mp3 files: {mp3_count}")
    print(f"Number of .wav files: {wav_count}")
    print(f"Number of other files: {other_count}")

if __name__ == "__main__":
    # Default path from original code
    default_path = r"c:\Users\ramak\OneDrive\Desktop\project_smartx\project_mew\Dev-of-an-AI-Based-Cat-Emotion-Recognition-System-Using-Facial-and-Vocal-Analysis_Feb_Batch-8_2026\data"
    
    # Allow command line argument for path, defaulting to the hardcoded path if not provided
    parser = argparse.ArgumentParser(description="Analyze audio file distribution in a directory.")
    parser.add_argument("path", nargs="?", default=default_path, help="Path to the data folder")
    
    args = parser.parse_args()
    analyze_audio_files(args.path)
