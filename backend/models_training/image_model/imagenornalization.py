# ...existing code...
"""
Image normalization, augmentation and EDA utilities (A1-A4).

Functions:
- acquire_images(image_dir): collect (path, label) pairs for .jpg/.jpeg/.png
- preprocess_image(img_path, size): resize + grayscale
- augment_image(img): rotation, flip, brightness
- save_images(images, base_name, output_dir, label): save images under output_dir/label
- plot_class_distribution(data): barplot of class counts
- main(image_dir, output_dir): example runner / CLI
"""
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter
from typing import List, Tuple

VALID_EXTS = ('*.jpg', '*.jpeg', '*.png')

# A1: Image Dataset Acquisition
def acquire_images(image_dir: str) -> List[Tuple[str, str]]:
    """
    Load labeled cat face images from directory.
    Returns list of (image_path, label).
    Expects image_dir/<label>/*.{jpg,png,jpeg}
    """
    data = []
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    for label in sorted(os.listdir(image_dir)):
        label_dir = os.path.join(image_dir, label)
        if os.path.isdir(label_dir):
            for ext in VALID_EXTS:
                for img_path in glob(os.path.join(label_dir, ext)):
                    data.append((img_path, label))
    return data

# A2: Image Preprocessing Pipeline
def preprocess_image(img_path: str, size: Tuple[int, int]=(128, 128)) -> np.ndarray:
    """
    Read image, resize and convert to grayscale uint8.
    Returns: grayscale image as uint8 array (0-255).
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray  # uint8

# A3: Augmentation Strategy
def augment_image(img: np.ndarray) -> List[np.ndarray]:
    """
    Apply rotation, horizontal flip and brightness shifts.
    Input: grayscale uint8 image.
    Returns: list of augmented uint8 images (includes original).
    """
    augmented = []
    h, w = img.shape[:2]
    # keep original
    augmented.append(img.copy())
    # Rotations
    for angle in (15, -15):
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)
    # Horizontal flip
    augmented.append(cv2.flip(img, 1))
    # Brightness shifts (beta)
    for beta in (30, -30):
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        augmented.append(bright)
    return augmented

# Save images
def save_images(images: List[np.ndarray], base_name: str, output_dir: str, label: str):
    """
    Save a list of uint8 images to output_dir/label with base_name.
    """
    target_dir = os.path.join(output_dir, label)
    os.makedirs(target_dir, exist_ok=True)
    for idx, img in enumerate(images):
        out_path = os.path.join(target_dir, f"{base_name}_aug_{idx}.jpg")
        cv2.imwrite(out_path, img)

# A4: EDA - Image Class Balance
def plot_class_distribution(data: List[Tuple[str, str]], show: bool = True):
    """
    Visualize distribution of emotions.
    """
    labels = [label for _, label in data]
    counter = Counter(labels)
    plt.figure(figsize=(6,4))
    plt.bar(counter.keys(), counter.values(), color='C0')
    plt.title("Class Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    if show:
        plt.show()

def main(image_dir: str, output_dir: str, preview: bool=False):
    data = acquire_images(image_dir)
    if not data:
        print("No images found in", image_dir); return
    plot_class_distribution(data, show=preview)
    for img_path, label in data:
        try:
            img = preprocess_image(img_path)
            augmented = augment_image(img)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_images(augmented, base_name, output_dir, label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image normalization & augmentation (A1-A4)")
    parser.add_argument("--image-dir", type=str,
                        default=r"C:\Users\ramak\OneDrive\Desktop\project_smartx\project_mew\Dev-of-an-AI-Based-Cat-Emotion-Recognition-System-Using-Facial-and-Vocal-Analysis_Feb_Batch-8_2026\uncleaned_datasets\cleaned_data\cat_classifieddataimage",
                        help="Root folder with class subfolders")
    parser.add_argument("--output-dir", type=str,
                        default=r"C:\Users\ramak\OneDrive\Desktop\project_smartx\project_mew\Dev-of-an-AI-Based-Cat-Emotion-Recognition-System-Using-Facial-and-Vocal-Analysis_Feb_Batch-8_2026\output_weekwise\week1output\image",
                        help="Directory to save processed/augmented images")
    parser.add_argument("--preview", action="store_true", help="Show class distribution plot")
    args = parser.parse_args()
    main(args.image_dir, args.output_dir, preview=args.preview)
# ...existing code...