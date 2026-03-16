"""
preprocess_images.py
Pipeline to load cat images, detect faces, resize (224x224), normalize, 
apply augmentations (rotation, flip, brightness), and save to processed directory.
"""

import os
import cv2
import numpy as np
import glob

# Paths
INPUT_DIR = "../data/raw/images"
OUTPUT_DIR = "../data/processed/images"
TARGET_SIZE = (224, 224)

# OpenCV Cat Face Haar Cascade
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
cat_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def apply_augmentations(image):
    """Applies rotation, horizontal flip, and brightness shift."""
    augmented_images = []
    
    # 1. Original (already processed/resized)
    augmented_images.append(image)
    
    # 2. Horizontal Flip
    flipped = cv2.flip(image, 1) # 1 means horizontal
    augmented_images.append(flipped)
    
    # 3. Rotation (+15 degrees and -15 degrees)
    rows, cols = image.shape[:2]
    M_plus = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    rotated_plus = cv2.warpAffine(image, M_plus, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    augmented_images.append(rotated_plus)
    
    M_minus = cv2.getRotationMatrix2D((cols/2, rows/2), -15, 1)
    rotated_minus = cv2.warpAffine(image, M_minus, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    augmented_images.append(rotated_minus)
    
    # 4. Brightness Shift (+30 and -30)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    v_plus = cv2.add(v, 30)
    hsv_plus = cv2.merge((h, s, v_plus))
    bright = cv2.cvtColor(hsv_plus, cv2.COLOR_HSV2RGB)
    augmented_images.append(bright)
    
    v_minus = cv2.subtract(v, 30)
    hsv_minus = cv2.merge((h, s, v_minus))
    dark = cv2.cvtColor(hsv_minus, cv2.COLOR_HSV2RGB)
    augmented_images.append(dark)
    
    return augmented_images

def process_image(img_path, output_class_dir):
    """
    Loads image, detects face, resizes to 224x224, converts to RGB, 
    normalizes internally prior to augmentations (rescaled for saving), 
    augments, and saves.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        return
        
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect face (use grayscale for detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check if cascade loaded successfully
    if not cat_cascade.empty():
        faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
    else:
        faces = []
        
    # Crop to face if detected, otherwise use full image
    if len(faces) > 0:
        (x, y, w, h) = faces[0] # Take the first detected cat face
        cropped = img_rgb[y:y+h, x:x+w]
    else:
        cropped = img_rgb
        
    # Resize to 224x224
    resized = cv2.resize(cropped, TARGET_SIZE)
    
    # Normalize pixel values (0 to 1) for the pipeline
    # Note: We scale it to 0-1 here as requested, but we multiply by 255.0 to 
    # generate valid image representations (for augmenting/saving to disc)
    normalized = resized.astype(np.float32) / 255.0
    rescaled = (normalized * 255.0).astype(np.uint8)
    
    # Augment
    aug_images = apply_augmentations(rescaled)
    
    # Save processed images
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    for idx, aug_img in enumerate(aug_images):
        save_path = os.path.join(output_class_dir, f"{base_name}_aug_{idx}.jpg")
        # OpenCV saves in BGR, so convert back from RGB before saving
        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} not found. Please place raw images here.")
        # Create directories to help the user
        os.makedirs(INPUT_DIR, exist_ok=True)
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    classes = os.listdir(INPUT_DIR)
    for cls in classes:
        cls_path = os.path.join(INPUT_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
            
        out_cls_path = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(out_cls_path, exist_ok=True)
        
        img_paths = glob.glob(os.path.join(cls_path, '*.*'))
        print(f"Processing {len(img_paths)} images for class '{cls}'...")
        for p in img_paths:
            # Avoid processing unknown files
            if p.lower().endswith(('.jpg', '.jpeg', '.png')):
                process_image(p, out_cls_path)
            
    print("Image preprocessing complete.")

if __name__ == "__main__":
    main()
