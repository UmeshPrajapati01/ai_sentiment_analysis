
import torch
import numpy as np
import librosa
from PIL import Image
from torchvision import transforms
from backend.inference.model_loader import loader

# Image Transforms (Must match training validation transforms)
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    if not loader.image_model:
        return "Model not loaded"
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_transforms(image).unsqueeze(0) # Add batch dimension
        image_tensor = image_tensor.to(loader.device)
        
        with torch.no_grad():
            outputs = loader.image_model(image_tensor)
            _, preds = torch.max(outputs, 1)
            
        class_idx = preds.item()
        if loader.image_classes and class_idx < len(loader.image_classes):
            return loader.image_classes[class_idx]
        return f"Class {class_idx}"
        
    except Exception as e:
        print(f"Image prediction error: {e}")
        return "Error"

def predict_audio(audio_path):
    if not loader.audio_model:
        return "Model not loaded"
        
    try:
        # Feature Extraction (Must match training logic exactly)
        # Using snippet from audio_model.py logic
        y, sr = librosa.load(audio_path, sr=22050, duration=3)
        
        if len(y) < 22050 * 3:
            padding = int(22050 * 3) - len(y)
            y = np.pad(y, (0, padding))
            
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)
        
        features = np.hstack([mfcc_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean])
        
        # Scale features
        features_scaled = loader.audio_scaler.transform([features])
        
        # Predict
        prediction_idx = loader.audio_model.predict(features_scaled)[0]
        prediction_label = loader.audio_label_encoder.inverse_transform([prediction_idx])[0]
        
        return prediction_label
        
    except Exception as e:
        print(f"Audio prediction error: {e}")
        return "Error"
