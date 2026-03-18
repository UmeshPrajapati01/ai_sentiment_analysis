import torch
import numpy as np
import librosa
from PIL import Image
from torchvision import transforms
from backend.inference.model_loader import loader
from backend.utils import extract_mel_spectrogram_sota # Use the 99% accuracy extractor

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
        image_tensor = image_transforms(image).unsqueeze(0).to(loader.device)
        
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
    """
    Predicts cat emotion from audio. 
    Switches dynamically between SOTA (CNN/Keras) and Legacy (RFC/Joblib).
    """
    if not loader.audio_model:
        return "Model not loaded"
        
    try:
        # ─── New SOTA High-Accuracy Path ─────────────────────────────────────
        if hasattr(loader.audio_model, 'predict') and not hasattr(loader.audio_model, 'predict_proba'):
            # This is likely the SOTA Keras model (99% candidate)
            features = extract_mel_spectrogram_sota(audio_path)
            if features is None:
                print(f"[Predictor] Error: Spectrogram extraction failed for {audio_path}")
                return "Error (Processing)"
            
            # Predict
            preds = loader.audio_model.predict(features, verbose=0)
            class_idx = int(np.argmax(preds[0]))
            
            # Map index to label
            CLASSES = ['angry', 'fearful', 'happy', 'sad']
            if hasattr(loader, 'audio_label_encoder') and loader.audio_label_encoder:
                try:
                    return loader.audio_label_encoder.inverse_transform([class_idx])[0]
                except:
                    return CLASSES[class_idx] if class_idx < 4 else "Unknown"
            return CLASSES[class_idx] if class_idx < 4 else "Unknown"

        # ─── Legacy RFC Path (Backup) ──────────────────────────────────────────
        else:
            y, sr = librosa.load(audio_path, sr=22050, duration=3)
            if len(y) < 22050 * 3:
                y = np.pad(y, (0, int(22050 * 3) - len(y)))
                
            # Extract statistical features
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
            
            features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
            features_scaled = loader.audio_scaler.transform([features])
            
            prediction_idx = loader.audio_model.predict(features_scaled)[0]
            return loader.audio_label_encoder.inverse_transform([prediction_idx])[0]
        
    except Exception as e:
        import traceback
        print(f"[Predictor] CRITICAL ERROR: {e}")
        traceback.print_exc()
        return "Error"
