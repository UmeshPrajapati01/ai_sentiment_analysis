"""
===============================================================================
CAT EMOTION RECOGNITION - MODEL FINE-TUNING SCRIPT
===============================================================================

This script fine-tunes two existing trained models:
1. Image Classification Model (PyTorch ResNet50)
2. Audio Classification Model (Scikit-learn RandomForest)

The models are fine-tuned on new labeled datasets and saved for production use.

Author: Senior ML Engineer
Date: 2026-02-15
===============================================================================
"""

import os
import sys
import time
import copy
import warnings
from pathlib import Path

# Data Science Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Audio Processing
import librosa

# Machine Learning - Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Deep Learning - PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================

# Determine Project Root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# ==========================================
# IMAGE MODEL CONFIGURATION
# ==========================================

# Existing Image Model Path
IMAGE_MODEL_PATH = PROJECT_ROOT / "backend/trained_modelimages/image_model/model.pth"
IMAGE_CLASSES_PATH = PROJECT_ROOT / "backend/trained_modelimages/image_model/classes.txt"

# New Image Training Data
IMAGE_DATA_DIR = PROJECT_ROOT / "data_analysis/datasets/imagefiles/cat_classifieddataimage"

# Image Output Directory (Fine-tuned)
IMAGE_OUTPUT_DIR = PROJECT_ROOT / "backend/trained_modelimages/image_model_finetuned"

# Image Hyperparameters
IMAGE_BATCH_SIZE = 16
IMAGE_NUM_EPOCHS = 15  # More epochs for fine-tuning
IMAGE_LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning
IMAGE_VAL_SPLIT = 0.2
IMAGE_NUM_WORKERS = 0
IMAGE_FREEZE_LAYERS = True  # Whether to freeze base layers during fine-tuning

# ==========================================
# AUDIO MODEL CONFIGURATION
# ==========================================

# Existing Audio Model Paths
AUDIO_MODEL_PATH = PROJECT_ROOT / "backend/trained_modelimages/audio_model/audio_classifier.pkl"
AUDIO_SCALER_PATH = PROJECT_ROOT / "backend/trained_modelimages/audio_model/scaler.pkl"
AUDIO_ENCODER_PATH = PROJECT_ROOT / "backend/trained_modelimages/audio_model/label_encoder.pkl"

# New Audio Training Data
AUDIO_DATA_DIR = PROJECT_ROOT / "data_analysis/datasets/audiofiles/classified_audio"

# Audio Output Directory (Fine-tuned)
AUDIO_OUTPUT_DIR = PROJECT_ROOT / "backend/trained_modelimages/audio_model_finetuned"

# Audio Processing Settings
SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_ESTIMATORS = 150  # Increase for better performance
RANDOM_STATE = 42

# ==========================================
# GENERAL SETTINGS
# ==========================================

# Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create output directories
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def save_training_plots(history, output_dir, prefix=""):
    """
    Save training and validation loss/accuracy plots.
    
    Args:
        history: Dictionary containing training metrics
        output_dir: Path to save plots
        prefix: Prefix for plot filenames
    """
    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], 'o-', label='Train Loss', linewidth=2)
    plt.plot(history['epoch'], history['val_loss'], 's-', label='Val Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], 'o-', label='Train Acc', linewidth=2)
    plt.plot(history['epoch'], history['val_acc'], 's-', label='Val Acc', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}training_curves.png', dpi=150)
    plt.close()
    print(f"✓ Training plots saved: {prefix}training_curves.png")


# ==========================================
# IMAGE MODEL FINE-TUNING
# ==========================================

class ImageModelFineTuner:
    """
    Class to handle fine-tuning of the ResNet50 image classification model.
    """
    
    def __init__(self):
        self.device = device
        self.model = None
        self.dataloaders = None
        self.dataset_sizes = None
        self.class_names = None
        
    def prepare_data(self):
        """
        Load and prepare image data with augmentation.
        Creates train/validation split from the new dataset.
        """
        print(f"Loading image data from: {IMAGE_DATA_DIR}")
        
        if not IMAGE_DATA_DIR.exists():
            raise FileNotFoundError(f"Image data directory not found: {IMAGE_DATA_DIR}")
        
        # Define data transforms with augmentation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # Load full dataset
        full_dataset = datasets.ImageFolder(str(IMAGE_DATA_DIR))
        self.class_names = full_dataset.classes
        print(f"✓ Classes found: {self.class_names}")
        
        # Split into train and validation
        num_data = len(full_dataset)
        num_val = int(IMAGE_VAL_SPLIT * num_data)
        num_train = num_data - num_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [num_train, num_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create wrapper class to apply transforms to subsets
        class DatasetFromSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform
                
            def __getitem__(self, index):
                x, y = self.subset[index]
                if self.transform:
                    x = self.transform(x)
                return x, y
            
            def __len__(self):
                return len(self.subset)
        
        # Apply transforms
        train_data = DatasetFromSubset(train_dataset, transform=data_transforms['train'])
        val_data = DatasetFromSubset(val_dataset, transform=data_transforms['val'])
        
        # Create dataloaders
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(
                train_data, batch_size=IMAGE_BATCH_SIZE, 
                shuffle=True, num_workers=IMAGE_NUM_WORKERS
            ),
            'val': torch.utils.data.DataLoader(
                val_data, batch_size=IMAGE_BATCH_SIZE, 
                shuffle=False, num_workers=IMAGE_NUM_WORKERS
            )
        }
        
        self.dataset_sizes = {'train': len(train_data), 'val': len(val_data)}
        
        print(f"✓ Training samples: {self.dataset_sizes['train']}")
        print(f"✓ Validation samples: {self.dataset_sizes['val']}")
        
    def load_pretrained_model(self):
        """
        Load the existing trained model and modify for new classes if needed.
        """
        print(f"Loading existing model from: {IMAGE_MODEL_PATH}")
        
        if not IMAGE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Pre-trained model not found: {IMAGE_MODEL_PATH}")
        
        # Initialize ResNet50 architecture
        self.model = models.resnet50(weights=None)
        
        # Modify final layer for current number of classes
        num_ftrs = self.model.fc.in_features
        num_classes = len(self.class_names)
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Load pre-trained weights
        try:
            state_dict = torch.load(IMAGE_MODEL_PATH, map_location=self.device)
            
            # Try to load weights (handle potential class mismatch)
            try:
                self.model.load_state_dict(state_dict)
                print("✓ Loaded pre-trained weights successfully")
            except RuntimeError as e:
                print(f"⚠ Warning: {e}")
                print("⚠ Class mismatch detected. Loading partial weights...")
                
                # Load all weights except the final layer
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() 
                                 if k in model_dict and 'fc' not in k}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                print("✓ Loaded partial weights (excluding final layer)")
                
        except Exception as e:
            print(f"⚠ Error loading weights: {e}")
            print("⚠ Starting with ImageNet pre-trained weights instead")
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Freeze base layers if specified
        if IMAGE_FREEZE_LAYERS:
            print("🔒 Freezing base layers (only training final layer)...")
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze the final layer
            for param in self.model.fc.parameters():
                param.requires_grad = True
                
            print("✓ Base layers frozen, final layer trainable")
        else:
            print("🔓 All layers trainable")
        
        self.model = self.model.to(self.device)
        
    def train_model(self):
        """
        Fine-tune the model on the new dataset.
        """
        print_section("STARTING IMAGE MODEL FINE-TUNING")
        
        criterion = nn.CrossEntropyLoss()
        
        # Only optimize parameters that require gradients
        params_to_update = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_update, lr=IMAGE_LEARNING_RATE)
        
        # Learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training loop
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(IMAGE_NUM_EPOCHS):
            print(f'\nEpoch {epoch + 1}/{IMAGE_NUM_EPOCHS}')
            print('-' * 40)
            
            # Each epoch has training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Backward pass + optimize
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                print(f'{phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
                
                # Record history
                if phase == 'train':
                    history['epoch'].append(epoch + 1)
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())
                
                # Save best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    print(f'★ New best model! Validation Accuracy: {best_acc:.4f}')
        
        time_elapsed = time.time() - since
        print(f'\n✓ Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'✓ Best validation accuracy: {best_acc:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        return history
    
    def save_model(self, history):
        """
        Save the fine-tuned model and training artifacts.
        """
        print_section("SAVING IMAGE MODEL ARTIFACTS")
        
        # Save model state
        model_path = IMAGE_OUTPUT_DIR / 'model.pth'
        torch.save(self.model.state_dict(), model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Save class names
        classes_path = IMAGE_OUTPUT_DIR / 'classes.txt'
        with open(classes_path, 'w') as f:
            for cls in self.class_names:
                f.write(f"{cls}\n")
        print(f"✓ Classes saved: {classes_path}")
        
        # Save training history
        history_df = pd.DataFrame(history)
        history_path = IMAGE_OUTPUT_DIR / 'training_history.csv'
        history_df.to_csv(history_path, index=False)
        print(f"✓ Training history saved: {history_path}")
        
        # Save plots
        save_training_plots(history, IMAGE_OUTPUT_DIR, prefix='image_')
        
        # Save metadata
        metadata = {
            'num_classes': len(self.class_names),
            'classes': self.class_names,
            'batch_size': IMAGE_BATCH_SIZE,
            'num_epochs': IMAGE_NUM_EPOCHS,
            'learning_rate': IMAGE_LEARNING_RATE,
            'best_val_acc': max(history['val_acc']),
            'freeze_layers': IMAGE_FREEZE_LAYERS
        }
        
        metadata_path = IMAGE_OUTPUT_DIR / 'metadata.txt'
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"✓ Metadata saved: {metadata_path}")
        
        print(f"\n✅ All image model artifacts saved to: {IMAGE_OUTPUT_DIR}")


# ==========================================
# AUDIO MODEL FINE-TUNING
# ==========================================

class AudioModelFineTuner:
    """
    Class to handle fine-tuning of the RandomForest audio classification model.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.classes = None
        
    def extract_features(self, file_path):
        """
        Extract audio features from a file.
        
        Features extracted:
        - MFCC (Mel-frequency cepstral coefficients)
        - Chroma
        - Mel Spectrogram
        - Spectral Contrast
        - Tonnetz
        
        Args:
            file_path: Path to audio file
            
        Returns:
            numpy array of concatenated features
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
            
            # Pad if shorter than DURATION
            if len(audio) < SAMPLE_RATE * DURATION:
                padding = int(SAMPLE_RATE * DURATION) - len(audio)
                audio = np.pad(audio, (0, padding))
            
            # Extract features
            # 1. MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            
            # 2. Chroma
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            chroma_mean = np.mean(chroma.T, axis=0)
            
            # 3. Mel Spectrogram
            mel = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_mean = np.mean(mel.T, axis=0)
            
            # 4. Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            contrast_mean = np.mean(contrast.T, axis=0)
            
            # 5. Tonnetz
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
            tonnetz_mean = np.mean(tonnetz.T, axis=0)
            
            # Concatenate all features
            features = np.hstack([mfcc_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean])
            
            return features
            
        except Exception as e:
            print(f"⚠ Error extracting features from {file_path}: {e}")
            return None
    
    def load_data(self):
        """
        Load audio data from directory structure and extract features.
        """
        print(f"Loading audio data from: {AUDIO_DATA_DIR}")
        
        if not AUDIO_DATA_DIR.exists():
            raise FileNotFoundError(f"Audio data directory not found: {AUDIO_DATA_DIR}")
        
        # Get class directories
        classes = [d.name for d in AUDIO_DATA_DIR.iterdir() if d.is_dir()]
        if not classes:
            raise ValueError("No class subdirectories found in audio data directory")
        
        self.classes = sorted(classes)
        print(f"✓ Detected {len(self.classes)} classes: {self.classes}")
        
        features_list = []
        labels_list = []
        
        # Extract features from all audio files
        for class_name in self.classes:
            class_dir = AUDIO_DATA_DIR / class_name
            audio_files = list(class_dir.glob("*.mp3")) + list(class_dir.glob("*.wav"))
            
            print(f"Processing '{class_name}': {len(audio_files)} files...")
            
            for file_path in audio_files:
                features = self.extract_features(file_path)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(class_name)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"✓ Data loaded. Features shape: {X.shape}, Labels shape: {y.shape}")
        
        return X, y
    
    def prepare_data(self):
        """
        Prepare data for training: encode labels, scale features, and split.
        """
        print_section("PREPARING AUDIO DATA")
        
        # Load and extract features
        X, y = self.load_data()
        
        if len(X) == 0:
            raise ValueError("No data loaded!")
        
        # Try to load existing label encoder
        try:
            if AUDIO_ENCODER_PATH.exists():
                self.label_encoder = joblib.load(AUDIO_ENCODER_PATH)
                print(f"✓ Loaded existing label encoder from {AUDIO_ENCODER_PATH}")
            else:
                self.label_encoder = LabelEncoder()
                print("✓ Created new label encoder")
        except:
            self.label_encoder = LabelEncoder()
            print("✓ Created new label encoder")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Try to load existing scaler
        try:
            if AUDIO_SCALER_PATH.exists():
                self.scaler = joblib.load(AUDIO_SCALER_PATH)
                print(f"✓ Loaded existing scaler from {AUDIO_SCALER_PATH}")
            else:
                self.scaler = StandardScaler()
                print("✓ Created new scaler")
        except:
            self.scaler = StandardScaler()
            print("✓ Created new scaler")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data into train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_scaled, y_encoded, 
            test_size=0.2, 
            random_state=RANDOM_STATE, 
            stratify=y_encoded
        )
        
        print(f"✓ Training set size: {self.X_train.shape[0]}")
        print(f"✓ Validation set size: {self.X_val.shape[0]}")
        
    def load_pretrained_model(self):
        """
        Load existing trained model or create new one.
        """
        print(f"Loading existing model from: {AUDIO_MODEL_PATH}")
        
        try:
            if AUDIO_MODEL_PATH.exists():
                self.model = joblib.load(AUDIO_MODEL_PATH)
                print("✓ Loaded pre-trained model successfully")
                
                # Update model parameters for fine-tuning
                self.model.n_estimators = N_ESTIMATORS
                print(f"✓ Updated n_estimators to {N_ESTIMATORS}")
            else:
                raise FileNotFoundError("Pre-trained model not found")
                
        except Exception as e:
            print(f"⚠ Could not load pre-trained model: {e}")
            print("⚠ Creating new RandomForest model...")
            self.model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, 
                random_state=RANDOM_STATE, 
                n_jobs=-1
            )
    
    def train_model(self):
        """
        Fine-tune the RandomForest model on new data.
        """
        print_section("STARTING AUDIO MODEL FINE-TUNING")
        
        print("Training RandomForest Classifier...")
        start_time = time.time()
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        print(f"✓ Training completed in {training_time:.2f}s")
        
    def evaluate_model(self):
        """
        Evaluate the fine-tuned model.
        """
        print_section("EVALUATING AUDIO MODEL")
        
        # Make predictions
        y_pred = self.model.predict(self.X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_val, y_pred)
        print(f"✓ Validation Accuracy: {accuracy:.4f}")
        
        # Classification report
        report = classification_report(
            self.y_val, y_pred, 
            target_names=self.label_encoder.classes_
        )
        print("\n" + report)
        
        return y_pred, accuracy, report
    
    def save_model(self, y_pred, accuracy, report):
        """
        Save the fine-tuned model and artifacts.
        """
        print_section("SAVING AUDIO MODEL ARTIFACTS")
        
        # Save model
        model_path = AUDIO_OUTPUT_DIR / 'audio_classifier.pkl'
        joblib.dump(self.model, model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = AUDIO_OUTPUT_DIR / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved: {scaler_path}")
        
        # Save label encoder
        encoder_path = AUDIO_OUTPUT_DIR / 'label_encoder.pkl'
        joblib.dump(self.label_encoder, encoder_path)
        print(f"✓ Label encoder saved: {encoder_path}")
        
        # Save classification report
        report_path = AUDIO_OUTPUT_DIR / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Validation Accuracy: {accuracy:.4f}\n\n")
            f.write(report)
        print(f"✓ Classification report saved: {report_path}")
        
        # Save confusion matrix plot
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(self.y_val, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix - Fine-tuned Audio Model', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        cm_path = AUDIO_OUTPUT_DIR / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"✓ Confusion matrix saved: {cm_path}")
        
        # Save feature importance plot
        feature_importances = pd.Series(self.model.feature_importances_)
        plt.figure(figsize=(10, 6))
        feature_importances.nlargest(20).plot(kind='barh', color='steelblue')
        plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.tight_layout()
        importance_path = AUDIO_OUTPUT_DIR / 'feature_importance.png'
        plt.savefig(importance_path, dpi=150)
        plt.close()
        print(f"✓ Feature importance saved: {importance_path}")
        
        # Save class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        class_names = [self.label_encoder.classes_[i] for i in unique]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=class_names, y=counts, palette='viridis')
        plt.title('Class Distribution in Training Data', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        dist_path = AUDIO_OUTPUT_DIR / 'class_distribution.png'
        plt.savefig(dist_path, dpi=150)
        plt.close()
        print(f"✓ Class distribution saved: {dist_path}")
        
        # Save metadata
        metadata = {
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'n_estimators': N_ESTIMATORS,
            'val_accuracy': accuracy,
            'sample_rate': SAMPLE_RATE,
            'duration': DURATION
        }
        
        metadata_path = AUDIO_OUTPUT_DIR / 'metadata.txt'
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"✓ Metadata saved: {metadata_path}")
        
        print(f"\n✅ All audio model artifacts saved to: {AUDIO_OUTPUT_DIR}")


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """
    Main function to orchestrate the fine-tuning process.
    """
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "CAT EMOTION RECOGNITION - MODEL FINE-TUNING" + " " * 20 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print(f"\nDevice: {device}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    
    # ==========================================
    # FINE-TUNE IMAGE MODEL
    # ==========================================
    
    try:
        print_section("PHASE 1: IMAGE MODEL FINE-TUNING")
        
        image_finetuner = ImageModelFineTuner()
        image_finetuner.prepare_data()
        image_finetuner.load_pretrained_model()
        image_history = image_finetuner.train_model()
        image_finetuner.save_model(image_history)
        
        print("\n✅ IMAGE MODEL FINE-TUNING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ ERROR in image model fine-tuning: {e}")
        import traceback
        traceback.print_exc()
    
    # ==========================================
    # FINE-TUNE AUDIO MODEL
    # ==========================================
    
    try:
        print_section("PHASE 2: AUDIO MODEL FINE-TUNING")
        
        audio_finetuner = AudioModelFineTuner()
        audio_finetuner.prepare_data()
        audio_finetuner.load_pretrained_model()
        audio_finetuner.train_model()
        y_pred, accuracy, report = audio_finetuner.evaluate_model()
        audio_finetuner.save_model(y_pred, accuracy, report)
        
        print("\n✅ AUDIO MODEL FINE-TUNING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ ERROR in audio model fine-tuning: {e}")
        import traceback
        traceback.print_exc()
    
    # ==========================================
    # SUMMARY
    # ==========================================
    
    print_section("FINE-TUNING SUMMARY")
    
    print("📊 Fine-tuned Models Saved:")
    print(f"   • Image Model: {IMAGE_OUTPUT_DIR}")
    print(f"   • Audio Model: {AUDIO_OUTPUT_DIR}")
    
    print("\n📈 Next Steps:")
    print("   1. Review the training curves and metrics in the output directories")
    print("   2. Test the fine-tuned models with sample data")
    print("   3. Deploy the models to production if performance is satisfactory")
    
    print("\n" + "█" * 80)
    print("█" + " " * 25 + "FINE-TUNING COMPLETED!" + " " * 31 + "█")
    print("█" * 80 + "\n")


if __name__ == '__main__':
    main()
