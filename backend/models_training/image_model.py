
import os
import sys
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

# ==========================================
# CONFIGURATION
# ==========================================

# Script location: backend/models_training/image_model.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Input Directory (Train Data)
# Using 'imagetraindata' which acts as our source for Train/Val split
DATA_DIR = PROJECT_ROOT / "data_analysis/test_traindeddata/image_data/imagetraindata"

# Output Directory
OUTPUT_DIR = PROJECT_ROOT / "backend/trained_modelimages/image_model"

# Hyperparameters
BATCH_SIZE = 16  # Small batch size for CPU/Memory efficiency
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2  # 20% for validation
NUM_WORKERS = 0  # 0 for Windows compatibility and simple debugging on CPU

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# DATA LOADING & TRANSFORMS
# ==========================================

def get_data_loaders(data_dir):
    """
    Prepares DataLoaders for training and validation.
    Applies transforms and splits the dataset.
    """
    print(f"Loading data from: {data_dir}")
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Define Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
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

    # Load Full Dataset
    try:
        full_dataset = datasets.ImageFolder(str(data_dir))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
        
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    
    # Split into Train and Validation
    num_data = len(full_dataset)
    num_val = int(VAL_SPLIT * num_data)
    num_train = num_data - num_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [num_train, num_val]
    )
    
    # Apply transforms individually (hacky but standard way requires wrapper or separate dir)
    # Since we split dynamically, we need to wrap the subset to apply transform
    # Defined a simple wrapper for assigning transforms to subsets
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

    train_data = DatasetFromSubset(train_dataset, transform=data_transforms['train'])
    val_data = DatasetFromSubset(val_dataset, transform=data_transforms['val'])

    image_datasets = {'train': train_data, 'val': val_data}
    dataset_sizes = {'train': len(train_data), 'val': len(val_data)}
    
    print(f"Training samples: {dataset_sizes['train']}")
    print(f"Validation samples: {dataset_sizes['val']}")

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                 shuffle=True, num_workers=NUM_WORKERS)
                  for x in ['train', 'val']}
                  
    return dataloaders, dataset_sizes, class_names

# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                history['epoch'].append(epoch + 1)
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# ==========================================
# PLOTTING
# ==========================================

def save_plots(history):
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'loss_curve.png')
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
    plt.plot(history['epoch'], history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'accuracy_curve.png')
    plt.close()
    print("Training plots saved.")

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("=" * 40)
    print("STARTING IMAGE TRAINING PIPELINE (ResNet50)")
    print("=" * 40)
    
    # 1. Prepare Data
    dataloaders, dataset_sizes, class_names = get_data_loaders(DATA_DIR)
    
    # 2. Setup Model (ResNet50)
    print("Initializing ResNet50 model...")
    model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Modify the last fully connected layer
    num_ftrs = model_ft.fc.in_features
    # Determine number of classes dynamically
    num_classes = len(class_names)
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    model_ft = model_ft.to(device)
    
    # 3. Setup Training Components
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    # 4. Train
    model_ft, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                   dataloaders, dataset_sizes, num_epochs=NUM_EPOCHS)
    
    # 5. Save Artifacts
    # Save Model
    save_path = OUTPUT_DIR / 'model.pth'
    torch.save(model_ft.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Save History
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / 'training_history.csv', index=False)
    print(f"Training history saved to {OUTPUT_DIR / 'training_history.csv'}")
    
    # Save Plots
    save_plots(history)
    
    # Save Class Names for inference
    with open(OUTPUT_DIR / 'classes.txt', 'w') as f:
        for cls in class_names:
            f.write(f"{cls}\n")
            
    print("=" * 40)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 40)

if __name__ == '__main__':
    main()
