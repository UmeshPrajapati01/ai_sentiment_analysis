# Standalone script: MeowMood vision training (A5-A8) - fixed path & plotting issues
"""
MeowMood - Vision Module Training Script (Tasks A5–A8)
Dataset expected at: uncleaned_datasets/cleaned_data/cat_classifieddataimage/
Output saved to: output_weekwise/week2output/image/
"""
import os
import sys
import time
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# ────────────────────────────────────────────────
# Global constants & configuration
# ────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

DATA_ROOT = r"uncleaned_datasets\cleaned_data\cat_classifieddataimage"
OUTPUT_ROOT = r"output_weekwise\week2output\image"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS_DEFAULT = 30
VALID_SPLIT = 0.20
NUM_CLASSES = 4  # Default number of emotion classes

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def resolve_data_dir(path: str) -> str:
    """
    Resolve dataset path to an absolute and existing directory.
    Tries given path (absolute/relative), then project-root relative.
    """
    p = os.path.expanduser(path)
    if os.path.isabs(p) and os.path.isdir(p):
        return p
    # project-root relative attempt (three levels up)
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    p1 = os.path.abspath(os.path.join(base, path))
    if os.path.isdir(p1):
        return p1
    # raw relative to cwd
    p2 = os.path.abspath(path)
    if os.path.isdir(p2):
        return p2
    raise FileNotFoundError(f"Dataset path not found. Tried:\n  {p}\n  {p1}\n  {p2}")


def stratified_class_split(dataset, test_size=VALID_SPLIT, random_state=SEED):
    """
    Stratified split of dataset indices into train and validation sets.
    """
    targets = np.array(dataset.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        stratify=targets,
        random_state=random_state
    )
    return list(train_idx), list(val_idx)


def get_transforms():
    """Train/Val transforms (ImageNet mean/std)."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


def get_num_workers():
    # Use 0 on Windows to avoid multiprocessing issues; else use 2
    return 0 if os.name == "nt" else 2


def list_sample_images(root_dir: str, n: int = 10):
    """Print up to n sample image paths for quick verification."""
    exts = ('.jpg', '.jpeg', '.png')
    samples = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(exts):
                samples.append(os.path.join(dirpath, fn))
                if len(samples) >= n:
                    break
        if len(samples) >= n:
            break
    print("\nSample image files (first {}):".format(n))
    if samples:
        for p in samples:
            print(p)
    else:
        print("  No image files (.jpg/.jpeg/.png) found under:", root_dir)
    return samples


def create_dataloaders(root_dir, batch_size=BATCH_SIZE, use_weighted_sampler=False):
    """
    Create train/val dataloaders. root_dir will be resolved to absolute path.
    """
    root_dir = resolve_data_dir(root_dir)
    print(f"Using dataset directory: {root_dir}")
    list_sample_images(root_dir, n=10)

    train_tf, val_tf = get_transforms()

    full_dataset = ImageFolder(root_dir)
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {root_dir}")

    train_idx, val_idx = stratified_class_split(full_dataset)

    train_dataset = ImageFolder(root_dir, transform=train_tf)
    val_dataset = ImageFolder(root_dir, transform=val_tf)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    pin_memory = torch.cuda.is_available()
    if use_weighted_sampler:
        train_targets = [full_dataset.targets[i] for i in train_idx]
        class_counts = np.bincount(train_targets, minlength=len(full_dataset.classes))
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = [class_weights[t] for t in train_targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=get_num_workers(), pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=get_num_workers(), pin_memory=pin_memory)

    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=get_num_workers(), pin_memory=pin_memory)

    class_names = full_dataset.classes
    return train_loader, val_loader, class_names


# ────────────────────────────────────────────────
# A5 – CNN Architecture Design
# ────────────────────────────────────────────────

def build_vision_model(base_arch="mobilenetv2", dropout_rate=0.4, n_classes=None):
    """
    Build pretrained base (MobileNetV2 default) with a custom head for NUM_CLASSES.
    """
    # resolve n_classes: prefer explicit arg, then global NUM_CLASSES, else 4
    if n_classes is None:
        try:
            n_classes = NUM_CLASSES
        except NameError:
            n_classes = 4
    if base_arch == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, n_classes)
        )
    elif base_arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, n_classes)
        )
    else:
        raise ValueError("Unsupported base architecture")

    return model.to(DEVICE)


def infer_output_classes_from_state(state_dict, base_arch="mobilenetv2"):
    """
    Inspect a state_dict to infer the classifier output dimension (number of classes).
    Returns an int if detected, otherwise None.
    """
    if base_arch == "mobilenetv2":
        key = "classifier.1.weight"
    elif base_arch == "resnet18":
        key = "fc.1.weight"
    else:
        # generic fallback: try to find a weight tensor with shape [C, ...] where key contains 'weight' and last dim > 0
        for k, v in state_dict.items():
            if k.endswith("weight") and isinstance(v, torch.Tensor) and v.dim() == 2:
                return v.shape[0]
        return None

    if key in state_dict:
        w = state_dict[key]
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            return w.shape[0]
    # fallback attempt
    for k, v in state_dict.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.dim() == 2:
            return v.shape[0]
    return None


# ────────────────────────────────────────────────
# Training & validation loops (A6)
# ────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return (total_loss / total) if total > 0 else 0.0, (correct / total * 100.0) if total > 0 else 0.0


@torch.inference_mode()
def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
    return (total_loss / total) if total > 0 else 0.0, (correct / total * 100.0) if total > 0 else 0.0, np.array(all_preds), np.array(all_targets)


# ────────────────────────────────────────────────
# A6 + A7 – Training + Hyperparameter Tuning
# ────────────────────────────────────────────────

def train_and_tune_vision_model(
    data_root,
    output_dir,
    base_arch="mobilenetv2",
    lr_candidates=(0.001, 0.0003, 0.0001),
    dropout_candidates=(0.3, 0.5),
    epochs=NUM_EPOCHS_DEFAULT,
    batch_size=BATCH_SIZE,
    patience_early_stop=6,
    use_balanced_sampling=True
):
    """
    Train and grid-search over (lr_candidates x dropout_candidates).
    Returns best model state, best val acc, class names, and experiments summary.
    """
    abs_root = resolve_data_dir(data_root)
    # create dataloaders using resolved path
    train_loader, val_loader, class_names = create_dataloaders(abs_root, batch_size=batch_size, use_weighted_sampler=use_balanced_sampling)
    n_classes = len(class_names)

    best_val_acc = -1.0
    best_model_state = None
    experiment_results = []

    for lr in lr_candidates:
        for drop in dropout_candidates:
            print(f"\n→ Starting experiment: {base_arch} | lr={lr:.1e} | dropout={drop}")

            model = build_vision_model(base_arch=base_arch, dropout_rate=drop, n_classes=n_classes)

            # Class-weighted loss (computed on full dataset)
            if use_balanced_sampling:
                full_ds = ImageFolder(abs_root)
                # ensure counts length equals n_classes
                counts = np.bincount(full_ds.targets, minlength=n_classes)
                weights = torch.tensor(1.0 / (counts + 1e-8), dtype=torch.float32, device=DEVICE)
                criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

            best_epoch_loss = float('inf')
            best_epoch_state = None
            epochs_no_improve = 0
            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
                va_loss, va_acc, _, _ = validate_epoch(model, val_loader, criterion)
                scheduler.step(va_loss)

                history["train_loss"].append(tr_loss)
                history["train_acc"].append(tr_acc)
                history["val_loss"].append(va_loss)
                history["val_acc"].append(va_acc)

                print(f"  Epoch {epoch:2d} | train {tr_loss:.4f} / {tr_acc:5.2f}% | val {va_loss:.4f} / {va_acc:5.2f}%")

                if va_loss < best_epoch_loss - 1e-6:
                    best_epoch_loss = va_loss
                    best_epoch_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience_early_stop:
                    print("   → Early stopping")
                    break

            # ensure we have weights to save
            if best_epoch_state is None:
                best_epoch_state = copy.deepcopy(model.state_dict())

            # Save this experiment artifacts
            exp_tag = f"{base_arch}_lr{lr:.1e}_drop{drop}"
            exp_folder = os.path.join(output_dir, exp_tag)
            os.makedirs(exp_folder, exist_ok=True)
            torch.save(best_epoch_state, os.path.join(exp_folder, "best_weights.pth"))
            with open(os.path.join(exp_folder, "history.json"), "w", encoding="utf8") as f:
                json.dump(history, f, indent=2)

            # Evaluate final on validation to obtain final metrics for this experiment
            # (rebuild model to match saved checkpoint classifier size then load weights)
            ckpt_out_classes = infer_output_classes_from_state(best_epoch_state, base_arch=base_arch)
            if ckpt_out_classes is None:
                ckpt_out_classes = n_classes
            temp_model = build_vision_model(base_arch=base_arch, dropout_rate=drop, n_classes=ckpt_out_classes)
            # load safely, allow classifier size mismatch without crashing
            load_res = temp_model.load_state_dict(best_epoch_state, strict=False)
            if getattr(load_res, "missing_keys", None) or getattr(load_res, "unexpected_keys", None):
                print(f"  Warning: checkpoint/state_dict mismatch: {load_res}")
            va_loss_final, va_acc_final, _, _ = validate_epoch(temp_model, val_loader, criterion)

            experiment_results.append({
                "experiment": exp_tag,
                "val_acc": float(va_acc_final),
                "val_loss": float(va_loss_final),
                "folder": exp_folder
            })

            if va_acc_final > best_val_acc:
                best_val_acc = float(va_acc_final)
                best_model_state = copy.deepcopy(best_epoch_state)
                best_exp_folder = exp_folder
                print(f"   → New best validation accuracy: {best_val_acc:.2f}% (saved)")

    # Save overall best model
    if best_model_state is not None:
        os.makedirs(output_dir, exist_ok=True)
        best_path = os.path.join(output_dir, "cat_emotion_best.pth")
        torch.save(best_model_state, best_path)
        print(f"\nBest model saved → Val acc: {best_val_acc:.2f}% -> {best_path}")

    # Save summary of experiments
    with open(os.path.join(output_dir, "experiments_summary.json"), "w", encoding="utf8") as f:
        json.dump(experiment_results, f, indent=2)

    return best_model_state, best_val_acc, class_names, experiment_results


# ────────────────────────────────────────────────
# A8 – Model Evaluation & Visualization
# ────────────────────────────────────────────────

def evaluate_model(model_state, class_names, data_root, output_dir, base_arch="mobilenetv2", batch_size=BATCH_SIZE):
    """
    Evaluate best model on validation set and save confusion matrix + per-class metrics.
    """
    abs_root = resolve_data_dir(data_root)
    _, val_tf = get_transforms()
    full_ds = ImageFolder(abs_root)
    _, val_idx = stratified_class_split(full_ds)
    val_ds = Subset(ImageFolder(abs_root, transform=val_tf), val_idx)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Infer output classes from the provided state_dict and rebuild model accordingly
    ckpt_out = infer_output_classes_from_state(model_state, base_arch=base_arch)
    if ckpt_out is None:
        ckpt_out = len(class_names)
    model = build_vision_model(base_arch=base_arch, dropout_rate=0.4, n_classes=ckpt_out)
    load_res = model.load_state_dict(model_state, strict=False)
    if getattr(load_res, "missing_keys", None) or getattr(load_res, "unexpected_keys", None):
        print(f"Warning: evaluate_model load_state_dict mismatch: {load_res}")
    model.eval()

    all_preds, all_gt = [], []
    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_gt.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_gt = np.array(all_gt)

    acc = 100.0 * (all_preds == all_gt).mean()
    print(f"\nFinal Validation Accuracy: {acc:.2f}%")

    # Classification report
    report = classification_report(all_gt, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:\n" + report)

    # Confusion Matrix plot
    cm = confusion_matrix(all_gt, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix – Accuracy {acc:.2f}%")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(cm_path, dpi=160, bbox_inches="tight")
    plt.close()

    # Per-class metrics
    p, r, f1, _ = precision_recall_fscore_support(all_gt, all_preds, zero_division=0)
    metrics_per_class = {class_names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
                         for i in range(len(class_names))}
    with open(os.path.join(output_dir, "val_metrics.json"), "w", encoding="utf8") as f:
        json.dump(metrics_per_class, f, indent=2)

    return acc, report, cm_path


def plot_training_curves(exp_folder_list, out_path):
    """
    Plot training/validation loss and accuracy using the first available history.json.
    """
    history = None
    for d in exp_folder_list:
        p = os.path.join(d, "history.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf8") as f:
                history = json.load(f)
            break
    if history is None:
        print("No history.json found in experiment folders; skipping training curves plot.")
        return

    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    plt.plot(epochs, history.get("train_loss", []), label="train_loss")
    plt.plot(epochs, history.get("val_loss", []), label="val_loss")
    plt.plot(epochs, history.get("train_acc", []), label="train_acc")
    plt.plot(epochs, history.get("val_acc", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Training Curves")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ────────────────────────────────────────────────
# Main execution
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MeowMood – Train & Evaluate Vision Model (A5–A8)")
    parser.add_argument("--data", type=str, default=DATA_ROOT, help="dataset root")
    parser.add_argument("--out", type=str, default=OUTPUT_ROOT, help="output directory")
    parser.add_argument("--base", type=str, default="mobilenetv2", choices=["mobilenetv2", "resnet18"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS_DEFAULT)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-balance", action="store_true", help="disable class balancing")
    args = parser.parse_args()

    # resolve data path once and use absolute path everywhere
    try:
        abs_data = resolve_data_dir(args.data)
    except FileNotFoundError as e:
        print(str(e)); sys.exit(1)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {abs_data}")
    print(f"Output:  {args.out}\n")

    best_state, best_acc, class_names, experiments = train_and_tune_vision_model(
        data_root=abs_data,
        output_dir=args.out,
        base_arch=args.base,
        epochs=args.epochs,
        batch_size=args.batch,
        use_balanced_sampling=not args.no_balance
    )

    if best_state is not None:
        evaluate_model(best_state, class_names, abs_data, args.out, base_arch=args.base, batch_size=args.batch)
        exp_dirs = [e.get("folder") for e in experiments if e.get("folder")]
        plot_training_curves(exp_dirs, os.path.join(args.out, "training_curves.png"))
        print("\nVision training pipeline (A5–A8) completed.")
    else:
        print("No model was trained successfully.")