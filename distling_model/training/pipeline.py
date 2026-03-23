"""
Training Pipeline - Dual Modality (Audio + Image)
===================================================
Orchestrates the full training flow for BOTH audio and image models:
  1. Audio: Supervised Training -> Distillation -> Evaluation
  2. Image: Supervised Training -> Distillation -> Evaluation
  3. Model Versioning + Prometheus Metrics

Designed for aggressive training to reach 80%+ accuracy.
Uses data augmentation, early stopping, and temperature-scheduled distillation.
"""

import os
import sys
import io
import json
import time
import logging
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_processing import (
    ClassifiedAudioDataset, ClassifiedImageDataset, AugmentedAudioDataset,
    NonClassifiedAudioDataset,
    analyze_dataset, create_data_splits, create_dataloaders
)
from training.models import AudioStudentModel, ImageStudentModel
from distillation import DistillationLoss, MockTeacher, GemmaTeacher, TemperatureScheduler

# Setup logging — force UTF-8 on console to avoid Windows cp1252 encoding errors
_console_handler = logging.StreamHandler(
    stream=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
)
_file_handler = logging.FileHandler(
    os.path.join(config.LOGS_DIR, 'training.log'), encoding='utf-8'
)
_formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
_console_handler.setFormatter(_formatter)
_file_handler.setFormatter(_formatter)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS (optional)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    TRAINING_LOSS = Gauge('training_loss', 'Current training loss')
    TRAINING_ACCURACY = Gauge('training_accuracy', 'Current training accuracy')
    VALIDATION_LOSS = Gauge('validation_loss', 'Current validation loss')
    VALIDATION_ACCURACY = Gauge('validation_accuracy', 'Current validation accuracy')
    EPOCHS_COMPLETED = Counter('epochs_completed_total', 'Total epochs completed')
    INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference latency',
                                  buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("prometheus_client not installed -- metrics disabled")


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, dataloader, device, class_names=None):
    """Full evaluation: accuracy, F1, confusion matrix, per-class report."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)

    num_classes = len(class_names) if class_names else None
    all_labels_list = list(range(num_classes)) if num_classes else None

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro',
                        labels=all_labels_list, zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted',
                           labels=all_labels_list, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=all_labels_list)

    report = classification_report(
        all_labels, all_preds,
        labels=all_labels_list,
        target_names=class_names if class_names else None,
        zero_division=0,
        output_dict=True
    )

    results = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "avg_loss": avg_loss,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    logger.info(f"[EVAL] acc={accuracy:.4f}, F1(macro)={f1_macro:.4f}, "
                 f"F1(weighted)={f1_weighted:.4f}, loss={avg_loss:.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# MODEL VERSIONING
# ──────────────────────────────────────────────────────────────────────────────

def save_model_version(model, model_type: str, metrics: dict, extra_info: dict = None):
    """Save model with version tracking."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    version_file = config.MODEL_VERSION_FILE
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"models": [], "latest_version": 0}

    version = registry["latest_version"] + 1
    filename = f"{model_type}_v{version}.pth"
    filepath = os.path.join(config.MODELS_DIR, filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        'version': version,
        'model_type': model_type,
        'metrics': metrics,
        'timestamp': datetime.datetime.now().isoformat(),
        'extra_info': extra_info or {}
    }, filepath)

    registry["latest_version"] = version
    registry["models"].append({
        "version": version,
        "file": filename,
        "type": model_type,
        "accuracy": metrics.get("accuracy", 0),
        "f1_macro": metrics.get("f1_macro", 0),
        "timestamp": datetime.datetime.now().isoformat()
    })

    with open(version_file, 'w') as f:
        json.dump(registry, f, indent=2)

    logger.info(f"[SAVE] Model saved: {filename} (v{version}) -- "
                 f"acc={metrics.get('accuracy', 0):.4f}")
    return filepath, version


def load_latest_model(model_class, model_type: str, **model_kwargs):
    """Load the latest version of a specific model type."""
    version_file = config.MODEL_VERSION_FILE
    if not os.path.exists(version_file):
        return None, None

    with open(version_file, 'r') as f:
        registry = json.load(f)

    matching = [m for m in registry["models"] if m["type"] == model_type]
    if not matching:
        return None, None

    latest = max(matching, key=lambda x: x["version"])
    filepath = os.path.join(config.MODELS_DIR, latest["file"])

    if not os.path.exists(filepath):
        return None, None

    model = model_class(**model_kwargs)
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"[LOAD] Loaded model: {latest['file']} (v{latest['version']})")
    return model, checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# SUPERVISED TRAINING (with early stopping + LR scheduling)
# ──────────────────────────────────────────────────────────────────────────────

def train_supervised(model, train_loader, val_loader, device,
                     num_epochs: int = 50, lr: float = 1e-3,
                     class_names=None, modality: str = "audio",
                     patience: int = 12):
    """
    Aggressive supervised training with:
    - ReduceLROnPlateau scheduler
    - Early stopping with patience
    - Best model restoration
    """
    logger.info(f">>> PHASE 1 [{modality.upper()}]: Supervised Training "
                f"({num_epochs} epochs, lr={lr}, patience={patience})")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None
    no_improve_count = 0

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / max(total, 1)
        train_loss = running_loss / max(len(train_loader), 1)

        # ── Validate ──
        val_results = evaluate_model(model, val_loader, device, class_names)
        val_acc = val_results["accuracy"]
        val_loss = val_results["avg_loss"]

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update Prometheus
        if METRICS_AVAILABLE:
            TRAINING_LOSS.set(train_loss)
            TRAINING_ACCURACY.set(train_acc)
            VALIDATION_LOSS.set(val_loss)
            VALIDATION_ACCURACY.set(val_acc)
            EPOCHS_COMPLETED.inc()

        logger.info(
            f"[{modality.upper()}] Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} F1: {val_results['f1_macro']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            logger.info(f"[{modality.upper()}] Early stopping at epoch {epoch+1} "
                        f"(no improvement for {patience} epochs)")
            break

        # Already reached target?
        if val_acc >= 0.85 and train_acc >= 0.95:
            logger.info(f"[{modality.upper()}] Target accuracy reached at epoch {epoch+1}!")
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    logger.info(f"[DONE] {modality.upper()} Supervised Training Complete. "
                f"Best Val Acc: {best_val_acc:.4f}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# DISTILLATION TRAINING (with Gemma teacher)
# ──────────────────────────────────────────────────────────────────────────────

def train_distillation(model, train_loader, val_loader, device,
                       teacher=None, num_epochs: int = 30,
                       lr: float = 5e-4, class_names=None,
                       modality: str = "audio", patience: int = 10):
    """
    Distillation phase: train student model using teacher soft labels.
    Pre-generates per-class soft labels from Gemma (cached).
    """
    logger.info(f">>> PHASE 2 [{modality.upper()}]: Distillation Training "
                f"({num_epochs} epochs)")

    model.to(device)

    # Determine class list and num_classes based on modality
    if modality == "image":
        classes = config.IMAGE_CLASSES
        num_classes = config.NUM_IMAGE_CLASSES
    else:
        classes = config.AUDIO_CLASSES
        num_classes = config.NUM_AUDIO_CLASSES

    # Setup teacher
    use_mock = True
    if teacher is None:
        gemma = GemmaTeacher(
            base_url=config.OLLAMA_BASE_URL,
            model_name=config.OLLAMA_MODEL_NAME,
            classes=classes
        )
        if gemma.is_available():
            logger.info(f"Using Gemma3:4B as teacher for {modality}")
            teacher = gemma
            use_mock = False
        else:
            logger.info(f"Gemma not available -- using MockTeacher for {modality}")
            teacher = MockTeacher(num_classes=num_classes)
    elif isinstance(teacher, MockTeacher):
        use_mock = True
    else:
        use_mock = False

    # ── Pre-generate Gemma soft labels (one per class, cached) ──
    gemma_class_logits = {}
    if not use_mock:
        logger.info(f"[TEACHER] Pre-generating per-class soft labels for {modality}...")
        for idx, cls_name in enumerate(classes):
            if modality == "audio":
                description = (
                    f"This is an audio recording of a cat that sounds {cls_name.lower()}. "
                    f"It clearly exhibits {cls_name.lower()} vocalisation patterns."
                )
            else:
                description = (
                    f"This is a photograph of a cat that looks {cls_name.lower()}. "
                    f"The cat's expression and body language clearly show "
                    f"{cls_name.lower()} emotion."
                )
            probs = teacher.generate_soft_labels(description, num_classes)
            probs = probs.clamp(min=1e-6)
            logits = torch.log(probs)
            gemma_class_logits[idx] = logits
            logger.info(
                f"  Class {idx} ({cls_name}): top prob={probs.max():.3f}"
            )

        # Validate
        all_uniform = all(
            (v.max() - v.min()).item() < 0.1 for v in gemma_class_logits.values()
        )
        if all_uniform:
            logger.warning("[TEACHER] Gemma returned uniform distributions -- "
                           "falling back to MockTeacher")
            use_mock = True
            teacher = MockTeacher(num_classes=num_classes)

    # Temperature scheduler
    temp_scheduler = TemperatureScheduler(
        initial_temp=config.TEMPERATURE,
        final_temp=1.5,
        total_epochs=num_epochs,
        strategy="cosine"
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_model_state = None
    no_improve_count = 0

    for epoch in range(num_epochs):
        model.train()
        current_temp = temp_scheduler.get_temperature(epoch)
        distill_criterion = DistillationLoss(temperature=current_temp, alpha=config.ALPHA)

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            student_logits = model(inputs)

            # Get teacher logits
            if use_mock:
                teacher_logits = teacher.generate_logits(labels).to(device)
            else:
                batch_teacher = torch.stack([
                    gemma_class_logits.get(lbl.item(), torch.zeros(num_classes))
                    for lbl in labels
                ]).to(device)
                teacher_logits = batch_teacher + torch.randn_like(batch_teacher) * 0.05

            loss = distill_criterion(student_logits, teacher_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / max(total, 1)
        train_loss = running_loss / max(len(train_loader), 1)
        scheduler.step()

        # Validate
        val_results = evaluate_model(model, val_loader, device, class_names)
        val_acc = val_results["accuracy"]

        if METRICS_AVAILABLE:
            TRAINING_LOSS.set(train_loss)
            TRAINING_ACCURACY.set(train_acc)
            VALIDATION_LOSS.set(val_results["avg_loss"])
            VALIDATION_ACCURACY.set(val_acc)
            EPOCHS_COMPLETED.inc()

        logger.info(
            f"[{modality.upper()} DISTILL] Epoch [{epoch+1}/{num_epochs}] "
            f"T={current_temp:.2f} | Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            logger.info(f"[{modality.upper()} DISTILL] Early stopping at epoch {epoch+1}")
            break

    # Restore best
    if best_model_state:
        model.load_state_dict(best_model_state)

    logger.info(f"[DONE] {modality.upper()} Distillation Complete. "
                f"Best Val Acc: {best_val_acc:.4f}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN ONE MODALITY (full pipeline for audio OR image)
# ──────────────────────────────────────────────────────────────────────────────

def train_modality(modality: str, device):
    """
    Train a single modality (audio or image) through:
      1. Supervised training
      2. Distillation
      3. Evaluation
      4. Version save
    Returns (model, test_supervised_results, test_distilled_results)
    """
    logger.info("=" * 60)
    logger.info(f">>> TRAINING {modality.upper()} MODEL")
    logger.info("=" * 60)

    if modality == "audio":
        classes = config.AUDIO_CLASSES
        num_classes = config.NUM_AUDIO_CLASSES

        # Load audio data with augmentation
        base_dataset = ClassifiedAudioDataset(
            root_dir=config.CLASSIFIED_AUDIO_DIR,
            classes=classes,
            sr=config.AUDIO_SAMPLE_RATE,
            n_mfcc=config.N_MFCC,
            max_len=config.AUDIO_MAX_LEN,
            hop_length=config.HOP_LENGTH
        )
        dataset = AugmentedAudioDataset(base_dataset, augment=config.DATA_AUGMENTATION)

        if len(dataset) == 0:
            logger.error("[ERROR] No audio samples found!")
            return None, None, None

        # Initialize model
        model = AudioStudentModel(num_classes=num_classes, n_mfcc=config.N_MFCC)

    elif modality == "image":
        classes = config.IMAGE_CLASSES
        num_classes = config.NUM_IMAGE_CLASSES

        # Load image data with augmentation
        dataset = ClassifiedImageDataset(
            root_dir=config.CLASSIFIED_IMAGE_DIR,
            classes=classes,
            size=config.IMAGE_SIZE,
            augment=config.DATA_AUGMENTATION
        )

        if len(dataset) == 0:
            logger.error("[ERROR] No image samples found!")
            return None, None, None

        # Initialize model
        model = ImageStudentModel(num_classes=num_classes)
    else:
        logger.error(f"Unknown modality: {modality}")
        return None, None, None

    # Analyze data
    analyze_dataset(dataset, name=f"Classified {modality.title()}")

    # Split data
    train_ds, val_ds, test_ds = create_data_splits(
        dataset,
        train_ratio=config.TRAIN_SPLIT,
        val_ratio=config.VAL_SPLIT,
        test_ratio=config.TEST_SPLIT,
        seed=config.RANDOM_SEED
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=config.BATCH_SIZE
    )

    logger.info(f"[MODEL] {modality.title()} model params: "
                 f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Phase 1: Supervised Training ──
    model = train_supervised(
        model, train_loader, val_loader, device,
        num_epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
        class_names=classes,
        modality=modality,
        patience=15
    )

    # Test evaluation after supervised
    logger.info(f"[EVAL] {modality.title()} Test Set (after supervised):")
    test_results_supervised = evaluate_model(model, test_loader, device, classes)

    # Save supervised version
    save_model_version(model, f"{modality}_student_supervised", test_results_supervised)

    # ── Phase 2: Distillation ──
    model = train_distillation(
        model, train_loader, val_loader, device,
        num_epochs=config.DISTILLATION_EPOCHS,
        lr=config.DISTILLATION_LR,
        class_names=classes,
        modality=modality,
        patience=12
    )

    # Test evaluation after distillation
    logger.info(f"[EVAL] {modality.title()} Test Set (after distillation):")
    test_results_distilled = evaluate_model(model, test_loader, device, classes)

    # Save distilled version
    save_model_version(
        model, f"{modality}_student_distilled", test_results_distilled,
        extra_info={
            "teacher": config.OLLAMA_MODEL_NAME,
            "temperature": config.TEMPERATURE,
            "alpha": config.ALPHA,
            "modality": modality
        }
    )

    return model, test_results_supervised, test_results_distilled


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE: BOTH MODALITIES
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    """Execute the full dual-modality training pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[INIT] Using device: {device}")
    logger.info(f"[INIT] Config: epochs={config.NUM_EPOCHS}, distill_epochs={config.DISTILLATION_EPOCHS}, "
                f"batch={config.BATCH_SIZE}, augment={config.DATA_AUGMENTATION}")

    # Start Prometheus metrics server
    if METRICS_AVAILABLE:
        try:
            start_http_server(config.PROMETHEUS_PORT)
            logger.info(f"[METRICS] Prometheus at http://localhost:{config.PROMETHEUS_PORT}/metrics")
        except OSError:
            logger.warning("Prometheus port already in use -- skipping")

    results_summary = {}

    # ══════════════════════════════════════════════════════
    # AUDIO MODEL
    # ══════════════════════════════════════════════════════
    audio_model, audio_sup, audio_dist = train_modality("audio", device)
    if audio_sup:
        results_summary["audio"] = {
            "supervised_acc": audio_sup["accuracy"],
            "distilled_acc": audio_dist["accuracy"] if audio_dist else 0,
            "supervised_f1": audio_sup["f1_macro"],
            "distilled_f1": audio_dist["f1_macro"] if audio_dist else 0,
        }

    # ══════════════════════════════════════════════════════
    # IMAGE MODEL
    # ══════════════════════════════════════════════════════
    image_model, image_sup, image_dist = train_modality("image", device)
    if image_sup:
        results_summary["image"] = {
            "supervised_acc": image_sup["accuracy"],
            "distilled_acc": image_dist["accuracy"] if image_dist else 0,
            "supervised_f1": image_sup["f1_macro"],
            "distilled_f1": image_dist["f1_macro"] if image_dist else 0,
        }

    # ══════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 70)
    logger.info(">>> FULL PIPELINE COMPLETE")
    logger.info("=" * 70)

    for mod, res in results_summary.items():
        logger.info(f"  [{mod.upper()}]")
        logger.info(f"    Supervised  Acc: {res['supervised_acc']:.4f}  F1: {res['supervised_f1']:.4f}")
        logger.info(f"    Distilled   Acc: {res['distilled_acc']:.4f}  F1: {res['distilled_f1']:.4f}")

    logger.info("=" * 70)

    # Save summary to file
    summary_path = os.path.join(config.LOGS_DIR, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"[SAVE] Summary saved to {summary_path}")

    return results_summary


if __name__ == "__main__":
    run_pipeline()
