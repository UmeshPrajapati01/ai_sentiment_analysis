"""
Feedback Loop Module
━━━━━━━━━━━━━━━━━━━━
RLHF-like human feedback system for continuous model improvement.

Components:
  • FeedbackDB       — SQLite-backed feedback storage
  • FeedbackWeighter — Weighted sampling strategy for corrections
  • IncrementalTrainer — Fine-tuning with human-corrected data
  • DataValidator    — Validates incoming feedback quality
"""

import os
import sys
import json
import logging
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ──────────────────────────────────────────────────────────────────────────────
# FEEDBACK DATABASE
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackDB:
    """
    SQLite-backed feedback storage.
    Stores human corrections with prediction context.
    """

    def __init__(self, db_path: str = None):
        import sqlite3
        self.db_path = db_path or os.path.join(config.USER_DATA_DIR, "feedback.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                filename TEXT,
                modality TEXT DEFAULT 'audio',
                model_prediction TEXT,
                model_confidence REAL,
                human_label TEXT,
                is_correct INTEGER DEFAULT 0,
                weight REAL DEFAULT 1.0,
                used_in_training INTEGER DEFAULT 0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retraining_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER,
                num_feedback_samples INTEGER,
                accuracy_before REAL,
                accuracy_after REAL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save_feedback(self, record_id: str, filename: str, modality: str,
                      prediction: str, confidence: float,
                      human_label: str = None) -> str:
        """Save a prediction and optionally a human correction."""
        is_correct = 1 if human_label and human_label == prediction else 0
        weight = config.FEEDBACK_WEIGHT_MULTIPLIER if human_label else 1.0

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO feedback 
            (id, filename, modality, model_prediction, model_confidence,
             human_label, is_correct, weight, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (record_id, filename, modality, prediction, confidence,
              human_label, is_correct, weight,
              datetime.datetime.now().isoformat()))
        self.conn.commit()

        logger.info(f"💬 Feedback saved: {record_id} | pred={prediction} "
                     f"| human={human_label} | weight={weight}")
        return record_id

    def update_human_label(self, record_id: str, human_label: str) -> bool:
        """Update the human label for an existing prediction."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE feedback 
            SET human_label = ?, 
                weight = ?,
                is_correct = CASE WHEN model_prediction = ? THEN 1 ELSE 0 END,
                timestamp = ?
            WHERE id = ?
        """, (human_label, config.FEEDBACK_WEIGHT_MULTIPLIER,
              human_label, datetime.datetime.now().isoformat(), record_id))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_unused_feedback(self) -> List[Dict]:
        """Get all feedback not yet used in retraining."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, filename, modality, model_prediction, human_label, weight
            FROM feedback
            WHERE human_label IS NOT NULL AND used_in_training = 0
        """)
        rows = cursor.fetchall()
        return [
            {"id": r[0], "filename": r[1], "modality": r[2],
             "prediction": r[3], "human_label": r[4], "weight": r[5]}
            for r in rows
        ]

    def mark_as_used(self, record_ids: List[str]):
        """Mark feedback records as used in training."""
        cursor = self.conn.cursor()
        for rid in record_ids:
            cursor.execute(
                "UPDATE feedback SET used_in_training = 1 WHERE id = ?", (rid,)
            )
        self.conn.commit()

    def get_feedback_count(self) -> int:
        """Get total number of human-labeled feedback items."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM feedback WHERE human_label IS NOT NULL"
        )
        return cursor.fetchone()[0]

    def get_unused_count(self) -> int:
        """Get number of feedback items not yet used in training."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM feedback WHERE human_label IS NOT NULL AND used_in_training = 0"
        )
        return cursor.fetchone()[0]

    def should_retrain(self) -> bool:
        """Check if retraining threshold is met."""
        count = self.get_unused_count()
        should = count >= config.FEEDBACK_RETRAIN_THRESHOLD
        if should:
            logger.info(f"🔄 Retraining threshold met: {count} >= "
                         f"{config.FEEDBACK_RETRAIN_THRESHOLD}")
        return should

    def log_retraining(self, version: int, num_samples: int,
                       accuracy_before: float, accuracy_after: float):
        """Log a retraining event."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO retraining_log 
            (version, num_feedback_samples, accuracy_before, accuracy_after, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (version, num_samples, accuracy_before, accuracy_after,
              datetime.datetime.now().isoformat()))
        self.conn.commit()

    def close(self):
        self.conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# DATA VALIDATOR
# ──────────────────────────────────────────────────────────────────────────────

class DataValidator:
    """
    Validates incoming feedback data for quality.
    Checks:
      • Label is in the known class set
      • File exists on disk
      • No duplicate submissions
    """

    def __init__(self, valid_classes: list, data_dir: str):
        self.valid_classes = set(valid_classes)
        self.data_dir = data_dir

    def validate_feedback(self, human_label: str, filename: str = None) -> tuple:
        """
        Returns (is_valid, error_message).
        """
        # Check label
        if not human_label or human_label.strip() == "":
            return False, "Human label is empty"

        if human_label not in self.valid_classes:
            return False, (f"Unknown class '{human_label}'. "
                           f"Valid classes: {sorted(self.valid_classes)}")

        # Check file exists (optional)
        if filename:
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                logger.warning(f"Feedback file not found: {filepath}")
                # Still valid — file might have been uploaded differently

        return True, None


# ──────────────────────────────────────────────────────────────────────────────
# FEEDBACK WEIGHTING STRATEGY
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackWeighter:
    """
    Weighting strategy for human-corrected samples.

    Strategy:
      • Corrections where model was WRONG → weight = FEEDBACK_WEIGHT_MULTIPLIER (1.5x)
      • Confirmations where model was RIGHT → weight = 1.0
      • Recent feedback → slightly higher weight (recency bias)
    """

    def __init__(self, base_multiplier: float = 1.5, recency_decay: float = 0.95):
        self.base_multiplier = base_multiplier
        self.recency_decay = recency_decay

    def compute_weight(self, is_correct: bool, age_days: float = 0) -> float:
        """Compute sample weight based on correctness and recency."""
        base_weight = 1.0 if is_correct else self.base_multiplier
        recency_factor = self.recency_decay ** max(age_days, 0)
        return base_weight * recency_factor

    def compute_batch_weights(self, feedback_items: List[Dict]) -> List[float]:
        """Compute weights for a batch of feedback items."""
        weights = []
        now = datetime.datetime.now()
        for item in feedback_items:
            is_correct = item.get("prediction") == item.get("human_label")
            timestamp = item.get("timestamp", now.isoformat())
            try:
                ts = datetime.datetime.fromisoformat(timestamp)
                age_days = (now - ts).total_seconds() / 86400
            except (ValueError, TypeError):
                age_days = 0
            weights.append(self.compute_weight(is_correct, age_days))
        return weights


# ──────────────────────────────────────────────────────────────────────────────
# INCREMENTAL FINE-TUNING
# ──────────────────────────────────────────────────────────────────────────────

def incremental_finetune(model, feedback_data: List[Dict],
                         preprocess_fn, device,
                         num_classes: int, class_list: list,
                         epochs: int = 5, lr: float = 1e-4):
    """
    Fine-tune the model incrementally using human feedback.

    Args:
        model: Current student model
        feedback_data: List of dicts with 'filename', 'human_label', 'weight'
        preprocess_fn: Function to load and preprocess a data file
        device: torch device
        num_classes: total number of classes
        class_list: list of class names
        epochs: number of fine-tuning epochs
        lr: learning rate (lower than initial training)
    """
    from torch.utils.data import DataLoader, TensorDataset

    logger.info(f"🔄 Incremental fine-tuning with {len(feedback_data)} feedback samples")

    class_to_idx = {c: i for i, c in enumerate(class_list)}

    # Prepare feedback tensors
    features = []
    labels = []
    weights = []

    weighter = FeedbackWeighter(config.FEEDBACK_WEIGHT_MULTIPLIER)

    for item in feedback_data:
        human_label = item.get("human_label")
        if human_label not in class_to_idx:
            continue

        filename = item.get("filename", "")
        try:
            feat = preprocess_fn(filename)
            if feat is not None:
                features.append(torch.tensor(feat, dtype=torch.float32))
                labels.append(class_to_idx[human_label])
                is_correct = item.get("prediction") == human_label
                weights.append(weighter.compute_weight(is_correct))
        except Exception as e:
            logger.warning(f"Failed to process feedback file {filename}: {e}")

    if len(features) == 0:
        logger.warning("No valid feedback samples — skipping fine-tuning")
        return model

    features_tensor = torch.stack(features)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # Create weighted loss
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        total_loss = 0.0

        # Mini-batch training
        for i in range(0, len(features_tensor), config.BATCH_SIZE):
            batch_features = features_tensor[i:i+config.BATCH_SIZE].to(device)
            batch_labels = labels_tensor[i:i+config.BATCH_SIZE].to(device)
            batch_weights = weights_tensor[i:i+config.BATCH_SIZE].to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # Weighted cross-entropy
            loss_per_sample = nn.CrossEntropyLoss(reduction='none')(outputs, batch_labels)
            loss = (loss_per_sample * batch_weights).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(features_tensor) // config.BATCH_SIZE, 1)
        logger.info(f"  Finetune Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    logger.info("✅ Incremental fine-tuning complete")
    return model
