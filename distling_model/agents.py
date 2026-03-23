"""
Agentic AI System
━━━━━━━━━━━━━━━━━
Modular AI agents that manage the pipeline autonomously.

Agents:
  • DataAgent       — Ingests, validates, and preprocesses data
  • TrainingAgent   — Manages supervised training + distillation
  • FeedbackAgent   — Handles RLHF feedback loop + retraining triggers
  • EvaluationAgent — Runs evaluation, tracks metrics
  • MonitoringAgent — Reports metrics to Prometheus

Communication: Event-based message bus (in-process queue).
Each agent reads/writes to a shared message queue.
"""

import os
import sys
import json
import time
import logging
import threading
from queue import Queue, Empty
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MESSAGE BUS
# ──────────────────────────────────────────────────────────────────────────────

class EventType(Enum):
    DATA_READY = "data_ready"
    DATA_ANALYZED = "data_analyzed"
    TRAINING_START = "training_start"
    TRAINING_COMPLETE = "training_complete"
    DISTILLATION_START = "distillation_start"
    DISTILLATION_COMPLETE = "distillation_complete"
    EVALUATION_REQUEST = "evaluation_request"
    EVALUATION_COMPLETE = "evaluation_complete"
    FEEDBACK_RECEIVED = "feedback_received"
    RETRAIN_TRIGGER = "retrain_trigger"
    RETRAIN_COMPLETE = "retrain_complete"
    METRICS_UPDATE = "metrics_update"
    PIPELINE_COMPLETE = "pipeline_complete"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    event_type: EventType
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """
    In-process event bus for inter-agent communication.
    Agents subscribe to event types and receive messages.
    """

    def __init__(self):
        self._subscribers: Dict[EventType, list] = {}
        self._history: list = []
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback):
        """Register a callback for an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def publish(self, message: AgentMessage):
        """Publish a message to all subscribers of its event type."""
        with self._lock:
            self._history.append(message)

        callbacks = self._subscribers.get(message.event_type, [])
        for cb in callbacks:
            try:
                cb(message)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")

        logger.debug(f"📨 [{message.source}] → {message.event_type.value}")

    def get_history(self) -> list:
        return list(self._history)


# Global message bus
bus = MessageBus()


# ──────────────────────────────────────────────────────────────────────────────
# BASE AGENT
# ──────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """Base class for all pipeline agents."""

    def __init__(self, name: str, bus: MessageBus):
        self.name = name
        self.bus = bus
        self.logger = logging.getLogger(f"agent.{name}")
        self.status = "idle"

    def send(self, event_type: EventType, data: dict = None):
        """Send a message on the bus."""
        msg = AgentMessage(event_type=event_type, source=self.name, data=data or {})
        self.bus.publish(msg)

    def listen(self, event_type: EventType, callback):
        """Subscribe to an event type."""
        self.bus.subscribe(event_type, callback)


# ──────────────────────────────────────────────────────────────────────────────
# DATA AGENT
# ──────────────────────────────────────────────────────────────────────────────

class DataAgent(BaseAgent):
    """
    Responsible for:
      • Scanning user_data directories
      • Loading and validating data files
      • Creating train/val/test splits
      • Publishing DATA_READY events
    """

    def __init__(self, bus: MessageBus):
        super().__init__("DataAgent", bus)

    def ingest_and_split(self):
        """Ingest all data and prepare splits."""
        self.status = "ingesting"
        self.logger.info("📂 Ingesting data...")

        from data_processing import (
            ClassifiedAudioDataset, analyze_dataset,
            create_data_splits, create_dataloaders
        )

        # Load classified audio
        audio_dataset = ClassifiedAudioDataset(
            root_dir=config.CLASSIFIED_AUDIO_DIR,
            classes=config.AUDIO_CLASSES,
            sr=config.AUDIO_SAMPLE_RATE,
            n_mfcc=config.N_MFCC,
            max_len=config.AUDIO_MAX_LEN,
            hop_length=config.HOP_LENGTH
        )

        if len(audio_dataset) == 0:
            self.send(EventType.ERROR, {"message": "No audio data found"})
            return None

        # Analyze
        stats = analyze_dataset(audio_dataset, "Classified Audio")
        self.send(EventType.DATA_ANALYZED, {"stats": stats})

        # Split
        train_ds, val_ds, test_ds = create_data_splits(
            audio_dataset,
            train_ratio=config.TRAIN_SPLIT,
            val_ratio=config.VAL_SPLIT,
            test_ratio=config.TEST_SPLIT,
            seed=config.RANDOM_SEED
        )

        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=config.BATCH_SIZE
        )

        data = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "num_classes": config.NUM_AUDIO_CLASSES,
            "class_names": config.AUDIO_CLASSES,
            "total_samples": len(audio_dataset)
        }

        self.status = "ready"
        self.send(EventType.DATA_READY, data)
        return data


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING AGENT
# ──────────────────────────────────────────────────────────────────────────────

class TrainingAgent(BaseAgent):
    """
    Responsible for:
      • Phase 1: Supervised training
      • Phase 2: Distillation training
      • Model saving and versioning
    """

    def __init__(self, bus: MessageBus):
        super().__init__("TrainingAgent", bus)
        self.model = None
        self.device = None

    def run_training(self, data: dict):
        """Execute full training pipeline."""
        import torch
        from training.models import AudioStudentModel
        from training.pipeline import train_supervised, train_distillation, save_model_version

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🖥️  Device: {self.device}")

        # Initialize model
        self.model = AudioStudentModel(
            num_classes=data["num_classes"],
            n_mfcc=config.N_MFCC
        )

        self.status = "training_supervised"
        self.send(EventType.TRAINING_START, {"phase": "supervised"})

        # Phase 1: Supervised
        self.model = train_supervised(
            self.model,
            data["train_loader"],
            data["val_loader"],
            self.device,
            num_epochs=config.NUM_EPOCHS,
            lr=config.LEARNING_RATE,
            class_names=data["class_names"]
        )

        self.send(EventType.TRAINING_COMPLETE, {"phase": "supervised"})

        # Phase 2: Distillation
        self.status = "training_distillation"
        self.send(EventType.DISTILLATION_START, {})

        self.model = train_distillation(
            self.model,
            data["train_loader"],
            data["val_loader"],
            self.device,
            num_epochs=config.DISTILLATION_EPOCHS,
            lr=config.DISTILLATION_LR,
            class_names=data["class_names"]
        )

        self.send(EventType.DISTILLATION_COMPLETE, {})
        self.status = "complete"

        # Request evaluation
        self.send(EventType.EVALUATION_REQUEST, {
            "model": self.model,
            "test_loader": data["test_loader"],
            "class_names": data["class_names"],
            "model_type": "audio_student_distilled"
        })

        return self.model


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION AGENT
# ──────────────────────────────────────────────────────────────────────────────

class EvaluationAgent(BaseAgent):
    """
    Responsible for:
      • Running evaluation metrics
      • Computing accuracy, F1, confusion matrix
      • Saving evaluation results
      • Triggering model versioning
    """

    def __init__(self, bus: MessageBus):
        super().__init__("EvaluationAgent", bus)
        self.results_history = []

    def evaluate(self, model, test_loader, device, class_names, model_type):
        """Run full evaluation and save results."""
        from training.pipeline import evaluate_model, save_model_version

        self.status = "evaluating"
        self.logger.info("📊 Running evaluation...")

        results = evaluate_model(model, test_loader, device, class_names)
        self.results_history.append(results)

        # Save model version
        filepath, version = save_model_version(
            model, model_type, results,
            extra_info={"teacher": config.OLLAMA_MODEL_NAME}
        )

        self.send(EventType.EVALUATION_COMPLETE, {
            "results": results,
            "version": version,
            "filepath": filepath
        })

        self.status = "idle"
        return results


# ──────────────────────────────────────────────────────────────────────────────
# FEEDBACK AGENT
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackAgent(BaseAgent):
    """
    Responsible for:
      • Receiving human corrections
      • Validating feedback quality
      • Storing in feedback DB
      • Triggering retraining when threshold met
    """

    def __init__(self, bus: MessageBus):
        super().__init__("FeedbackAgent", bus)
        from feedback_loop import FeedbackDB, DataValidator
        self.db = FeedbackDB()
        self.validator = DataValidator(
            valid_classes=config.AUDIO_CLASSES,
            data_dir=config.USER_DATA_DIR
        )

    def process_feedback(self, record_id: str, human_label: str,
                         filename: str = None):
        """Process incoming human feedback."""
        # Validate
        is_valid, error = self.validator.validate_feedback(human_label, filename)
        if not is_valid:
            self.logger.warning(f"Invalid feedback: {error}")
            return {"status": "error", "message": error}

        # Store
        self.db.update_human_label(record_id, human_label)

        self.send(EventType.FEEDBACK_RECEIVED, {
            "record_id": record_id,
            "human_label": human_label,
            "total_unused": self.db.get_unused_count()
        })

        # Check retrain threshold
        if self.db.should_retrain():
            self.send(EventType.RETRAIN_TRIGGER, {
                "unused_count": self.db.get_unused_count()
            })

        return {"status": "success", "message": "Feedback recorded"}

    def get_stats(self) -> dict:
        return {
            "total_feedback": self.db.get_feedback_count(),
            "unused_feedback": self.db.get_unused_count(),
            "threshold": config.FEEDBACK_RETRAIN_THRESHOLD,
            "will_retrain": self.db.should_retrain()
        }


# ──────────────────────────────────────────────────────────────────────────────
# MONITORING AGENT
# ──────────────────────────────────────────────────────────────────────────────

class MonitoringAgent(BaseAgent):
    """
    Responsible for:
      • Exporting metrics to Prometheus
      • Tracking drift detection
      • Logging inference latency
    """

    def __init__(self, bus: MessageBus):
        super().__init__("MonitoringAgent", bus)
        self.metrics = {
            "training_loss": 0.0,
            "training_accuracy": 0.0,
            "validation_accuracy": 0.0,
            "feedback_count": 0,
            "model_version": 0,
            "inference_latency_avg": 0.0,
            "predictions_total": 0,
            "drift_score": 0.0
        }

        # Listen for events
        self.listen(EventType.EVALUATION_COMPLETE, self._on_evaluation)
        self.listen(EventType.FEEDBACK_RECEIVED, self._on_feedback)
        self.listen(EventType.METRICS_UPDATE, self._on_metrics)

    def _on_evaluation(self, msg: AgentMessage):
        results = msg.data.get("results", {})
        self.metrics["validation_accuracy"] = results.get("accuracy", 0)
        self.metrics["model_version"] = msg.data.get("version", 0)

    def _on_feedback(self, msg: AgentMessage):
        self.metrics["feedback_count"] = msg.data.get("total_unused", 0)

    def _on_metrics(self, msg: AgentMessage):
        self.metrics.update(msg.data)

    def get_dashboard_data(self) -> dict:
        """Get current metrics for dashboard display."""
        return {**self.metrics}

    def check_drift(self, recent_predictions: list, baseline_distribution: dict) -> float:
        """
        Simple drift detection: compare recent prediction distribution
        against baseline distribution using KL divergence approximation.
        """
        if not recent_predictions or not baseline_distribution:
            return 0.0

        # Count recent predictions
        recent_dist = {}
        for pred in recent_predictions:
            recent_dist[pred] = recent_dist.get(pred, 0) + 1

        total = len(recent_predictions)
        for k in recent_dist:
            recent_dist[k] /= total

        # Simple absolute difference drift score
        drift_score = 0.0
        all_keys = set(list(recent_dist.keys()) + list(baseline_distribution.keys()))
        for key in all_keys:
            p = recent_dist.get(key, 0)
            q = baseline_distribution.get(key, 0)
            drift_score += abs(p - q)

        self.metrics["drift_score"] = drift_score
        return drift_score


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    """
    Coordinates all agents for the complete pipeline.

    Architecture:
    ┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
    │  Data Agent   │────▶│ Training Agent  │────▶│ Evaluation Agent │
    └──────────────┘     └────────────────┘     └─────────────────┘
           │                      ▲                       │
           │                      │                       │
           ▼                      │                       ▼
    ┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
    │ Monitoring   │◀────│ Feedback Agent  │◀────│   Message Bus   │
    │   Agent      │     └────────────────┘     └─────────────────┘
    └──────────────┘
    """

    def __init__(self):
        self.bus = MessageBus()
        self.data_agent = DataAgent(self.bus)
        self.training_agent = TrainingAgent(self.bus)
        self.evaluation_agent = EvaluationAgent(self.bus)
        self.feedback_agent = FeedbackAgent(self.bus)
        self.monitoring_agent = MonitoringAgent(self.bus)

    def run_full_pipeline(self):
        """Execute the complete pipeline end-to-end."""
        import torch
        logger.info("=" * 60)
        logger.info("🚀 AGENTIC PIPELINE ORCHESTRATOR STARTING")
        logger.info("=" * 60)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Step 1: Data Agent ingests data
        data = self.data_agent.ingest_and_split()
        if data is None:
            logger.error("❌ Data ingestion failed. Aborting.")
            return

        # Step 2: Training Agent runs training + distillation
        model = self.training_agent.run_training(data)

        # Step 3: Evaluation Agent evaluates
        results = self.evaluation_agent.evaluate(
            model,
            data["test_loader"],
            device,
            data["class_names"],
            "audio_student_distilled"
        )

        # Step 4: Report
        self.bus.publish(AgentMessage(
            event_type=EventType.PIPELINE_COMPLETE,
            source="Orchestrator",
            data={"results": results}
        ))

        logger.info("=" * 60)
        logger.info("🎯 PIPELINE COMPLETE")
        logger.info(f"   Accuracy:  {results['accuracy']:.4f}")
        logger.info(f"   F1 Macro:  {results['f1_macro']:.4f}")
        logger.info("=" * 60)

        return model, results
