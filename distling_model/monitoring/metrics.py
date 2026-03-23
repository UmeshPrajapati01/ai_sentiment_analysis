"""
Monitoring Metrics Module
━━━━━━━━━━━━━━━━━━━━━━━━
Prometheus metrics exporter for the training pipeline.
Exports metrics on port 8001 (separate from backend API on 8000).
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        start_http_server, Gauge, Counter, Histogram, Info
    )

    # ── Training Metrics ──
    TRAINING_LOSS = Gauge(
        'training_loss', 'Current training loss value'
    )
    TRAINING_ACCURACY = Gauge(
        'training_accuracy', 'Current training accuracy (0-1)'
    )
    VALIDATION_LOSS = Gauge(
        'validation_loss', 'Current validation loss value'
    )
    VALIDATION_ACCURACY = Gauge(
        'validation_accuracy', 'Current validation accuracy (0-1)'
    )

    # ── Epoch Tracking ──
    EPOCHS_COMPLETED = Counter(
        'epochs_completed_total', 'Total number of training epochs completed'
    )

    # ── Inference Metrics ──
    INFERENCE_LATENCY = Histogram(
        'inference_latency_seconds', 'Model inference latency in seconds',
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    )
    PREDICTIONS_COUNTER = Counter(
        'predictions_total', 'Total predictions made',
        ['modality', 'class_name']
    )

    # ── Feedback Metrics ──
    FEEDBACK_TOTAL = Counter(
        'feedback_submissions_total', 'Total human feedback submissions'
    )
    FEEDBACK_CORRECTIONS = Counter(
        'feedback_corrections_total', 'Total corrections (model was wrong)'
    )

    # ── Model Info ──
    MODEL_VERSION = Gauge(
        'model_version', 'Current deployed model version number'
    )
    MODEL_ACCURACY = Gauge(
        'model_accuracy', 'Latest model evaluation accuracy'
    )

    # ── Drift Detection ──
    DRIFT_SCORE = Gauge(
        'drift_score', 'Current prediction distribution drift score'
    )

    # ── System Info ──
    SYSTEM_INFO = Info(
        'distill_ai_system', 'System information'
    )
    SYSTEM_INFO.info({
        'teacher_model': config.OLLAMA_MODEL_NAME,
        'project': 'distill_ai_studio',
        'version': '1.0.0'
    })

    METRICS_AVAILABLE = True

    def start_metrics_server(port: int = None):
        """Start the Prometheus metrics HTTP server."""
        port = port or config.PROMETHEUS_PORT
        try:
            start_http_server(port)
            logger.info(f"📡 Prometheus metrics server started on port {port}")
            logger.info(f"   View metrics: http://localhost:{port}/metrics")
        except OSError as e:
            logger.warning(f"Could not start metrics server on port {port}: {e}")

except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("prometheus_client not installed — monitoring disabled")

    def start_metrics_server(port=None):
        logger.warning("Monitoring disabled — install prometheus_client")
