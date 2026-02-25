"""
Distillation Module
━━━━━━━━━━━━━━━━━━
Implements Knowledge Distillation from Gemma3:4B (teacher) to student model.

Knowledge Distillation Loss Formula:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_total = α · L_hard + (1 - α) · T² · L_soft

Where:
  L_hard = CrossEntropy(student_logits, true_labels)
  L_soft = KLDivergence(softmax(student_logits/T), softmax(teacher_logits/T))
  T      = Temperature (higher → softer probability distributions)
  α      = Blending coefficient (0.5 = equal weight)

Temperature Strategy:
  • T=1.0  → standard softmax (no smoothing)
  • T=3.0  → moderate smoothing (good default)
  • T=5.0+ → very soft distributions (more inter-class knowledge)
  • Adaptive: start high, anneal toward 1.0 over training
"""

import logging
import requests
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# DISTILLATION LOSS
# ──────────────────────────────────────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    Combined knowledge distillation loss.

    L = α · CE(student, labels) + (1-α) · T² · KL(soft_student ∥ soft_teacher)
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                true_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits: (batch, num_classes) raw logits from student
            teacher_logits: (batch, num_classes) raw logits from teacher
            true_labels:    (batch,) integer class labels
        Returns:
            Combined distillation loss (scalar)
        """
        # Hard loss: standard cross-entropy with true labels
        hard_loss = self.ce_loss(student_logits, true_labels)

        # Soft loss: KL divergence between softened distributions
        T = self.temperature
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (T ** 2)

        # Combined
        total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
        return total_loss


# ──────────────────────────────────────────────────────────────────────────────
# TEACHER MODEL INTERFACE (Gemma3:4B via Ollama)
# ──────────────────────────────────────────────────────────────────────────────

class GemmaTeacher:
    """
    Interface to Gemma3:4B running locally via Ollama.
    Generates soft labels for knowledge distillation.
    """

    def __init__(self, base_url: str = "http://localhost:11434",
                 model_name: str = "gemma3:4b",
                 classes: list = None):
        self.base_url = base_url
        self.model_name = model_name
        self.classes = classes or []

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                available = any(self.model_name in name for name in model_names)
                if available:
                    logger.info(f"[OK] Teacher model '{self.model_name}' is available")
                else:
                    logger.warning(f"[WARN] Model '{self.model_name}' not found. "
                                    f"Available: {model_names}")
                return available
            return False
        except Exception as e:
            logger.error(f"[ERROR] Cannot connect to Ollama: {e}")
            return False

    def classify_audio_description(self, audio_description: str) -> dict:
        """
        Ask Gemma to classify an audio sample based on its acoustic description.
        Returns a dict of class probabilities (soft labels).
        """
        prompt = self._build_classification_prompt(audio_description, modality="audio")
        return self._query_model(prompt)

    def classify_image_description(self, image_description: str) -> dict:
        """
        Ask Gemma to classify an image based on its description.
        Returns a dict of class probabilities (soft labels).
        """
        prompt = self._build_classification_prompt(image_description, modality="image")
        return self._query_model(prompt)

    def generate_soft_labels(self, description: str, num_classes: int) -> torch.Tensor:
        """
        Generate soft label distribution from teacher model.
        Returns a tensor of shape (num_classes,) with probabilities.
        """
        result = self._query_model(
            self._build_classification_prompt(description, modality="general")
        )

        # Convert to tensor
        probs = torch.zeros(num_classes)
        for cls_name, prob in result.items():
            if cls_name in self.classes:
                idx = self.classes.index(cls_name)
                if idx < num_classes:
                    probs[idx] = prob

        # Normalize
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            # Uniform fallback
            probs = torch.ones(num_classes) / num_classes

        return probs

    def _build_classification_prompt(self, description: str, modality: str) -> str:
        classes_str = ", ".join(self.classes)
        return f"""You are a classification expert. Given the following {modality} description, 
assign probability scores (0.0 to 1.0) to each of these categories: [{classes_str}].

Description: {description}

Respond ONLY with a JSON object mapping each category to its probability. 
The probabilities must sum to 1.0. Example format:
{{"Angry": 0.3, "Happy": 0.5, "Resting": 0.2}}

Your response (JSON only):"""

    def _query_model(self, prompt: str) -> dict:
        """Send prompt to Ollama and parse the JSON response."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=60
            )

            if resp.status_code == 200:
                text = resp.json().get("response", "")
                # Try to extract JSON from response
                return self._parse_json_response(text)
            else:
                logger.warning(f"Ollama returned status {resp.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error querying teacher model: {e}")
            return {}

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON dict from model response text."""
        try:
            # Try direct parse
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        import re
        pattern = r'\{[^}]+\}'
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

        logger.warning(f"Could not parse teacher response: {text[:200]}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# MOCK TEACHER (for development without Ollama)
# ──────────────────────────────────────────────────────────────────────────────

class MockTeacher:
    """
    Generates synthetic teacher logits for pipeline testing.
    Simulates a teacher that is 80% correct with soft probability distributions.
    """

    def __init__(self, num_classes: int = 10, noise_level: float = 0.3):
        self.num_classes = num_classes
        self.noise_level = noise_level

    def generate_logits(self, true_labels: torch.Tensor) -> torch.Tensor:
        """
        Generate mock teacher logits that peak at the true class with noise.
        """
        batch_size = true_labels.size(0)
        logits = torch.randn(batch_size, self.num_classes) * self.noise_level

        # Make the correct class have the highest logit (80% of the time)
        for i, label in enumerate(true_labels):
            if torch.rand(1).item() < 0.8:
                logits[i, label] += 3.0  # Strong signal for correct class
            else:
                # Occasionally "wrong" — picks a random class
                wrong = torch.randint(0, self.num_classes, (1,)).item()
                logits[i, wrong] += 3.0

        return logits


# ──────────────────────────────────────────────────────────────────────────────
# TEMPERATURE SCHEDULING
# ──────────────────────────────────────────────────────────────────────────────

class TemperatureScheduler:
    """
    Adaptive temperature scheduling for distillation.
    Starts high (more teacher knowledge) and anneals toward 1.0 (standard softmax).
    """

    def __init__(self, initial_temp: float = 5.0, final_temp: float = 1.0,
                 total_epochs: int = 15, strategy: str = "linear"):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_epochs = total_epochs
        self.strategy = strategy

    def get_temperature(self, epoch: int) -> float:
        """Get temperature for the given epoch."""
        progress = min(epoch / max(self.total_epochs - 1, 1), 1.0)

        if self.strategy == "linear":
            return self.initial_temp + (self.final_temp - self.initial_temp) * progress
        elif self.strategy == "cosine":
            return self.final_temp + (self.initial_temp - self.final_temp) * \
                   0.5 * (1 + np.cos(np.pi * progress))
        elif self.strategy == "step":
            # Step down at 1/3 and 2/3 of training
            if progress < 0.33:
                return self.initial_temp
            elif progress < 0.66:
                return (self.initial_temp + self.final_temp) / 2
            else:
                return self.final_temp
        else:
            return self.initial_temp
