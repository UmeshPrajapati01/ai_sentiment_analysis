"""
Student Model Architectures
━━━━━━━━━━━━━━━━━━━━━━━━━━
Lightweight models designed to be distilled from Gemma3:4B teacher.
  • AudioStudentModel  — CNN on MFCC features for audio classification
  • ImageStudentModel  — Lightweight CNN for image classification
  • MultimodalStudent  — Fused audio+image student model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioStudentModel(nn.Module):
    """
    Compact CNN for audio classification using MFCC input.
    Input shape: (batch, 1, n_mfcc, time_steps)  e.g. (B, 1, 40, 94)
    """

    def __init__(self, num_classes: int = 10, n_mfcc: int = 40):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_embeddings(self, x):
        """Return intermediate embeddings (before final classifier layer)."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # Pass through all but last linear layer
        for layer in list(self.classifier.children())[:-1]:
            x = layer(x)
        return x


class ImageStudentModel(nn.Module):
    """
    Lightweight CNN for image classification.
    Input shape: (batch, 3, 224, 224)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # -> 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # -> 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # -> 28x28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MultimodalStudentModel(nn.Module):
    """
    Fused multimodal student model: combines audio + image branches.
    Each branch produces an embedding, which are concatenated and classified.
    """

    def __init__(self, num_classes: int = 10, n_mfcc: int = 40):
        super().__init__()

        # Audio branch (from MFCC)
        self.audio_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.audio_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Image branch
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.image_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, audio_input, image_input):
        audio_emb = self.audio_fc(self.audio_branch(audio_input))
        image_emb = self.image_fc(self.image_branch(image_input))
        fused = torch.cat([audio_emb, image_emb], dim=1)
        return self.fusion_classifier(fused)

    def forward_audio_only(self, audio_input):
        """Classify with audio input only (image zeroed)."""
        audio_emb = self.audio_fc(self.audio_branch(audio_input))
        image_emb = torch.zeros(audio_input.size(0), 128, device=audio_input.device)
        fused = torch.cat([audio_emb, image_emb], dim=1)
        return self.fusion_classifier(fused)

    def forward_image_only(self, image_input):
        """Classify with image input only (audio zeroed)."""
        audio_emb = torch.zeros(image_input.size(0), 128, device=image_input.device)
        image_emb = self.image_fc(self.image_branch(image_input))
        fused = torch.cat([audio_emb, image_emb], dim=1)
        return self.fusion_classifier(fused)
