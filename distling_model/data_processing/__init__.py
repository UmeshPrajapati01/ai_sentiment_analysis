"""
Data Processing Module
━━━━━━━━━━━━━━━━━━━━━
Handles ingestion, preprocessing, and train/test split for:
  • Classified audio   → MFCC feature extraction
  • Non-classified audio → raw feature extraction (for teacher labeling)
  • Classified images  → resize + normalize
  • Non-classified images → raw preprocessing (for teacher labeling)
"""

import os
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# AUDIO PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def extract_mfcc(file_path: str, sr: int = 16000, n_mfcc: int = 40,
                 max_len: int = 48000, hop_length: int = 512) -> np.ndarray:
    """
    Extract MFCC features from an audio file.
    Returns a 2D numpy array of shape (n_mfcc, time_steps).
    Pads or truncates to fixed length.
    """
    import librosa
    try:
        y, loaded_sr = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        logger.warning(f"Failed to load audio {file_path}: {e}")
        # Return silence
        y = np.zeros(max_len, dtype=np.float32)
        loaded_sr = sr

    # Pad or truncate
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        y = y[:max_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc.astype(np.float32)


def extract_mel_spectrogram(file_path: str, sr: int = 16000, n_mels: int = 128,
                            max_len: int = 48000, hop_length: int = 512) -> np.ndarray:
    """
    Extract mel-spectrogram from an audio file.
    Returns shape (n_mels, time_steps).
    """
    import librosa
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        logger.warning(f"Failed to load audio {file_path}: {e}")
        y = np.zeros(max_len, dtype=np.float32)

    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        y = y[:max_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(file_path: str, size: tuple = (224, 224)) -> np.ndarray:
    """
    Load and preprocess an image to a normalized numpy array.
    Returns shape (3, H, W) normalized to [0, 1].
    """
    from PIL import Image
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        # HWC -> CHW
        arr = arr.transpose(2, 0, 1)
        return arr
    except Exception as e:
        logger.warning(f"Failed to load image {file_path}: {e}")
        return np.zeros((3, size[0], size[1]), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET CLASSES
# ──────────────────────────────────────────────────────────────────────────────

class ClassifiedAudioDataset(Dataset):
    """
    Loads classified audio files organized as:
      classified_audio/ClassName/file.wav
    """
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

    def __init__(self, root_dir: str, classes: list, sr: int = 16000,
                 n_mfcc: int = 40, max_len: int = 48000, hop_length: int = 512):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.hop_length = hop_length
        self.samples = []

        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                logger.warning(f"Class directory not found: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.AUDIO_EXTENSIONS:
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls_name]
                    ))

        logger.info(f"ClassifiedAudioDataset: loaded {len(self.samples)} samples "
                     f"across {len(classes)} classes from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mfcc = extract_mfcc(path, sr=self.sr, n_mfcc=self.n_mfcc,
                            max_len=self.max_len, hop_length=self.hop_length)
        # Add channel dimension: (1, n_mfcc, time_steps)
        mfcc = np.expand_dims(mfcc, axis=0)
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class NonClassifiedAudioDataset(Dataset):
    """
    Loads non-classified audio files (flat directory).
    Returns features + file paths (no labels).
    """
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

    def __init__(self, root_dir: str, sr: int = 16000,
                 n_mfcc: int = 40, max_len: int = 48000, hop_length: int = 512):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.hop_length = hop_length
        self.files = []

        if os.path.isdir(root_dir):
            for fname in os.listdir(root_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.AUDIO_EXTENSIONS:
                    self.files.append(os.path.join(root_dir, fname))

        logger.info(f"NonClassifiedAudioDataset: {len(self.files)} files from {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        mfcc = extract_mfcc(path, sr=self.sr, n_mfcc=self.n_mfcc,
                            max_len=self.max_len, hop_length=self.hop_length)
        mfcc = np.expand_dims(mfcc, axis=0)
        return torch.tensor(mfcc, dtype=torch.float32), path


class ClassifiedImageDataset(Dataset):
    """
    Loads classified images organized as:
      classified_imagedata/ClassName/file.jpg
    With optional data augmentation for training.
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, root_dir: str, classes: list, size: tuple = (224, 224),
                 augment: bool = False):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.size = size
        self.augment = augment
        self.samples = []

        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.IMAGE_EXTENSIONS:
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls_name]
                    ))

        logger.info(f"ClassifiedImageDataset: loaded {len(self.samples)} samples "
                     f"across {len(classes)} classes from {root_dir} "
                     f"(augment={augment})")

    def __len__(self):
        return len(self.samples)

    def _augment_image(self, arr: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a CHW image array."""
        # Random horizontal flip
        if random.random() < 0.5:
            arr = arr[:, :, ::-1].copy()

        # Random vertical flip (less common, but helps)
        if random.random() < 0.15:
            arr = arr[:, ::-1, :].copy()

        # Random brightness/contrast adjustment
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            arr = np.clip(arr * brightness, 0, 1)

        # Random noise injection
        if random.random() < 0.3:
            noise = np.random.randn(*arr.shape).astype(np.float32) * 0.02
            arr = np.clip(arr + noise, 0, 1)

        # Random channel-wise color shift
        if random.random() < 0.3:
            for c in range(3):
                shift = random.uniform(-0.05, 0.05)
                arr[c] = np.clip(arr[c] + shift, 0, 1)

        return arr.astype(np.float32)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = preprocess_image(path, size=self.size)
        if self.augment:
            img = self._augment_image(img)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class AugmentedAudioDataset(Dataset):
    """
    Wraps ClassifiedAudioDataset with on-the-fly audio augmentation.
    Augmentations: time-shift, noise injection, volume change.
    """
    def __init__(self, base_dataset: ClassifiedAudioDataset, augment: bool = True):
        self.base = base_dataset
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def _augment_mfcc(self, mfcc: torch.Tensor) -> torch.Tensor:
        """Apply augmentation on MFCC features (1, n_mfcc, time)."""
        # Random time shift
        if random.random() < 0.4:
            shift = random.randint(-5, 5)
            mfcc = torch.roll(mfcc, shifts=shift, dims=2)

        # Random noise
        if random.random() < 0.4:
            noise = torch.randn_like(mfcc) * 0.05
            mfcc = mfcc + noise

        # Random volume (scale)
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            mfcc = mfcc * scale

        # Frequency masking (mask a few MFCC bands)
        if random.random() < 0.3:
            n_bands = random.randint(1, 4)
            start = random.randint(0, mfcc.shape[1] - n_bands)
            mfcc[:, start:start+n_bands, :] = 0

        # Time masking (mask a few time steps)
        if random.random() < 0.3:
            n_steps = random.randint(1, 8)
            start = random.randint(0, max(mfcc.shape[2] - n_steps, 1))
            mfcc[:, :, start:start+n_steps] = 0

        return mfcc

    def __getitem__(self, idx):
        mfcc, label = self.base[idx]
        if self.augment:
            mfcc = self._augment_mfcc(mfcc)
        return mfcc, label


# ──────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE: ANALYSIS + SPLITTING
# ──────────────────────────────────────────────────────────────────────────────

def analyze_dataset(dataset: Dataset, name: str = "Dataset") -> dict:
    """Analyze dataset for class distribution and basic stats (fast — label only)."""
    stats = {"name": name, "total_samples": len(dataset), "class_distribution": {}}

    # For wrapped datasets (AugmentedAudioDataset), get the base
    base = getattr(dataset, 'base', dataset)
    if hasattr(base, 'samples'):
        label_counts = {}
        for _, label in base.samples:
            if isinstance(label, int):
                label_counts[label] = label_counts.get(label, 0) + 1
        stats["class_distribution"] = label_counts
        logger.info(f"[STATS] {name}: {len(dataset)} samples, distribution: {label_counts}")
        return stats

    # Fallback: iterate (slow for large datasets)
    label_counts = {}
    for i in range(min(len(dataset), 500)):  # cap at 500 to avoid slowness
        try:
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            elif isinstance(label, str):
                continue
            label_counts[label] = label_counts.get(label, 0) + 1
        except Exception:
            continue

    stats["class_distribution"] = label_counts
    logger.info(f"[STATS] {name}: {len(dataset)} samples, distribution: {label_counts}")
    return stats


def create_data_splits(dataset: Dataset, train_ratio: float = 0.8,
                       val_ratio: float = 0.1, test_ratio: float = 0.1,
                       seed: int = 42):
    """
    Split a dataset into train/validation/test sets.
    Returns (train_dataset, val_dataset, test_dataset).
    """
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    logger.info(f"[SPLIT] train={train_size}, val={val_size}, test={test_size}")
    return train_ds, val_ds, test_ds


def create_dataloaders(train_ds, val_ds, test_ds, batch_size: int = 16):
    """Create DataLoaders for train/val/test splits."""
    import torch
    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=use_pin)
    return train_loader, val_loader, test_loader

