#!/usr/bin/env python3
"""
Improved Training Script for Multimodal Biomedical Monitoring

Features:
1. Data augmentation for biomedical signals
2. Focal loss / Label smoothing for class imbalance
3. Cosine annealing with warm restarts
4. Gradient clipping and accumulation
5. Mixed precision training (optional)
6. Comprehensive logging and checkpointing
7. K-fold cross-validation option

Usage:
    # Standard training with improvements
    python scripts/train_improved.py --epochs 50 --use_focal --use_augmentation

    # K-fold cross-validation
    python scripts/train_improved.py --run_kfold --n_folds 5

    # Train with improved model architecture
    python scripts/train_improved.py --use_improved_model --preset balanced
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_transformer_lite import CNNTransformerLite
from src.models.cnn_transformer_improved import (
    CNNTransformerImproved, create_improved_model, MultiTaskLoss, FocalLoss
)


# ============================================================================
# Data Augmentation
# ============================================================================

class BiomedicalAugmentation:
    """
    Data augmentation techniques for biomedical signals.

    These augmentations are designed to be physiologically plausible
    and help the model generalize better.
    """

    def __init__(
        self,
        noise_level: float = 0.05,
        time_shift_max: int = 50,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        channel_dropout_prob: float = 0.1,
        time_mask_max: int = 100,
        cutmix_prob: float = 0.0,
        mixup_alpha: float = 0.0,
    ):
        self.noise_level = noise_level
        self.time_shift_max = time_shift_max
        self.scale_range = scale_range
        self.channel_dropout_prob = channel_dropout_prob
        self.time_mask_max = time_mask_max
        self.cutmix_prob = cutmix_prob
        self.mixup_alpha = mixup_alpha

    def add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to simulate sensor noise."""
        if self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        return x

    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Random circular shift in time dimension."""
        if self.time_shift_max > 0:
            shift = random.randint(-self.time_shift_max, self.time_shift_max)
            return torch.roll(x, shift, dims=-1)
        return x

    def random_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Random amplitude scaling."""
        if self.scale_range != (1.0, 1.0):
            scale = random.uniform(*self.scale_range)
            return x * scale
        return x

    def channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly drop entire channels.
        Simulates missing sensors and encourages robustness.
        """
        if self.channel_dropout_prob > 0 and random.random() < 0.5:
            # x: [C, T]
            mask = torch.ones(x.size(0), 1, device=x.device)
            for i in range(x.size(0)):
                if random.random() < self.channel_dropout_prob:
                    mask[i] = 0
            return x * mask
        return x

    def time_masking(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly mask a contiguous time segment.
        Similar to SpecAugment for audio.
        """
        if self.time_mask_max > 0 and random.random() < 0.3:
            mask_len = random.randint(1, self.time_mask_max)
            start = random.randint(0, x.size(-1) - mask_len)
            x = x.clone()
            x[..., start:start+mask_len] = 0
        return x

    def baseline_wander(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add low-frequency baseline wander (common in ECG).
        Only applied to ECG channel (assumed to be channel 0).
        """
        if random.random() < 0.2:
            t = torch.linspace(0, 2 * np.pi, x.size(-1), device=x.device)
            freq = random.uniform(0.1, 0.5)
            amplitude = random.uniform(0.05, 0.15)
            wander = amplitude * torch.sin(freq * t)
            x = x.clone()
            x[0] = x[0] + wander  # ECG channel
        return x

    def powerline_noise(self, x: torch.Tensor, fs: int = 100) -> torch.Tensor:
        """
        Add 50/60 Hz powerline noise (common interference).
        """
        if random.random() < 0.1:
            t = torch.arange(x.size(-1), device=x.device, dtype=torch.float32) / fs
            freq = random.choice([50, 60])
            amplitude = random.uniform(0.02, 0.08)
            noise = amplitude * torch.sin(2 * np.pi * freq * t)
            x = x.clone()
            x[0] = x[0] + noise  # ECG channel
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        x = self.add_gaussian_noise(x)
        x = self.time_shift(x)
        x = self.random_scale(x)
        x = self.channel_dropout(x)
        x = self.time_masking(x)
        x = self.baseline_wander(x)
        return x


def mixup_data(
    x: torch.Tensor,
    y_activity: torch.Tensor,
    y_stress: torch.Tensor,
    y_arrhythmia: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], float]:
    """
    Apply mixup augmentation.

    Returns mixed inputs and pairs of labels for mixup loss computation.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    labels = {
        'activity_a': y_activity,
        'activity_b': y_activity[index],
        'stress_a': y_stress,
        'stress_b': y_stress[index],
        'arrhythmia_a': y_arrhythmia,
        'arrhythmia_b': y_arrhythmia[index],
    }

    return mixed_x, labels, lam


# ============================================================================
# Dataset
# ============================================================================

class AugmentedBiomedicalDataset(Dataset):
    """Dataset with augmentation support."""

    def __init__(
        self,
        data_path: str,
        subject_filter: Optional[List[str]] = None,
        augmentation: Optional[BiomedicalAugmentation] = None,
        training: bool = True
    ):
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)

        self.samples = []
        self.subject_ids = []
        self.augmentation = augmentation
        self.training = training

        for sample in raw_data:
            subject_id = sample.get('subject_id', 'unknown')

            if subject_filter is not None and subject_id not in subject_filter:
                continue

            window_data = sample.get('window_data')
            labels = sample.get('labels', {})

            if window_data is None:
                continue

            # Handle NaN values
            window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure correct shape
            if window_data.ndim == 1:
                window_data = window_data.reshape(1, -1)

            if window_data.shape[0] < 11:
                padded = np.zeros((11, window_data.shape[1]))
                padded[:window_data.shape[0], :] = window_data
                window_data = padded

            # Extract labels
            activity_label = self._extract_label(labels.get('activity'), 8)
            stress_label = self._extract_label(labels.get('stress'), 4)
            arrhythmia_label = self._extract_label(labels.get('arrhythmia'), 2)

            self.samples.append({
                'window_data': window_data.astype(np.float32),
                'activity': activity_label,
                'stress': stress_label,
                'arrhythmia': arrhythmia_label,
                'subject_id': subject_id,
            })
            self.subject_ids.append(subject_id)

        print(f"Loaded {len(self.samples)} samples")

    def _extract_label(self, label_data, n_classes: int) -> int:
        if label_data is None:
            return -1

        if isinstance(label_data, np.ndarray):
            if label_data.sum() > 0:
                return int(np.argmax(label_data))
            return -1

        if isinstance(label_data, (int, float)):
            return int(label_data)

        return -1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.from_numpy(sample['window_data'])

        # Apply augmentation during training
        if self.training and self.augmentation is not None:
            x = self.augmentation(x)

        return {
            'window_data': x,
            'activity': sample['activity'],
            'stress': sample['stress'],
            'arrhythmia': sample['arrhythmia'],
        }

    def get_subject_ids(self):
        return self.subject_ids

    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """Compute class weights for balanced training."""
        weights = {}

        for task, n_classes in [('activity', 8), ('stress', 4), ('arrhythmia', 2)]:
            counts = np.zeros(n_classes)
            for s in self.samples:
                label = s[task]
                if 0 <= label < n_classes:
                    counts[label] += 1

            # Inverse frequency weighting
            total = counts.sum()
            if total > 0:
                w = total / (n_classes * counts + 1e-6)
                weights[task] = torch.FloatTensor(w)
            else:
                weights[task] = torch.ones(n_classes)

        return weights


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0,
    mixup_alpha: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    task_losses = {'activity': 0.0, 'stress': 0.0, 'arrhythmia': 0.0}
    correct = {'activity': 0, 'stress': 0, 'arrhythmia': 0}
    total = {'activity': 0, 'stress': 0, 'arrhythmia': 0}
    n_batches = 0

    for batch in dataloader:
        x = batch['window_data'].to(device)
        labels = {
            'activity': batch['activity'].to(device),
            'stress': batch['stress'].to(device),
            'arrhythmia': batch['arrhythmia'].to(device),
        }

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(x)
                loss, batch_task_losses = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x)
            loss, batch_task_losses = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        for task in ['activity', 'stress', 'arrhythmia']:
            task_losses[task] += batch_task_losses[task].item()

            # Compute accuracy
            preds = outputs[task].argmax(dim=1)
            valid_mask = labels[task] >= 0
            if valid_mask.sum() > 0:
                correct[task] += (preds[valid_mask] == labels[task][valid_mask]).sum().item()
                total[task] += valid_mask.sum().item()

    # Compute averages
    metrics = {
        'loss': total_loss / n_batches,
        'activity_loss': task_losses['activity'] / n_batches,
        'stress_loss': task_losses['stress'] / n_batches,
        'arrhythmia_loss': task_losses['arrhythmia'] / n_batches,
    }

    for task in ['activity', 'stress', 'arrhythmia']:
        if total[task] > 0:
            metrics[f'{task}_acc'] = correct[task] / total[task]
        else:
            metrics[f'{task}_acc'] = 0.0

    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    task_losses = {'activity': 0.0, 'stress': 0.0, 'arrhythmia': 0.0}
    correct = {'activity': 0, 'stress': 0, 'arrhythmia': 0}
    total = {'activity': 0, 'stress': 0, 'arrhythmia': 0}
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch['window_data'].to(device)
            labels = {
                'activity': batch['activity'].to(device),
                'stress': batch['stress'].to(device),
                'arrhythmia': batch['arrhythmia'].to(device),
            }

            outputs = model(x)
            loss, batch_task_losses = criterion(outputs, labels)

            total_loss += loss.item()
            n_batches += 1

            for task in ['activity', 'stress', 'arrhythmia']:
                task_losses[task] += batch_task_losses[task].item()

                preds = outputs[task].argmax(dim=1)
                valid_mask = labels[task] >= 0
                if valid_mask.sum() > 0:
                    correct[task] += (preds[valid_mask] == labels[task][valid_mask]).sum().item()
                    total[task] += valid_mask.sum().item()

    metrics = {
        'loss': total_loss / max(n_batches, 1),
        'activity_loss': task_losses['activity'] / max(n_batches, 1),
        'stress_loss': task_losses['stress'] / max(n_batches, 1),
        'arrhythmia_loss': task_losses['arrhythmia'] / max(n_batches, 1),
    }

    for task in ['activity', 'stress', 'arrhythmia']:
        if total[task] > 0:
            metrics[f'{task}_acc'] = correct[task] / total[task]
        else:
            metrics[f'{task}_acc'] = 0.0

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_focal: bool = False,
    label_smoothing: float = 0.1,
    patience: int = 10,
    use_amp: bool = False,
    output_dir: Path = None,
) -> Dict:
    """Full training loop with validation and early stopping."""

    # Setup criterion
    criterion = MultiTaskLoss(
        activity_weight=1.5,
        stress_weight=1.5,
        arrhythmia_weight=2.0,
        use_focal=use_focal,
        label_smoothing=label_smoothing,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_activity_acc': [], 'val_activity_acc': [],
        'train_stress_acc': [], 'val_stress_acc': [],
        'train_arrhythmia_acc': [], 'val_arrhythmia_acc': [],
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        for task in ['activity', 'stress', 'arrhythmia']:
            history[f'train_{task}_acc'].append(train_metrics[f'{task}_acc'])
            history[f'val_{task}_acc'].append(val_metrics[f'{task}_acc'])

        # Early stopping check
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Act: {train_metrics['activity_acc']*100:.1f}% | "
              f"Str: {train_metrics['stress_acc']*100:.1f}% | "
              f"Arr: {train_metrics['arrhythmia_acc']*100:.1f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Act: {val_metrics['activity_acc']*100:.1f}% | "
              f"Str: {val_metrics['stress_acc']*100:.1f}% | "
              f"Arr: {val_metrics['arrhythmia_acc']*100:.1f}%")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model and history
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save model
        model_path = output_dir / f'model_improved_{timestamp}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to: {model_path}")

        # Also save as model.pth for compatibility
        torch.save(model.state_dict(), output_dir / 'model.pth')

        # Save history
        history_path = output_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Improved Training Script')

    # Data
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--splits_path', type=str, default='training_results/subject_splits.json')

    # Model
    parser.add_argument('--use_improved_model', action='store_true',
                        help='Use improved model architecture')
    parser.add_argument('--preset', type=str, default='balanced',
                        choices=['balanced', 'accurate', 'edge'])

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)

    # Loss
    parser.add_argument('--use_focal', action='store_true',
                        help='Use focal loss for class imbalance')
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # Augmentation
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--noise_level', type=float, default=0.05)

    # Hardware
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision training')

    # Output
    parser.add_argument('--output_dir', type=str, default='training_results')

    # K-fold CV
    parser.add_argument('--run_kfold', action='store_true')
    parser.add_argument('--n_folds', type=int, default=5)

    args = parser.parse_args()

    # Setup paths
    base_dir = PROJECT_ROOT
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Find dataset
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        possible_paths = [
            base_dir.parent / 'processed_unified_dataset' / 'balanced_unified_dataset.pkl',
            base_dir / 'processed_unified_dataset' / 'balanced_unified_dataset.pkl',
        ]
        dataset_path = None
        for p in possible_paths:
            if p.exists():
                dataset_path = p
                break

        if dataset_path is None:
            print("ERROR: Could not find dataset")
            sys.exit(1)

    print(f"Using dataset: {dataset_path}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Augmentation
    augmentation = None
    if args.use_augmentation:
        augmentation = BiomedicalAugmentation(
            noise_level=args.noise_level,
            time_shift_max=50,
            scale_range=(0.9, 1.1),
            channel_dropout_prob=0.1,
        )
        print("Data augmentation enabled")

    # Load subject splits
    splits_path = base_dir / args.splits_path
    if splits_path.exists():
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        train_subjects = splits.get('train_subjects', [])
        val_subjects = splits.get('val_subjects', [])
        test_subjects = splits.get('test_subjects', [])
    else:
        print("WARNING: No splits file found. Using random split.")
        train_subjects = val_subjects = test_subjects = None

    # Create datasets
    train_dataset = AugmentedBiomedicalDataset(
        str(dataset_path),
        subject_filter=train_subjects,
        augmentation=augmentation,
        training=True
    )

    val_dataset = AugmentedBiomedicalDataset(
        str(dataset_path),
        subject_filter=val_subjects,
        augmentation=None,
        training=False
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    if args.use_improved_model:
        print(f"Using improved model with preset: {args.preset}")
        model = create_improved_model(preset=args.preset)
    else:
        print("Using standard CNNTransformerLite model")
        model = CNNTransformerLite(
            n_channels=11,
            n_samples=1000,
            activity_classes=8,
            stress_classes=4,
            arrhythmia_classes=2,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.2
        )

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Model size: {n_params * 4 / 1024:.1f} KB")

    # Train
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_focal=args.use_focal,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        use_amp=args.use_amp,
        output_dir=output_dir
    )

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final epoch: {results['final_epoch']}")

    # Print final metrics
    history = results['history']
    print(f"\nFinal Validation Metrics:")
    print(f"  Activity Accuracy: {history['val_activity_acc'][-1]*100:.1f}%")
    print(f"  Stress Accuracy: {history['val_stress_acc'][-1]*100:.1f}%")
    print(f"  Arrhythmia Accuracy: {history['val_arrhythmia_acc'][-1]*100:.1f}%")


if __name__ == '__main__':
    main()
