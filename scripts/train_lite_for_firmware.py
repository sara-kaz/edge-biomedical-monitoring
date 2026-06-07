#!/usr/bin/env python3
"""
Train CNNTransformerLite model for ESP32 firmware deployment.

This trains the LITE model (matching the firmware C++ inference engine architecture)
with 5 channels, then exports weights to .inc files for flashing.

The firmware implements: 3-conv CNN → projection → FC heads (no transformer on device).
CNNTransformerLite uses a transformer during training but the export script only
extracts the CNN + projection + head weights.

Usage:
    python scripts/train_lite_for_firmware.py
    python scripts/train_lite_for_firmware.py --epochs 100
"""

import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_transformer_lite import CNNTransformerLite

# Activity class merging: 8 → 4
ACTIVITY_MERGE_MAP = {
    0: 0,  # sitting -> sedentary
    1: 1,  # walking -> walking
    2: 2,  # cycling -> cycling
    3: 0,  # driving -> sedentary
    4: 0,  # working -> sedentary
    5: 3,  # stairs -> high_intensity
    6: 3,  # table_soccer -> high_intensity
    7: 0,  # lunch -> sedentary
}
MERGED_ACTIVITY_NAMES = {0: 'sedentary', 1: 'walking', 2: 'cycling', 3: 'high_intensity'}
N_MERGED_ACTIVITY = 4


# =============================================================================
# Dataset (reused from train_v3.py)
# =============================================================================

class BiomedicalDatasetLite(Dataset):
    def __init__(self, data_path, subject_filter=None, binary_stress=True,
                 augment=False, normalize=True, oversample=False, merge_activity=True):
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)

        self.binary_stress = binary_stress
        self.augment = augment
        self.normalize = normalize
        self.merge_activity = merge_activity
        self.n_channels = None
        self.samples = []
        self.activity_labels = []
        self.stress_labels = []
        self.arrhythmia_labels = []

        # Compute normalization stats
        if normalize:
            all_data = []
            for sample in raw_data:
                if subject_filter and sample.get('subject_id') not in subject_filter:
                    continue
                wd = sample.get('window_data')
                if wd is not None:
                    all_data.append(wd)
            if all_data:
                stacked = np.stack(all_data[:5000])
                self.channel_means = np.nanmean(stacked, axis=(0, 2), keepdims=True).squeeze(0)
                self.channel_stds = np.nanstd(stacked, axis=(0, 2), keepdims=True).squeeze(0)
                self.channel_stds[self.channel_stds < 1e-6] = 1.0
            else:
                self.channel_means = self.channel_stds = None

        # Load samples
        for sample in raw_data:
            sid = sample.get('subject_id', 'unknown')
            if subject_filter is not None and sid not in subject_filter:
                continue
            window_data = sample.get('window_data')
            labels = sample.get('labels', {})
            if window_data is None:
                continue
            window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)
            if self.n_channels is None:
                self.n_channels = window_data.shape[0]

            activity = self._decode(labels.get('activity'))
            stress = self._decode(labels.get('stress'))
            arrhythmia = self._decode(labels.get('arrhythmia'))

            if self.merge_activity and activity >= 0:
                activity = ACTIVITY_MERGE_MAP.get(activity, -1)
            if self.binary_stress and stress >= 0:
                stress = 1 if stress == 1 else 0

            self.samples.append(window_data.astype(np.float32))
            self.activity_labels.append(activity)
            self.stress_labels.append(stress)
            self.arrhythmia_labels.append(arrhythmia)

        if oversample:
            self._oversample()

        print(f"  Loaded {len(self.samples)} samples, {self.n_channels} channels")
        self._print_stats()

    def _decode(self, label_data):
        if label_data is None:
            return -1
        if isinstance(label_data, np.ndarray):
            return int(np.argmax(label_data)) if label_data.sum() > 0 else -1
        if isinstance(label_data, (int, float)):
            return int(label_data)
        return -1

    def _oversample(self):
        """SMOTE-like oversampling."""
        n_act = N_MERGED_ACTIVITY if self.merge_activity else 8
        valid_act = [(i, self.activity_labels[i]) for i in range(len(self.samples))
                     if self.activity_labels[i] >= 0]
        if valid_act:
            act_counts = Counter(a for _, a in valid_act)
            sorted_counts = sorted(act_counts.values())
            median_count = sorted_counts[len(sorted_counts) // 2]
            max_count = sorted_counts[-1]
            target = min(int(np.sqrt(median_count * max_count)), max_count // 2)
            for cls in range(n_act):
                cls_idx = [i for i, a in valid_act if a == cls]
                if len(cls_idx) < 2:
                    continue
                need = min(target - len(cls_idx), len(cls_idx) * 10)
                if need <= 0:
                    continue
                for _ in range(need):
                    i1, i2 = random.sample(cls_idx, 2)
                    lam = random.uniform(0.3, 0.7)
                    syn = self.samples[i1] * lam + self.samples[i2] * (1 - lam)
                    syn += np.random.randn(*syn.shape).astype(np.float32) * 0.02
                    self.samples.append(syn)
                    self.activity_labels.append(cls)
                    self.stress_labels.append(self.stress_labels[i1])
                    self.arrhythmia_labels.append(self.arrhythmia_labels[i1])

        # Oversample arrhythmia minority
        arr_valid = [(i, self.arrhythmia_labels[i]) for i in range(len(self.samples))
                     if self.arrhythmia_labels[i] >= 0]
        if arr_valid:
            arr_counts = Counter(a for _, a in arr_valid)
            if 1 in arr_counts and 0 in arr_counts:
                target = int(arr_counts[0] * 0.4)
                cls1 = [i for i, a in arr_valid if a == 1]
                need = target - len(cls1)
                if need > 0 and len(cls1) >= 2:
                    for _ in range(need):
                        i1, i2 = random.sample(cls1, 2)
                        lam = random.uniform(0.3, 0.7)
                        syn = self.samples[i1] * lam + self.samples[i2] * (1 - lam)
                        syn += np.random.randn(*syn.shape).astype(np.float32) * 0.02
                        self.samples.append(syn)
                        self.activity_labels.append(self.activity_labels[i1])
                        self.stress_labels.append(self.stress_labels[i1])
                        self.arrhythmia_labels.append(1)

        # Oversample stress minority
        str_valid = [(i, self.stress_labels[i]) for i in range(len(self.samples))
                     if self.stress_labels[i] >= 0]
        if str_valid:
            str_counts = Counter(a for _, a in str_valid)
            if 1 in str_counts and 0 in str_counts:
                target = int(str_counts[0] * 0.8)
                cls1 = [i for i, a in str_valid if a == 1]
                need = target - len(cls1)
                if need > 0 and len(cls1) >= 2:
                    for _ in range(need):
                        i1, i2 = random.sample(cls1, 2)
                        lam = random.uniform(0.3, 0.7)
                        syn = self.samples[i1] * lam + self.samples[i2] * (1 - lam)
                        syn += np.random.randn(*syn.shape).astype(np.float32) * 0.02
                        self.samples.append(syn)
                        self.activity_labels.append(self.activity_labels[i1])
                        self.stress_labels.append(1)
                        self.arrhythmia_labels.append(self.arrhythmia_labels[i1])

    def _print_stats(self):
        act = [a for a in self.activity_labels if a >= 0]
        strs = [s for s in self.stress_labels if s >= 0]
        arr = [a for a in self.arrhythmia_labels if a >= 0]
        print(f"  Activity ({len(act)}): {dict(Counter(act))}")
        print(f"  Stress ({len(strs)}): {dict(Counter(strs))}")
        print(f"  Arrhythmia ({len(arr)}): {dict(Counter(arr))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].copy()
        if self.normalize and self.channel_means is not None:
            x = (x - self.channel_means) / self.channel_stds
        x = torch.from_numpy(x.astype(np.float32))
        if self.augment:
            if random.random() < 0.5:
                x = x + torch.randn_like(x) * 0.05
            if random.random() < 0.3:
                x = torch.roll(x, random.randint(-50, 50), dims=-1)
            if random.random() < 0.4:
                for ch in range(x.size(0)):
                    x[ch] *= random.uniform(0.8, 1.2)
            if random.random() < 0.3:
                ml = random.randint(20, 100)
                st = random.randint(0, x.size(-1) - ml)
                x[..., st:st+ml] = 0
            if random.random() < 0.15:
                x[random.randint(0, x.size(0)-1)] = 0
        return {
            'window_data': x,
            'activity': self.activity_labels[idx],
            'stress': self.stress_labels[idx],
            'arrhythmia': self.arrhythmia_labels[idx],
        }

    def get_sample_weights(self):
        weights = torch.ones(len(self.samples))
        act_counts = Counter(a for a in self.activity_labels if a >= 0)
        str_counts = Counter(s for s in self.stress_labels if s >= 0)
        arr_counts = Counter(a for a in self.arrhythmia_labels if a >= 0)
        for i in range(len(self.samples)):
            w = 1.0
            a, s, ar = self.activity_labels[i], self.stress_labels[i], self.arrhythmia_labels[i]
            if a >= 0 and a in act_counts:
                w = max(w, max(act_counts.values()) / act_counts[a])
            if s >= 0 and s in str_counts:
                w = max(w, max(str_counts.values()) / str_counts[s])
            if ar >= 0 and ar in arr_counts:
                w = max(w, max(arr_counts.values()) / arr_counts[ar])
            weights[i] = w
        return weights

    def get_class_weights(self):
        result = {}
        n_act = N_MERGED_ACTIVITY if self.merge_activity else 8
        n_str = 2 if self.binary_stress else 4
        for task, n_cls, labels in [
            ('activity', n_act, self.activity_labels),
            ('stress', n_str, self.stress_labels),
            ('arrhythmia', 2, self.arrhythmia_labels),
        ]:
            counts = np.zeros(n_cls)
            for lbl in labels:
                if 0 <= lbl < n_cls:
                    counts[lbl] += 1
            total = counts.sum()
            if total > 0:
                w = np.sqrt(total / (n_cls * np.maximum(counts, 1)))
                w = np.clip(w, 0.5, 5.0)
                result[task] = torch.FloatTensor(w)
            else:
                result[task] = torch.ones(n_cls)
        return result


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce = nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean() if self.reduction == 'mean' else focal


# =============================================================================
# Training
# =============================================================================

def train_lite_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Find dataset
    dataset_path = None
    for p in [
        PROJECT_ROOT / 'processed_unified_dataset' / 'unified_dataset.pkl',
        PROJECT_ROOT / 'data' / 'processed_unified_dataset' / 'unified_dataset.pkl',
        PROJECT_ROOT.parent / 'multimodal-biomedical-monitoring-improved' / 'processed_unified_dataset' / 'unified_dataset.pkl',
    ]:
        if p.exists():
            dataset_path = str(p)
            break
    if not dataset_path:
        print("ERROR: Cannot find unified_dataset.pkl")
        sys.exit(1)
    print(f"Dataset: {dataset_path}")

    # Load splits
    splits_path = PROJECT_ROOT / 'training_results' / 'subject_splits_v3.json'
    if not splits_path.exists():
        splits_path = PROJECT_ROOT / 'training_results' / 'subject_splits.json'
    with open(splits_path) as f:
        splits = json.load(f)
    train_subjects = splits['train_subjects']
    val_subjects = splits['val_subjects']
    test_subjects = splits['test_subjects']
    print(f"Splits: Train={len(train_subjects)}, Val={len(val_subjects)}, Test={len(test_subjects)}")

    activity_classes = N_MERGED_ACTIVITY
    stress_classes = 2

    # Create datasets
    print("\n--- Training set ---")
    train_ds = BiomedicalDatasetLite(dataset_path, train_subjects, augment=True, oversample=True)
    print("\n--- Validation set ---")
    val_ds = BiomedicalDatasetLite(dataset_path, val_subjects, augment=False, oversample=False)
    print("\n--- Test set ---")
    test_ds = BiomedicalDatasetLite(dataset_path, test_subjects, augment=False, oversample=False)

    n_channels = train_ds.n_channels
    print(f"\nChannels: {n_channels}, Activity: {activity_classes}, Stress: {stress_classes}")

    # Data loaders
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model — CNNTransformerLite with 5 channels
    # CRITICAL: use_transformer=False so training matches firmware inference path
    # (firmware does: CNN → global avg pool → projection → heads, NO transformer)
    model = CNNTransformerLite(
        n_channels=n_channels,
        n_samples=1000,
        activity_classes=activity_classes,
        stress_classes=stress_classes,
        arrhythmia_classes=2,
        d_model=64,
        nhead=4,
        num_layers=0,  # No transformer — matches firmware
        dim_feedforward=128,
        dropout=args.dropout,
        use_transformer=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,} ({n_params * 4 / 1024:.1f} KB)")

    # Loss functions
    class_weights = train_ds.get_class_weights()
    loss_fns = {
        'activity': FocalLoss(gamma=2.5, weight=class_weights['activity'].to(device)),
        'stress': FocalLoss(gamma=1.0, weight=class_weights['stress'].to(device)),
        'arrhythmia': FocalLoss(gamma=2.5, weight=class_weights['arrhythmia'].to(device)),
    }
    task_weights = {'activity': 1.0, 'stress': 1.5, 'arrhythmia': 2.0}

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                           steps_per_epoch=len(train_loader), pct_start=0.1)

    # Training loop
    best_val_score = 0
    patience_counter = 0
    best_model_state = None
    output_dir = PROJECT_ROOT / 'training_results'
    output_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for batch in train_loader:
            x = batch['window_data'].to(device)
            outputs = model(x)

            loss = torch.tensor(0.0, device=device)
            for task in ['activity', 'stress', 'arrhythmia']:
                labels = batch[task]
                valid = labels >= 0
                if valid.sum() == 0:
                    continue
                pred = outputs[task][valid]
                tgt = labels[valid].to(device)
                loss += task_weights[task] * loss_fns[task](pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_metrics = evaluate(model, val_loader, device, activity_classes, stress_classes)
        val_score = (val_metrics['activity_f1'] + val_metrics['stress_f1'] + val_metrics['arrhythmia_f1']) / 3

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_train_loss:.4f} | "
                  f"Val Act-F1: {val_metrics['activity_f1']:.3f} "
                  f"Str-F1: {val_metrics['stress_f1']:.3f} "
                  f"Arr-F1: {val_metrics['arrhythmia_f1']:.3f} | "
                  f"Score: {val_score:.3f}")

        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    model.to(device)

    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test_metrics = evaluate(model, test_loader, device, activity_classes, stress_classes)
    print(f"  Activity F1-macro: {test_metrics['activity_f1']:.4f}")
    print(f"  Activity Accuracy: {test_metrics['activity_acc']:.4f}")
    print(f"  Stress F1-macro:   {test_metrics['stress_f1']:.4f}")
    print(f"  Stress Accuracy:   {test_metrics['stress_acc']:.4f}")
    print(f"  Arrhythmia F1:     {test_metrics['arrhythmia_f1']:.4f}")
    print(f"  Arrhythmia Acc:    {test_metrics['arrhythmia_acc']:.4f}")

    # Save model
    save_path = output_dir / 'model_lite_firmware.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_channels': n_channels,
        'activity_classes': activity_classes,
        'stress_classes': stress_classes,
        'arrhythmia_classes': 2,
        'test_metrics': test_metrics,
        'architecture': 'CNNTransformerLite',
        'best_val_score': best_val_score,
    }, save_path)
    print(f"\nModel saved: {save_path}")

    # Export weights for firmware
    print("\n" + "=" * 60)
    print("EXPORTING WEIGHTS FOR ESP32 FIRMWARE")
    print("=" * 60)
    export_lite_weights(model, n_channels, activity_classes, stress_classes)

    return test_metrics


def evaluate(model, loader, device, n_act, n_str):
    """Evaluate model and return metrics."""
    from sklearn.metrics import f1_score, accuracy_score
    model.eval()
    all_preds = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_labels = {'activity': [], 'stress': [], 'arrhythmia': []}

    with torch.no_grad():
        for batch in loader:
            x = batch['window_data'].to(device)
            outputs = model(x)
            for task in ['activity', 'stress', 'arrhythmia']:
                labels = batch[task]
                valid = labels >= 0
                if valid.sum() > 0:
                    preds = outputs[task][valid].argmax(dim=1).cpu().numpy()
                    all_preds[task].extend(preds)
                    all_labels[task].extend(labels[valid].numpy())

    metrics = {}
    for task in ['activity', 'stress', 'arrhythmia']:
        if all_preds[task]:
            metrics[f'{task}_f1'] = f1_score(all_labels[task], all_preds[task], average='macro', zero_division=0)
            metrics[f'{task}_acc'] = accuracy_score(all_labels[task], all_preds[task])
        else:
            metrics[f'{task}_f1'] = 0
            metrics[f'{task}_acc'] = 0
    return metrics


def export_lite_weights(model, n_channels, activity_classes, stress_classes):
    """Export CNNTransformerLite weights to firmware .inc files."""
    output_dir = PROJECT_ROOT / 'firmware' / 'esp32' / 'src' / 'inference' / 'weights'
    output_dir.mkdir(parents=True, exist_ok=True)

    sd = model.state_dict()

    def fuse_conv_bn(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
        if conv_b is None:
            conv_b = torch.zeros(conv_w.shape[0])
        scale = bn_w / torch.sqrt(bn_var + eps)
        fused_w = conv_w * scale.view(-1, 1, 1)
        fused_b = (conv_b - bn_mean) * scale + bn_b
        return fused_w.numpy(), fused_b.numpy()

    def write_inc(arr, path, name):
        flat = arr.flatten()
        lines = []
        for i in range(0, len(flat), 8):
            chunk = flat[i:i+8]
            lines.append(', '.join(f'{v:.6f}f' for v in chunk))
        with open(path, 'w') as f:
            f.write(f"// {name}: shape {list(arr.shape)}\n")
            f.write(',\n'.join(lines))
        print(f"  {name}: {list(arr.shape)} ({arr.size} params)")

    total = 0

    # Conv1: cnn_layers.0 (Conv1d) + cnn_layers.1 (BatchNorm1d)
    w, b = fuse_conv_bn(
        sd['cnn_layers.0.weight'], sd.get('cnn_layers.0.bias'),
        sd['cnn_layers.1.weight'], sd['cnn_layers.1.bias'],
        sd['cnn_layers.1.running_mean'], sd['cnn_layers.1.running_var'])
    write_inc(w, output_dir / 'conv1_weight.inc', 'conv1 (fused with bn)')
    write_inc(b, output_dir / 'conv1_bias.inc', 'conv1 (fused with bn) bias')
    total += w.size + b.size

    # Conv2: cnn_layers.4 + cnn_layers.5
    w, b = fuse_conv_bn(
        sd['cnn_layers.4.weight'], sd.get('cnn_layers.4.bias'),
        sd['cnn_layers.5.weight'], sd['cnn_layers.5.bias'],
        sd['cnn_layers.5.running_mean'], sd['cnn_layers.5.running_var'])
    write_inc(w, output_dir / 'conv2_weight.inc', 'conv2 (fused with bn)')
    write_inc(b, output_dir / 'conv2_bias.inc', 'conv2 (fused with bn) bias')
    total += w.size + b.size

    # Conv3: cnn_layers.8 + cnn_layers.9
    w, b = fuse_conv_bn(
        sd['cnn_layers.8.weight'], sd.get('cnn_layers.8.bias'),
        sd['cnn_layers.9.weight'], sd['cnn_layers.9.bias'],
        sd['cnn_layers.9.running_mean'], sd['cnn_layers.9.running_var'])
    write_inc(w, output_dir / 'conv3_weight.inc', 'conv3 (fused with bn)')
    write_inc(b, output_dir / 'conv3_bias.inc', 'conv3 (fused with bn) bias')
    total += w.size + b.size

    # Projection: channel_projection
    w = sd['channel_projection.weight'].numpy()
    b = sd['channel_projection.bias'].numpy()
    write_inc(w, output_dir / 'projection_weight.inc', 'channel_projection')
    write_inc(b, output_dir / 'projection_bias.inc', 'channel_projection bias')
    total += w.size + b.size

    # Activity head: indices 0 and 3
    w = sd['task_heads.activity.0.weight'].numpy()
    b = sd['task_heads.activity.0.bias'].numpy()
    write_inc(w, output_dir / 'activity_head_fc1_weight.inc', 'activity_head_fc1')
    write_inc(b, output_dir / 'activity_head_fc1_bias.inc', 'activity_head_fc1 bias')
    total += w.size + b.size

    w = sd['task_heads.activity.3.weight'].numpy()
    b = sd['task_heads.activity.3.bias'].numpy()
    write_inc(w, output_dir / 'activity_head_fc2_weight.inc', 'activity_head_fc2')
    write_inc(b, output_dir / 'activity_head_fc2_bias.inc', 'activity_head_fc2 bias')
    total += w.size + b.size

    # Stress head
    w = sd['task_heads.stress.0.weight'].numpy()
    b = sd['task_heads.stress.0.bias'].numpy()
    write_inc(w, output_dir / 'stress_head_fc1_weight.inc', 'stress_head_fc1')
    write_inc(b, output_dir / 'stress_head_fc1_bias.inc', 'stress_head_fc1 bias')
    total += w.size + b.size

    w = sd['task_heads.stress.3.weight'].numpy()
    b = sd['task_heads.stress.3.bias'].numpy()
    write_inc(w, output_dir / 'stress_head_fc2_weight.inc', 'stress_head_fc2')
    write_inc(b, output_dir / 'stress_head_fc2_bias.inc', 'stress_head_fc2 bias')
    total += w.size + b.size

    # Arrhythmia head
    w = sd['task_heads.arrhythmia.0.weight'].numpy()
    b = sd['task_heads.arrhythmia.0.bias'].numpy()
    write_inc(w, output_dir / 'arrhythmia_head_fc1_weight.inc', 'arrhythmia_head_fc1')
    write_inc(b, output_dir / 'arrhythmia_head_fc1_bias.inc', 'arrhythmia_head_fc1 bias')
    total += w.size + b.size

    w = sd['task_heads.arrhythmia.3.weight'].numpy()
    b = sd['task_heads.arrhythmia.3.bias'].numpy()
    write_inc(w, output_dir / 'arrhythmia_head_fc2_weight.inc', 'arrhythmia_head_fc2')
    write_inc(b, output_dir / 'arrhythmia_head_fc2_bias.inc', 'arrhythmia_head_fc2 bias')
    total += w.size + b.size

    print(f"\n  Total params exported: {total:,}")
    print(f"  Model size: {total * 4 / 1024:.1f} KB (float32)")
    print(f"\n  Weights exported to: {output_dir}")

    # Print firmware config changes needed
    conv1_shape = list(sd['cnn_layers.0.weight'].shape)
    conv2_shape = list(sd['cnn_layers.4.weight'].shape)
    conv3_shape = list(sd['cnn_layers.8.weight'].shape)
    proj_shape = list(sd['channel_projection.weight'].shape)

    print(f"\n  === FIRMWARE CONFIG UPDATES NEEDED ===")
    print(f"  INPUT_CHANNELS = {n_channels}")
    print(f"  CONV1: [{conv1_shape[0]}, {conv1_shape[1]}, {conv1_shape[2]}]")
    print(f"  CONV2: [{conv2_shape[0]}, {conv2_shape[1]}, {conv2_shape[2]}]")
    print(f"  CONV3: [{conv3_shape[0]}, {conv3_shape[1]}, {conv3_shape[2]}]")
    print(f"  PROJ: [{proj_shape[0]}, {proj_shape[1]}]")
    print(f"  ACT_FC2_OUT = {activity_classes}")
    print(f"  STRESS_FC2_OUT = {stress_classes}")
    print(f"  ARR_FC2_OUT = 2")


def main():
    parser = argparse.ArgumentParser(description='Train CNNTransformerLite for ESP32')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    train_lite_model(args)


if __name__ == '__main__':
    main()
