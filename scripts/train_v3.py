#!/usr/bin/env python3
"""
Training V3 — Addresses root causes of low F1-macro scores.

Key fixes over V2:
1. SMOTE-like synthetic oversampling (not exact copies)
2. Separate per-task evaluation with proper class handling
3. Activity class merging (8 -> 4 classes) to handle extreme sparsity
4. Stronger augmentation with Mixup
5. Post-training threshold optimization for binary tasks
6. Better early stopping on per-task F1-macro (not combined accuracy)
7. Class-balanced focal loss with higher gamma for extreme imbalance
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_fscore_support
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Activity class merging: 8 -> 4
# =============================================================================
# Original: 0=sitting, 1=walking, 2=cycling, 3=driving, 4=working, 5=stairs, 6=table_soccer, 7=lunch
# Merged:   0=sedentary(sit/drive/work/lunch), 1=walking, 2=cycling, 3=high_intensity(stairs/table_soccer)
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
# Dataset
# =============================================================================

class BiomedicalDatasetV3(Dataset):
    """
    V3 dataset with:
    - Activity class merging (8->4)
    - SMOTE-like synthetic oversampling
    - Mixup support
    - Per-channel z-normalization
    """

    def __init__(
        self,
        data_path: str,
        subject_filter: Optional[List[str]] = None,
        binary_stress: bool = True,
        augment: bool = False,
        normalize: bool = True,
        oversample: bool = False,
        merge_activity: bool = True,
    ):
        print(f"Loading dataset from {data_path}...")
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
        self.subject_ids = []

        # First pass: compute normalization stats
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
                self.channel_means = None
                self.channel_stds = None

        # Second pass: load samples
        for sample in raw_data:
            subject_id = sample.get('subject_id', 'unknown')
            if subject_filter is not None and subject_id not in subject_filter:
                continue

            window_data = sample.get('window_data')
            labels = sample.get('labels', {})

            if window_data is None:
                continue

            window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)

            if self.n_channels is None:
                self.n_channels = window_data.shape[0]

            # Extract labels
            activity = self._decode_label(labels.get('activity'))
            stress = self._decode_label(labels.get('stress'))
            arrhythmia = self._decode_label(labels.get('arrhythmia'))

            # Merge activity classes
            if self.merge_activity and activity >= 0:
                activity = ACTIVITY_MERGE_MAP.get(activity, -1)

            # Binary stress
            if self.binary_stress and stress >= 0:
                stress = 1 if stress == 1 else 0

            self.samples.append(window_data.astype(np.float32))
            self.activity_labels.append(activity)
            self.stress_labels.append(stress)
            self.arrhythmia_labels.append(arrhythmia)
            self.subject_ids.append(subject_id)

        # SMOTE-like synthetic oversampling
        if oversample:
            self._synthetic_oversample()

        print(f"Loaded {len(self.samples)} samples, {self.n_channels} channels")
        self._print_label_stats()

    def _decode_label(self, label_data) -> int:
        if label_data is None:
            return -1
        if isinstance(label_data, np.ndarray):
            if label_data.sum() > 0:
                return int(np.argmax(label_data))
            return -1
        if isinstance(label_data, (int, float)):
            return int(label_data)
        return -1

    def _synthetic_oversample(self):
        """SMOTE-like oversampling with capped target to avoid extreme duplication."""
        n_activity = N_MERGED_ACTIVITY if self.merge_activity else 8

        # Oversample activity classes — cap at median count (not max)
        valid_act = [(i, self.activity_labels[i]) for i in range(len(self.samples)) if self.activity_labels[i] >= 0]
        if valid_act:
            act_counts = Counter(a for _, a in valid_act)
            sorted_counts = sorted(act_counts.values())
            # Target = geometric mean of median and max, capped at 5x minority
            median_count = sorted_counts[len(sorted_counts) // 2]
            max_count = sorted_counts[-1]
            target_count = min(int(np.sqrt(median_count * max_count)), max_count // 2)

            for cls in range(n_activity):
                cls_indices = [i for i, a in valid_act if a == cls]
                if len(cls_indices) < 2:
                    continue
                need = min(target_count - len(cls_indices), len(cls_indices) * 10)  # Max 10x oversample
                if need <= 0:
                    continue

                print(f"    Activity class {cls}: {len(cls_indices)} -> {len(cls_indices)+need} (target={target_count})")
                for _ in range(need):
                    i1, i2 = random.sample(cls_indices, 2)
                    lam = random.uniform(0.3, 0.7)
                    synthetic = self.samples[i1] * lam + self.samples[i2] * (1 - lam)
                    synthetic += np.random.randn(*synthetic.shape).astype(np.float32) * 0.02

                    self.samples.append(synthetic)
                    self.activity_labels.append(cls)
                    self.stress_labels.append(self.stress_labels[i1])
                    self.arrhythmia_labels.append(self.arrhythmia_labels[i1])
                    self.subject_ids.append(self.subject_ids[i1])

        # Oversample arrhythmia minority (class 1)
        arr_valid = [(i, self.arrhythmia_labels[i]) for i in range(len(self.samples)) if self.arrhythmia_labels[i] >= 0]
        if arr_valid:
            arr_counts = Counter(a for _, a in arr_valid)
            if 1 in arr_counts and 0 in arr_counts:
                # Oversample minority to 40% of majority (not 100% — avoid overfitting)
                target = int(arr_counts[0] * 0.4)
                cls1_indices = [i for i, a in arr_valid if a == 1]
                need = target - len(cls1_indices)
                if need > 0 and len(cls1_indices) >= 2:
                    for _ in range(need):
                        i1, i2 = random.sample(cls1_indices, 2)
                        lam = random.uniform(0.3, 0.7)
                        synthetic = self.samples[i1] * lam + self.samples[i2] * (1 - lam)
                        synthetic += np.random.randn(*synthetic.shape).astype(np.float32) * 0.02
                        self.samples.append(synthetic)
                        self.activity_labels.append(self.activity_labels[i1])
                        self.stress_labels.append(self.stress_labels[i1])
                        self.arrhythmia_labels.append(1)
                        self.subject_ids.append(self.subject_ids[i1])

        # Oversample stress minority (class 1) — more aggressively
        str_valid = [(i, self.stress_labels[i]) for i in range(len(self.samples)) if self.stress_labels[i] >= 0]
        if str_valid:
            str_counts = Counter(a for _, a in str_valid)
            if 1 in str_counts and 0 in str_counts:
                target = int(str_counts[0] * 0.8)
                cls1_indices = [i for i, a in str_valid if a == 1]
                need = target - len(cls1_indices)
                if need > 0 and len(cls1_indices) >= 2:
                    for _ in range(need):
                        i1, i2 = random.sample(cls1_indices, 2)
                        lam = random.uniform(0.3, 0.7)
                        synthetic = self.samples[i1] * lam + self.samples[i2] * (1 - lam)
                        synthetic += np.random.randn(*synthetic.shape).astype(np.float32) * 0.02
                        self.samples.append(synthetic)
                        self.activity_labels.append(self.activity_labels[i1])
                        self.stress_labels.append(1)
                        self.arrhythmia_labels.append(self.arrhythmia_labels[i1])
                        self.subject_ids.append(self.subject_ids[i1])

        print(f"  After synthetic oversampling: {len(self.samples)} samples")

    def _print_label_stats(self):
        act_valid = [a for a in self.activity_labels if a >= 0]
        str_valid = [s for s in self.stress_labels if s >= 0]
        arr_valid = [a for a in self.arrhythmia_labels if a >= 0]
        act_names = MERGED_ACTIVITY_NAMES if self.merge_activity else {}
        act_dist = dict(Counter(act_valid))
        if act_names:
            act_dist_named = {act_names.get(k, k): v for k, v in sorted(act_dist.items())}
        else:
            act_dist_named = act_dist
        print(f"  Activity ({len(act_valid)} labeled): {act_dist_named}")
        print(f"  Stress   ({len(str_valid)} labeled): {dict(Counter(str_valid))}")
        print(f"  Arrhythmia ({len(arr_valid)} labeled): {dict(Counter(arr_valid))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].copy()

        # Per-channel z-score normalization
        if self.normalize and self.channel_means is not None:
            x = (x - self.channel_means) / self.channel_stds

        x = torch.from_numpy(x)

        # Augmentation
        if self.augment:
            x = self._augment(x)

        return {
            'window_data': x,
            'activity': self.activity_labels[idx],
            'stress': self.stress_labels[idx],
            'arrhythmia': self.arrhythmia_labels[idx],
        }

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced augmentation."""
        # Gaussian noise
        if random.random() < 0.5:
            x = x + torch.randn_like(x) * 0.05

        # Random time shift
        if random.random() < 0.3:
            shift = random.randint(-50, 50)
            x = torch.roll(x, shift, dims=-1)

        # Random amplitude scaling per channel
        if random.random() < 0.4:
            for ch in range(x.size(0)):
                scale = random.uniform(0.8, 1.2)
                x[ch] *= scale

        # Time warping (simple stretch/compress)
        if random.random() < 0.2:
            T = x.size(-1)
            indices = torch.linspace(0, T-1, T)
            warp = random.uniform(0.9, 1.1)
            warped = torch.linspace(0, (T-1)*warp, T).clamp(0, T-1).long()
            x = x[:, warped]

        # Time masking
        if random.random() < 0.3:
            mask_len = random.randint(20, 100)
            start = random.randint(0, x.size(-1) - mask_len)
            x[..., start:start + mask_len] = 0

        # Channel dropout
        if random.random() < 0.15:
            ch = random.randint(0, x.size(0) - 1)
            x[ch] = 0

        return x

    def get_sample_weights(self) -> torch.Tensor:
        """Multi-task aware sample weighting."""
        weights = torch.ones(len(self.samples))

        n_act = N_MERGED_ACTIVITY if self.merge_activity else 8

        # Weight by all tasks combined
        act_counts = Counter(a for a in self.activity_labels if a >= 0)
        str_counts = Counter(s for s in self.stress_labels if s >= 0)
        arr_counts = Counter(a for a in self.arrhythmia_labels if a >= 0)

        for i in range(len(self.samples)):
            w = 1.0
            a = self.activity_labels[i]
            s = self.stress_labels[i]
            ar = self.arrhythmia_labels[i]

            if a >= 0 and a in act_counts:
                max_act = max(act_counts.values())
                w = max(w, max_act / act_counts[a])

            if s >= 0 and s in str_counts:
                max_str = max(str_counts.values())
                w = max(w, max_str / str_counts[s])

            if ar >= 0 and ar in arr_counts:
                max_arr = max(arr_counts.values())
                w = max(w, max_arr / arr_counts[ar])

            weights[i] = w

        return weights

    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """Compute class weights: sqrt-smoothed for activity/arrhythmia,
        direct inverse-frequency for stress (stronger minority emphasis)."""
        result = {}
        n_act = N_MERGED_ACTIVITY if self.merge_activity else 8
        stress_n = 2 if self.binary_stress else 4

        for task, n_cls, labels in [
            ('activity', n_act, self.activity_labels),
            ('stress', stress_n, self.stress_labels),
            ('arrhythmia', 2, self.arrhythmia_labels),
        ]:
            counts = np.zeros(n_cls)
            for lbl in labels:
                if 0 <= lbl < n_cls:
                    counts[lbl] += 1
            total = counts.sum()
            if total > 0:
                # Sqrt-smoothed inverse frequency for all tasks (uniform treatment)
                w = np.sqrt(total / (n_cls * np.maximum(counts, 1)))
                w = np.clip(w, 0.5, 5.0)
                result[task] = torch.FloatTensor(w)
            else:
                result[task] = torch.ones(n_cls)

        return result


# =============================================================================
# Model — Same as V2 but with configurable activity classes
# =============================================================================

class CNNTransformerV3(nn.Module):
    def __init__(
        self,
        n_channels: int = 5,
        n_samples: int = 1000,
        activity_classes: int = 4,  # Merged
        stress_classes: int = 2,
        arrhythmia_classes: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        use_transformer: bool = True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.d_model = d_model
        self.use_transformer = use_transformer and num_layers > 0

        # Input batch norm
        self.input_bn = nn.BatchNorm1d(n_channels)

        # CNN backbone
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.cnn_dropout = nn.Dropout(dropout)

        # Squeeze-and-Excitation
        self.se_fc1 = nn.Linear(64, 16)
        self.se_fc2 = nn.Linear(16, 64)

        # Projection
        self.channel_projection = nn.Linear(64, d_model)
        self.proj_ln = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)

        # Transformer
        if self.use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        else:
            self.transformer = None

        # Task heads — wider hidden for activity
        self.task_heads = nn.ModuleDict({
            "activity": self._make_head(d_model, 64, activity_classes, dropout),
            "stress": self._make_head(d_model, 48, stress_classes, dropout),
            "arrhythmia": self._make_head(d_model, 48, arrhythmia_classes, dropout),
        })

        self._init_weights()

    def _make_head(self, in_dim, hidden, n_classes, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, n_classes),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _se_attention(self, x):
        w = x.mean(dim=-1)
        w = F.relu(self.se_fc1(w))
        w = torch.sigmoid(self.se_fc2(w))
        return x * w.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_bn(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.cnn_dropout(x)
        x = self._se_attention(x)

        if self.use_transformer and self.transformer is not None:
            seq = x.transpose(1, 2)
            seq = self.channel_projection(seq)
            seq = self.proj_ln(seq)
            seq = self.proj_drop(seq)

            pe = self._pos_encoding(seq.shape[1], self.d_model, seq.device)
            seq = seq + pe

            enc = self.transformer(seq)
            z = enc.mean(dim=1)
        else:
            pooled = x.mean(dim=-1)
            z = self.channel_projection(pooled)
            z = self.proj_ln(z)
            z = self.proj_drop(z)

        return {
            "activity": self.task_heads["activity"](z),
            "stress": self.task_heads["stress"](z),
            "arrhythmia": self.task_heads["arrhythmia"](z),
        }

    @staticmethod
    def _pos_encoding(seq_len, d_model, device):
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


# =============================================================================
# Loss — Class-Balanced Focal Loss
# =============================================================================

class ClassBalancedFocalLoss(nn.Module):
    """Focal loss with class-balanced weighting (Cui et al. 2019)."""
    def __init__(self, gamma=2.5, weight=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        valid = target != self.ignore_index
        if not valid.any():
            return torch.tensor(0.0, device=input.device, requires_grad=True)

        input = input[valid]
        target = target[valid]

        ce = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class MultiTaskLossV3(nn.Module):
    def __init__(self, class_weights, task_weights=None, focal_gamma=2.5,
                 stress_focal_gamma=None, use_focal=True):
        super().__init__()
        if task_weights is None:
            task_weights = {'activity': 1.0, 'stress': 1.5, 'arrhythmia': 2.0}
        self.task_weights = task_weights

        # Per-task gamma: lower gamma for stress preserves gradient from minority class
        task_gammas = {
            'activity': focal_gamma,
            'stress': stress_focal_gamma if stress_focal_gamma is not None else focal_gamma,
            'arrhythmia': focal_gamma,
        }

        if use_focal:
            self.losses = nn.ModuleDict({
                task: ClassBalancedFocalLoss(
                    gamma=task_gammas[task],
                    weight=class_weights.get(task),
                    ignore_index=-1
                )
                for task in ['activity', 'stress', 'arrhythmia']
            })
        else:
            self.losses = nn.ModuleDict({
                task: nn.CrossEntropyLoss(weight=class_weights.get(task), ignore_index=-1)
                for task in ['activity', 'stress', 'arrhythmia']
            })

    def forward(self, outputs, labels):
        task_losses = {}
        total = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        for task in ['activity', 'stress', 'arrhythmia']:
            loss = self.losses[task](outputs[task], labels[task])
            task_losses[task] = loss
            total = total + self.task_weights[task] * loss
        return total, task_losses


# =============================================================================
# Training with Mixup
# =============================================================================

def mixup_batch(x, labels, alpha=0.2):
    """Mixup augmentation for time series."""
    if alpha <= 0:
        return x, labels

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    # For labels, we'll return both sets and handle in loss
    mixed_labels = {}
    for task in ['activity', 'stress', 'arrhythmia']:
        mixed_labels[task] = (labels[task], labels[task][index], lam)

    return mixed_x, mixed_labels


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None,
                max_grad_norm=1.0, use_mixup=True, mixup_alpha=0.2):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        x = batch['window_data'].float().to(device)
        labels = {t: batch[t].long().to(device) for t in ['activity', 'stress', 'arrhythmia']}

        optimizer.zero_grad()

        if use_mixup and random.random() < 0.5:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(x.size(0), device=device)
            mixed_x = lam * x + (1 - lam) * x[index]

            outputs = model(mixed_x)

            loss1, _ = criterion(outputs, labels)
            labels2 = {t: labels[t][index] for t in ['activity', 'stress', 'arrhythmia']}
            loss2, _ = criterion(outputs, labels2)
            loss = lam * loss1 + (1 - lam) * loss2
        else:
            outputs = model(x)
            loss, _ = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return {'loss': total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_labels = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_probs = {'activity': [], 'stress': [], 'arrhythmia': []}
    n_batches = 0

    for batch in loader:
        x = batch['window_data'].float().to(device)
        labels = {t: batch[t].long().to(device) for t in ['activity', 'stress', 'arrhythmia']}

        outputs = model(x)
        loss, _ = criterion(outputs, labels)
        total_loss += loss.item()
        n_batches += 1

        for task in ['activity', 'stress', 'arrhythmia']:
            valid = labels[task] >= 0
            if valid.sum() > 0:
                probs = F.softmax(outputs[task][valid], dim=1)
                preds = probs.argmax(dim=1)
                all_preds[task].extend(preds.cpu().numpy())
                all_labels[task].extend(labels[task][valid].cpu().numpy())
                all_probs[task].append(probs.cpu().numpy())

    metrics = {'loss': total_loss / max(n_batches, 1)}

    for task in ['activity', 'stress', 'arrhythmia']:
        if all_labels[task]:
            y_true = np.array(all_labels[task])
            y_pred = np.array(all_preds[task])
            y_probs = np.concatenate(all_probs[task], axis=0)

            acc = (y_true == y_pred).mean()
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            metrics[f'{task}_acc'] = float(acc)
            metrics[f'{task}_f1_macro'] = float(f1_macro)
            metrics[f'{task}_f1_weighted'] = float(f1_weighted)

            n_classes = y_probs.shape[1]
            try:
                if n_classes == 2:
                    auc = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
                metrics[f'{task}_auc'] = float(auc)
            except:
                pass

            metrics[f'{task}_report'] = classification_report(
                y_true, y_pred, zero_division=0, output_dict=True
            )
        else:
            metrics[f'{task}_acc'] = 0.0
            metrics[f'{task}_f1_macro'] = 0.0

    return metrics


def optimize_thresholds(model, loader, device):
    """Find optimal decision thresholds for binary tasks."""
    model.eval()
    all_probs = {'stress': [], 'arrhythmia': []}
    all_labels = {'stress': [], 'arrhythmia': []}

    with torch.no_grad():
        for batch in loader:
            x = batch['window_data'].float().to(device)
            labels = {t: batch[t].long().to(device) for t in ['stress', 'arrhythmia']}
            outputs = model(x)

            for task in ['stress', 'arrhythmia']:
                valid = labels[task] >= 0
                if valid.sum() > 0:
                    probs = F.softmax(outputs[task][valid], dim=1)[:, 1]
                    all_probs[task].extend(probs.cpu().numpy())
                    all_labels[task].extend(labels[task][valid].cpu().numpy())

    thresholds = {}
    for task in ['stress', 'arrhythmia']:
        if not all_labels[task]:
            thresholds[task] = 0.5
            continue

        y_true = np.array(all_labels[task])
        y_probs = np.array(all_probs[task])

        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.02):
            y_pred = (y_probs >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        thresholds[task] = float(best_thresh)
        print(f"  {task}: optimal threshold = {best_thresh:.2f} (F1-macro = {best_f1:.3f})")

    return thresholds


@torch.no_grad()
def evaluate_with_thresholds(model, loader, criterion, device, thresholds):
    """Evaluate using optimized thresholds for binary tasks."""
    model.eval()
    total_loss = 0
    all_preds = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_labels = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_probs = {'activity': [], 'stress': [], 'arrhythmia': []}
    n_batches = 0

    for batch in loader:
        x = batch['window_data'].float().to(device)
        labels = {t: batch[t].long().to(device) for t in ['activity', 'stress', 'arrhythmia']}

        outputs = model(x)
        loss, _ = criterion(outputs, labels)
        total_loss += loss.item()
        n_batches += 1

        for task in ['activity', 'stress', 'arrhythmia']:
            valid = labels[task] >= 0
            if valid.sum() > 0:
                probs = F.softmax(outputs[task][valid], dim=1)
                all_probs[task].append(probs.cpu().numpy())
                all_labels[task].extend(labels[task][valid].cpu().numpy())

                if task in thresholds and probs.shape[1] == 2:
                    # Use optimized threshold
                    preds = (probs[:, 1] >= thresholds[task]).long()
                else:
                    preds = probs.argmax(dim=1)
                all_preds[task].extend(preds.cpu().numpy())

    metrics = {'loss': total_loss / max(n_batches, 1)}

    for task in ['activity', 'stress', 'arrhythmia']:
        if all_labels[task]:
            y_true = np.array(all_labels[task])
            y_pred = np.array(all_preds[task])
            y_probs = np.concatenate(all_probs[task], axis=0)

            acc = (y_true == y_pred).mean()
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            metrics[f'{task}_acc'] = float(acc)
            metrics[f'{task}_f1_macro'] = float(f1_macro)
            metrics[f'{task}_f1_weighted'] = float(f1_weighted)

            n_classes = y_probs.shape[1]
            try:
                if n_classes == 2:
                    auc = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
                metrics[f'{task}_auc'] = float(auc)
            except:
                pass

            metrics[f'{task}_report'] = classification_report(
                y_true, y_pred, zero_division=0, output_dict=True
            )
            metrics[f'{task}_confusion'] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Find dataset
    dataset_path = args.dataset_path
    if not dataset_path:
        candidates = [
            PROJECT_ROOT.parent / 'processed_unified_dataset' / 'unified_dataset.pkl',
            Path('/Users/HP/Desktop/University/Thesis/Code/multimodal-biomedical-monitoring-improved/processed_unified_dataset/unified_dataset.pkl'),
            PROJECT_ROOT / 'processed_unified_dataset' / 'unified_dataset.pkl',
        ]
        for p in candidates:
            if p.exists():
                dataset_path = str(p)
                break

    if not dataset_path or not Path(dataset_path).exists():
        print("ERROR: Cannot find dataset. Use --dataset_path")
        sys.exit(1)

    print(f"Dataset: {dataset_path}")

    # Load splits — prefer v3 splits (includes UCI HAR subjects)
    splits_path = PROJECT_ROOT / 'training_results' / 'subject_splits_v3.json'
    if not splits_path.exists():
        splits_path = PROJECT_ROOT / 'training_results' / 'subject_splits.json'
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        train_subjects = splits['train_subjects']
        val_subjects = splits['val_subjects']
        test_subjects = splits['test_subjects']
        print(f"Using splits: {splits_path.name}")
        print(f"Subjects — Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")
    else:
        print("ERROR: No subject_splits.json found")
        sys.exit(1)

    binary_stress = args.binary_stress
    stress_classes = 2 if binary_stress else 4
    activity_classes = N_MERGED_ACTIVITY if args.merge_activity else 8

    # Create datasets
    train_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=train_subjects,
        binary_stress=binary_stress, augment=True, normalize=True,
        oversample=True, merge_activity=args.merge_activity,
    )
    val_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=val_subjects,
        binary_stress=binary_stress, augment=False, normalize=True,
        oversample=False, merge_activity=args.merge_activity,
    )
    test_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=test_subjects,
        binary_stress=binary_stress, augment=False, normalize=True,
        oversample=False, merge_activity=args.merge_activity,
    )

    n_channels = train_ds.n_channels
    print(f"\nChannels: {n_channels}, Activity classes: {activity_classes}")

    # Weighted sampler
    if args.use_weighted_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = CNNTransformerV3(
        n_channels=n_channels,
        n_samples=1000,
        activity_classes=activity_classes,
        stress_classes=stress_classes,
        arrhythmia_classes=2,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        use_transformer=args.use_transformer,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,} ({n_params * 4 / 1024:.1f} KB)")

    # Class weights
    class_weights = train_ds.get_class_weights()
    for task, w in class_weights.items():
        class_weights[task] = w.to(device)
        print(f"  {task} class weights: {w.numpy().round(3)}")

    # Loss
    criterion = MultiTaskLossV3(
        class_weights=class_weights,
        task_weights={'activity': args.w_activity, 'stress': args.w_stress, 'arrhythmia': args.w_arrhythmia},
        focal_gamma=args.focal_gamma,
        stress_focal_gamma=args.stress_focal_gamma,
        use_focal=args.use_focal,
    )

    # Optimizer with discriminative learning rates
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # OneCycleLR for better convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos',
    )

    # Training loop
    best_val_score = -1
    best_state = None
    patience_counter = 0
    history = []

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training V3 for {args.epochs} epochs")
    print(f"{'='*70}")

    for epoch in range(args.epochs):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler,
            use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
        )

        val_metrics = evaluate(model, val_loader, criterion, device)

        # Combined F1-macro score for early stopping
        val_score = (
            val_metrics.get('activity_f1_macro', 0) * 0.33 +
            val_metrics.get('stress_f1_macro', 0) * 0.34 +
            val_metrics.get('arrhythmia_f1_macro', 0) * 0.33
        )

        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val — Act: {val_metrics.get('activity_acc', 0)*100:.1f}% (F1m: {val_metrics.get('activity_f1_macro', 0)*100:.1f}%) | "
                  f"Str: {val_metrics.get('stress_acc', 0)*100:.1f}% (F1m: {val_metrics.get('stress_f1_macro', 0)*100:.1f}%) | "
                  f"Arr: {val_metrics.get('arrhythmia_acc', 0)*100:.1f}% (F1m: {val_metrics.get('arrhythmia_f1_macro', 0)*100:.1f}%)")
            print(f"  Val Combined F1-macro: {val_score*100:.1f}%")

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  ★ New best! {val_score*100:.1f}%")
        else:
            patience_counter += 1

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_activity_f1': val_metrics.get('activity_f1_macro', 0),
            'val_stress_f1': val_metrics.get('stress_f1_macro', 0),
            'val_arrhythmia_f1': val_metrics.get('arrhythmia_f1_macro', 0),
            'val_combined_f1': val_score,
        })

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Phase 2 stress fine-tuning disabled — empirically shown to overfit
    # and degrade stress accuracy (68.6% → 61.1%) on small stress dataset

    # Threshold optimization on validation set
    print(f"\n{'='*70}")
    print("THRESHOLD OPTIMIZATION (on validation set)")
    print(f"{'='*70}")
    thresholds = optimize_thresholds(model, val_loader, device)

    # Final test evaluation
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION")
    print(f"{'='*70}")

    # Without threshold optimization
    test_metrics_default = evaluate(model, test_loader, criterion, device)

    # With threshold optimization
    test_metrics = evaluate_with_thresholds(model, test_loader, criterion, device, thresholds)

    print(f"\n{'Task':<15} {'Acc':>8} {'F1-macro':>10} {'F1-wt':>10} {'AUC':>8}")
    print(f"{'-'*55}")
    for task in ['activity', 'stress', 'arrhythmia']:
        acc = test_metrics.get(f'{task}_acc', 0)
        f1m = test_metrics.get(f'{task}_f1_macro', 0)
        f1w = test_metrics.get(f'{task}_f1_weighted', 0)
        auc = test_metrics.get(f'{task}_auc', 0)
        print(f"  {task:<13} {acc*100:>7.1f}% {f1m*100:>9.1f}% {f1w*100:>9.1f}% {auc:>7.3f}")

    # Per-class reports
    for task in ['activity', 'stress', 'arrhythmia']:
        report = test_metrics.get(f'{task}_report')
        if report:
            print(f"\n  {task.upper()} per-class:")
            for cls, vals in report.items():
                if cls in ('accuracy', 'macro avg', 'weighted avg'):
                    continue
                if isinstance(vals, dict):
                    cls_name = MERGED_ACTIVITY_NAMES.get(int(cls), cls) if task == 'activity' and args.merge_activity else cls
                    print(f"    {cls_name}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1-score']:.3f} N={int(vals['support'])}")

    # Confusion matrices
    for task in ['stress', 'arrhythmia']:
        cm = test_metrics.get(f'{task}_confusion')
        if cm:
            print(f"\n  {task.upper()} Confusion Matrix:")
            for row in cm:
                print(f"    {row}")

    # Comparison: default vs optimized thresholds
    print(f"\n{'='*70}")
    print("DEFAULT vs OPTIMIZED THRESHOLDS")
    print(f"{'='*70}")
    for task in ['stress', 'arrhythmia']:
        f1_def = test_metrics_default.get(f'{task}_f1_macro', 0)
        f1_opt = test_metrics.get(f'{task}_f1_macro', 0)
        print(f"  {task}: F1-macro {f1_def*100:.1f}% -> {f1_opt*100:.1f}% (threshold={thresholds.get(task, 0.5):.2f})")

    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = output_dir / f'model_v3_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_channels': n_channels,
        'activity_classes': activity_classes,
        'stress_classes': stress_classes,
        'n_params': n_params,
        'thresholds': thresholds,
        'merge_activity': args.merge_activity,
        'test_metrics': {k: v for k, v in test_metrics.items() if not k.endswith('_report')},
        'best_val_score': best_val_score,
    }, model_path)
    print(f"\nModel saved: {model_path}")

    torch.save(model.state_dict(), output_dir / 'model_v3.pth')

    # Save full results
    results = {
        'timestamp': timestamp,
        'args': vars(args),
        'n_channels': n_channels,
        'activity_classes': activity_classes,
        'n_params': n_params,
        'best_val_score': best_val_score,
        'thresholds': thresholds,
        'test_metrics': {k: v for k, v in test_metrics.items() if not k.endswith('_report')},
        'test_metrics_default_threshold': {k: v for k, v in test_metrics_default.items() if not k.endswith('_report')},
        'history': history,
    }
    for task in ['activity', 'stress', 'arrhythmia']:
        for metrics_dict, suffix in [(test_metrics, ''), (test_metrics_default, '_default')]:
            report = metrics_dict.get(f'{task}_report')
            if report:
                results[f'{task}_report{suffix}'] = report

    results_path = output_dir / f'training_v3_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Training V3 — Fixed imbalance pipeline')

    # Data
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--binary_stress', action='store_true', default=True)
    parser.add_argument('--four_class_stress', action='store_true', default=False)
    parser.add_argument('--merge_activity', action='store_true', default=True,
                        help='Merge 8 activity classes to 4 (default: True)')
    parser.add_argument('--no_merge_activity', action='store_true', default=False)
    parser.add_argument('--use_weighted_sampler', action='store_true', default=True)

    # Model
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_ff', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--use_transformer', action='store_true', default=True)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--use_mixup', action='store_true', default=True)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)

    # Loss
    parser.add_argument('--use_focal', action='store_true', default=True)
    parser.add_argument('--focal_gamma', type=float, default=2.5)
    parser.add_argument('--stress_focal_gamma', type=float, default=1.0,
                        help='Focal gamma for stress (lower preserves gradient from minority)')
    parser.add_argument('--w_activity', type=float, default=1.0)
    parser.add_argument('--w_stress', type=float, default=1.5)
    parser.add_argument('--w_arrhythmia', type=float, default=2.0)

    # Output
    parser.add_argument('--output_dir', type=str, default='training_results')

    args = parser.parse_args()

    if args.four_class_stress:
        args.binary_stress = False
    if args.no_merge_activity:
        args.merge_activity = False

    train(args)


if __name__ == '__main__':
    main()
