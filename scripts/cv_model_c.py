#!/usr/bin/env python3
"""
5-fold stratified subject-wise cross-validation for Model C architecture.
FULL training configuration matching the held-out model:
  - Phase 1: 150 epochs, oversampling, focal loss, mixup
  - Phase 2: 60 epochs, frozen backbone, fine-tune stress/arrhythmia heads
  - Batch size 64, lr 3e-4, patience 25
"""

import json
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from train_v3 import (
    CNNTransformerV3, BiomedicalDatasetV3, ClassBalancedFocalLoss,
    MultiTaskLossV3, N_MERGED_ACTIVITY, MERGED_ACTIVITY_NAMES,
    train_epoch, evaluate,
)


def get_subject_groups(dataset_path):
    """Group subjects by which tasks they have labels for."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    subj_tasks = defaultdict(lambda: set())
    for w in data:
        sid = w.get('subject_id', 'unknown')
        labels = w.get('labels', {})
        for task in ['activity', 'stress', 'arrhythmia']:
            lbl = labels.get(task, None)
            if lbl is not None:
                if isinstance(lbl, np.ndarray) and lbl.sum() > 0:
                    subj_tasks[sid].add(task)
                elif isinstance(lbl, (int, float)) and lbl >= 0:
                    subj_tasks[sid].add(task)

    groups = defaultdict(list)
    for sid, tasks in subj_tasks.items():
        key = '_'.join(sorted(tasks))
        groups[key].append(sid)

    return groups


def stratified_subject_splits(groups, n_folds=5, seed=42):
    """Create n_folds splits where each fold has subjects from each group."""
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_folds)]

    for group_name, subjects in groups.items():
        subjects = sorted(subjects)
        rng.shuffle(subjects)
        for i, s in enumerate(subjects):
            folds[i % n_folds].append(s)

    return folds


def run_fold(fold_idx, train_subjects, val_subjects, test_subjects, dataset_path, device):
    """Run one fold of CV with FULL training configuration."""
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx+1}")
    print(f"{'='*70}")
    print(f"  Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")
    fold_start = time.time()

    # --- Datasets (WITH oversampling, matching held-out training) ---
    train_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=train_subjects,
        binary_stress=True, augment=True, normalize=True,
        oversample=True, merge_activity=True,
    )
    val_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=val_subjects,
        binary_stress=True, augment=False, normalize=True,
        oversample=False, merge_activity=True,
    )
    test_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=test_subjects,
        binary_stress=True, augment=False, normalize=True,
        oversample=False, merge_activity=True,
    )

    n_channels = train_ds.n_channels
    print(f"  Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # --- Model ---
    model = CNNTransformerV3(
        n_channels=n_channels, n_samples=1000,
        activity_classes=4, stress_classes=2, arrhythmia_classes=2,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.3, use_transformer=True,
    ).to(device)

    class_weights = train_ds.get_class_weights()
    for task, w in class_weights.items():
        class_weights[task] = w.to(device)

    # === PHASE 1: Full end-to-end training (150 epochs, matching held-out) ===
    print("  Phase 1: Full training (150 epochs max, patience 25)")

    criterion = MultiTaskLossV3(
        class_weights=class_weights,
        task_weights={'activity': 1.0, 'stress': 1.5, 'arrhythmia': 2.0},
        focal_gamma=2.5, stress_focal_gamma=1.0, use_focal=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        epochs=150, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos',
    )

    best_val_score = -1
    best_state = None
    patience_counter = 0

    for epoch in range(150):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler, use_mixup=True, mixup_alpha=0.2,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Combined F1-macro score (only tasks with valid samples)
        scores = []
        for task in ['activity', 'stress', 'arrhythmia']:
            f1 = val_metrics.get(f'{task}_f1_macro', 0)
            if f1 > 0:
                scores.append(f1)
        val_score = np.mean(scores) if scores else 0

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 25:
            print(f"  Phase 1 early stop at epoch {epoch+1} (best={best_val_score*100:.1f}%)")
            break

        if (epoch + 1) % 25 == 0:
            elapsed = (time.time() - fold_start) / 60
            print(f"  Epoch {epoch+1}: val_score={val_score*100:.1f}% best={best_val_score*100:.1f}% ({elapsed:.0f}min)")

    if best_state:
        model.load_state_dict(best_state)

    # === PHASE 2: Fine-tune stress/arrhythmia heads (60 epochs) ===
    print("  Phase 2: Fine-tuning stress/arrhythmia heads (60 epochs max)")

    for name, param in model.named_parameters():
        if 'task_heads.stress' in name or 'task_heads.arrhythmia' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    ft_criterion = MultiTaskLossV3(
        class_weights=class_weights,
        task_weights={'activity': 0.0, 'stress': 2.0, 'arrhythmia': 2.0},
        focal_gamma=1.5, stress_focal_gamma=0.5, use_focal=True,
    )
    ft_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-3,
    )
    ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ft_optimizer, T_max=60, eta_min=1e-5,
    )

    best_ft_score = -1
    best_ft_state = None
    ft_patience = 0

    for epoch in range(60):
        model.train()
        for batch in train_loader:
            x = batch['window_data'].float().to(device)
            labels = {t: batch[t].long().to(device) for t in ['activity', 'stress', 'arrhythmia']}
            ft_optimizer.zero_grad()
            outputs = model(x)
            loss, _ = ft_criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            ft_optimizer.step()
        ft_scheduler.step()

        eval_crit = MultiTaskLossV3(
            class_weights=class_weights,
            task_weights={'activity': 1.0, 'stress': 1.0, 'arrhythmia': 1.0},
            focal_gamma=2.5, use_focal=True,
        )
        val_m = evaluate(model, val_loader, eval_crit, device)
        scores = []
        for task in ['stress', 'arrhythmia']:
            f1 = val_m.get(f'{task}_f1_macro', 0)
            if f1 > 0:
                scores.append(f1)
        ft_score = np.mean(scores) if scores else 0

        if ft_score > best_ft_score:
            best_ft_score = ft_score
            best_ft_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ft_patience = 0
        else:
            ft_patience += 1

        if ft_patience >= 15:
            print(f"  Phase 2 early stop at epoch {epoch+1} (best={best_ft_score*100:.1f}%)")
            break

        if (epoch + 1) % 20 == 0:
            elapsed = (time.time() - fold_start) / 60
            print(f"  FT Epoch {epoch+1}: ft_score={ft_score*100:.1f}% ({elapsed:.0f}min)")

    if best_ft_state:
        model.load_state_dict(best_ft_state)

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    # === EVALUATE TEST ===
    eval_crit = MultiTaskLossV3(
        class_weights=class_weights,
        task_weights={'activity': 1.0, 'stress': 1.0, 'arrhythmia': 1.0},
        focal_gamma=2.5, use_focal=True,
    )
    test_metrics = evaluate(model, test_loader, eval_crit, device)

    result = {}
    for task in ['activity', 'stress', 'arrhythmia']:
        result[f'{task}_acc'] = test_metrics.get(f'{task}_acc', 0)
        result[f'{task}_f1_macro'] = test_metrics.get(f'{task}_f1_macro', 0)
        result[f'{task}_f1_weighted'] = test_metrics.get(f'{task}_f1_weighted', 0)
        result[f'{task}_auc'] = test_metrics.get(f'{task}_auc', 0)

    fold_time = (time.time() - fold_start) / 60
    print(f"\n  Fold {fold_idx+1} results ({fold_time:.0f} min):")
    for task in ['activity', 'stress', 'arrhythmia']:
        acc = result[f'{task}_acc']
        f1m = result[f'{task}_f1_macro']
        f1w = result[f'{task}_f1_weighted']
        auc_v = result[f'{task}_auc']
        if acc > 0:
            print(f"    {task}: acc={acc*100:.1f}%  F1m={f1m*100:.1f}%  F1w={f1w*100:.1f}%  AUC={auc_v:.3f}")
        else:
            print(f"    {task}: no test samples")

    # Save intermediate results after each fold
    interim_path = PROJECT_ROOT / 'training_results' / f'cv_fold_{fold_idx+1}_results.json'
    with open(interim_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    dataset_path = None
    candidates = [
        PROJECT_ROOT.parent / 'processed_unified_dataset' / 'unified_dataset.pkl',
        Path('/Users/HP/Desktop/University/Thesis/Code/multimodal-biomedical-monitoring-improved/processed_unified_dataset/unified_dataset.pkl'),
    ]
    for p in candidates:
        if p.exists():
            dataset_path = str(p)
            break

    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get subject groups
    groups = get_subject_groups(dataset_path)
    print("\nSubject groups:")
    for name, sids in sorted(groups.items()):
        print(f"  {name}: {len(sids)} subjects")

    # Create stratified folds
    n_folds = 5
    folds = stratified_subject_splits(groups, n_folds=n_folds)
    for i, f in enumerate(folds):
        print(f"  Fold {i+1} test: {len(f)} subjects")

    total_start = time.time()
    fold_results = []
    for fold_idx in range(n_folds):
        test_subjects = folds[fold_idx]
        other = []
        for j in range(n_folds):
            if j != fold_idx:
                other.extend(folds[j])
        n_val = max(3, len(other) // 5)
        val_subjects = other[:n_val]
        train_subjects = other[n_val:]

        result = run_fold(fold_idx, train_subjects, val_subjects, test_subjects,
                         dataset_path, device)
        fold_results.append(result)

    total_time = (time.time() - total_start) / 3600
    # Summary
    print(f"\n{'='*70}")
    print(f"5-FOLD CV SUMMARY (mean ± std) — Total time: {total_time:.1f} hours")
    print(f"{'='*70}")

    summary = {}
    for task in ['activity', 'stress', 'arrhythmia']:
        for metric in ['acc', 'f1_macro', 'f1_weighted', 'auc']:
            key = f'{task}_{metric}'
            vals = [r[key] for r in fold_results if r[key] > 0]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                summary[f'{key}_mean'] = float(mean_val)
                summary[f'{key}_std'] = float(std_val)
                summary[f'{key}_values'] = [float(v) for v in vals]
                summary[f'{key}_n_folds'] = len(vals)

    for task in ['activity', 'stress', 'arrhythmia']:
        print(f"  {task}:")
        for metric in ['acc', 'f1_macro', 'f1_weighted', 'auc']:
            key = f'{task}_{metric}'
            if f'{key}_mean' in summary:
                m = summary[f'{key}_mean']
                s = summary[f'{key}_std']
                n = summary[f'{key}_n_folds']
                if metric == 'auc':
                    print(f"    {metric}: {m:.3f} ± {s:.3f} ({n} folds)")
                else:
                    print(f"    {metric}: {m*100:.1f}% ± {s*100:.1f}% ({n} folds)")

    results_path = PROJECT_ROOT / 'training_results' / 'cv_5fold_merged_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'n_folds': n_folds,
            'fold_results': fold_results,
            'summary': summary,
            'folds': folds,
            'total_hours': total_time,
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
