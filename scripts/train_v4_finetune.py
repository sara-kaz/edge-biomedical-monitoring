#!/usr/bin/env python3
"""
Training V4 — Two-phase fine-tuning to improve stress/arrhythmia
WITHOUT degrading activity accuracy.

Strategy:
  Phase 1: Load Model A checkpoint (94.2% activity)
  Phase 2: Freeze backbone + activity head, fine-tune ONLY stress/arrhythmia
           heads with higher LR, lower focal gamma, more aggressive weighting.
  Phase 3: Optional SWA (Stochastic Weight Averaging) for smoother generalization.

This guarantees activity accuracy is preserved (frozen layers) while giving
stress and arrhythmia heads maximal freedom to improve.
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
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_fscore_support
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import model and dataset from train_v3
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from train_v3 import (
    CNNTransformerV3, BiomedicalDatasetV3, ClassBalancedFocalLoss,
    MultiTaskLossV3, evaluate, evaluate_with_thresholds, optimize_thresholds,
    N_MERGED_ACTIVITY, MERGED_ACTIVITY_NAMES, ACTIVITY_MERGE_MAP,
)


def train_epoch_frozen(model, loader, criterion, optimizer, device,
                       frozen_tasks={'activity'}, max_grad_norm=1.0,
                       use_mixup=True, mixup_alpha=0.2):
    """Train only unfrozen task heads (stress/arrhythmia)."""
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

            # Only compute loss for unfrozen tasks
            loss1, _ = criterion(outputs, labels)
            labels2 = {t: labels[t][index] for t in ['activity', 'stress', 'arrhythmia']}
            loss2, _ = criterion(outputs, labels2)
            loss = lam * loss1 + (1 - lam) * loss2
        else:
            outputs = model(x)
            loss, _ = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_grad_norm
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {'loss': total_loss / max(n_batches, 1)}


def freeze_backbone_and_activity(model):
    """Freeze everything except stress and arrhythmia heads."""
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if 'task_heads.stress' in name or 'task_heads.arrhythmia' in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"  Frozen: {frozen_count:,} params")
    print(f"  Trainable (stress+arrhythmia heads): {trainable_count:,} params")
    return trainable_count


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def train_v4(args):
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

    # Load splits
    splits_path = PROJECT_ROOT / 'training_results' / 'subject_splits_v3.json'
    if not splits_path.exists():
        splits_path = PROJECT_ROOT / 'training_results' / 'subject_splits.json'
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        train_subjects = splits['train_subjects']
        val_subjects = splits['val_subjects']
        test_subjects = splits['test_subjects']
        print(f"Subjects — Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")
    else:
        print("ERROR: No subject_splits.json found")
        sys.exit(1)

    # Create datasets
    activity_classes = N_MERGED_ACTIVITY
    stress_classes = 2

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
    print(f"Channels: {n_channels}, Activity: {activity_classes} cls, Stress: {stress_classes} cls")

    # Weighted sampler — weight more toward stress/arrhythmia minority
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
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
        use_transformer=True,
    ).to(device)

    # Load Model A checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = str(PROJECT_ROOT / 'training_results' / 'model_v3_2026-03-12_02-37-39.pth')

    print(f"\nLoading Model A checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    # Verify Model A performance before fine-tuning
    class_weights = train_ds.get_class_weights()
    for task, w in class_weights.items():
        class_weights[task] = w.to(device)

    # Use standard loss for evaluation
    eval_criterion = MultiTaskLossV3(
        class_weights=class_weights,
        task_weights={'activity': 1.0, 'stress': 1.0, 'arrhythmia': 1.0},
        focal_gamma=2.5,
        use_focal=True,
    )

    print("\n" + "=" * 70)
    print("BASELINE (Model A) — Before fine-tuning")
    print("=" * 70)
    baseline_metrics = evaluate(model, test_loader, eval_criterion, device)
    for task in ['activity', 'stress', 'arrhythmia']:
        acc = baseline_metrics.get(f'{task}_acc', 0)
        f1m = baseline_metrics.get(f'{task}_f1_macro', 0)
        auc = baseline_metrics.get(f'{task}_auc', 0)
        print(f"  {task:<13} Acc: {acc*100:.1f}%  F1m: {f1m*100:.1f}%  AUC: {auc:.3f}")

    baseline_activity_acc = baseline_metrics.get('activity_acc', 0)

    # =========================================================================
    # PHASE 2: Fine-tune stress/arrhythmia heads (backbone + activity frozen)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Fine-tune stress/arrhythmia heads (backbone frozen)")
    print("=" * 70)

    n_trainable = freeze_backbone_and_activity(model)

    # Loss for fine-tuning — lower gamma for stress to preserve minority gradient
    ft_criterion = MultiTaskLossV3(
        class_weights=class_weights,
        task_weights={
            'activity': 0.0,  # Don't waste gradient on frozen head
            'stress': args.ft_w_stress,
            'arrhythmia': args.ft_w_arrhythmia,
        },
        focal_gamma=args.ft_focal_gamma,
        stress_focal_gamma=args.ft_stress_gamma,
        use_focal=True,
    )

    # Higher LR for fine-tuning heads
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.ft_lr,
        weight_decay=args.ft_weight_decay,
    )

    # Cosine annealing schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.ft_epochs, eta_min=args.ft_lr * 0.01
    )

    best_val_score = -1
    best_state = None
    patience_counter = 0
    history = []

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(exist_ok=True)

    for epoch in range(args.ft_epochs):
        t0 = time.time()

        train_metrics = train_epoch_frozen(
            model, train_loader, ft_criterion, optimizer, device,
            frozen_tasks={'activity'},
            use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
        )

        scheduler.step()

        val_metrics = evaluate(model, val_loader, eval_criterion, device)

        # Score: only stress + arrhythmia F1 (activity is frozen)
        val_score = (
            val_metrics.get('stress_f1_macro', 0) * 0.5 +
            val_metrics.get('arrhythmia_f1_macro', 0) * 0.5
        )

        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{args.ft_epochs} ({elapsed:.1f}s) | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val — Act: {val_metrics.get('activity_acc', 0)*100:.1f}% | "
                  f"Str: {val_metrics.get('stress_acc', 0)*100:.1f}% (F1m: {val_metrics.get('stress_f1_macro', 0)*100:.1f}%, AUC: {val_metrics.get('stress_auc', 0):.3f}) | "
                  f"Arr: {val_metrics.get('arrhythmia_acc', 0)*100:.1f}% (F1m: {val_metrics.get('arrhythmia_f1_macro', 0)*100:.1f}%, AUC: {val_metrics.get('arrhythmia_auc', 0):.3f})")
            print(f"  Val Score (stress+arr F1m): {val_score*100:.1f}%")

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
            'phase': 'finetune',
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_activity_acc': val_metrics.get('activity_acc', 0),
            'val_stress_f1': val_metrics.get('stress_f1_macro', 0),
            'val_stress_auc': val_metrics.get('stress_auc', 0),
            'val_arrhythmia_f1': val_metrics.get('arrhythmia_f1_macro', 0),
            'val_arrhythmia_auc': val_metrics.get('arrhythmia_auc', 0),
            'val_score': val_score,
        })

        if patience_counter >= args.ft_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Restore best fine-tuned model
    if best_state:
        model.load_state_dict(best_state)

    # =========================================================================
    # PHASE 3 (optional): SWA over last N epochs
    # =========================================================================
    if args.use_swa:
        print("\n" + "=" * 70)
        print("PHASE 3: Stochastic Weight Averaging (SWA)")
        print("=" * 70)

        # Unfreeze for SWA collection (but use same frozen criterion)
        # Actually keep frozen — SWA only on the trainable params
        swa_model = AveragedModel(model)
        swa_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.swa_lr,
            weight_decay=args.ft_weight_decay,
        )
        swa_scheduler = SWALR(swa_optimizer, swa_lr=args.swa_lr)

        for epoch in range(args.swa_epochs):
            t0 = time.time()
            train_metrics = train_epoch_frozen(
                model, train_loader, ft_criterion, swa_optimizer, device,
                frozen_tasks={'activity'},
                use_mixup=False,  # No mixup during SWA
            )
            swa_model.update_parameters(model)
            swa_scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                elapsed = time.time() - t0
                print(f"  SWA Epoch {epoch+1}/{args.swa_epochs} ({elapsed:.1f}s) Loss: {train_metrics['loss']:.4f}")

        # Update batch norm stats
        print("  Updating BN stats...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # Use SWA model for final evaluation
        model = swa_model.module

    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION — Model A + fine-tuned heads")
    print("=" * 70)

    # Threshold optimization
    thresholds = optimize_thresholds(model, val_loader, device)

    # Test with optimized thresholds
    test_metrics = evaluate_with_thresholds(model, test_loader, eval_criterion, device, thresholds)

    # Test with default thresholds
    test_metrics_default = evaluate(model, test_loader, eval_criterion, device)

    print(f"\n{'Task':<15} {'Acc':>8} {'F1-macro':>10} {'F1-wt':>10} {'AUC':>8}")
    print(f"{'-'*55}")
    for task in ['activity', 'stress', 'arrhythmia']:
        acc = test_metrics.get(f'{task}_acc', 0)
        f1m = test_metrics.get(f'{task}_f1_macro', 0)
        f1w = test_metrics.get(f'{task}_f1_weighted', 0)
        auc = test_metrics.get(f'{task}_auc', 0)
        print(f"  {task:<13} {acc*100:>7.1f}% {f1m*100:>9.1f}% {f1w*100:>9.1f}% {auc:>7.3f}")

    # Compare with baseline
    print(f"\n{'='*70}")
    print("COMPARISON: Baseline (Model A) vs Fine-tuned")
    print(f"{'='*70}")
    print(f"{'Task':<15} {'Baseline Acc':>12} {'FT Acc':>10} {'Δ':>6} | {'Base AUC':>10} {'FT AUC':>10} {'Δ':>6}")
    for task in ['activity', 'stress', 'arrhythmia']:
        b_acc = baseline_metrics.get(f'{task}_acc', 0)
        f_acc = test_metrics_default.get(f'{task}_acc', 0)
        b_auc = baseline_metrics.get(f'{task}_auc', 0)
        f_auc = test_metrics_default.get(f'{task}_auc', 0)
        print(f"  {task:<13} {b_acc*100:>11.1f}% {f_acc*100:>9.1f}% {(f_acc-b_acc)*100:>+5.1f} | "
              f"{b_auc:>9.3f} {f_auc:>9.3f} {(f_auc-b_auc):>+6.3f}")

    # Verify activity didn't degrade
    ft_activity_acc = test_metrics_default.get('activity_acc', 0)
    if ft_activity_acc < baseline_activity_acc - 0.005:
        print(f"\n⚠️  WARNING: Activity accuracy degraded! {baseline_activity_acc*100:.1f}% → {ft_activity_acc*100:.1f}%")
        print("  This should not happen with frozen backbone+activity. Check implementation.")
    else:
        print(f"\n✅ Activity accuracy preserved: {baseline_activity_acc*100:.1f}% → {ft_activity_acc*100:.1f}%")

    # Per-class reports
    for task in ['stress', 'arrhythmia']:
        report = test_metrics_default.get(f'{task}_report')
        if report:
            print(f"\n  {task.upper()} per-class (default threshold):")
            for cls, vals in report.items():
                if cls in ('accuracy', 'macro avg', 'weighted avg'):
                    continue
                if isinstance(vals, dict):
                    print(f"    Class {cls}: P={vals['precision']:.3f} R={vals['recall']:.3f} "
                          f"F1={vals['f1-score']:.3f} N={int(vals['support'])}")

    # Confusion matrices
    for task in ['stress', 'arrhythmia']:
        cm = test_metrics.get(f'{task}_confusion')
        if cm:
            print(f"\n  {task.upper()} Confusion Matrix (optimized threshold):")
            for row in cm:
                print(f"    {row}")

    # Save model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = output_dir / f'model_v4_finetuned_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_channels': n_channels,
        'activity_classes': activity_classes,
        'stress_classes': stress_classes,
        'n_params': sum(p.numel() for p in model.parameters()),
        'thresholds': thresholds,
        'merge_activity': True,
        'test_metrics': {k: v for k, v in test_metrics.items() if not k.endswith('_report')},
        'baseline_checkpoint': checkpoint_path,
        'best_val_score': best_val_score,
    }, model_path)
    print(f"\nModel saved: {model_path}")

    # Save results
    results = {
        'timestamp': timestamp,
        'strategy': 'two_phase_finetune',
        'baseline_checkpoint': checkpoint_path,
        'args': vars(args),
        'n_channels': n_channels,
        'activity_classes': activity_classes,
        'n_params': sum(p.numel() for p in model.parameters()),
        'best_val_score': best_val_score,
        'thresholds': thresholds,
        'baseline_metrics': {k: v for k, v in baseline_metrics.items() if not k.endswith('_report')},
        'test_metrics': {k: v for k, v in test_metrics.items() if not k.endswith('_report')},
        'test_metrics_default_threshold': {k: v for k, v in test_metrics_default.items() if not k.endswith('_report')},
        'history': history,
    }
    for task in ['activity', 'stress', 'arrhythmia']:
        for metrics_dict, suffix in [(test_metrics, ''), (test_metrics_default, '_default')]:
            report = metrics_dict.get(f'{task}_report')
            if report:
                results[f'{task}_report{suffix}'] = report

    results_path = output_dir / f'training_v4_finetune_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='V4 — Two-phase fine-tuning for stress/arrhythmia improvement')

    # Data
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)

    # Model architecture (must match checkpoint)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_ff', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to Model A checkpoint (default: auto-detect)')

    # Fine-tuning hyperparams
    parser.add_argument('--ft_epochs', type=int, default=80,
                        help='Fine-tuning epochs')
    parser.add_argument('--ft_patience', type=int, default=20,
                        help='Early stopping patience for fine-tuning')
    parser.add_argument('--ft_lr', type=float, default=1e-3,
                        help='Learning rate for head fine-tuning (higher than full training)')
    parser.add_argument('--ft_weight_decay', type=float, default=1e-3)
    parser.add_argument('--ft_focal_gamma', type=float, default=1.5,
                        help='Focal gamma for fine-tuning (lower = stronger minority gradient)')
    parser.add_argument('--ft_stress_gamma', type=float, default=0.5,
                        help='Stress-specific focal gamma (very low for max minority attention)')
    parser.add_argument('--ft_w_stress', type=float, default=2.0,
                        help='Stress task weight during fine-tuning')
    parser.add_argument('--ft_w_arrhythmia', type=float, default=2.0,
                        help='Arrhythmia task weight during fine-tuning')

    # Augmentation
    parser.add_argument('--use_mixup', action='store_true', default=True)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)

    # SWA
    parser.add_argument('--use_swa', action='store_true', default=False)
    parser.add_argument('--swa_epochs', type=int, default=20)
    parser.add_argument('--swa_lr', type=float, default=5e-4)

    # Output
    parser.add_argument('--output_dir', type=str, default='training_results')

    args = parser.parse_args()
    train_v4(args)


if __name__ == '__main__':
    main()
