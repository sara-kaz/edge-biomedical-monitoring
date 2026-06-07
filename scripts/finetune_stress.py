#!/usr/bin/env python3
"""
Fine-tune ONLY the stress head on the best V3 model.
Backbone and other heads are frozen.
"""

import json
import pickle
import random
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, roc_auc_score, classification_report

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from train_v3 import (
    BiomedicalDatasetV3, CNNTransformerV3, ClassBalancedFocalLoss,
    MultiTaskLossV3, evaluate, evaluate_with_thresholds, optimize_thresholds,
    N_MERGED_ACTIVITY
)


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load the good V3 checkpoint
    ckpt_path = PROJECT_ROOT / 'training_results' / 'model_v3_2026-03-12_02-37-39.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"Loaded checkpoint: {ckpt_path.name}")
    print(f"  Activity acc: {ckpt['test_metrics']['activity_acc']*100:.1f}%")
    print(f"  Stress F1-macro: {ckpt['test_metrics']['stress_f1_macro']*100:.1f}%")
    print(f"  Arrhythmia acc: {ckpt['test_metrics']['arrhythmia_acc']*100:.1f}%")

    # Find dataset
    candidates = [
        PROJECT_ROOT.parent / 'processed_unified_dataset' / 'unified_dataset.pkl',
        Path('/Users/HP/Desktop/University/Thesis/Code/multimodal-biomedical-monitoring-improved/processed_unified_dataset/unified_dataset.pkl'),
    ]
    dataset_path = str(next(p for p in candidates if p.exists()))

    # Load splits
    with open(PROJECT_ROOT / 'training_results' / 'subject_splits_v3.json') as f:
        splits = json.load(f)

    # Create datasets — stress-focused oversampling
    train_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=splits['train_subjects'],
        binary_stress=True, augment=True, normalize=True,
        oversample=True, merge_activity=True,
    )
    val_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=splits['val_subjects'],
        binary_stress=True, augment=False, normalize=True,
        oversample=False, merge_activity=True,
    )
    test_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=splits['test_subjects'],
        binary_stress=True, augment=False, normalize=True,
        oversample=False, merge_activity=True,
    )

    # Create stress-focused sampler: weight stressed samples 4x
    weights = torch.ones(len(train_ds))
    for i in range(len(train_ds)):
        if train_ds.stress_labels[i] == 1:
            weights[i] = 4.0
        elif train_ds.stress_labels[i] == 0:
            weights[i] = 1.0
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Build model and load weights
    model = CNNTransformerV3(
        n_channels=5, n_samples=1000,
        activity_classes=4, stress_classes=2, arrhythmia_classes=2,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.3, use_transformer=True,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    print("Model weights loaded successfully")

    # For evaluation criterion (all tasks)
    class_weights = train_ds.get_class_weights()
    for task, w in class_weights.items():
        class_weights[task] = w.to(device)
    criterion = MultiTaskLossV3(class_weights=class_weights, task_weights={'activity': 1.0, 'stress': 1.5, 'arrhythmia': 2.0})

    # Evaluate baseline
    print("\n=== BASELINE (before fine-tuning) ===")
    base_metrics = evaluate(model, test_loader, criterion, device)
    for task in ['activity', 'stress', 'arrhythmia']:
        acc = base_metrics.get(f'{task}_acc', 0)
        f1m = base_metrics.get(f'{task}_f1_macro', 0)
        auc = base_metrics.get(f'{task}_auc', 0)
        print(f"  {task:13s}: Acc={acc*100:.1f}%  F1m={f1m*100:.1f}%  AUC={auc:.3f}")

    # =========================================================================
    # Phase 2: Fine-tune ONLY stress head
    # =========================================================================
    print(f"\n{'='*60}")
    print("STRESS HEAD FINE-TUNING")
    print(f"{'='*60}")

    # Freeze everything except stress head
    for name, param in model.named_parameters():
        if 'task_heads.stress' not in name:
            param.requires_grad = False

    stress_params = list(model.task_heads['stress'].parameters())
    n_stress_params = sum(p.numel() for p in stress_params)
    print(f"Trainable params: {n_stress_params}")

    # Strong class weighting for stress: inverse frequency
    str_counts = Counter(s for s in train_ds.stress_labels if s >= 0)
    total_str = sum(str_counts.values())
    stress_w = torch.FloatTensor([
        total_str / (2 * str_counts.get(0, 1)),
        total_str / (2 * str_counts.get(1, 1)),
    ]).to(device)
    print(f"Stress class weights: {stress_w.cpu().numpy()}")

    # Try multiple configs and pick the best
    configs = [
        {'lr': 5e-4, 'gamma': 2.0, 'epochs': 30, 'name': 'lr5e-4_g2'},
        {'lr': 1e-3, 'gamma': 2.5, 'epochs': 30, 'name': 'lr1e-3_g2.5'},
        {'lr': 2e-3, 'gamma': 3.0, 'epochs': 30, 'name': 'lr2e-3_g3'},
        {'lr': 5e-4, 'gamma': 1.5, 'epochs': 40, 'name': 'lr5e-4_g1.5'},
    ]

    best_overall_state = None
    best_overall_f1 = 0
    best_config_name = ""

    # Save original stress head state to restore between configs
    original_stress_state = {k: v.clone() for k, v in model.task_heads['stress'].state_dict().items()}

    for config in configs:
        # Reset stress head to original
        model.task_heads['stress'].load_state_dict(original_stress_state)

        stress_loss = ClassBalancedFocalLoss(gamma=config['gamma'], weight=stress_w, ignore_index=-1)
        optimizer = torch.optim.AdamW(stress_params, lr=config['lr'], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

        best_f1 = 0
        best_state = None

        for epoch in range(config['epochs']):
            model.train()
            for batch in train_loader:
                x = batch['window_data'].float().to(device)
                stress_labels = batch['stress'].long().to(device)

                optimizer.zero_grad()
                outputs = model(x)
                loss = stress_loss(outputs['stress'], stress_labels)

                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()

            scheduler.step()

            # Evaluate on val
            model.eval()
            all_preds, all_labels, all_probs = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['window_data'].float().to(device)
                    s = batch['stress'].long().to(device)
                    out = model(x)
                    valid = s >= 0
                    if valid.sum() > 0:
                        probs = F.softmax(out['stress'][valid], dim=1)
                        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
                        all_labels.extend(s[valid].cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())

            if all_labels:
                val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Config {config['name']}: best val stress F1-macro = {best_f1*100:.1f}%")

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall_state = best_state
            best_config_name = config['name']

    print(f"\nBest config: {best_config_name} (val stress F1 = {best_overall_f1*100:.1f}%)")

    # Load best state
    if best_overall_state:
        model.load_state_dict(best_overall_state)

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    # Threshold optimization
    print(f"\n{'='*60}")
    print("THRESHOLD OPTIMIZATION")
    print(f"{'='*60}")
    thresholds = optimize_thresholds(model, val_loader, device)

    # Final test
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")

    test_default = evaluate(model, test_loader, criterion, device)
    test_opt = evaluate_with_thresholds(model, test_loader, criterion, device, thresholds)

    print(f"\n{'Task':<15} {'Acc':>8} {'F1-macro':>10} {'F1-wt':>10} {'AUC':>8}")
    print(f"{'-'*55}")
    for task in ['activity', 'stress', 'arrhythmia']:
        acc = test_opt.get(f'{task}_acc', 0)
        f1m = test_opt.get(f'{task}_f1_macro', 0)
        f1w = test_opt.get(f'{task}_f1_weighted', 0)
        auc = test_opt.get(f'{task}_auc', 0)
        print(f"  {task:<13} {acc*100:>7.1f}% {f1m*100:>9.1f}% {f1w*100:>9.1f}% {auc:>7.3f}")

    # Per-class
    for task in ['activity', 'stress', 'arrhythmia']:
        report = test_opt.get(f'{task}_report')
        if report:
            print(f"\n  {task.upper()} per-class:")
            for cls, vals in report.items():
                if cls in ('accuracy', 'macro avg', 'weighted avg'):
                    continue
                if isinstance(vals, dict):
                    print(f"    Class {cls}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1-score']:.3f} N={int(vals['support'])}")

    # Comparison
    print(f"\n{'='*60}")
    print("IMPROVEMENT OVER BASELINE")
    print(f"{'='*60}")
    for task in ['activity', 'stress', 'arrhythmia']:
        base_f1 = base_metrics.get(f'{task}_f1_macro', 0)
        new_f1 = test_opt.get(f'{task}_f1_macro', 0)
        base_acc = base_metrics.get(f'{task}_acc', 0)
        new_acc = test_opt.get(f'{task}_acc', 0)
        delta_f1 = (new_f1 - base_f1) * 100
        delta_acc = (new_acc - base_acc) * 100
        print(f"  {task:13s}: Acc {base_acc*100:.1f}% -> {new_acc*100:.1f}% ({delta_acc:+.1f}%) | F1m {base_f1*100:.1f}% -> {new_f1*100:.1f}% ({delta_f1:+.1f}%)")

    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = PROJECT_ROOT / 'training_results' / f'model_v3_stress_ft_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_channels': 5,
        'activity_classes': 4,
        'stress_classes': 2,
        'n_params': sum(p.numel() for p in model.parameters()),
        'thresholds': thresholds,
        'best_config': best_config_name,
        'test_metrics': {k: v for k, v in test_opt.items() if not k.endswith('_report')},
        'test_metrics_default': {k: v for k, v in test_default.items() if not k.endswith('_report')},
    }, save_path)
    print(f"\nSaved: {save_path}")

    # Also overwrite model_v3.pth
    torch.save(model.state_dict(), PROJECT_ROOT / 'training_results' / 'model_v3.pth')

    # Save results JSON
    results = {
        'timestamp': timestamp,
        'best_config': best_config_name,
        'thresholds': thresholds,
        'baseline_metrics': {k: v for k, v in base_metrics.items() if not k.endswith('_report')},
        'test_metrics': {k: v for k, v in test_opt.items() if not k.endswith('_report')},
        'test_metrics_default': {k: v for k, v in test_default.items() if not k.endswith('_report')},
    }
    with open(PROJECT_ROOT / 'training_results' / f'stress_finetune_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
