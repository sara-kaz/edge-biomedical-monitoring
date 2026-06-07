#!/usr/bin/env python3
"""
Phase 4: Fine-tune ONLY the arrhythmia head on Model V5.
Backbone, activity head, and stress head are all frozen.

Goal: Improve arrhythmia abnormal recall from 36% to 60%+ while
maintaining specificity and not degrading other tasks.

Strategy:
  1. Load Model V5 (Phase 3 stress fine-tuned)
  2. Extract backbone features for all arrhythmia-labeled windows (forward hook)
  3. Retrain arrhythmia head on CPU with heavy class weighting
  4. Sweep hyperparameters (lr, gamma, class weight boost)
  5. Save best as Model V6
"""

import json
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from train_v3 import (
    BiomedicalDatasetV3, CNNTransformerV3, ClassBalancedFocalLoss,
    MultiTaskLossV3, N_MERGED_ACTIVITY
)
from torch.utils.data import DataLoader


def extract_features(model, loader, device, task='arrhythmia'):
    """Extract frozen backbone features for arrhythmia-labeled windows."""
    model.eval()

    # Hook to capture input to the arrhythmia head
    features_list = []
    labels_list = []

    def hook_fn(module, input, output):
        features_list.append(input[0].detach().cpu())

    # Register hook on first layer of arrhythmia head
    hook = model.task_heads[task][0].register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            x = batch['window_data'].float().to(device)
            labels = batch[task].long()
            _ = model(x)

            # Get features from hook for valid samples
            batch_features = features_list[-1]
            valid = labels >= 0
            if valid.sum() > 0:
                # We need to figure out which samples in the batch have valid labels
                # The hook captures ALL samples, but we only want valid ones
                pass

    hook.remove()

    # Now do it properly: collect all features and labels, filter valid
    features_list = []
    labels_list = []
    hook = model.task_heads[task][0].register_forward_hook(hook_fn)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            x = batch['window_data'].float().to(device)
            labels = batch[task].long()
            features_list.clear()
            _ = model(x)

            batch_features = features_list[0]  # shape: [batch_size, feature_dim]
            valid = labels >= 0
            if valid.sum() > 0:
                all_features.append(batch_features[valid])
                all_labels.append(labels[valid])

    hook.remove()

    features = torch.cat(all_features, dim=0)  # [N, 64]
    labels = torch.cat(all_labels, dim=0)       # [N]

    return features, labels


def train_head_on_features(features, labels, head, configs, val_features=None, val_labels=None):
    """Train a classification head on pre-extracted features."""
    device = 'cpu'  # CPU training for small head

    best_overall = {'f1': 0, 'state': None, 'config': '', 'metrics': {}}

    original_state = {k: v.clone() for k, v in head.state_dict().items()}

    # Class distribution
    counts = Counter(labels.numpy().tolist())
    total = sum(counts.values())
    print(f"\n  Class distribution: {dict(counts)}")
    print(f"  Imbalance ratio: 1:{counts[0]/max(counts[1],1):.1f}")

    for config in configs:
        # Reset head
        head.load_state_dict(original_state)
        head.to(device)
        head.train()

        # Class weights
        boost = config.get('boost', 1.0)
        w0 = total / (2 * counts[0])
        w1 = (total / (2 * counts[1])) * boost
        class_weight = torch.FloatTensor([w0, w1]).to(device)

        # Loss
        gamma = config.get('gamma', 2.0)
        loss_fn = ClassBalancedFocalLoss(gamma=gamma, weight=class_weight, ignore_index=-1)

        # Optimizer
        lr = config.get('lr', 1e-3)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=config.get('wd', 1e-3))
        epochs = config.get('epochs', 60)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        batch_size = config.get('batch_size', 128)
        n = features.shape[0]

        best_val_f1 = 0
        best_state = None

        for epoch in range(epochs):
            head.train()
            # Shuffle
            perm = torch.randperm(n)
            epoch_loss = 0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start:start+batch_size]
                x = features[idx].to(device)
                y = labels[idx].to(device)

                optimizer.zero_grad()
                out = head(x)
                loss = loss_fn(out, y)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validate
            if val_features is not None and val_labels is not None:
                head.eval()
                with torch.no_grad():
                    val_out = head(val_features.to(device))
                    val_probs = F.softmax(val_out, dim=1)
                    val_preds = val_probs.argmax(dim=1).numpy()
                    vl = val_labels.numpy()
                    val_f1 = f1_score(vl, val_preds, average='macro', zero_division=0)

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            else:
                # No val set, use training F1
                head.eval()
                with torch.no_grad():
                    train_out = head(features.to(device))
                    train_preds = train_out.argmax(dim=1).numpy()
                    train_f1 = f1_score(labels.numpy(), train_preds, average='macro', zero_division=0)
                    if train_f1 > best_val_f1:
                        best_val_f1 = train_f1
                        best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        # Evaluate best state on test features (using val as proxy)
        if best_state:
            head.load_state_dict(best_state)
        head.eval()

        eval_feat = val_features if val_features is not None else features
        eval_lab = val_labels if val_labels is not None else labels

        with torch.no_grad():
            out = head(eval_feat.to(device))
            probs = F.softmax(out, dim=1)
            preds = probs.argmax(dim=1).numpy()
            y = eval_lab.numpy()

            acc = (preds == y).mean()
            f1m = f1_score(y, preds, average='macro', zero_division=0)
            try:
                auc = roc_auc_score(y, probs[:, 1].numpy())
            except:
                auc = 0.0

            cm = confusion_matrix(y, preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                recall, spec = 0, 0

        config_name = config.get('name', str(config))
        print(f"  Config {config_name}: Acc={acc*100:.1f}% F1m={f1m*100:.1f}% AUC={auc:.3f} "
              f"Recall={recall*100:.1f}% Spec={spec*100:.1f}% [w={class_weight.numpy()}]")

        if f1m > best_overall['f1']:
            best_overall = {
                'f1': f1m,
                'state': {k: v.cpu().clone() for k, v in head.state_dict().items()},
                'config': config_name,
                'metrics': {
                    'acc': float(acc), 'f1_macro': float(f1m), 'auc': float(auc),
                    'recall': float(recall), 'specificity': float(spec),
                    'confusion': cm.tolist(),
                },
            }

    return best_overall


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Model V5
    ckpt_path = PROJECT_ROOT / 'training_results' / 'model_v5_stress_finetune.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"Loaded: {ckpt_path.name}")

    # Find dataset
    candidates = [
        PROJECT_ROOT.parent / 'processed_unified_dataset' / 'unified_dataset.pkl',
        Path('/Users/HP/Desktop/University/Thesis/Code/multimodal-biomedical-monitoring-improved/processed_unified_dataset/unified_dataset.pkl'),
    ]
    dataset_path = str(next(p for p in candidates if p.exists()))

    # Load splits
    with open(PROJECT_ROOT / 'training_results' / 'subject_splits_v3.json') as f:
        splits = json.load(f)

    # Create datasets
    train_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=splits['train_subjects'],
        binary_stress=True, augment=False, normalize=True,
        oversample=False, merge_activity=True,
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

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Build model
    model = CNNTransformerV3(
        n_channels=5, n_samples=1000,
        activity_classes=4, stress_classes=2, arrhythmia_classes=2,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.3, use_transformer=True,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Evaluate baseline (Model V5)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("BASELINE — Model V5 (before arrhythmia fine-tuning)")
    print(f"{'='*60}")

    from eval_model_c import evaluate_detailed
    baseline = evaluate_detailed(model, test_loader, device)

    for task in ['activity', 'stress', 'arrhythmia']:
        r = baseline[task]
        print(f"  {task:<13} Acc={r['acc']*100:.1f}% F1m={r['f1_macro']*100:.1f}% AUC={r['auc']:.3f}")

    print(f"\n  ARRHYTHMIA confusion matrix:")
    cm = np.array(baseline['arrhythmia']['confusion'])
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    print(f"    TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"    Recall={tp/(tp+fn)*100:.1f}% Specificity={tn/(tn+fp)*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Extract backbone features for arrhythmia-labeled windows
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("EXTRACTING BACKBONE FEATURES")
    print(f"{'='*60}")

    print("  Extracting train features...")
    train_feat, train_lab = extract_features(model, train_loader, device, 'arrhythmia')
    print(f"  Train: {train_feat.shape[0]} samples, {train_feat.shape[1]}-dim features")

    print("  Extracting val features...")
    val_feat, val_lab = extract_features(model, val_loader, device, 'arrhythmia')
    print(f"  Val: {val_feat.shape[0]} samples, {val_feat.shape[1]}-dim features")

    print("  Extracting test features...")
    test_feat, test_lab = extract_features(model, test_loader, device, 'arrhythmia')
    print(f"  Test: {test_feat.shape[0]} samples, {test_feat.shape[1]}-dim features")

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Train arrhythmia head with heavy class weighting
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("ARRHYTHMIA HEAD FINE-TUNING (Phase 4)")
    print(f"{'='*60}")

    # Get the arrhythmia head
    arr_head = model.task_heads['arrhythmia']
    arr_params = sum(p.numel() for p in arr_head.parameters())
    print(f"  Arrhythmia head params: {arr_params}")
    print(f"  Head architecture:")
    for name, module in arr_head.named_modules():
        if name:
            print(f"    {name}: {module}")

    # Sweep configurations
    configs = [
        # Conservative: moderate boost
        {'lr': 1e-3, 'gamma': 2.0, 'boost': 2.0, 'epochs': 80, 'name': 'lr1e-3_g2_b2'},
        {'lr': 1e-3, 'gamma': 2.5, 'boost': 3.0, 'epochs': 80, 'name': 'lr1e-3_g2.5_b3'},
        # Aggressive: high class weight boost
        {'lr': 2e-3, 'gamma': 2.0, 'boost': 4.0, 'epochs': 80, 'name': 'lr2e-3_g2_b4'},
        {'lr': 3e-3, 'gamma': 2.0, 'boost': 5.0, 'epochs': 80, 'name': 'lr3e-3_g2_b5'},
        # Very aggressive
        {'lr': 5e-3, 'gamma': 1.5, 'boost': 6.0, 'epochs': 100, 'name': 'lr5e-3_g1.5_b6'},
        {'lr': 3e-3, 'gamma': 3.0, 'boost': 4.0, 'epochs': 100, 'name': 'lr3e-3_g3_b4'},
        # Different optimizers
        {'lr': 1e-3, 'gamma': 2.0, 'boost': 8.0, 'epochs': 80, 'name': 'lr1e-3_g2_b8'},
        {'lr': 2e-3, 'gamma': 2.0, 'boost': 10.0, 'epochs': 100, 'name': 'lr2e-3_g2_b10'},
    ]

    best = train_head_on_features(
        train_feat, train_lab, arr_head, configs,
        val_features=val_feat, val_labels=val_lab
    )

    print(f"\n  Best config: {best['config']}")
    print(f"  Val metrics: {best['metrics']}")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Load best head and evaluate on test set
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION — Model V6")
    print(f"{'='*60}")

    arr_head.load_state_dict(best['state'])
    model.task_heads['arrhythmia'] = arr_head.to(device)

    final = evaluate_detailed(model, test_loader, device)

    print(f"\n{'Task':<15} {'Samples':>8} {'Acc':>8} {'F1-macro':>10} {'AUC':>8}")
    print("-" * 55)
    for task in ['activity', 'stress', 'arrhythmia']:
        r = final[task]
        print(f"  {task:<13} {r['n_samples']:>7} {r['acc']*100:>7.1f}% {r['f1_macro']*100:>9.1f}% {r['auc']:>7.3f}")

    # Per-class arrhythmia
    report = final['arrhythmia']['report']
    print(f"\n  ARRHYTHMIA per-class:")
    for cls_key, vals in report.items():
        if cls_key in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        if isinstance(vals, dict):
            name = ['Normal', 'Abnormal'][int(cls_key)]
            print(f"    {name}: P={vals['precision']:.3f} R={vals['recall']:.3f} "
                  f"F1={vals['f1-score']:.3f} N={int(vals['support'])}")

    cm = np.array(final['arrhythmia']['confusion'])
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"    Recall:      {tp/(tp+fn)*100:.1f}%")
    print(f"    Specificity:  {tn/(tn+fp)*100:.1f}%")
    print(f"    PPV:          {tp/(tp+fp)*100:.1f}%")
    print(f"    NPV:          {tn/(tn+fn)*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("IMPROVEMENT: Model V5 → Model V6")
    print(f"{'='*60}")
    for task in ['activity', 'stress', 'arrhythmia']:
        b = baseline[task]
        f = final[task]
        d_acc = (f['acc'] - b['acc']) * 100
        d_f1 = (f['f1_macro'] - b['f1_macro']) * 100
        d_auc = f['auc'] - b['auc']
        print(f"  {task:<13} Acc: {b['acc']*100:.1f}% → {f['acc']*100:.1f}% ({d_acc:+.1f}%) "
              f" F1m: {b['f1_macro']*100:.1f}% → {f['f1_macro']*100:.1f}% ({d_f1:+.1f}%) "
              f" AUC: {b['auc']:.3f} → {f['auc']:.3f} ({d_auc:+.3f})")

    # Check that stress and activity didn't degrade
    stress_ok = abs(final['stress']['acc'] - baseline['stress']['acc']) < 0.001
    activity_ok = abs(final['activity']['acc'] - baseline['activity']['acc']) < 0.001
    print(f"\n  Activity preserved: {'✓' if activity_ok else '✗'}")
    print(f"  Stress preserved:   {'✓' if stress_ok else '✗'}")

    # ═══════════════════════════════════════════════════════════════
    # Step 6: Save Model V6
    # ═══════════════════════════════════════════════════════════════
    save_path = PROJECT_ROOT / 'training_results' / 'model_v6_arrhythmia_finetune.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_channels': 5,
        'activity_classes': 4,
        'stress_classes': 2,
        'arrhythmia_classes': 2,
        'n_params': n_params,
        'phase': 'Phase 4: arrhythmia head fine-tune on Model V5',
        'best_config': best['config'],
        'val_metrics': best['metrics'],
        'test_metrics': {
            task: {
                'acc': final[task]['acc'],
                'f1_macro': final[task]['f1_macro'],
                'f1_weighted': final[task]['f1_weighted'],
                'auc': final[task]['auc'],
                'confusion': final[task]['confusion'],
                'n_samples': final[task]['n_samples'],
            } for task in ['activity', 'stress', 'arrhythmia']
        },
        'baseline_metrics': {
            task: {
                'acc': baseline[task]['acc'],
                'f1_macro': baseline[task]['f1_macro'],
                'auc': baseline[task]['auc'],
            } for task in ['activity', 'stress', 'arrhythmia']
        },
    }, save_path)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
