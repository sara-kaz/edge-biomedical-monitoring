#!/usr/bin/env python3
"""
Evaluate Model V5 (three-phase fine-tuned) on the v3 test set.
Generates:
  - ROC curves (stress + arrhythmia)
  - Confusion matrices (all three tasks)
  - Bootstrap 95% confidence intervals
  - Per-dataset performance breakdown
  - Per-subject consistency analysis
"""

import json
import sys
import os
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, roc_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from train_v3 import (
    CNNTransformerV3, BiomedicalDatasetV3,
    N_MERGED_ACTIVITY, MERGED_ACTIVITY_NAMES,
)
from torch.utils.data import DataLoader


def evaluate_detailed(model, loader, device):
    """Evaluate with full per-class metrics, collecting subject IDs."""
    model.eval()
    all_preds = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_labels = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_probs = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_subject_ids = []

    with torch.no_grad():
        for batch in loader:
            x = batch['window_data'].float().to(device)
            labels = {t: batch[t].long().to(device) for t in ['activity', 'stress', 'arrhythmia']}
            outputs = model(x)

            batch_size = x.size(0)
            all_subject_ids.extend(['unknown'] * batch_size)

            for task in ['activity', 'stress', 'arrhythmia']:
                valid = labels[task] >= 0
                if valid.sum() > 0:
                    probs = F.softmax(outputs[task][valid], dim=1)
                    preds = probs.argmax(dim=1)
                    all_preds[task].extend(preds.cpu().numpy())
                    all_labels[task].extend(labels[task][valid].cpu().numpy())
                    all_probs[task].append(probs.cpu().numpy())

    results = {}
    for task in ['activity', 'stress', 'arrhythmia']:
        if not all_labels[task]:
            continue

        y_true = np.array(all_labels[task])
        y_pred = np.array(all_preds[task])
        y_probs = np.concatenate(all_probs[task], axis=0)

        acc = (y_true == y_pred).mean()
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        n_classes = y_probs.shape[1]
        try:
            if n_classes == 2:
                auc_val = roc_auc_score(y_true, y_probs[:, 1])
            else:
                auc_val = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        except Exception:
            auc_val = 0.0

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

        results[task] = {
            'acc': float(acc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'auc': float(auc_val),
            'confusion': cm.tolist(),
            'report': report,
            'n_samples': len(y_true),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probs': y_probs,
        }

    return results


def bootstrap_ci(y_true, y_pred, y_probs, n_bootstrap=2000, ci=0.95):
    """Bootstrap 95% confidence intervals for accuracy, F1-macro, AUC."""
    n = len(y_true)
    rng = np.random.default_rng(42)
    accs, f1s, aucs = [], [], []

    n_cls = y_probs.shape[1]

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        bt_true = y_true[idx]
        bt_pred = y_pred[idx]
        bt_probs = y_probs[idx]

        accs.append(float((bt_true == bt_pred).mean()))
        f1s.append(float(f1_score(bt_true, bt_pred, average='macro', zero_division=0)))

        try:
            if n_cls == 2:
                aucs.append(float(roc_auc_score(bt_true, bt_probs[:, 1])))
            else:
                aucs.append(float(roc_auc_score(bt_true, bt_probs,
                                                multi_class='ovr', average='weighted')))
        except Exception:
            pass

    alpha = (1 - ci) / 2
    def ci_bounds(vals):
        if not vals:
            return (0, 0)
        vals = sorted(vals)
        lo = vals[int(alpha * len(vals))]
        hi = vals[int((1 - alpha) * len(vals))]
        return (round(lo, 4), round(hi, 4))

    return {
        'acc_ci': ci_bounds(accs),
        'f1_macro_ci': ci_bounds(f1s),
        'auc_ci': ci_bounds(aucs),
    }


def plot_roc_curves(results, output_dir):
    """Generate publication-quality ROC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'stress': '#2196F3', 'arrhythmia': '#E91E63'}
    titles = {'stress': 'Stress Detection', 'arrhythmia': 'Arrhythmia Detection'}

    for idx, task in enumerate(['stress', 'arrhythmia']):
        if task not in results:
            continue
        r = results[task]
        y_true = r['y_true']
        y_probs = r['y_probs']

        if y_probs.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)

            axes[idx].plot(fpr, tpr, color=colors[task], linewidth=2.5,
                          label=f'ROC (AUC = {roc_auc:.3f})')
            axes[idx].fill_between(fpr, tpr, alpha=0.1, color=colors[task])
            axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
            axes[idx].set_xlabel('False Positive Rate', fontsize=12)
            axes[idx].set_ylabel('True Positive Rate', fontsize=12)
            axes[idx].set_title(f'{titles[task]} ROC Curve', fontsize=13, fontweight='bold')
            axes[idx].legend(fontsize=11, loc='lower right')
            axes[idx].grid(True, alpha=0.2)
            axes[idx].set_xlim([-0.02, 1.02])
            axes[idx].set_ylim([-0.02, 1.02])

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(output_dir / f'roc_curves_model_v5.{ext}', dpi=300, bbox_inches='tight')
    print(f"  ROC curves saved to {output_dir / 'roc_curves_model_v5.pdf'}")


def plot_confusion_matrices(results, output_dir):
    """Generate publication-quality confusion matrix plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    task_labels = {
        'activity': ['Sedentary', 'Walking', 'Cycling', 'High-Int.'],
        'stress': ['Non-stressed', 'Stressed'],
        'arrhythmia': ['Normal', 'Abnormal'],
    }

    for idx, task in enumerate(['activity', 'stress', 'arrhythmia']):
        if task not in results:
            continue
        r = results[task]
        cm = np.array(r['confusion'])
        labels = task_labels[task]

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = axes[idx].imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        axes[idx].set_title(f'{task.capitalize()}\n(Acc={r["acc"]*100:.1f}%, AUC={r["auc"]:.3f})',
                           fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('True', fontsize=10)

        tick_marks = np.arange(len(labels))
        axes[idx].set_xticks(tick_marks)
        axes[idx].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[idx].set_yticks(tick_marks)
        axes[idx].set_yticklabels(labels, fontsize=8)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                axes[idx].text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1%})',
                             ha='center', va='center', color=color, fontsize=7)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(output_dir / f'confusion_matrices_model_v5.{ext}', dpi=300, bbox_inches='tight')
    print(f"  Confusion matrices saved")


def per_dataset_breakdown(model, dataset_path, test_subjects, device):
    """Evaluate performance per source dataset."""
    print("\n" + "=" * 70)
    print("PER-DATASET PERFORMANCE BREAKDOWN")
    print("=" * 70)

    # Load raw data to get source info
    with open(dataset_path, 'rb') as f:
        raw_data = pickle.load(f)

    # Group test samples by source dataset
    source_map = {}  # subject_id -> source prefix
    for sample in raw_data:
        sid = sample.get('subject_id', 'unknown')
        if sid in test_subjects:
            if sid.startswith('UCI_'):
                source_map[sid] = 'UCI-HAR'
            elif sid.startswith('WS_'):
                source_map[sid] = 'Wearable-Stress'
            elif sid.startswith('S'):
                source_map[sid] = 'WESAD'
            elif sid.isdigit() and int(sid) >= 200:
                source_map[sid] = 'MIT-BIH'
            else:
                source_map[sid] = 'PPG-DaLiA'

    # Evaluate per source
    sources = sorted(set(source_map.values()))
    results_by_source = {}

    for source in sources:
        source_subjects = [s for s, src in source_map.items() if src == source]
        if not source_subjects:
            continue

        ds = BiomedicalDatasetV3(
            dataset_path, subject_filter=source_subjects,
            binary_stress=True, augment=False, normalize=True,
            oversample=False, merge_activity=True,
        )
        if len(ds) == 0:
            continue

        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        r = evaluate_detailed(model, loader, device)

        results_by_source[source] = {}
        print(f"\n  {source} ({len(source_subjects)} subjects, {len(ds)} windows):")
        for task in ['activity', 'stress', 'arrhythmia']:
            if task in r:
                t = r[task]
                print(f"    {task:<12} N={t['n_samples']:>5} Acc={t['acc']*100:.1f}% "
                      f"F1m={t['f1_macro']*100:.1f}% AUC={t['auc']:.3f}")
                results_by_source[source][task] = {
                    'acc': t['acc'], 'f1_macro': t['f1_macro'],
                    'auc': t['auc'], 'n_samples': t['n_samples'],
                }

    return results_by_source


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Find dataset
    dataset_path = None
    candidates = [
        PROJECT_ROOT.parent / 'processed_unified_dataset' / 'unified_dataset.pkl',
        Path('/Users/HP/Desktop/University/Thesis/Code/multimodal-biomedical-monitoring-improved/processed_unified_dataset/unified_dataset.pkl'),
    ]
    for p in candidates:
        if p.exists():
            dataset_path = str(p)
            break
    print(f"Dataset: {dataset_path}")

    # Use V3 splits (includes UCI HAR + Wearable Stress)
    splits_path = PROJECT_ROOT / 'training_results' / 'subject_splits_v3.json'
    print(f"Splits: {splits_path}")
    with open(splits_path) as f:
        splits = json.load(f)
    test_subjects = splits['test_subjects']
    print(f"Test subjects: {len(test_subjects)}")

    # Create test dataset
    test_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=test_subjects,
        binary_stress=True, augment=False, normalize=True,
        oversample=False, merge_activity=True,
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Load Model V5
    model_path = PROJECT_ROOT / 'training_results' / 'model_v5_stress_finetune.pth'
    print(f"\nLoading Model V5: {model_path}")

    model = CNNTransformerV3(
        n_channels=5, n_samples=1000,
        activity_classes=4, stress_classes=2, arrhythmia_classes=2,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.3, use_transformer=True,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ═══════════════════════════════════════════════════════════════
    # Main evaluation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL V5 — Evaluated on v3 test set")
    print("=" * 70)

    results = evaluate_detailed(model, test_loader, device)

    print(f"\n{'Task':<15} {'Samples':>8} {'Acc':>8} {'F1-macro':>10} {'F1-wt':>10} {'AUC':>8}")
    print("-" * 65)
    for task in ['activity', 'stress', 'arrhythmia']:
        r = results[task]
        print(f"  {task:<13} {r['n_samples']:>7} {r['acc']*100:>7.1f}% {r['f1_macro']*100:>9.1f}% "
              f"{r['f1_weighted']*100:>9.1f}% {r['auc']:>7.3f}")

    # Per-class breakdown
    for task in ['activity', 'stress', 'arrhythmia']:
        r = results[task]
        report = r['report']
        print(f"\n  {task.upper()} per-class:")
        for cls_key, vals in report.items():
            if cls_key in ('accuracy', 'macro avg', 'weighted avg'):
                continue
            if isinstance(vals, dict):
                if task == 'activity':
                    cls_name = MERGED_ACTIVITY_NAMES.get(int(cls_key), cls_key)
                elif task == 'stress':
                    cls_name = ['Non-stressed', 'Stressed'][int(cls_key)]
                else:
                    cls_name = ['Normal', 'Abnormal'][int(cls_key)]
                print(f"    {cls_name}: P={vals['precision']:.3f} R={vals['recall']:.3f} "
                      f"F1={vals['f1-score']:.3f} N={int(vals['support'])}")

    # Confusion matrices
    for task in ['stress', 'arrhythmia']:
        r = results[task]
        print(f"\n  {task.upper()} Confusion Matrix:")
        for row in r['confusion']:
            print(f"    {row}")

    # Clinical metrics for arrhythmia
    if 'arrhythmia' in results:
        r = results['arrhythmia']
        cm = np.array(r['confusion'])
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        print(f"\n  ARRHYTHMIA Clinical Metrics:")
        print(f"    Sensitivity: {sens*100:.1f}%")
        print(f"    Specificity: {spec*100:.1f}%")
        print(f"    AUC-ROC: {r['auc']:.3f}")
        print(f"    PPV: {ppv*100:.1f}%")
        print(f"    NPV: {npv*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Bootstrap confidence intervals
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (2000 iterations)")
    print("=" * 70)

    ci_results = {}
    for task in ['activity', 'stress', 'arrhythmia']:
        r = results[task]
        ci = bootstrap_ci(r['y_true'], r['y_pred'], r['y_probs'])
        ci_results[task] = ci
        print(f"\n  {task.upper()} (N={r['n_samples']}):")
        print(f"    Accuracy:  {r['acc']*100:.1f}% "
              f"[{ci['acc_ci'][0]*100:.1f}%, {ci['acc_ci'][1]*100:.1f}%]")
        print(f"    F1-macro:  {r['f1_macro']*100:.1f}% "
              f"[{ci['f1_macro_ci'][0]*100:.1f}%, {ci['f1_macro_ci'][1]*100:.1f}%]")
        print(f"    AUC:       {r['auc']:.3f} "
              f"[{ci['auc_ci'][0]:.3f}, {ci['auc_ci'][1]:.3f}]")

    # ═══════════════════════════════════════════════════════════════
    # Per-dataset breakdown
    # ═══════════════════════════════════════════════════════════════
    source_results = per_dataset_breakdown(model, dataset_path, test_subjects, device)

    # ═══════════════════════════════════════════════════════════════
    # Generate figures
    # ═══════════════════════════════════════════════════════════════
    output_dir = PROJECT_ROOT / 'docs' / 'thesis' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also save to journal paper figures
    journal_fig_dir = PROJECT_ROOT / 'docs' / 'thesis' / 'journal_paper' / 'figures'
    journal_fig_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curves(results, output_dir)
    plot_confusion_matrices(results, output_dir)

    # Copy to journal paper figures too
    plot_roc_curves(results, journal_fig_dir)
    plot_confusion_matrices(results, journal_fig_dir)

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    save_results = {}
    for task in ['activity', 'stress', 'arrhythmia']:
        r = results[task]
        save_results[task] = {
            'acc': r['acc'], 'f1_macro': r['f1_macro'],
            'f1_weighted': r['f1_weighted'], 'auc': r['auc'],
            'confusion': r['confusion'], 'n_samples': r['n_samples'],
            'report': r['report'],
            'bootstrap_ci_95': ci_results[task],
        }
    save_results['per_dataset'] = source_results

    results_path = PROJECT_ROOT / 'training_results' / 'model_v5_eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == '__main__':
    main()
