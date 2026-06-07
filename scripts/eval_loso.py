#!/usr/bin/env python3
"""
Leave-One-Subject-Out (LOSO) evaluation of Model C.
Evaluates the pre-trained model on each subject individually, then averages.
No retraining — this measures generalization of the trained model.
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
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from train_v3 import (
    CNNTransformerV3, BiomedicalDatasetV3,
    N_MERGED_ACTIVITY, MERGED_ACTIVITY_NAMES,
)


def get_subject_tasks(dataset_path):
    """Get subjects and which tasks they have labels for."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    subj_tasks = defaultdict(lambda: set())
    subj_counts = defaultdict(int)
    for w in data:
        sid = w.get('subject_id', 'unknown')
        subj_counts[sid] += 1
        labels = w.get('labels', {})
        for task in ['activity', 'stress', 'arrhythmia']:
            lbl = labels.get(task, None)
            if lbl is not None:
                if isinstance(lbl, np.ndarray) and lbl.sum() > 0:
                    subj_tasks[sid].add(task)
                elif isinstance(lbl, (int, float)) and lbl >= 0:
                    subj_tasks[sid].add(task)

    return dict(subj_tasks), dict(subj_counts)


def evaluate_subject(model, dataset_path, subject_id, device, global_norm_stats=None):
    """Evaluate model on a single subject using global normalization."""
    ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=[subject_id],
        binary_stress=True, augment=False, normalize=False,  # We'll normalize manually
        oversample=False, merge_activity=True,
    )
    # Apply global normalization stats
    if global_norm_stats is not None:
        ds.normalize = True
        ds.channel_means = global_norm_stats['means']
        ds.channel_stds = global_norm_stats['stds']

    if len(ds) == 0:
        return None

    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    model.eval()
    all_preds = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_labels = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_probs = {'activity': [], 'stress': [], 'arrhythmia': []}

    with torch.no_grad():
        for batch in loader:
            x = batch['window_data'].float().to(device)
            labels = {t: batch[t].long().to(device) for t in ['activity', 'stress', 'arrhythmia']}
            outputs = model(x)

            for task in ['activity', 'stress', 'arrhythmia']:
                valid = labels[task] >= 0
                if valid.sum() > 0:
                    probs = F.softmax(outputs[task][valid], dim=1)
                    preds = probs.argmax(dim=1)
                    all_preds[task].extend(preds.cpu().numpy())
                    all_labels[task].extend(labels[task][valid].cpu().numpy())
                    all_probs[task].append(probs.cpu().numpy())

    result = {'subject': subject_id, 'n_windows': len(ds)}
    for task in ['activity', 'stress', 'arrhythmia']:
        if not all_labels[task]:
            continue

        y_true = np.array(all_labels[task])
        y_pred = np.array(all_preds[task])
        y_probs = np.concatenate(all_probs[task], axis=0)

        acc = (y_true == y_pred).mean()
        f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        n_classes = y_probs.shape[1]
        try:
            if n_classes == 2:
                # Need both classes present for AUC
                if len(np.unique(y_true)) > 1:
                    auc_val = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    auc_val = float('nan')
            else:
                if len(np.unique(y_true)) > 1:
                    auc_val = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
                else:
                    auc_val = float('nan')
        except:
            auc_val = float('nan')

        cm = confusion_matrix(y_true, y_pred)

        # Clinical metrics for binary tasks
        if n_classes == 2 and cm.shape == (2, 2):
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        else:
            sens = spec = float('nan')

        result[f'{task}_acc'] = float(acc)
        result[f'{task}_f1_macro'] = float(f1m)
        result[f'{task}_f1_weighted'] = float(f1w)
        result[f'{task}_auc'] = float(auc_val)
        result[f'{task}_sensitivity'] = float(sens)
        result[f'{task}_specificity'] = float(spec)
        result[f'{task}_n_samples'] = len(y_true)
        result[f'{task}_class_dist'] = {str(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))}

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
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load Model C
    model_path = PROJECT_ROOT / 'training_results' / 'model_v4_finetuned_2026-03-15_05-22-08.pth'
    print(f"Model: {model_path}")

    model = CNNTransformerV3(
        n_channels=5, n_samples=1000,
        activity_classes=4, stress_classes=2, arrhythmia_classes=2,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.3, use_transformer=True,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Get subjects and their tasks
    subj_tasks, subj_counts = get_subject_tasks(dataset_path)

    # Identify which subjects were in training vs test
    splits_path = PROJECT_ROOT / 'training_results' / 'subject_splits.json'
    with open(splits_path) as f:
        splits = json.load(f)
    train_subjects = set(splits['train_subjects'] + splits['val_subjects'])
    test_subjects_original = set(splits['test_subjects'])

    print(f"\nTotal subjects: {len(subj_tasks)}")
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Original test subjects: {len(test_subjects_original)}")

    # Compute GLOBAL normalization stats (matching training time)
    print("\nComputing global normalization stats...")
    global_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=list(train_subjects),
        binary_stress=True, augment=False, normalize=True,
        oversample=False, merge_activity=True,
    )
    global_norm_stats = {
        'means': global_ds.channel_means,
        'stds': global_ds.channel_stds,
    }
    print(f"  Global means shape: {global_norm_stats['means'].shape}")
    print(f"  Global stds: {global_norm_stats['stds'].flatten()}")
    del global_ds

    # Evaluate ALL subjects (both train and test, reported separately)
    all_results = []
    start = time.time()

    for i, (sid, tasks) in enumerate(sorted(subj_tasks.items())):
        result = evaluate_subject(model, dataset_path, sid, device, global_norm_stats)
        if result:
            result['in_training'] = sid in train_subjects
            result['tasks'] = sorted(tasks)
            all_results.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  Evaluated {i+1}/{len(subj_tasks)} subjects ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"\nEvaluation complete: {len(all_results)} subjects in {elapsed:.0f}s")

    # ========================
    # REPORT: Test subjects only (unseen during training)
    # ========================
    print(f"\n{'='*70}")
    print("LOSO RESULTS — TEST SUBJECTS ONLY (unseen during training)")
    print(f"{'='*70}")

    test_results = [r for r in all_results if not r['in_training']]
    _report_summary(test_results, "Test")

    # ========================
    # REPORT: ALL subjects
    # ========================
    print(f"\n{'='*70}")
    print("LOSO RESULTS — ALL SUBJECTS")
    print(f"{'='*70}")

    _report_summary(all_results, "All")

    # ========================
    # REPORT: Per-task breakdown by subject
    # ========================
    print(f"\n{'='*70}")
    print("PER-SUBJECT DETAIL")
    print(f"{'='*70}")

    for task in ['stress', 'arrhythmia']:
        task_results = [r for r in all_results if f'{task}_acc' in r]
        print(f"\n  {task.upper()} ({len(task_results)} subjects):")
        print(f"  {'Subject':<12} {'Split':<6} {'Acc':>7} {'AUC':>7} {'Sens':>7} {'Spec':>7} {'N':>6}")
        print(f"  {'-'*55}")
        for r in sorted(task_results, key=lambda x: x[f'{task}_acc'], reverse=True):
            split = "TEST" if not r['in_training'] else "train"
            acc = r[f'{task}_acc']
            auc_v = r[f'{task}_auc']
            sens = r[f'{task}_sensitivity']
            spec = r[f'{task}_specificity']
            n = r[f'{task}_n_samples']
            auc_str = f"{auc_v:.3f}" if not np.isnan(auc_v) else "  N/A"
            sens_str = f"{sens*100:.1f}%" if not np.isnan(sens) else "  N/A"
            spec_str = f"{spec*100:.1f}%" if not np.isnan(spec) else "  N/A"
            print(f"  {r['subject']:<12} {split:<6} {acc*100:>6.1f}% {auc_str:>7} {sens_str:>7} {spec_str:>7} {n:>6}")

    # Save
    save_data = {
        'model': str(model_path),
        'n_subjects': len(all_results),
        'n_test_subjects': len(test_results),
        'results': all_results,
    }

    # Compute summary stats for saving
    for group_name, group_results in [('test', test_results), ('all', all_results)]:
        summary = {}
        for task in ['activity', 'stress', 'arrhythmia']:
            for metric in ['acc', 'f1_macro', 'f1_weighted', 'auc', 'sensitivity', 'specificity']:
                key = f'{task}_{metric}'
                vals = [r[key] for r in group_results if key in r and not np.isnan(r[key])]
                if vals:
                    summary[f'{key}_mean'] = float(np.mean(vals))
                    summary[f'{key}_std'] = float(np.std(vals))
                    summary[f'{key}_n'] = len(vals)
        save_data[f'{group_name}_summary'] = summary

    results_path = PROJECT_ROOT / 'training_results' / 'loso_eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


def _report_summary(results, label):
    """Print summary for a group of results."""
    for task in ['activity', 'stress', 'arrhythmia']:
        task_results = [r for r in results if f'{task}_acc' in r]
        if not task_results:
            continue

        accs = [r[f'{task}_acc'] for r in task_results]
        f1ms = [r[f'{task}_f1_macro'] for r in task_results]
        f1ws = [r[f'{task}_f1_weighted'] for r in task_results]
        aucs = [r[f'{task}_auc'] for r in task_results if not np.isnan(r[f'{task}_auc'])]
        sens_vals = [r[f'{task}_sensitivity'] for r in task_results if not np.isnan(r.get(f'{task}_sensitivity', float('nan')))]
        spec_vals = [r[f'{task}_specificity'] for r in task_results if not np.isnan(r.get(f'{task}_specificity', float('nan')))]

        total_samples = sum(r[f'{task}_n_samples'] for r in task_results)

        print(f"\n  {task.upper()} ({len(task_results)} subjects, {total_samples} windows):")
        print(f"    Accuracy:    {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
        print(f"    F1-weighted: {np.mean(f1ws)*100:.1f}% ± {np.std(f1ws)*100:.1f}%")
        print(f"    F1-macro:    {np.mean(f1ms)*100:.1f}% ± {np.std(f1ms)*100:.1f}%")
        if aucs:
            print(f"    AUC:         {np.mean(aucs):.3f} ± {np.std(aucs):.3f} ({len(aucs)} subjects with both classes)")
        if sens_vals:
            print(f"    Sensitivity: {np.mean(sens_vals)*100:.1f}% ± {np.std(sens_vals)*100:.1f}%")
        if spec_vals:
            print(f"    Specificity: {np.mean(spec_vals)*100:.1f}% ± {np.std(spec_vals)*100:.1f}%")


if __name__ == '__main__':
    main()
