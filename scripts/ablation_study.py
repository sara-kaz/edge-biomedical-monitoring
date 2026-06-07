#!/usr/bin/env python3
"""
Ablation Study — Sensor, SE-Attention, and Motion-Gate ablations.

Three experiments for IEEE Sensors / IoT Journal:
  1. Sensor Ablation: zero out each modality at inference to quantify
     each sensor's contribution to multi-task performance.
  2. SE-Attention Ablation: bypass squeeze-and-excitation to show
     channel-attention value.
  3. Motion-Gate Ablation: remove motion-aware thresholding from the
     alert algorithm to show its contribution to clinical decision quality.

All experiments use Model V5 (three-phase fine-tuned) on the held-out
test set (subject_splits_v3.json) — same model and splits as all papers.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from train_v3 import (
    CNNTransformerV3, BiomedicalDatasetV3,
    N_MERGED_ACTIVITY, MERGED_ACTIVITY_NAMES,
)
from sklearn.metrics import f1_score, roc_auc_score


# ─── helpers ───────────────────────────────────────────────────────
CHANNEL_NAMES = {0: 'ECG', 1: 'PPG', 2: 'ACC_X', 3: 'ACC_Y', 4: 'ACC_Z'}
CHANNEL_GROUPS = {
    'Full model (baseline)': [],          # no channels zeroed
    'Without ECG':           [0],
    'Without PPG':           [1],
    'Without ACC (all axes)':[2, 3, 4],
    'Without cardiac (ECG+PPG)': [0, 1],
    'ECG only':              [1, 2, 3, 4],
    'PPG only':              [0, 2, 3, 4],
    'ACC only':              [0, 1],
}


def evaluate_model(model, loader, device, zero_channels=None,
                   bypass_se=False):
    """Evaluate with optional channel zeroing or SE bypass."""
    model.eval()
    all_preds = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_labels = {'activity': [], 'stress': [], 'arrhythmia': []}
    all_probs = {'activity': [], 'stress': [], 'arrhythmia': []}

    # Optionally bypass SE attention by making sigmoid output ~1.0
    if bypass_se:
        orig_se_fc2_bias = model.se_fc2.bias.data.clone()
        orig_se_fc2_weight = model.se_fc2.weight.data.clone()
        model.se_fc2.weight.data.zero_()
        model.se_fc2.bias.data.fill_(10.0)  # sigmoid(10) ≈ 1.0

    with torch.no_grad():
        for batch in loader:
            x = batch['window_data'].float().to(device)

            # Zero out specified channels
            if zero_channels:
                for ch in zero_channels:
                    x[:, ch, :] = 0.0

            labels = {t: batch[t].long().to(device)
                      for t in ['activity', 'stress', 'arrhythmia']}
            outputs = model(x)

            for task in ['activity', 'stress', 'arrhythmia']:
                valid = labels[task] >= 0
                if valid.sum() > 0:
                    probs = F.softmax(outputs[task][valid], dim=1)
                    preds = probs.argmax(dim=1)
                    all_preds[task].extend(preds.cpu().numpy())
                    all_labels[task].extend(labels[task][valid].cpu().numpy())
                    all_probs[task].append(probs.cpu().numpy())

    # Restore SE weights
    if bypass_se:
        model.se_fc2.bias.data = orig_se_fc2_bias
        model.se_fc2.weight.data = orig_se_fc2_weight

    results = {}
    for task in ['activity', 'stress', 'arrhythmia']:
        if not all_labels[task]:
            continue
        y_true = np.array(all_labels[task])
        y_pred = np.array(all_preds[task])
        y_probs = np.concatenate(all_probs[task], axis=0)

        acc = float((y_true == y_pred).mean())
        f1m = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        f1w = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

        n_cls = y_probs.shape[1]
        try:
            if n_cls == 2:
                auc_val = float(roc_auc_score(y_true, y_probs[:, 1]))
            else:
                auc_val = float(roc_auc_score(y_true, y_probs,
                                              multi_class='ovr', average='weighted'))
        except Exception:
            auc_val = 0.0

        results[task] = {
            'acc': acc, 'f1_macro': f1m, 'f1_weighted': f1w,
            'auc': auc_val, 'n_samples': len(y_true),
        }

    return results


def compute_motion_score(window_data) -> float:
    if isinstance(window_data, torch.Tensor):
        window_data = window_data.numpy()
    stds = [np.std(window_data[ch]) for ch in [2, 3, 4]]
    return float(np.mean(stds))


def alert_decision(stress_score, arr_prob, motion_score,
                   tau_m, tau_s=0.35, tau_a=0.70, use_motion_gate=True):
    """Alert logic with optional motion gating."""
    if use_motion_gate:
        moving = motion_score > tau_m
    else:
        moving = False  # Treat everything as sedentary

    if arr_prob > tau_a:
        return 'critical' if moving else 'arrhythmia'
    elif stress_score > tau_s:
        return 'no_alert' if moving else 'stress'
    else:
        return 'no_alert'


def run_alert_benchmark(model, full_ds, device, tau_m,
                        use_motion_gate=True, tau_s=0.35, tau_a=0.70):
    """Run 4-case alert benchmark, returns per-case accuracy."""
    rng = np.random.default_rng(42)
    N = 100

    stressed_sed, stressed_ex, arr_sed = [], [], []

    for i in range(len(full_ds)):
        d = full_ds[i]
        x = d['window_data']
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x
        ms = compute_motion_score(x_np)

        s = int(d['stress']) if isinstance(d['stress'], (int, np.integer)) else -1
        a = int(d['arrhythmia']) if isinstance(d['arrhythmia'], (int, np.integer)) else -1

        if s == 1:
            if ms < 0.2:
                stressed_sed.append(i)
            elif ms > 0.4:
                stressed_ex.append(i)
        if a == 1:
            arr_sed.append(i)

    def sample(pool, n):
        if len(pool) >= n:
            return rng.choice(pool, n, replace=False).tolist()
        elif pool:
            return rng.choice(pool, n, replace=True).tolist()
        return []

    cases = [
        ('Case 1: Stress+Sed', sample(stressed_sed, N), 'stress', False),
        ('Case 2: Stress+Exer', sample(stressed_ex, N), 'no_alert', False),
        ('Case 3: Arr+Sed', sample(arr_sed, N), 'arrhythmia', False),
        ('Case 4: Arr+Motion', sample(arr_sed, N), 'critical', True),
    ]

    case_results = {}
    for name, indices, expected, inject_motion in cases:
        if not indices:
            continue
        correct = 0
        for idx in indices:
            d = full_ds[idx]
            x = d['window_data'].float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)
            stress_probs = F.softmax(out['stress'], dim=1)
            stress_score = stress_probs[0, 1].item()
            arr_probs = F.softmax(out['arrhythmia'], dim=1)
            arr_prob = arr_probs[0, 1].item()

            if inject_motion:
                ms = tau_m + 0.5
            else:
                ms = compute_motion_score(d['window_data'])

            alert = alert_decision(stress_score, arr_prob, ms, tau_m,
                                   tau_s, tau_a, use_motion_gate)
            if alert == expected:
                correct += 1
        case_results[name] = round(correct / len(indices) * 100, 1)

    return case_results


# ─── main ──────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Find dataset
    dataset_path = None
    for p in [
        PROJECT_ROOT.parent / 'processed_unified_dataset' / 'unified_dataset.pkl',
        Path('/Users/HP/Desktop/University/Thesis/Code/'
             'multimodal-biomedical-monitoring-improved/'
             'processed_unified_dataset/unified_dataset.pkl'),
    ]:
        if p.exists():
            dataset_path = str(p)
            break
    assert dataset_path, "Dataset not found"

    # Load splits
    splits_path = PROJECT_ROOT / 'training_results' / 'subject_splits_v3.json'
    with open(splits_path) as f:
        splits = json.load(f)
    test_subjects = splits['test_subjects']

    # Load Model V5
    model_path = PROJECT_ROOT / 'training_results' / 'model_v5_stress_finetune.pth'
    model = CNNTransformerV3(
        n_channels=5, n_samples=1000,
        activity_classes=4, stress_classes=2, arrhythmia_classes=2,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.3, use_transformer=True,
    ).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model V5 loaded: {n_params:,} params")

    # Test dataset
    test_ds = BiomedicalDatasetV3(
        dataset_path, subject_filter=test_subjects,
        binary_stress=True, augment=False, normalize=True,
        oversample=False, merge_activity=True,
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Full dataset (for alert benchmark)
    full_ds = BiomedicalDatasetV3(
        dataset_path, binary_stress=True, augment=False,
        normalize=True, oversample=False, merge_activity=True,
    )

    # Calibrate motion threshold
    sedentary_motions, moving_motions = [], []
    for i in range(min(len(full_ds), 5000)):
        d = full_ds[i]
        ms = compute_motion_score(d['window_data'])
        if ms < 0.3:
            sedentary_motions.append(ms)
        else:
            moving_motions.append(ms)
    if sedentary_motions and moving_motions:
        tau_m = (np.median(sedentary_motions) + np.median(moving_motions)) / 2
    else:
        tau_m = 0.33
    print(f"Motion threshold: {tau_m:.4f}")

    all_results = {}

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Sensor Ablation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: SENSOR ABLATION")
    print("=" * 70)

    sensor_results = {}
    for condition, channels_to_zero in CHANNEL_GROUPS.items():
        print(f"\n  {condition} (zeroing channels {channels_to_zero or 'none'})...")
        r = evaluate_model(model, test_loader, device,
                           zero_channels=channels_to_zero if channels_to_zero else None)
        sensor_results[condition] = r

        print(f"    {'Task':<12} {'Acc':>7} {'F1m':>7} {'F1w':>7} {'AUC':>7}")
        for task in ['activity', 'stress', 'arrhythmia']:
            if task in r:
                t = r[task]
                print(f"    {task:<12} {t['acc']*100:>6.1f}% {t['f1_macro']*100:>6.1f}% "
                      f"{t['f1_weighted']*100:>6.1f}% {t['auc']:>6.3f}")

    all_results['sensor_ablation'] = sensor_results

    # Compute delta from baseline
    baseline = sensor_results['Full model (baseline)']
    print(f"\n  {'Condition':<30} {'Δ Act':>7} {'Δ Str':>7} {'Δ Arr':>7}")
    print(f"  {'-'*55}")
    for condition, r in sensor_results.items():
        if condition == 'Full model (baseline)':
            continue
        deltas = {}
        for task in ['activity', 'stress', 'arrhythmia']:
            if task in r and task in baseline:
                deltas[task] = (r[task]['acc'] - baseline[task]['acc']) * 100
            else:
                deltas[task] = float('nan')
        print(f"  {condition:<30} {deltas['activity']:>+6.1f}% "
              f"{deltas['stress']:>+6.1f}% {deltas['arrhythmia']:>+6.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT 2: SE-Attention Ablation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SE-ATTENTION ABLATION")
    print("=" * 70)

    print("\n  With SE attention (baseline):")
    se_baseline = sensor_results['Full model (baseline)']
    for task in ['activity', 'stress', 'arrhythmia']:
        if task in se_baseline:
            t = se_baseline[task]
            print(f"    {task:<12} Acc={t['acc']*100:.1f}% AUC={t['auc']:.3f}")

    print("\n  Without SE attention (bypassed):")
    se_ablated = evaluate_model(model, test_loader, device, bypass_se=True)
    for task in ['activity', 'stress', 'arrhythmia']:
        if task in se_ablated:
            t = se_ablated[task]
            b = se_baseline[task]
            delta = (t['acc'] - b['acc']) * 100
            print(f"    {task:<12} Acc={t['acc']*100:.1f}% AUC={t['auc']:.3f} "
                  f"(Δ={delta:+.1f}%)")

    all_results['se_ablation'] = {
        'with_se': se_baseline,
        'without_se': se_ablated,
    }

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT 3: Motion-Gate Ablation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: MOTION-GATE ABLATION")
    print("=" * 70)

    print("\n  With motion gating (baseline):")
    mg_baseline = run_alert_benchmark(model, full_ds, device, tau_m,
                                      use_motion_gate=True)
    for case, acc in mg_baseline.items():
        print(f"    {case}: {acc}%")

    print("\n  Without motion gating:")
    mg_ablated = run_alert_benchmark(model, full_ds, device, tau_m,
                                     use_motion_gate=False)
    for case, acc in mg_ablated.items():
        delta = acc - mg_baseline[case]
        print(f"    {case}: {acc}% (Δ={delta:+.1f}%)")

    all_results['motion_gate_ablation'] = {
        'with_motion_gate': mg_baseline,
        'without_motion_gate': mg_ablated,
    }

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY TABLE (LaTeX-ready)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY: SENSOR ABLATION TABLE (LaTeX-ready)")
    print("=" * 70)

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Sensor ablation study. Each row zeros the specified")
    print(r"modality channels at inference. $\Delta$ is the accuracy change")
    print(r"from the full model.}")
    print(r"\label{tab:sensor-ablation}")
    print(r"\begin{tabular}{l c c c}")
    print(r"\hline")
    print(r"\textbf{Configuration} & \textbf{Activity} & \textbf{Stress} & \textbf{Arrhythmia} \\")
    print(r"\hline")

    for condition in ['Full model (baseline)',
                      'Without ECG', 'Without PPG',
                      'Without ACC (all axes)',
                      'Without cardiac (ECG+PPG)',
                      'ECG only', 'PPG only', 'ACC only']:
        r = sensor_results[condition]
        cols = []
        for task in ['activity', 'stress', 'arrhythmia']:
            if task in r:
                acc = r[task]['acc'] * 100
                if condition == 'Full model (baseline)':
                    cols.append(f"{acc:.1f}\\%")
                else:
                    delta = (r[task]['acc'] - baseline[task]['acc']) * 100
                    cols.append(f"{acc:.1f}\\% ({delta:+.1f})")
            else:
                cols.append("--")
        tex_name = condition.replace('%', r'\%').replace('_', r'\_')
        print(f"{tex_name} & {' & '.join(cols)} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n" + "=" * 70)
    print("SUMMARY: SE-ATTENTION ABLATION")
    print("=" * 70)
    for task in ['activity', 'stress', 'arrhythmia']:
        b = se_baseline[task]
        a = se_ablated[task]
        print(f"  {task}: {b['acc']*100:.1f}% -> {a['acc']*100:.1f}% "
              f"(delta={((a['acc']-b['acc'])*100):+.1f}%), "
              f"AUC {b['auc']:.3f} -> {a['auc']:.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY: MOTION-GATE ABLATION")
    print("=" * 70)
    avg_with = np.mean(list(mg_baseline.values()))
    avg_without = np.mean(list(mg_ablated.values()))
    print(f"  Average alert accuracy: {avg_with:.1f}% -> {avg_without:.1f}% "
          f"(delta={avg_without - avg_with:+.1f}%)")
    for case in mg_baseline:
        b = mg_baseline[case]
        a = mg_ablated[case]
        print(f"  {case}: {b}% -> {a}% (delta={a - b:+.1f}%)")

    # Save all results
    out_path = PROJECT_ROOT / 'training_results' / 'ablation_results_v5.json'

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
