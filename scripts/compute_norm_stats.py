#!/usr/bin/env python3
"""Compute per-channel normalization stats from training dataset."""
import pickle, numpy as np, json, sys

DATASET = '/Users/HP/Desktop/University/Thesis/Code/multimodal-biomedical-monitoring-improved/processed_unified_dataset/unified_dataset.pkl'

print(f"Loading dataset...")
with open(DATASET, 'rb') as f:
    raw_data = pickle.load(f)

print(f"Total samples: {len(raw_data)}")

# Get first sample to understand shape
for s in raw_data:
    wd = s.get('window_data')
    if wd is not None:
        print(f"window_data shape: {wd.shape}, dtype: {wd.dtype}")
        break

# Collect all window data (up to 5000 like training does)
all_data = []
for sample in raw_data:
    wd = sample.get('window_data')
    if wd is not None:
        all_data.append(wd)

print(f"Windows with data: {len(all_data)}")

stacked = np.stack(all_data[:5000])
print(f"Stacked shape: {stacked.shape}")  # should be (N, 5, 1000)

channel_names = ['ECG', 'PPG', 'AccX', 'AccY', 'AccZ']

print("\n=== Per-Channel Training Statistics ===")
print(f"{'Channel':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'P5':>12} {'P95':>12}")
print("-" * 82)

channel_means = np.nanmean(stacked, axis=(0, 2), keepdims=True).squeeze(0)  # (5, 1)
channel_stds = np.nanstd(stacked, axis=(0, 2), keepdims=True).squeeze(0)    # (5, 1)
channel_stds[channel_stds < 1e-6] = 1.0

stats = {}
for ch in range(5):
    ch_data = stacked[:, ch, :].flatten()
    ch_data = ch_data[~np.isnan(ch_data)]
    mean = float(channel_means[ch, 0])
    std = float(channel_stds[ch, 0])
    mn = float(np.min(ch_data)) if len(ch_data) > 0 else float('nan')
    mx = float(np.max(ch_data)) if len(ch_data) > 0 else float('nan')
    p5 = float(np.percentile(ch_data, 5)) if len(ch_data) > 0 else float('nan')
    p95 = float(np.percentile(ch_data, 95)) if len(ch_data) > 0 else float('nan')

    print(f"{channel_names[ch]:<10} {mean:>12.6f} {std:>12.6f} {mn:>12.4f} {mx:>12.4f} {p5:>12.4f} {p95:>12.4f}")

    stats[channel_names[ch]] = {
        'mean': mean, 'std': std, 'min': mn, 'max': mx, 'p5': p5, 'p95': p95
    }

# Also show what happens after normalization
print("\n=== After Global Z-Score (what the model sees) ===")
print(f"{'Channel':<10} {'Mean':>12} {'Std':>12} {'P5':>12} {'P95':>12}")
print("-" * 58)
for ch in range(5):
    ch_data = stacked[:, ch, :].flatten()
    ch_data = ch_data[~np.isnan(ch_data)]
    normed = (ch_data - float(channel_means[ch, 0])) / float(channel_stds[ch, 0])
    print(f"{channel_names[ch]:<10} {np.mean(normed):>12.6f} {np.std(normed):>12.6f} {np.percentile(normed, 5):>12.4f} {np.percentile(normed, 95):>12.4f}")

# Now show what the firmware's hardware sensors produce
print("\n=== What Hardware Sensors Produce (pre-normalization) ===")
print("ECG:  (adc - 2048) / 2048  →  range [-1, 1], typical ~[-0.5, 0.5]")
print("PPG:  (red - 131072) / 131072  →  if red=117934: (117934-131072)/131072 = -0.100")
print("AccX: raw g  →  ~[-0.3, 0.3]")
print("AccY: raw g  →  ~[-0.3, 0.3]")
print("AccZ: raw g  →  ~[0.9, 1.2] (gravity)")

# Save stats
out_path = '/Users/HP/Desktop/University/Thesis/Code/multimodal-biomedical-monitoring/edge_deployment/training_norm_stats.json'
with open(out_path, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"\nSaved to {out_path}")
