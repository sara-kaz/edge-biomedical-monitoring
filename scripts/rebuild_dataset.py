#!/usr/bin/env python3
"""
Rebuild unified dataset with additional stress datasets.

Adds Wearable Acute Stress (PhysioNet, 2025) to the existing
PPG-DaLiA + MIT-BIH + WESAD pipeline.
"""

import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_integration import UnifiedBiomedicalDataProcessor

# Data paths
DATA_DIR = PROJECT_ROOT.parent / "data"
OUTPUT_DIR = PROJECT_ROOT.parent / "multimodal-biomedical-monitoring-improved" / "processed_unified_dataset"

ppg_dalia_path = DATA_DIR / "ppg+dalia" / "PPG_FieldStudy"
mit_bih_path = DATA_DIR / "mit-bih-arrhythmia-database-1.0.0"
wesad_path = DATA_DIR / "WESAD"
wearable_stress_path = DATA_DIR / "wearable_stress_2025" / "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
uci_har_path = DATA_DIR / "UCI HAR Dataset"

print(f"Output: {OUTPUT_DIR}")
print(f"PPG-DaLiA: {ppg_dalia_path} (exists: {ppg_dalia_path.exists()})")
print(f"MIT-BIH: {mit_bih_path} (exists: {mit_bih_path.exists()})")
print(f"WESAD: {wesad_path} (exists: {wesad_path.exists()})")
print(f"Wearable Stress: {wearable_stress_path} (exists: {wearable_stress_path.exists()})")
print(f"UCI HAR: {uci_har_path} (exists: {uci_har_path.exists()})")

processor = UnifiedBiomedicalDataProcessor(output_dir=str(OUTPUT_DIR))

# Use original 3-source configuration (matching the committed combine_all_datasets)
all_windows, summary = processor.combine_all_datasets(
    ppg_dalia_path=str(ppg_dalia_path) if ppg_dalia_path.exists() else None,
    mit_bih_path=str(mit_bih_path) if mit_bih_path.exists() else None,
    wesad_path=str(wesad_path) if wesad_path.exists() else None,
)

# Print stress label distribution
from collections import Counter
import numpy as np

stress_subjects = set()
stress_counts = Counter()
for w in all_windows:
    stress_vec = w['labels'].get('stress', np.zeros(4))
    if stress_vec.sum() > 0:
        stress_idx = int(np.argmax(stress_vec))
        stress_counts[stress_idx] += 1
        stress_subjects.add(w.get('subject_id', 'unknown'))

print(f"\nStress label distribution: {dict(stress_counts)}")
print(f"Subjects with stress labels: {len(stress_subjects)}")
print(f"Stress subjects: {sorted(stress_subjects)}")
