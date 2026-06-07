"""
Create Synthetic/Constructed Test Samples for 4 Alert Cases

This script generates *synthetic (constructed)* windows for Cases 1-4 by taking a real
window as a base and applying controlled perturbations to create:

- Case 1: Stress + Sedentary (expected: psychological_stress)
- Case 2: Stress + Exercise  (expected: no_alert)
- Case 3: Arrhythmia + Sedentary (expected: arrhythmia_detected)
- Case 4: Arrhythmia + High Motion (expected: critical_alert)

The output format matches what `scripts/test_accuracy_4_cases.py` expects:
- `window_data`: numpy array shaped [11, 1000]
- `labels`: one-hot vectors for activity/stress/arrhythmia
- `metadata`: includes `case` and `expected_alert` (plus extra helpful fields)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make paths work no matter where you run from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pickle
import json
from datetime import datetime
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')


class RealisticTestSampleGenerator:
    """Generate realistic test samples that work with the trained model"""

    def __init__(self, original_dataset_path: str = None, seed: int = 42, profile: str = "realistic"):
        """
        Args:
            original_dataset_path: Path to a dataset .pkl to use as a *base* for constructing samples.
                If None, tries common candidates in priority order.
            seed: Random seed for reproducible generation.
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.profile = profile

        if original_dataset_path is None:
            candidates = [
                Path('processed_unified_dataset/cleaned_unified_dataset.pkl'),
                Path('processed_unified_dataset/balanced_unified_dataset.pkl'),
                Path('processed_unified_dataset/unified_dataset.pkl'),
            ]
            found = next((p for p in candidates if p.exists()), None)
            if found is None:
                raise FileNotFoundError(
                    "Could not find a base dataset. Tried:\n"
                    + "\n".join([f"- {p}" for p in candidates])
                )
            self.original_dataset_path = str(found)
        else:
            self.original_dataset_path = str(original_dataset_path)
        self.original_dataset = None

        # Activity classes
        self.activity_classes = [
            'Sitting', 'Walking', 'Cycling', 'Driving',
            'Working', 'Stairs', 'Table Soccer', 'Lunch'
        ]
        self.stress_classes = ['Baseline', 'Stress', 'Amusement', 'Meditation']
        self.arrhythmia_classes = ['Normal', 'Abnormal']

        # Profile controls how "hard" the synthetic perturbations are.
        # - realistic: mild, closer to training distribution
        # - stress_test: stronger perturbations (can be out-of-distribution)
        if self.profile not in ("realistic", "stress_test"):
            raise ValueError(f"Unknown profile: {self.profile}. Use 'realistic' or 'stress_test'.")

        # Load original dataset for reference
        self._load_original_dataset()

    def _load_original_dataset(self):
        """Load original dataset to understand the data structure"""
        print("🔄 Loading original dataset for reference...")
        with open(self.original_dataset_path, 'rb') as f:
            self.original_dataset = pickle.load(f)
        print(f"✅ Loaded {len(self.original_dataset)} original samples")
        print(f"📁 Base dataset: {self.original_dataset_path}")

        # Analyze the data structure
        sample = self.original_dataset[0]
        print(f"📊 Original data structure:")
        print(f"  window_data shape: {sample['window_data'].shape}")
        print(f"  labels keys: {list(sample['labels'].keys())}")
        print(f"  activity shape: {sample['labels']['activity'].shape}")
        print(f"  stress shape: {sample['labels']['stress'].shape}")
        print(f"  arrhythmia shape: {sample['labels']['arrhythmia'].shape}")

        # Build fast index lookups for in-distribution sampling.
        # We primarily key by (stress_id, arrhythmia_id) since those define the alert cases.
        self._index_by_stress_arrhythmia = {}
        self._index_by_dataset_arrhythmia = {}  # Also index by dataset for MIT-BIH lookup
        for i, s in enumerate(self.original_dataset):
            try:
                labels = s.get('labels', {})
                stress_i = int(np.argmax(labels['stress']))
                arr_i = int(np.argmax(labels['arrhythmia']))
                ds = str(s.get('dataset', '')).lower()
                self._index_by_stress_arrhythmia.setdefault((stress_i, arr_i), []).append(i)
                self._index_by_dataset_arrhythmia.setdefault((ds, arr_i), []).append(i)
            except Exception:
                continue

        print("📌 Base-sample availability by (stress, arrhythmia):")
        for key in sorted(self._index_by_stress_arrhythmia.keys()):
            print(f"  {key}: {len(self._index_by_stress_arrhythmia[key])}")

        # Count MIT-BIH abnormal samples specifically
        mitbih_abnormal = sum(
            1
            for i, s in enumerate(self.original_dataset)
            if 'mit' in str(s.get('dataset', '')).lower() or 'bih' in str(s.get('dataset', '')).lower()
            if int(np.argmax(s.get('labels', {}).get('arrhythmia', [1, 0]))) == 1
        )
        print(f"📌 MIT-BIH abnormal samples available: {mitbih_abnormal}")

    def create_realistic_sample(
        self,
        activity_id: int,
        stress_id: int,
        arrhythmia_id: int,
        prefer_mitbih_for_arrhythmia: bool = True,
    ) -> Dict:
        """
        Create a synthetic/constructed sample.

        Key idea: keep things in-distribution by sampling a *base* window that already matches
        the target (stress_id, arrhythmia_id) whenever possible, then only adjust motion
        (and lightly adjust other channels if needed).

        For arrhythmia cases, prefer MIT-BIH samples which have real arrhythmia patterns.
        """

        # For abnormal arrhythmia, strongly prefer MIT-BIH samples
        if arrhythmia_id == 1 and prefer_mitbih_for_arrhythmia:
            # Look for MIT-BIH abnormal samples first
            mitbih_candidates = []
            for i, s in enumerate(self.original_dataset):
                ds = str(s.get('dataset', '')).lower()
                if 'mit' in ds or 'bih' in ds:
                    labels = s.get('labels', {})
                    if int(np.argmax(labels.get('arrhythmia', [1, 0]))) == 1:
                        mitbih_candidates.append(i)

            if mitbih_candidates:
                base_idx = int(self.rng.choice(mitbih_candidates))
            else:
                # Fall back to any abnormal sample
                candidates = self._index_by_stress_arrhythmia.get((int(stress_id), int(arrhythmia_id)), [])
                if candidates:
                    base_idx = int(self.rng.choice(candidates))
                else:
                    base_idx = int(self.rng.integers(0, len(self.original_dataset)))
        else:
            # Prefer in-distribution base sample with matching stress/arrhythmia.
            candidates = self._index_by_stress_arrhythmia.get((int(stress_id), int(arrhythmia_id)), [])
            if candidates:
                base_idx = int(self.rng.choice(candidates))
            else:
                base_idx = int(self.rng.integers(0, len(self.original_dataset)))

        base_sample = self.original_dataset[base_idx]
        base_sensor_data = base_sample['window_data'].copy()
        base_labels = base_sample.get('labels', {})
        base_stress_id = int(np.argmax(base_labels.get('stress', np.zeros(4)))) if 'stress' in base_labels else -1
        base_arrhythmia_id = int(np.argmax(base_labels.get('arrhythmia', np.zeros(2)))) if 'arrhythmia' in base_labels else -1
        base_dataset = str(base_sample.get('dataset', '')).lower()

        # If base sample is from MIT-BIH and has real arrhythmia, don't add synthetic arrhythmia patterns
        skip_arrhythmia_mod = (
            arrhythmia_id == 1 and
            base_arrhythmia_id == 1 and
            ('mit' in base_dataset or 'bih' in base_dataset)
        )

        # Modify the sensor data based on the desired labels
        modified_sensor_data = self._modify_sensor_data(
            base_sensor_data,
            activity_id,
            stress_id,
            arrhythmia_id,
            base_stress_id=base_stress_id,
            base_arrhythmia_id=base_arrhythmia_id,
            skip_arrhythmia_modification=skip_arrhythmia_mod,
        )

        # Create labels
        activity_label = np.zeros(8)
        activity_label[activity_id] = 1

        stress_label = np.zeros(4)
        stress_label[stress_id] = 1

        arrhythmia_label = np.zeros(2)
        arrhythmia_label[arrhythmia_id] = 1

        # Create sample
        sample = {
            'window_data': modified_sensor_data,
            'labels': {
                'activity': activity_label,
                'stress': stress_label,
                'arrhythmia': arrhythmia_label
            },
            'metadata': {
                # These are filled by `generate_case_samples`:
                # - case
                # - expected_alert
                # And we always include explicit label ids/names for debugging.
                'activity_id': int(activity_id),
                'activity_name': self.activity_classes[int(activity_id)],
                'stress_id': int(stress_id),
                'stress_name': self.stress_classes[int(stress_id)],
                'arrhythmia_id': int(arrhythmia_id),
                'arrhythmia_name': self.arrhythmia_classes[int(arrhythmia_id)],
                'synthetic': True,
                'base_dataset_path': self.original_dataset_path,
                'base_sample_index': int(base_idx),
                'base_stress_id': int(base_stress_id),
                'base_arrhythmia_id': int(base_arrhythmia_id),
            }
        }

        return sample

    def _modify_sensor_data(
        self,
        base_data: np.ndarray,
        activity_id: int,
        stress_id: int,
        arrhythmia_id: int,
        base_stress_id: int,
        base_arrhythmia_id: int,
        skip_arrhythmia_modification: bool = False,
    ) -> np.ndarray:
        """Modify sensor data to reflect the desired activity, stress, and arrhythmia

        Args:
            skip_arrhythmia_modification: If True, don't modify ECG/PPG for arrhythmia
                (useful when base sample already has real arrhythmia pattern from MIT-BIH)
        """
        modified_data = base_data.copy()
        n = modified_data.shape[1]

        # Activity modifications - each activity has a DISTINCT motion signature
        # This is critical for the model to learn activity-specific patterns
        t = np.arange(n) / 100.0  # 100 Hz, 10 seconds

        if activity_id == 1:  # Walking - regular, rhythmic gait pattern ~1.5-2.5 Hz
            if self.profile == "realistic":
                motion_factor = 1.2 + float(self.rng.uniform(0, 0.3))
                noise_std = 0.06
            else:
                motion_factor = 1.5 + float(self.rng.uniform(0, 0.5))
                noise_std = 0.10
            # Walking has a characteristic ~2 Hz gait frequency with vertical dominance
            gait_freq = 1.8 + float(self.rng.uniform(-0.3, 0.3))
            phase = float(self.rng.uniform(0, 2 * np.pi))
            # Vertical (Z) dominant, with smaller lateral (X, Y) components
            modified_data[2] = 0.1 * modified_data[2] + motion_factor * 0.4 * np.sin(2 * np.pi * gait_freq * t + phase)
            modified_data[3] = 0.1 * modified_data[3] + motion_factor * 0.3 * np.sin(2 * np.pi * gait_freq * t + phase + np.pi / 4)
            modified_data[4] = 0.1 * modified_data[4] + motion_factor * 1.0 * np.sin(2 * np.pi * gait_freq * t + phase)  # Z dominant
            modified_data[2:5] += self.rng.normal(0, noise_std, (3, n))
            # Slight respiration increase
            modified_data[6] += 0.02 * np.sin(2 * np.pi * 0.25 * t)

        elif activity_id == 2:  # Cycling - smooth, periodic leg motion ~1-1.5 Hz (cadence)
            if self.profile == "realistic":
                motion_factor = 1.0 + float(self.rng.uniform(0, 0.3))
                noise_std = 0.04
            else:
                motion_factor = 1.3 + float(self.rng.uniform(0, 0.5))
                noise_std = 0.08
            # Cycling has lower, smoother frequency from pedaling, less vertical bounce
            cadence_freq = 1.2 + float(self.rng.uniform(-0.2, 0.2))
            phase = float(self.rng.uniform(0, 2 * np.pi))
            # More circular motion in X-Y plane, less Z
            modified_data[2] = 0.1 * modified_data[2] + motion_factor * 0.6 * np.sin(2 * np.pi * cadence_freq * t + phase)
            modified_data[3] = 0.1 * modified_data[3] + motion_factor * 0.6 * np.cos(2 * np.pi * cadence_freq * t + phase)
            modified_data[4] = 0.1 * modified_data[4] + motion_factor * 0.3 * np.sin(2 * np.pi * cadence_freq * 2 * t)  # Small Z
            modified_data[2:5] += self.rng.normal(0, noise_std, (3, n))
            # Higher respiration for cardio
            modified_data[6] += 0.04 * np.sin(2 * np.pi * 0.30 * t)

        elif activity_id == 5:  # Stairs - irregular, higher impact ~1-2 Hz with asymmetry
            if self.profile == "realistic":
                motion_factor = 1.5 + float(self.rng.uniform(0, 0.4))
                noise_std = 0.10
            else:
                motion_factor = 1.8 + float(self.rng.uniform(0, 0.6))
                noise_std = 0.15
            # Stairs have irregular rhythm with high vertical acceleration (step impacts)
            step_freq = 1.5 + float(self.rng.uniform(-0.3, 0.3))
            phase = float(self.rng.uniform(0, 2 * np.pi))
            # High Z (vertical) with asymmetric pattern (step-by-step variation)
            base_pattern = np.sin(2 * np.pi * step_freq * t + phase)
            # Add asymmetry (different left/right step)
            asymmetry = 0.3 * np.sin(2 * np.pi * step_freq / 2 * t + phase)
            modified_data[2] = 0.1 * modified_data[2] + motion_factor * 0.5 * (base_pattern + asymmetry)
            modified_data[3] = 0.1 * modified_data[3] + motion_factor * 0.4 * base_pattern
            modified_data[4] = 0.1 * modified_data[4] + motion_factor * 1.2 * np.abs(base_pattern)  # High impact Z
            modified_data[2:5] += self.rng.normal(0, noise_std, (3, n))
            # Higher respiration for effort
            modified_data[6] += 0.05 * np.sin(2 * np.pi * 0.35 * t)

        elif activity_id == 0:  # Sitting - minimal movement, relaxed posture
            # Very low, random noise - baseline sedentary
            modified_data[2:5] = 0.05 * modified_data[2:5]
            modified_data[2:5] += self.rng.normal(0, 0.008, (3, n))
            # Slow, relaxed respiration
            modified_data[6] = 0.8 * modified_data[6] + 0.01 * np.sin(2 * np.pi * 0.20 * t)

        elif activity_id == 3:  # Driving - vehicle vibration + steering micro-movements
            # Characteristic low-frequency vibration from vehicle (~10-15 Hz engine, but aliased/filtered)
            # Plus very subtle steering adjustments
            vehicle_vib = 0.03 * np.sin(2 * np.pi * 8.0 * t) + 0.02 * np.sin(2 * np.pi * 12.0 * t)
            steering_adj = 0.015 * np.sin(2 * np.pi * 0.1 * t + float(self.rng.uniform(0, np.pi)))
            modified_data[2] = 0.05 * modified_data[2] + vehicle_vib + steering_adj
            modified_data[3] = 0.05 * modified_data[3] + 0.8 * vehicle_vib
            modified_data[4] = 0.05 * modified_data[4] + 1.2 * vehicle_vib  # More Z from road
            modified_data[2:5] += self.rng.normal(0, 0.012, (3, n))
            # Slightly elevated alertness (subtle HR increase through respiration proxy)
            modified_data[6] = 0.9 * modified_data[6] + 0.015 * np.sin(2 * np.pi * 0.22 * t)

        elif activity_id == 4:  # Working - typing/mouse micro-movements
            # Characteristic typing rhythm ~2-4 Hz (finger movements), very low amplitude
            typing_freq = 3.0 + float(self.rng.uniform(-0.5, 0.5))
            typing_pattern = 0.025 * np.sin(2 * np.pi * typing_freq * t)
            # Intermittent bursts (typing in spurts)
            burst = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)  # ~20 sec cycles
            modified_data[2] = 0.05 * modified_data[2] + burst * typing_pattern
            modified_data[3] = 0.05 * modified_data[3] + 0.7 * burst * typing_pattern
            modified_data[4] = 0.05 * modified_data[4]  # Minimal Z
            modified_data[2:5] += self.rng.normal(0, 0.010, (3, n))
            # Focused work - slightly faster respiration
            modified_data[6] = 0.85 * modified_data[6] + 0.012 * np.sin(2 * np.pi * 0.23 * t)

        elif activity_id == 7:  # Lunch - eating movements (hand-to-mouth cycles)
            # Characteristic eating rhythm ~0.3-0.5 Hz (bites/chews), with arm movements
            eating_freq = 0.4 + float(self.rng.uniform(-0.1, 0.1))
            chewing_freq = 1.5 + float(self.rng.uniform(-0.3, 0.3))
            # Arm movement to mouth
            arm_motion = 0.04 * np.sin(2 * np.pi * eating_freq * t)
            # Subtle chewing vibration
            chew_motion = 0.015 * np.sin(2 * np.pi * chewing_freq * t)
            modified_data[2] = 0.05 * modified_data[2] + arm_motion + 0.5 * chew_motion
            modified_data[3] = 0.05 * modified_data[3] + 0.8 * arm_motion
            modified_data[4] = 0.05 * modified_data[4] + 0.3 * arm_motion  # Some Z from arm lift
            modified_data[2:5] += self.rng.normal(0, 0.015, (3, n))
            # Eating - relaxed respiration, possibly slightly irregular from swallowing
            modified_data[6] = 0.85 * modified_data[6] + 0.018 * np.sin(2 * np.pi * 0.18 * t)

        elif activity_id == 6:  # Table Soccer - high frequency, erratic, burst-like motion
            if self.profile == "realistic":
                motion_factor = 1.7 + float(self.rng.uniform(0, 0.5))
                noise_std = 0.12
            else:
                motion_factor = 2.2 + float(self.rng.uniform(0, 0.8))
                noise_std = 0.18
            # Table soccer has HIGH frequency (arm movements 3-6 Hz), ERRATIC pattern
            # Mix of multiple frequencies to create chaotic appearance
            freq1 = 3.5 + float(self.rng.uniform(-0.5, 0.5))
            freq2 = 5.0 + float(self.rng.uniform(-0.5, 0.5))
            phase1 = float(self.rng.uniform(0, 2 * np.pi))
            phase2 = float(self.rng.uniform(0, 2 * np.pi))
            # Erratic multi-frequency pattern with bursts
            burst_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)  # Activity bursts
            pattern1 = np.sin(2 * np.pi * freq1 * t + phase1)
            pattern2 = np.sin(2 * np.pi * freq2 * t + phase2)
            modified_data[2] = 0.1 * modified_data[2] + motion_factor * burst_envelope * (0.6 * pattern1 + 0.4 * pattern2)
            modified_data[3] = 0.1 * modified_data[3] + motion_factor * burst_envelope * (0.5 * pattern1 + 0.5 * pattern2)
            modified_data[4] = 0.1 * modified_data[4] + motion_factor * 0.4 * burst_envelope * pattern1  # Less Z
            modified_data[2:5] += self.rng.normal(0, noise_std, (3, n))

        # Stress modifications (only if base does NOT already match target)
        if stress_id == 1 and base_stress_id != 1:  # Stress
            # IMPORTANT: keep stress from looking like arrhythmia.
            # Stress should mainly affect EDA/respiration and add only mild ECG/PPG noise.
            if self.profile == "realistic":
                ecg_noise = 0.02
                ppg_noise = 0.02
                eda_shift = float(self.rng.uniform(0.05, 0.15))
                eda_noise = 0.02
                resp_amp = 0.03
            else:
                ecg_noise = 0.08
                ppg_noise = 0.08
                eda_shift = float(self.rng.uniform(0.10, 0.25))
                eda_noise = 0.05
                resp_amp = 0.06

            modified_data[0] += self.rng.normal(0, ecg_noise, n)  # ECG
            modified_data[1] += self.rng.normal(0, ppg_noise, n)  # PPG

            # EDA channel (index 5): upward shift + noise
            modified_data[5] += eda_shift
            modified_data[5] += self.rng.normal(0, eda_noise, n)

            # Respiration channel (index 6): slightly faster / higher amplitude breathing
            t = np.arange(n) / 100.0
            modified_data[6] += resp_amp * np.sin(2 * np.pi * (0.30 + float(self.rng.uniform(-0.05, 0.05))) * t)

        elif stress_id == 0:  # Baseline
            # Keep baseline characteristics
            pass

        elif stress_id == 2:  # Amusement
            # Slightly elevated but regular
            modified_data[0] *= 1.1
            modified_data[1] *= 1.1

        elif stress_id == 3:  # Meditation
            # Reduced variability
            modified_data[0] *= 0.9
            modified_data[1] *= 0.9

        # Arrhythmia modifications (only if base does NOT already match target AND not skipped)
        if arrhythmia_id == 1 and base_arrhythmia_id != 1 and not skip_arrhythmia_modification:  # Abnormal
            # Abnormal rhythm: sparse ectopic-like spikes + mild irregular baseline wander.
            if self.profile == "realistic":
                wander_amp = 0.08
                n_spikes = 10
                ecg_spike = 0.45
                ppg_spike = 0.25
            else:
                wander_amp = 0.15
                n_spikes = 24
                ecg_spike = 0.55
                ppg_spike = 0.35

            t = np.arange(n) / 100.0
            wander = wander_amp * np.sin(2 * np.pi * (0.12 + float(self.rng.uniform(-0.03, 0.03))) * t)
            modified_data[0] += wander
            modified_data[1] += 0.5 * wander

            spike_indices = self.rng.choice(n, size=n_spikes, replace=False)
            modified_data[0, spike_indices] += self.rng.normal(0, ecg_spike, n_spikes)
            modified_data[1, spike_indices] += self.rng.normal(0, ppg_spike, n_spikes)

        # Final safety: remove NaN/inf if created by perturbations
        modified_data = np.nan_to_num(modified_data, nan=0.0, posinf=0.0, neginf=0.0)

        return modified_data

    def generate_case_samples(self, case_name: str, num_samples: int) -> List[Dict]:
        """Generate samples for a specific case"""
        samples = []

        if case_name == 'case1_stress_sedentary':
            # Case 1: High stress + Sedentary activities
            for i in range(num_samples):
                # Random sedentary activity
                activity_id = int(self.rng.choice([0, 3, 4, 7]))  # Sitting, Driving, Working, Lunch
                stress_id = 1  # Stress
                arrhythmia_id = 0  # Normal

                sample = self.create_realistic_sample(activity_id, stress_id, arrhythmia_id)
                sample['metadata'].update({
                    'case': 'case1_stress_sedentary',
                    'expected_alert': 'psychological_stress',
                })
                samples.append(sample)

        elif case_name == 'case2_stress_exercise':
            # Case 2: High stress + Exercise activities
            for i in range(num_samples):
                # Random exercise activity
                activity_id = int(self.rng.choice([1, 2, 5]))  # Walking, Cycling, Stairs
                stress_id = 1  # Stress
                arrhythmia_id = 0  # Normal

                sample = self.create_realistic_sample(activity_id, stress_id, arrhythmia_id)
                sample['metadata'].update({
                    'case': 'case2_stress_exercise',
                    'expected_alert': 'no_alert',
                })
                samples.append(sample)

        elif case_name == 'case3_arrhythmia_sedentary':
            # Case 3: Arrhythmia + Sedentary
            for i in range(num_samples):
                # Random sedentary activity
                activity_id = int(self.rng.choice([0, 3, 4, 7]))  # Sitting, Driving, Working, Lunch
                stress_id = 0  # Baseline (stress doesn't matter for arrhythmia alert)
                arrhythmia_id = 1  # Abnormal

                sample = self.create_realistic_sample(activity_id, stress_id, arrhythmia_id)
                sample['metadata'].update({
                    'case': 'case3_arrhythmia_sedentary',
                    'expected_alert': 'arrhythmia_detected',
                })
                samples.append(sample)

        elif case_name == 'case4_arrhythmia_motion':
            # Case 4: Arrhythmia + High motion
            for i in range(num_samples):
                # Random high motion activity
                activity_id = int(self.rng.choice([1, 2, 5, 6]))  # Walking, Cycling, Stairs, Table Soccer
                stress_id = 0  # Baseline (moderate stress)
                arrhythmia_id = 1  # Abnormal

                sample = self.create_realistic_sample(activity_id, stress_id, arrhythmia_id)
                sample['metadata'].update({
                    'case': 'case4_arrhythmia_motion',
                    'expected_alert': 'critical_alert',
                })
                samples.append(sample)

        return samples

    def generate_all_test_samples(self, samples_per_case: int = 250):
        """Generate all test samples for the 4 cases"""
        print(f"🔄 Generating {samples_per_case * 4} realistic test samples...")

        all_samples = []
        case_names = [
            'case1_stress_sedentary',
            'case2_stress_exercise',
            'case3_arrhythmia_sedentary',
            'case4_arrhythmia_motion'
        ]

        for case_name in case_names:
            print(f"  Generating {samples_per_case} samples for {case_name}...")
            case_samples = self.generate_case_samples(case_name, samples_per_case)
            all_samples.extend(case_samples)

        print(f"✅ Generated {len(all_samples)} total samples")
        return all_samples

    def save_test_samples(self, samples: List[Dict], filename: str = 'realistic_test_samples_1000.pkl'):
        """Save test samples to file"""
        print(f"💾 Saving test samples to {filename}...")

        with open(filename, 'wb') as f:
            pickle.dump(samples, f)

        print(f"✅ Saved {len(samples)} samples to {filename}")

        # Also save metadata
        metadata = {
            'total_samples': len(samples),
            'samples_per_case': len(samples) // 4,
            'cases': [
                'case1_stress_sedentary',
                'case2_stress_exercise',
                'case3_arrhythmia_sedentary',
                'case4_arrhythmia_motion'
            ],
            'synthetic': True,
            'generator': 'create_realistic_test_samples.py',
            'base_dataset_path': self.original_dataset_path,
            'seed': self.seed,
            'generated_at': datetime.now().isoformat(),
            'description': 'Synthetic/constructed test samples for 4 alert cases (base window + perturbations)'
        }

        metadata_filename = filename.replace('.pkl', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved metadata to {metadata_filename}")

    def print_sample_summary(self, samples: List[Dict]):
        """Print summary of generated samples"""
        print(f"\n📊 REALISTIC TEST SAMPLE SUMMARY")
        print("=" * 50)

        case_counts = {}
        for sample in samples:
            case = sample['metadata']['case']
            case_counts[case] = case_counts.get(case, 0) + 1

        for case, count in case_counts.items():
            print(f"  {case}: {count} samples")

        print(f"\nTotal samples: {len(samples)}")

        # Show sample structure
        if samples:
            sample = samples[0]
            print(f"\nSample structure:")
            print(f"  window_data shape: {sample['window_data'].shape}")
            print(f"  activity label: {sample['labels']['activity']}")
            print(f"  stress label: {sample['labels']['stress']}")
            print(f"  arrhythmia label: {sample['labels']['arrhythmia']}")
            print(f"  metadata: {sample['metadata']}")


def main():
    """Main function to generate realistic test samples"""
    print("🧪 GENERATING 1000 REALISTIC TEST SAMPLES FOR 4 ALERT CASES")
    print("=" * 70)
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic/constructed test samples for 4 alert cases')
    parser.add_argument('--base_dataset_path', type=str, default=None,
                        help='Path to base dataset .pkl (optional; will auto-detect if omitted)')
    parser.add_argument('--samples_per_case', type=int, default=250,
                        help='Number of samples per case (default: 250)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--profile', type=str, default='realistic',
                        choices=['realistic', 'stress_test'],
                        help='Perturbation profile: realistic or stress_test (default: realistic)')
    parser.add_argument('--output', type=str, default='realistic_test_samples_1000.pkl',
                        help='Output .pkl filename (default: realistic_test_samples_1000.pkl)')
    args = parser.parse_args()

    generator = RealisticTestSampleGenerator(
        original_dataset_path=args.base_dataset_path,
        seed=args.seed,
        profile=args.profile
    )

    # Generate samples
    samples = generator.generate_all_test_samples(samples_per_case=int(args.samples_per_case))

    # Print summary
    generator.print_sample_summary(samples)

    # Save samples
    generator.save_test_samples(samples, args.output)

    print(f"\n🎉 Realistic test sample generation complete!")
    print(f"   These samples are based on the original dataset structure")
    print(f"   and should work better with the trained model")

    return samples


if __name__ == "__main__":
    main()

