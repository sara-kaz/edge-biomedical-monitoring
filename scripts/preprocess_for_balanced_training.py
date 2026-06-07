"""
Preprocessing Steps Before Training
Creates a balanced dataset from the cleaned unified dataset
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
from collections import defaultdict
import copy
import warnings

warnings.filterwarnings('ignore')


class PreprocessingForBalancedTraining:
    """Preprocess the cleaned unified dataset for balanced training"""

    def __init__(self, dataset_path='processed_unified_dataset/cleaned_unified_dataset.pkl'):
        self.dataset_path = dataset_path
        self.dataset = None

        # Activity classes
        self.activity_classes = [
            'Sitting', 'Walking', 'Cycling', 'Driving',
            'Working', 'Stairs', 'Table Soccer', 'Lunch'
        ]

        # Stress classes
        self.stress_classes = [
            'Baseline', 'Stress', 'Amusement', 'Meditation'
        ]

        # Arrhythmia classes
        self.arrhythmia_classes = ['Normal', 'Abnormal']

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load the cleaned unified dataset"""
        print("🔄 Loading cleaned unified dataset...")
        with open(self.dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        print(f"✅ Loaded {len(self.dataset)} samples")

        # Clean NaN/inf values from dataset
        self._clean_dataset()

        # Analyze original distribution
        self._analyze_original_distribution()

    def _clean_dataset(self):
        """Clean NaN/inf values from all samples in the dataset"""
        print("\n🧹 Cleaning NaN/inf values from dataset...")
        cleaned_count = 0

        for i, sample in enumerate(self.dataset):
            if not isinstance(sample, dict) or 'window_data' not in sample:
                continue

            window_data = sample['window_data']

            # Check if sample has NaN/inf
            if np.isnan(window_data).any() or np.isinf(window_data).any():
                cleaned_count += 1

                # Clean channel-wise: replace NaN/inf with channel mean
                for channel_idx in range(window_data.shape[0]):
                    channel_data = window_data[channel_idx]

                    # Check if channel has NaN/inf
                    if np.isnan(channel_data).any() or np.isinf(channel_data).any():
                        # Calculate mean of valid values
                        valid_values = channel_data[np.isfinite(channel_data)]
                        if len(valid_values) > 0:
                            channel_mean = np.mean(valid_values)
                            # Replace NaN/inf with mean
                            channel_data = np.nan_to_num(
                                channel_data,
                                nan=channel_mean,
                                posinf=channel_mean,
                                neginf=channel_mean,
                            )
                        else:
                            # If all values are invalid, use zeros
                            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)

                        window_data[channel_idx] = channel_data

                # Final safety check - ensure ALL NaN/inf are replaced
                window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)

                # Double-check: verify no NaN/inf remain
                if np.isnan(window_data).any() or np.isinf(window_data).any():
                    # Force replace with zeros if still present
                    window_data = np.where(np.isfinite(window_data), window_data, 0.0)

                # Update sample with cleaned data (make sure it's a copy)
                sample['window_data'] = window_data.copy()

        print(f"✅ Cleaned {cleaned_count} samples with NaN/inf values")
        print(f"   All samples are now clean and ready for training")

    def _analyze_original_distribution(self):
        """Analyze the original class distribution"""
        print("\n📊 Analyzing original class distribution...")

        activity_counts = np.zeros(len(self.activity_classes))
        stress_counts = np.zeros(len(self.stress_classes))
        arrhythmia_counts = np.zeros(len(self.arrhythmia_classes))

        for sample in self.dataset:
            if not isinstance(sample, dict) or 'labels' not in sample:
                continue

            labels = sample.get('labels', {})
            if not all(key in labels for key in ['activity', 'stress', 'arrhythmia']):
                continue

            activity_idx = np.argmax(labels['activity'])
            stress_idx = np.argmax(labels['stress'])
            arrhythmia_idx = np.argmax(labels['arrhythmia'])

            activity_counts[activity_idx] += 1
            stress_counts[stress_idx] += 1
            arrhythmia_counts[arrhythmia_idx] += 1

        print(f"📊 Original Activity Distribution:")
        for i, (class_name, count) in enumerate(zip(self.activity_classes, activity_counts)):
            percentage = (count / len(self.dataset)) * 100
            print(f"  {class_name}: {int(count)} ({percentage:.1f}%)")

        print(f"\n📊 Original Stress Distribution:")
        for i, (class_name, count) in enumerate(zip(self.stress_classes, stress_counts)):
            percentage = (count / len(self.dataset)) * 100
            print(f"  {class_name}: {int(count)} ({percentage:.1f}%)")

        print(f"\n📊 Original Arrhythmia Distribution:")
        for i, (class_name, count) in enumerate(zip(self.arrhythmia_classes, arrhythmia_counts)):
            percentage = (count / len(self.dataset)) * 100
            print(f"  {class_name}: {int(count)} ({percentage:.1f}%)")

        # Store for later use
        self.original_activity_counts = activity_counts
        self.original_stress_counts = stress_counts
        self.original_arrhythmia_counts = arrhythmia_counts

    def step1_smart_resampling(self, target_activity_ratio=0.2, arrhythmia_oversample_factor=3.0):
        """Step 1: Smart resampling based on critical tasks"""
        print(f"\n🔄 Step 1: Smart resampling...")
        print(f"  Target activity ratio: {target_activity_ratio}")
        print(f"  Arrhythmia oversample factor: {arrhythmia_oversample_factor}")

        # Group samples by labels
        activity_groups = defaultdict(list)
        arrhythmia_groups = defaultdict(list)

        for i, sample in enumerate(self.dataset):
            if not isinstance(sample, dict) or 'labels' not in sample:
                continue

            labels = sample.get('labels', {})
            if not all(key in labels for key in ['activity', 'stress', 'arrhythmia']):
                continue

            activity_idx = np.argmax(labels['activity'])
            arrhythmia_idx = np.argmax(labels['arrhythmia'])

            activity_groups[activity_idx].append(i)
            arrhythmia_groups[arrhythmia_idx].append(i)

        # Strategy 1: Aggressively undersample sitting
        sitting_indices = activity_groups[0]  # Sitting is class 0
        target_sitting_count = int(len(sitting_indices) * target_activity_ratio)
        selected_sitting = np.random.choice(sitting_indices, target_sitting_count, replace=False)

        # Strategy 2: Aggressively oversample arrhythmia abnormal cases to get 50/50 balance
        abnormal_indices = arrhythmia_groups[1]  # Abnormal is class 1
        normal_indices = arrhythmia_groups[0]     # Normal is class 0

        # Calculate oversampling needed to get close to 50/50
        normal_count = len(normal_indices)
        abnormal_count = len(abnormal_indices)

        if abnormal_count > 0:
            # Target: approximately equal normal and abnormal (50/50)
            # We need to oversample abnormal to match the normal count
            # But also apply the oversample factor for safety
            base_oversample = int(len(abnormal_indices) * arrhythmia_oversample_factor)
            target_abnormal_count = max(base_oversample, normal_count)

            # If we're still far from 50/50, add even more
            # After adding all samples, we'll have: normal_count + (existing abnormal in other activities)
            # We want: total_normal ≈ total_abnormal
            # So we need: target_abnormal ≈ normal_count (after resampling)
            # Estimate: after resampling, normal samples will be roughly normal_count
            # So we need abnormal_count ≈ normal_count
            target_abnormal_count = max(target_abnormal_count, int(normal_count * 1.2))  # 20% buffer

            print(f"  Normal arrhythmia samples: {normal_count}")
            print(f"  Abnormal arrhythmia samples: {abnormal_count}")
            print(f"  Target abnormal after oversampling: {target_abnormal_count} (targeting ~50/50 balance)")
            oversampled_abnormal = np.random.choice(abnormal_indices, target_abnormal_count, replace=True)
        else:
            oversampled_abnormal = []

        # Combine all samples with better arrhythmia balance
        resampled_indices = []

        # Group samples by activity AND arrhythmia for better balance
        normal_by_activity = defaultdict(list)
        abnormal_by_activity = defaultdict(list)

        for activity_idx in range(len(self.activity_classes)):
            if activity_idx in activity_groups:
                for sample_idx in activity_groups[activity_idx]:
                    sample = self.dataset[sample_idx]
                    if isinstance(sample, dict) and 'labels' in sample:
                        labels = sample.get('labels', {})
                        if 'arrhythmia' in labels:
                            arrhythmia_idx = np.argmax(labels['arrhythmia'])
                            if arrhythmia_idx == 0:  # Normal
                                normal_by_activity[activity_idx].append(sample_idx)
                            else:  # Abnormal
                                abnormal_by_activity[activity_idx].append(sample_idx)

        # Add samples with better balance
        for activity_idx in range(len(self.activity_classes)):
            normal_samples = normal_by_activity.get(activity_idx, [])
            abnormal_samples = abnormal_by_activity.get(activity_idx, [])

            # For sitting (activity 0), only add selected samples
            if activity_idx == 0:
                selected_normal = [s for s in selected_sitting if s in normal_samples]
                selected_abnormal = [s for s in selected_sitting if s in abnormal_samples]
                resampled_indices.extend(selected_normal)
                resampled_indices.extend(selected_abnormal)
            else:
                # For other activities, add all samples but balance arrhythmia
                resampled_indices.extend(normal_samples)
                resampled_indices.extend(abnormal_samples)

        # Add oversampled abnormal cases to improve overall balance
        resampled_indices.extend(oversampled_abnormal)

        # Remove duplicates while preserving order
        resampled_indices = list(dict.fromkeys(resampled_indices))

        print(f"📊 Resampling results:")
        print(f"  Original samples: {len(self.dataset)}")
        print(f"  Resampled samples: {len(resampled_indices)}")
        print(f"  Sitting samples: {len(selected_sitting)} (reduced from {len(sitting_indices)})")
        print(f"  Abnormal arrhythmia: {len(oversampled_abnormal)} (increased from {len(abnormal_indices)})")

        return resampled_indices

    def step2_data_augmentation(self, resampled_indices, target_samples_per_minority_class=1000):
        """Step 2: Data augmentation for minority classes"""
        print(f"\n🔄 Step 2: Data augmentation for minority classes...")

        # Identify minority classes
        minority_activity_classes = []
        for i, count in enumerate(self.original_activity_counts):
            if count < target_samples_per_minority_class and i != 0:  # Exclude sitting
                minority_activity_classes.append(i)

        print(f"📊 Minority activity classes: {[self.activity_classes[i] for i in minority_activity_classes]}")

        # Group samples by activity class
        activity_groups = defaultdict(list)
        for idx in resampled_indices:
            sample = self.dataset[idx]
            if isinstance(sample, dict) and 'labels' in sample:
                labels = sample.get('labels', {})
                if 'activity' in labels:
                    activity_idx = np.argmax(labels['activity'])
                    activity_groups[activity_idx].append(idx)

        # Create augmented samples
        augmented_samples = []
        augmentation_types = ['jitter', 'scaling', 'time_warp', 'rotation']

        for activity_idx in minority_activity_classes:
            if activity_idx in activity_groups:
                original_samples = activity_groups[activity_idx]
                samples_needed = target_samples_per_minority_class - len(original_samples)

                if samples_needed > 0:
                    print(f"  Augmenting {self.activity_classes[activity_idx]}: {len(original_samples)} → {target_samples_per_minority_class}")

                    # Add original samples
                    for idx in original_samples:
                        augmented_samples.append(self.dataset[idx])

                    # Generate augmented samples
                    for _ in range(samples_needed):
                        # Pick a random original sample
                        original_idx = np.random.choice(original_samples)
                        original_sample = self.dataset[original_idx]

                        # Create augmented sample
                        augmented_sample = copy.deepcopy(original_sample)

                        # Apply augmentation
                        aug_type = np.random.choice(augmentation_types)
                        augmented_window_data = self._augment_time_series(
                            original_sample['window_data'], aug_type
                        )

                        # Clean augmented data immediately (augmentation can introduce NaN/inf)
                        if np.isnan(augmented_window_data).any() or np.isinf(augmented_window_data).any():
                            # Clean channel-wise
                            for channel_idx in range(augmented_window_data.shape[0]):
                                channel_data = augmented_window_data[channel_idx]
                                if np.isnan(channel_data).any() or np.isinf(channel_data).any():
                                    valid_values = channel_data[np.isfinite(channel_data)]
                                    if len(valid_values) > 0:
                                        channel_mean = np.mean(valid_values)
                                        channel_data = np.nan_to_num(
                                            channel_data,
                                            nan=channel_mean,
                                            posinf=channel_mean,
                                            neginf=channel_mean,
                                        )
                                    else:
                                        channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
                                    augmented_window_data[channel_idx] = channel_data
                            augmented_window_data = np.nan_to_num(augmented_window_data, nan=0.0, posinf=0.0, neginf=0.0)

                        # Final verification - use np.where for absolute guarantee
                        if np.isnan(augmented_window_data).any() or np.isinf(augmented_window_data).any():
                            augmented_window_data = np.where(np.isfinite(augmented_window_data), augmented_window_data, 0.0)

                        augmented_sample['window_data'] = augmented_window_data.copy()

                        # Add metadata
                        augmented_sample['metadata'] = {
                            'original_idx': original_idx,
                            'augmentation_type': aug_type,
                            'augmented_at': datetime.now().isoformat()
                        }

                        augmented_samples.append(augmented_sample)
                else:
                    # Use original samples if already sufficient
                    for idx in original_samples:
                        augmented_samples.append(self.dataset[idx])

        print(f"✅ Created augmented dataset with {len(augmented_samples)} samples")
        return augmented_samples

    def _augment_time_series(self, signal, augmentation_type='jitter'):
        """Apply data augmentation to time series signals"""
        augmented_signal = signal.copy()

        if augmentation_type == 'jitter':
            # Add Gaussian noise
            noise_std = 0.05 * np.std(signal)
            noise = np.random.normal(0, noise_std, signal.shape)
            augmented_signal = signal + noise

        elif augmentation_type == 'scaling':
            # Scale the signal
            scale_factor = np.random.uniform(0.9, 1.1)
            augmented_signal = signal * scale_factor

        elif augmentation_type == 'time_warp':
            # Time warping
            warp_factor = np.random.uniform(0.9, 1.1)
            if len(signal.shape) == 2:  # Multi-channel
                warped_signal = np.zeros_like(signal)
                for ch in range(signal.shape[0]):
                    indices = np.linspace(0, len(signal[ch]) - 1, int(len(signal[ch]) * warp_factor))
                    if len(indices) <= len(signal[ch]):
                        warped_signal[ch, :len(indices)] = np.interp(indices, np.arange(len(signal[ch])), signal[ch])[:len(indices)]
                    else:
                        warped_signal[ch] = signal[ch]
                augmented_signal = warped_signal
            else:
                indices = np.linspace(0, len(signal) - 1, int(len(signal) * warp_factor))
                if len(indices) <= len(signal):
                    augmented_signal = np.interp(indices, np.arange(len(signal)), signal)[:len(indices)]
                else:
                    augmented_signal = signal

        elif augmentation_type == 'rotation':
            # Rotation for accelerometer data (channels 6-8)
            if len(signal.shape) == 2 and signal.shape[0] >= 9:  # Has IMU data
                rotation_angle = np.random.uniform(-0.1, 0.1)  # Small rotation
                cos_angle, sin_angle = np.cos(rotation_angle), np.sin(rotation_angle)

                # Apply rotation to IMU channels
                for ch in range(6, min(9, signal.shape[0])):
                    if ch + 1 < signal.shape[0]:
                        x, y = signal[ch], signal[ch + 1]
                        signal[ch] = x * cos_angle - y * sin_angle
                        signal[ch + 1] = x * sin_angle + y * cos_angle

        return augmented_signal

    def step3_create_balanced_dataset(self, resampled_indices, augmented_samples):
        """Step 3: Create final balanced dataset"""
        print(f"\n🔄 Step 3: Creating final balanced dataset...")

        # Combine resampled and augmented samples
        balanced_dataset = []

        # Add resampled samples
        for idx in resampled_indices:
            balanced_dataset.append(self.dataset[idx])

        # Add augmented samples
        balanced_dataset.extend(augmented_samples)

        # Final cleaning pass - ensure ALL samples are clean (augmentation might have introduced NaN/inf)
        print("🧹 Final cleaning pass on balanced dataset...")
        cleaned_count = 0
        for sample in balanced_dataset:
            if not isinstance(sample, dict) or 'window_data' not in sample:
                continue

            window_data = sample['window_data']
            if np.isnan(window_data).any() or np.isinf(window_data).any():
                cleaned_count += 1
                # Clean channel-wise
                for channel_idx in range(window_data.shape[0]):
                    channel_data = window_data[channel_idx]
                    if np.isnan(channel_data).any() or np.isinf(channel_data).any():
                        valid_values = channel_data[np.isfinite(channel_data)]
                        if len(valid_values) > 0:
                            channel_mean = np.mean(valid_values)
                            channel_data = np.nan_to_num(
                                channel_data,
                                nan=channel_mean,
                                posinf=channel_mean,
                                neginf=channel_mean,
                            )
                        else:
                            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
                        window_data[channel_idx] = channel_data
                window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)

                # Double-check: verify no NaN/inf remain
                if np.isnan(window_data).any() or np.isinf(window_data).any():
                    # Force replace with zeros if still present
                    window_data = np.where(np.isfinite(window_data), window_data, 0.0)

                sample['window_data'] = window_data.copy()

        if cleaned_count > 0:
            print(f"✅ Cleaned {cleaned_count} additional samples in final pass")
        print(f"✅ All {len(balanced_dataset)} samples are clean")

        # Shuffle the final dataset
        np.random.shuffle(balanced_dataset)

        print(f"✅ Final balanced dataset created with {len(balanced_dataset)} samples")

        # Analyze final distribution
        self._analyze_final_distribution(balanced_dataset)

        return balanced_dataset

    def _analyze_final_distribution(self, balanced_dataset):
        """Analyze the final balanced dataset distribution"""
        print(f"\n📊 Final balanced dataset distribution:")

        activity_counts = np.zeros(len(self.activity_classes))
        stress_counts = np.zeros(len(self.stress_classes))
        arrhythmia_counts = np.zeros(len(self.arrhythmia_classes))

        for sample in balanced_dataset:
            if not isinstance(sample, dict) or 'labels' not in sample:
                continue

            labels = sample.get('labels', {})
            if not all(key in labels for key in ['activity', 'stress', 'arrhythmia']):
                continue

            activity_idx = np.argmax(labels['activity'])
            stress_idx = np.argmax(labels['stress'])
            arrhythmia_idx = np.argmax(labels['arrhythmia'])

            activity_counts[activity_idx] += 1
            stress_counts[stress_idx] += 1
            arrhythmia_counts[arrhythmia_idx] += 1

        print(f"📊 Final Activity Distribution:")
        for i, (class_name, count) in enumerate(zip(self.activity_classes, activity_counts)):
            percentage = (count / len(balanced_dataset)) * 100
            print(f"  {class_name}: {int(count)} ({percentage:.1f}%)")

        print(f"\n📊 Final Stress Distribution:")
        for i, (class_name, count) in enumerate(zip(self.stress_classes, stress_counts)):
            percentage = (count / len(balanced_dataset)) * 100
            print(f"  {class_name}: {int(count)} ({percentage:.1f}%)")

        print(f"\n📊 Final Arrhythmia Distribution:")
        for i, (class_name, count) in enumerate(zip(self.arrhythmia_classes, arrhythmia_counts)):
            percentage = (count / len(balanced_dataset)) * 100
            print(f"  {class_name}: {int(count)} ({percentage:.1f}%)")

    def save_balanced_dataset(self, balanced_dataset, output_path='processed_unified_dataset/balanced_unified_dataset.pkl'):
        """Save the balanced dataset"""
        print(f"\n💾 Saving balanced dataset to {output_path}...")

        with open(output_path, 'wb') as f:
            pickle.dump(balanced_dataset, f)

        # Save metadata
        metadata = {
            'total_samples': len(balanced_dataset),
            'original_samples': len(self.dataset),
            'preprocessing_steps': [
                'Data cleaning (NaN/inf removal with channel-wise mean replacement)',
                'Smart resampling (sitting reduction, arrhythmia oversampling)',
                'Data augmentation for minority classes',
                'Final balanced dataset creation'
            ],
            'created_at': datetime.now().isoformat()
        }

        metadata_path = output_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Balanced dataset saved to {output_path}")
        print(f"✅ Metadata saved to {metadata_path}")

        return output_path, metadata_path

    def run_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("🚀 STARTING PREPROCESSING PIPELINE")
        print("=" * 70)

        # Step 1: Smart resampling
        resampled_indices = self.step1_smart_resampling(
            target_activity_ratio=0.2,  # Reduce sitting to 20%
            arrhythmia_oversample_factor=10.0  # Much more aggressive oversampling for 50/50 balance
        )

        # Step 2: Data augmentation
        augmented_samples = self.step2_data_augmentation(
            resampled_indices,
            target_samples_per_minority_class=1000
        )

        # Step 3: Create balanced dataset
        balanced_dataset = self.step3_create_balanced_dataset(
            resampled_indices, augmented_samples
        )

        # Save balanced dataset
        output_path, metadata_path = self.save_balanced_dataset(balanced_dataset)

        print(f"\n🎯 PREPROCESSING PIPELINE COMPLETE!")
        print(f"  Original dataset: {len(self.dataset)} samples")
        print(f"  Balanced dataset: {len(balanced_dataset)} samples")
        print(f"  Output file: {output_path}")
        print(f"  Metadata file: {metadata_path}")

        return balanced_dataset, output_path, metadata_path


def main():
    """Run the preprocessing pipeline"""
    preprocessor = PreprocessingForBalancedTraining()
    balanced_dataset, output_path, metadata_path = preprocessor.run_preprocessing_pipeline()
    return balanced_dataset, output_path, metadata_path


if __name__ == "__main__":
    main()

