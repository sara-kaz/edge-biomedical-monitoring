"""
#2
Multimodal Biomedical Dataset Integration Pipeline
Following Exact Specifications for Edge Intelligence Thesis Project

Unified Format:
- Sampling Rate: 100 Hz (all signals)
- Window Size: 10 seconds = 1000 samples
- Shape: [channels × 1000 samples]
- Labels: One-hot encoded vectors
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from scipy.interpolate import interp1d
import h5py
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class UnifiedBiomedicalDataProcessor:
    def __init__(self, output_dir="processed_unified_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # UNIFIED SPECIFICATIONS
        self.TARGET_FS = 100  # Hz - unified sampling rate
        self.WINDOW_LENGTH = 10  # seconds
        self.WINDOW_SAMPLES = 1000  # samples (10s × 100Hz)
        self.OVERLAP = 0.5  # 50% overlap
        
        # Unified channel mapping
        self.CHANNEL_MAPPING = {
            'ecg': 0,
            'ppg': 1, 
            'accel_x': 2,
            'accel_y': 3,
            'accel_z': 4
        }
        self.N_CHANNELS = 5
        
        # Label encodings (one-hot)
        self.LABEL_ENCODINGS = {
            'activity': {
                'sitting': 0, 'walking': 1, 'cycling': 2, 'driving': 3, 
                'working': 4, 'stairs': 5, 'table_soccer': 6, 'lunch': 7
            },
            'stress': {
                'baseline': 0, 'stress': 1, 'amusement': 2, 'meditation': 3
            },
            'arrhythmia': {
                'normal': 0, 'abnormal': 1  # Binary classification
            }
        }
        
    def create_empty_window(self):
        """Create empty window with NaN for missing channels"""
        return np.full((self.N_CHANNELS, self.WINDOW_SAMPLES), np.nan)
    
    def resample_signal(self, signal_data, original_fs):
        """Resample signal to unified 100 Hz"""
        if original_fs == self.TARGET_FS:
            return signal_data
        
        # Calculate new length
        new_length = int(len(signal_data) * self.TARGET_FS / original_fs)
        
        # Use scipy.signal.resample for high-quality resampling
        resampled = signal.resample(signal_data, new_length)
        return resampled
    
    def normalize_signal(self, signal_data):
        """Z-score normalization: (x - mean) / std"""
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        
        if std_val > 1e-8:  # Avoid division by zero
            normalized = (signal_data - mean_val) / std_val
        else:
            normalized = signal_data - mean_val
            
        return normalized
    
    def create_windows(self, signals_dict, labels_dict=None):
        """
        Create 10-second windows with 50% overlap
        
        Args:
            signals_dict: {'ecg': array, 'ppg': array, 'accel_x': array, ...}
            labels_dict: {'activity': value, 'stress': value, 'arrhythmia': value}
        
        Returns:
            List of windowed samples
        """
        # Find the longest signal to determine number of windows
        max_length = 0
        for signal_name, signal_data in signals_dict.items():
            if signal_data is not None:
                max_length = max(max_length, len(signal_data))
        
        if max_length < self.WINDOW_SAMPLES:
            return []  # Signal too short
        
        # Calculate window parameters
        step_size = int(self.WINDOW_SAMPLES * (1 - self.OVERLAP))  # 500 samples
        n_windows = (max_length - self.WINDOW_SAMPLES) // step_size + 1
        
        windowed_samples = []
        
        for window_idx in range(n_windows):
            start_idx = window_idx * step_size
            end_idx = start_idx + self.WINDOW_SAMPLES
            
            # Create empty window
            window_data = self.create_empty_window()
            
            # Fill available channels
            for signal_name, signal_data in signals_dict.items():
                if signal_data is not None and signal_name in self.CHANNEL_MAPPING:
                    channel_idx = self.CHANNEL_MAPPING[signal_name]
                    if end_idx <= len(signal_data):
                        window_data[channel_idx, :] = signal_data[start_idx:end_idx]
                    else:
                        # Handle edge case where signal is shorter
                        available_samples = len(signal_data) - start_idx
                        if available_samples > 0:
                            window_data[channel_idx, :available_samples] = signal_data[start_idx:]
            
            # Create one-hot labels
            one_hot_labels = self.create_one_hot_labels(labels_dict)
            
            sample = {
                'window_data': window_data,  # Shape: [5, 1000]
                'labels': one_hot_labels,
                'window_index': window_idx,
                'start_time': start_idx / self.TARGET_FS
            }
            
            windowed_samples.append(sample)
        
        return windowed_samples
    
    def create_one_hot_labels(self, labels_dict):
        """Create one-hot encoded labels"""
        one_hot = {}
        
        if labels_dict is None:
            labels_dict = {}
        
        for label_type, encoding_map in self.LABEL_ENCODINGS.items():
            n_classes = len(encoding_map)
            one_hot_vector = np.zeros(n_classes)
            
            if label_type in labels_dict and labels_dict[label_type] is not None:
                label_value = labels_dict[label_type]
                if label_value in encoding_map:
                    class_idx = encoding_map[label_value]
                    one_hot_vector[class_idx] = 1
            
            one_hot[label_type] = one_hot_vector
        
        return one_hot
    
    def process_ppg_dalia(self, data_path):
        """Process PPG-DaLiA Dataset"""
        print("📊 Processing PPG-DaLiA Dataset...")
        
        all_windows = []
        data_path = Path(data_path)
        
        print(f"Looking for files in: {data_path}")
        pkl_files = list(data_path.glob("S*/S*.pkl"))
        print(f"Found {len(pkl_files)} PKL files")
        
        for pkl_file in pkl_files:
            subject_id = pkl_file.stem
            print(f"  Processing {subject_id}")
            
            with open(pkl_file, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            # Extract signals
            # Note: PPG-DaLiA has chest ECG, wrist PPG/ACC
            signals = {}
            
            # ECG from chest sensor (64 Hz)
            if 'chest' in subject_data['signal']:
                ecg_raw = subject_data['signal']['chest']['ECG'].flatten()
                ecg_resampled = self.resample_signal(ecg_raw, 64)
                signals['ecg'] = self.normalize_signal(ecg_resampled)
            
            # PPG from wrist (64 Hz) 
            ppg_raw = subject_data['signal']['wrist']['BVP'].flatten()
            ppg_resampled = self.resample_signal(ppg_raw, 64)
            signals['ppg'] = self.normalize_signal(ppg_resampled)
            
            # Accelerometer from wrist (32 Hz, 3-axis)
            acc_raw = subject_data['signal']['wrist']['ACC']  # [N, 3]
            acc_x_resampled = self.resample_signal(acc_raw[:, 0], 32)
            acc_y_resampled = self.resample_signal(acc_raw[:, 1], 32)
            acc_z_resampled = self.resample_signal(acc_raw[:, 2], 32)
            
            signals['accel_x'] = self.normalize_signal(acc_x_resampled)
            signals['accel_y'] = self.normalize_signal(acc_y_resampled)
            signals['accel_z'] = self.normalize_signal(acc_z_resampled)
            
            # Extract activity labels
            activity_labels = subject_data.get('activity', None)
            
            # Map activity labels to our standard format
            activity_mapping = {
                0: 'sitting', 1: 'walking', 2: 'cycling', 3: 'driving',
                4: 'working', 5: 'stairs', 6: 'table_soccer', 7: 'lunch'
            }
            
            if activity_labels is not None:
                # Get majority activity for each potential window
                step_size = int(self.WINDOW_SAMPLES * (1 - self.OVERLAP))
                signal_length = len(signals['ppg'])
                
                for start_idx in range(0, signal_length - self.WINDOW_SAMPLES + 1, step_size):
                    end_idx = start_idx + self.WINDOW_SAMPLES
                    
                    # Get activity labels for this window
                    window_activity_indices = np.arange(start_idx, end_idx)
                    window_activities = []
                    
                    for idx in window_activity_indices:
                        if idx < len(activity_labels):
                            activity_code = int(activity_labels[idx])  # Convert to int
                            if activity_code in activity_mapping:
                                window_activities.append(activity_mapping[activity_code])
                    
                    # Use majority vote
                    if window_activities:
                        majority_activity = max(set(window_activities), key=window_activities.count)
                    else:
                        majority_activity = 'sitting'  # default
                    
                    labels = {'activity': majority_activity}
                    
                    # Create window for this segment
                    window_signals = {}
                    for sig_name, sig_data in signals.items():
                        if end_idx <= len(sig_data):
                            window_signals[sig_name] = sig_data[start_idx:end_idx]
                    
                    if len(window_signals) > 0:
                        windows = self.create_windows(window_signals, labels)
                        for window in windows:
                            window['subject_id'] = subject_id
                            window['dataset'] = 'PPG-DaLiA'
                        all_windows.extend(windows)
            else:
                # No activity labels available, create windows without activity labels
                labels = {'activity': None}
                windows = self.create_windows(signals, labels)
                for window in windows:
                    window['subject_id'] = subject_id
                    window['dataset'] = 'PPG-DaLiA'
                all_windows.extend(windows)
        
        print(f"  ✅ PPG-DaLiA: {len(all_windows)} windows created")
        return all_windows
    
    def process_mit_bih(self, data_path):
        """Process MIT-BIH Arrhythmia Dataset"""
        print("Processing MIT-BIH Arrhythmia Dataset...")
        
        all_windows = []
        data_path = Path(data_path)
        
        for dat_file in data_path.glob("*.dat"):
            record_name = dat_file.stem
            if record_name.startswith('.'):
                continue
            
            print(f"  Processing {record_name}")
            
            try:
                import wfdb
                
                # Read record and annotations
                record = wfdb.rdrecord(str(dat_file.parent / record_name))
                annotation = wfdb.rdann(str(dat_file.parent / record_name), 'atr')
                
                # Extract ECG (Lead II preferred, or first available lead)
                ecg_raw = record.p_signal[:, 0]  # First lead
                
                # Resample from 360 Hz to 100 Hz
                ecg_resampled = self.resample_signal(ecg_raw, record.fs)
                ecg_normalized = self.normalize_signal(ecg_resampled)
                
                # Create binary arrhythmia labels (normal vs abnormal)
                arrhythmia_labels = self.create_mit_bih_labels(
                    annotation, len(ecg_normalized), record.fs
                )
                
                # Create windows
                step_size = int(self.WINDOW_SAMPLES * (1 - self.OVERLAP))
                
                for start_idx in range(0, len(ecg_normalized) - self.WINDOW_SAMPLES + 1, step_size):
                    end_idx = start_idx + self.WINDOW_SAMPLES
                    
                    # Get majority arrhythmia label for this window
                    window_labels = arrhythmia_labels[start_idx:end_idx]
                    majority_label = 'abnormal' if np.mean(window_labels) > 0.5 else 'normal'
                    
                    # Create signals dict (only ECG available)
                    signals = {
                        'ecg': ecg_normalized[start_idx:end_idx],
                        'ppg': None,
                        'accel_x': None,
                        'accel_y': None,
                        'accel_z': None
                    }
                    
                    labels = {'arrhythmia': majority_label}
                    
                    windows = self.create_windows(signals, labels)
                    for window in windows:
                        window['subject_id'] = record_name
                        window['dataset'] = 'MIT-BIH'
                    all_windows.extend(windows)
                    
            except ImportError:
                print("  ⚠️  wfdb library not found. Install with: pip install wfdb")
                continue
            except Exception as e:
                print(f"  ❌ Error processing {record_name}: {e}")
                continue
        
        print(f"  ✅ MIT-BIH: {len(all_windows)} windows created")
        return all_windows
    
    def create_mit_bih_labels(self, annotation, signal_length, original_fs):
        """Create binary arrhythmia labels (normal vs abnormal)"""
        
        # AAMI arrhythmia classification
        normal_beats = ['N', 'L', 'R', 'e', 'j']  # Normal beats
        abnormal_beats = ['A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']  # Abnormal
        
        # Create sample-level labels (resampled to 100 Hz)
        labels = np.zeros(signal_length)
        
        for sample, symbol in zip(annotation.sample, annotation.symbol):
            # Convert sample index to 100 Hz
            new_sample = int(sample * self.TARGET_FS / original_fs)
            if new_sample < signal_length:
                if symbol in abnormal_beats:
                    # Mark as abnormal (with context window)
                    start_ctx = max(0, new_sample - 50)  # 0.5s before
                    end_ctx = min(signal_length, new_sample + 50)  # 0.5s after
                    labels[start_ctx:end_ctx] = 1
        
        return labels
    
    def process_wesad(self, data_path):
        """Process WESAD Dataset"""
        print("Processing WESAD Dataset...")
        
        all_windows = []
        data_path = Path(data_path)
        
        print(f"Looking for files in: {data_path}")
        pkl_files = list(data_path.glob("S*/S*.pkl"))
        print(f"Found {len(pkl_files)} PKL files")
        
        for pkl_file in pkl_files:
            subject_id = pkl_file.stem
            print(f"  Processing {subject_id}")
            
            with open(pkl_file, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            # Extract signals
            chest_data = subject_data['signal']['chest']
            wrist_data = subject_data['signal']['wrist']
            
            signals = {}
            
            # ECG from chest (700 Hz)
            ecg_raw = chest_data['ECG'].flatten()
            ecg_resampled = self.resample_signal(ecg_raw, 700)
            signals['ecg'] = self.normalize_signal(ecg_resampled)
            
            # PPG from wrist (64 Hz)
            ppg_raw = wrist_data['BVP'].flatten() 
            ppg_resampled = self.resample_signal(ppg_raw, 64)
            signals['ppg'] = self.normalize_signal(ppg_resampled)
            
            # Accelerometer from wrist (32 Hz, 3-axis)
            acc_raw = wrist_data['ACC']  # [N, 3]
            acc_x_resampled = self.resample_signal(acc_raw[:, 0], 32)
            acc_y_resampled = self.resample_signal(acc_raw[:, 1], 32) 
            acc_z_resampled = self.resample_signal(acc_raw[:, 2], 32)
            
            signals['accel_x'] = self.normalize_signal(acc_x_resampled)
            signals['accel_y'] = self.normalize_signal(acc_y_resampled)
            signals['accel_z'] = self.normalize_signal(acc_z_resampled)
            
            # Extract stress labels (700 Hz)
            stress_labels_raw = subject_data['label'].flatten()
            
            # Map WESAD stress labels: 0=baseline, 1=stress, 2=amusement, 3=meditation
            stress_mapping = {0: 'baseline', 1: 'stress', 2: 'amusement', 3: 'meditation'}
            
            # Resample stress labels to 100 Hz
            stress_labels_resampled = self.resample_signal(stress_labels_raw.astype(float), 700)
            stress_labels_resampled = np.round(stress_labels_resampled).astype(int)
            
            # Create windows
            step_size = int(self.WINDOW_SAMPLES * (1 - self.OVERLAP))
            signal_length = len(signals['ecg'])
            
            for start_idx in range(0, signal_length - self.WINDOW_SAMPLES + 1, step_size):
                end_idx = start_idx + self.WINDOW_SAMPLES
                
                # Get majority stress label for this window
                if start_idx < len(stress_labels_resampled):
                    window_stress_labels = stress_labels_resampled[start_idx:end_idx]
                    majority_stress_code = int(np.median(window_stress_labels))  # Use median
                    
                    if majority_stress_code in stress_mapping:
                        majority_stress = stress_mapping[majority_stress_code]
                    else:
                        majority_stress = 'baseline'  # default
                else:
                    majority_stress = 'baseline'
                
                # Create signals dict for this window
                window_signals = {}
                for sig_name, sig_data in signals.items():
                    if end_idx <= len(sig_data):
                        window_signals[sig_name] = sig_data[start_idx:end_idx]
                
                labels = {'stress': majority_stress}
                
                if len(window_signals) > 0:
                    windows = self.create_windows(window_signals, labels)
                    for window in windows:
                        window['subject_id'] = subject_id
                        window['dataset'] = 'WESAD'
                    all_windows.extend(windows)
        
        print(f"  ✅ WESAD: {len(all_windows)} windows created")
        return all_windows

    def process_wearable_stress(self, data_path):
        """
        Process Wearable Acute Stress dataset (PhysioNet, 2025).

        Empatica E4 signals: BVP (PPG) at 64 Hz, ACC at 32 Hz.
        No ECG — ECG channel will be NaN/zero.

        Protocol phases for v1 (S01-S18):
          Baseline, Stroop, First Rest, TMCT, Second Rest,
          Real Opinion, Opposite Opinion, Subtract
        For v2 (f01-f18):
          Baseline, TMCT, First Rest, Real Opinion,
          Opposite Opinion, Second Rest, Subtract

        Binary stress: Baseline/Rest → 0, Task phases → 1
        """
        from datetime import datetime

        print("Processing Wearable Acute Stress Dataset...")
        data_path = Path(data_path)
        stress_dir = data_path / "Wearable_Dataset" / "STRESS"

        if not stress_dir.exists():
            print(f"  ⚠️ Stress directory not found: {stress_dir}")
            return []

        # Phase definitions: True = stressed task, False = baseline/rest
        v1_phases = [
            ('Baseline', False), ('Stroop', True), ('First Rest', False),
            ('TMCT', True), ('Second Rest', False),
            ('Real Opinion', True), ('Opposite Opinion', True), ('Subtract', True)
        ]
        v2_phases = [
            ('Baseline', False), ('TMCT', True), ('First Rest', False),
            ('Real Opinion', True), ('Opposite Opinion', True),
            ('Second Rest', False), ('Subtract', True)
        ]

        all_windows = []
        subject_dirs = sorted([d for d in stress_dir.iterdir() if d.is_dir()])

        for subj_dir in subject_dirs:
            subject_id = subj_dir.name

            # Skip known bad data
            if subject_id == 'f07':  # PPG sensor was covered
                print(f"  Skipping {subject_id} (PPG sensor covered)")
                continue

            bvp_path = subj_dir / "BVP.csv"
            acc_path = subj_dir / "ACC.csv"
            tags_path = subj_dir / "tags.csv"

            if not all(p.exists() for p in [bvp_path, acc_path, tags_path]):
                print(f"  Skipping {subject_id} (missing files)")
                continue

            try:
                # Read BVP (PPG) — row 0: start time, row 1: sample rate, rest: data
                bvp_lines = bvp_path.read_text().strip().split('\n')
                bvp_start_str = bvp_lines[0].strip()
                bvp_sr = float(bvp_lines[1].strip())
                bvp_data = np.array([float(x) for x in bvp_lines[2:]])
                bvp_start = datetime.strptime(bvp_start_str, "%Y-%m-%d %H:%M:%S")

                # Read ACC (3-axis) — row 0 has 3 comma-separated timestamps
                acc_lines = acc_path.read_text().strip().split('\n')
                # ACC start time: first of the 3 comma-separated timestamps
                acc_start_str = acc_lines[0].strip().split(',')[0].strip()
                acc_sr = float(acc_lines[1].strip().split(',')[0])
                acc_data = []
                for line in acc_lines[2:]:
                    vals = line.strip().split(',')
                    if len(vals) >= 3:
                        acc_data.append([float(v) for v in vals[:3]])
                acc_data = np.array(acc_data)
                acc_start = datetime.strptime(acc_start_str, "%Y-%m-%d %H:%M:%S")

                # Handle S02 duplicated data
                if subject_id == 'S02':
                    max_bvp = min(len(bvp_data), 99090)
                    bvp_data = bvp_data[:max_bvp]
                    max_acc = min(len(acc_data), 49544)
                    acc_data = acc_data[:max_acc]

                # Read tags (timestamps marking phase transitions)
                tag_lines = tags_path.read_text().strip().split('\n')
                tag_times = []
                for t in tag_lines:
                    t = t.strip()
                    if t:
                        try:
                            tag_times.append(datetime.strptime(t, "%Y-%m-%d %H:%M:%S"))
                        except ValueError:
                            continue

                if len(tag_times) < 2:
                    print(f"  Skipping {subject_id} (insufficient tags: {len(tag_times)})")
                    continue

                # Determine phase list based on subject ID
                is_v2 = subject_id.startswith('f')
                phases = v2_phases if is_v2 else v1_phases

                # Map tags to phases: first N tags = phase starts, where N = len(phases)
                n_phases = min(len(phases), len(tag_times))

                # Resample PPG to 100 Hz
                ppg_resampled = self.resample_signal(bvp_data, bvp_sr)
                ppg_resampled = self.normalize_signal(ppg_resampled)

                # Resample ACC to 100 Hz
                acc_x = self.resample_signal(acc_data[:, 0], acc_sr)
                acc_y = self.resample_signal(acc_data[:, 1], acc_sr)
                acc_z = self.resample_signal(acc_data[:, 2], acc_sr)
                acc_x = self.normalize_signal(acc_x)
                acc_y = self.normalize_signal(acc_y)
                acc_z = self.normalize_signal(acc_z)

                # Total signal length at 100 Hz
                signal_length = min(len(ppg_resampled), len(acc_x))

                # Create stress label array at 100 Hz (default: -1 = unlabeled)
                stress_labels = np.full(signal_length, -1, dtype=int)

                for phase_idx in range(n_phases):
                    phase_name, is_stressed = phases[phase_idx]

                    # Phase start in seconds from recording start
                    phase_start_sec = (tag_times[phase_idx] - bvp_start).total_seconds()

                    # Phase end: next tag or end of recording
                    if phase_idx + 1 < len(tag_times):
                        phase_end_sec = (tag_times[phase_idx + 1] - bvp_start).total_seconds()
                    else:
                        phase_end_sec = signal_length / self.TARGET_FS

                    # Convert to sample indices at 100 Hz
                    start_sample = max(0, int(phase_start_sec * self.TARGET_FS))
                    end_sample = min(signal_length, int(phase_end_sec * self.TARGET_FS))

                    if start_sample < end_sample:
                        stress_labels[start_sample:end_sample] = 1 if is_stressed else 0

                # Create windows with 50% overlap
                step_size = int(self.WINDOW_SAMPLES * (1 - self.OVERLAP))
                n_subj_windows = 0

                for start_idx in range(0, signal_length - self.WINDOW_SAMPLES + 1, step_size):
                    end_idx = start_idx + self.WINDOW_SAMPLES

                    # Get majority stress label for window
                    window_labels = stress_labels[start_idx:end_idx]
                    labeled_mask = window_labels >= 0

                    if labeled_mask.sum() < self.WINDOW_SAMPLES * 0.5:
                        continue  # Skip mostly unlabeled windows

                    majority_label = int(np.median(window_labels[labeled_mask]))
                    stress_str = 'stress' if majority_label == 1 else 'baseline'

                    # Build window data
                    window_data = self.create_empty_window()
                    # Channel 0 (ECG): leave as NaN — not available in E4
                    window_data[0, :] = 0.0  # Zero-fill ECG

                    if end_idx <= len(ppg_resampled):
                        window_data[1, :] = ppg_resampled[start_idx:end_idx]
                    if end_idx <= len(acc_x):
                        window_data[2, :] = acc_x[start_idx:end_idx]
                        window_data[3, :] = acc_y[start_idx:end_idx]
                        window_data[4, :] = acc_z[start_idx:end_idx]

                    labels = {'stress': stress_str}
                    one_hot_labels = self.create_one_hot_labels(labels)

                    sample = {
                        'window_data': window_data,
                        'labels': one_hot_labels,
                        'window_index': n_subj_windows,
                        'start_time': start_idx / self.TARGET_FS,
                        'subject_id': f"WS_{subject_id}",
                        'dataset': 'WearableStress'
                    }
                    all_windows.append(sample)
                    n_subj_windows += 1

                print(f"  {subject_id}: {n_subj_windows} windows (PPG@{bvp_sr}Hz, ACC@{acc_sr}Hz, {len(tag_times)} tags)")

            except Exception as e:
                print(f"  ⚠️ Error processing {subject_id}: {e}")
                continue

        print(f"  ✅ Wearable Stress: {len(all_windows)} windows from {len(subject_dirs)} subjects")
        return all_windows

    def process_uci_har(self, data_path):
        """Process UCI HAR Dataset for activity recognition.

        UCI HAR provides 128-sample windows at 50Hz (2.56s each).
        We reconstruct continuous signals per subject+activity,
        resample to 100Hz, and window into 10s (1000 sample) windows.
        """
        print("Processing UCI HAR Dataset...")

        all_windows = []
        data_path = Path(data_path)

        uci_activity_mapping = {
            1: 'walking',       # WALKING -> walking
            2: 'stairs',        # WALKING_UPSTAIRS -> stairs
            3: 'stairs',        # WALKING_DOWNSTAIRS -> stairs
            4: 'sitting',       # SITTING -> sitting
            5: 'working',       # STANDING -> working (closest sedentary)
            6: 'sitting',       # LAYING -> sitting (closest sedentary)
        }

        for split in ['train', 'test']:
            split_dir = data_path / split
            if not split_dir.exists():
                continue

            inertial_dir = split_dir / 'Inertial Signals'

            acc_x = np.loadtxt(inertial_dir / f'total_acc_x_{split}.txt')
            acc_y = np.loadtxt(inertial_dir / f'total_acc_y_{split}.txt')
            acc_z = np.loadtxt(inertial_dir / f'total_acc_z_{split}.txt')

            labels = np.loadtxt(split_dir / f'y_{split}.txt', dtype=int)
            subjects = np.loadtxt(split_dir / f'subject_{split}.txt', dtype=int)

            print(f"  {split}: {len(labels)} windows from {len(np.unique(subjects))} subjects")

            stride = 64  # 50% overlap at 50Hz

            for subj_id in np.unique(subjects):
                subj_mask = subjects == subj_id
                subj_indices = np.where(subj_mask)[0]
                subj_labels = labels[subj_indices]

                i = 0
                while i < len(subj_indices):
                    current_activity = subj_labels[i]
                    if current_activity not in uci_activity_mapping:
                        i += 1
                        continue

                    run_end = i + 1
                    while run_end < len(subj_indices) and subj_labels[run_end] == current_activity:
                        if subj_indices[run_end] != subj_indices[run_end - 1] + 1:
                            break
                        run_end += 1

                    run_length = run_end - i

                    if run_length >= 8:
                        total_samples = 128 + (run_length - 1) * stride

                        continuous_ax = np.zeros(total_samples)
                        continuous_ay = np.zeros(total_samples)
                        continuous_az = np.zeros(total_samples)

                        for j in range(run_length):
                            idx = subj_indices[i + j]
                            start = j * stride
                            continuous_ax[start:start + 128] = acc_x[idx]
                            continuous_ay[start:start + 128] = acc_y[idx]
                            continuous_az[start:start + 128] = acc_z[idx]

                        ax_100 = self.resample_signal(continuous_ax, 50)
                        ay_100 = self.resample_signal(continuous_ay, 50)
                        az_100 = self.resample_signal(continuous_az, 50)

                        ax_100 = self.normalize_signal(ax_100)
                        ay_100 = self.normalize_signal(ay_100)
                        az_100 = self.normalize_signal(az_100)

                        step_size = int(self.WINDOW_SAMPLES * (1 - self.OVERLAP))

                        for start_idx in range(0, len(ax_100) - self.WINDOW_SAMPLES + 1, step_size):
                            end_idx = start_idx + self.WINDOW_SAMPLES

                            window_data = self.create_empty_window()
                            window_data[self.CHANNEL_MAPPING['accel_x'], :] = ax_100[start_idx:end_idx]
                            window_data[self.CHANNEL_MAPPING['accel_y'], :] = ay_100[start_idx:end_idx]
                            window_data[self.CHANNEL_MAPPING['accel_z'], :] = az_100[start_idx:end_idx]

                            for ch_name in ['ecg', 'ppg']:
                                window_data[self.CHANNEL_MAPPING[ch_name], :] = 0.0

                            activity_name = uci_activity_mapping[current_activity]
                            one_hot_labels = self.create_one_hot_labels({'activity': activity_name})

                            sample = {
                                'window_data': window_data,
                                'labels': one_hot_labels,
                                'window_index': len(all_windows),
                                'start_time': start_idx / self.TARGET_FS,
                                'subject_id': f'UCI_S{subj_id:02d}',
                                'dataset': 'UCI-HAR'
                            }
                            all_windows.append(sample)

                    i = run_end

        print(f"  ✅ UCI-HAR: {len(all_windows)} windows created")
        return all_windows

    def combine_all_datasets(self, ppg_dalia_path=None, mit_bih_path=None, wesad_path=None,
                             wearable_stress_path=None, uci_har_path=None):
        """
        Main processing pipeline - combines all datasets into unified format

        Returns:
            unified_dataset: List of samples with format:
                {
                    'window_data': np.array([5, 1000]),  # [channels × samples]
                    'labels': {
                        'activity': one_hot_vector,
                        'stress': one_hot_vector,
                        'arrhythmia': one_hot_vector
                    },
                    'subject_id': str,
                    'dataset': str,
                    'window_index': int,
                    'start_time': float
                }
        """

        print("Starting Unified Dataset Creation...")
        print(f"- Target Format: {self.N_CHANNELS} channels × {self.WINDOW_SAMPLES} samples")
        print(f"- Sampling Rate: {self.TARGET_FS} Hz")
        print(f"- Window Length: {self.WINDOW_LENGTH} seconds")
        print(f"- Overlap: {self.OVERLAP * 100}%")

        all_windows = []

        # Process each dataset
        if ppg_dalia_path:
            ppg_windows = self.process_ppg_dalia(ppg_dalia_path)
            all_windows.extend(ppg_windows)

        if mit_bih_path:
            mit_windows = self.process_mit_bih(mit_bih_path)
            all_windows.extend(mit_windows)

        if wesad_path:
            wesad_windows = self.process_wesad(wesad_path)
            all_windows.extend(wesad_windows)

        if wearable_stress_path:
            ws_windows = self.process_wearable_stress(wearable_stress_path)
            all_windows.extend(ws_windows)

        if uci_har_path:
            uci_windows = self.process_uci_har(uci_har_path)
            all_windows.extend(uci_windows)
        
        print(f"\n✅ Total Windows Created: {len(all_windows)}")
        
        # Save unified dataset
        self.save_unified_dataset(all_windows)
        
        # Create summary statistics
        summary = self.create_summary_statistics(all_windows)
        
        return all_windows, summary
    
    def save_unified_dataset(self, unified_dataset):
        """Save unified dataset"""
        
        # Save as pickle
        save_path = self.output_dir / 'unified_dataset.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(unified_dataset, f)
        
        print(f"✅ Unified dataset saved: {save_path}")
        
        # Save metadata
        metadata = {
            'format': {
                'channels': self.N_CHANNELS,
                'samples_per_window': self.WINDOW_SAMPLES,
                'sampling_rate_hz': self.TARGET_FS,
                'window_length_sec': self.WINDOW_LENGTH,
                'overlap': self.OVERLAP
            },
            'channel_mapping': self.CHANNEL_MAPPING,
            'label_encodings': self.LABEL_ENCODINGS,
            'total_windows': len(unified_dataset)
        }
        
        metadata_path = self.output_dir / 'dataset_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"📋 Metadata saved: {metadata_path}")
    
    def create_summary_statistics(self, unified_dataset):
        """Create summary statistics"""
        
        summary = {
            'total_windows': len(unified_dataset),
            'datasets': {},
            'subjects': set(),
            'label_distributions': {}
        }
        
        # Analyze by dataset
        for window in unified_dataset:
            dataset = window['dataset']
            subject = window['subject_id']
            
            if dataset not in summary['datasets']:
                summary['datasets'][dataset] = {'count': 0, 'subjects': set()}
            
            summary['datasets'][dataset]['count'] += 1
            summary['datasets'][dataset]['subjects'].add(subject)
            summary['subjects'].add(f"{dataset}_{subject}")
        
        # Convert sets to lists
        for dataset_info in summary['datasets'].values():
            dataset_info['subjects'] = list(dataset_info['subjects'])
        summary['subjects'] = list(summary['subjects'])
        
        # Analyze label distributions
        for label_type in self.LABEL_ENCODINGS.keys():
            summary['label_distributions'][label_type] = {}
            
            for window in unified_dataset:
                one_hot = window['labels'][label_type]
                if np.sum(one_hot) > 0:  # Has valid label
                    class_idx = np.argmax(one_hot)
                    # Find class name
                    for class_name, idx in self.LABEL_ENCODINGS[label_type].items():
                        if idx == class_idx:
                            if class_name not in summary['label_distributions'][label_type]:
                                summary['label_distributions'][label_type][class_name] = 0
                            summary['label_distributions'][label_type][class_name] += 1
                            break
        
        # Print summary
        print(f"\n--- DATASET SUMMARY ---")
        print(f"Total Windows: {summary['total_windows']}")
        print(f"Total Subjects: {len(summary['subjects'])}")
        
        for dataset, info in summary['datasets'].items():
            print(f"{dataset}: {info['count']} windows, {len(info['subjects'])} subjects")
        
        for label_type, distribution in summary['label_distributions'].items():
            if distribution:
                print(f"{label_type.title()} Labels: {distribution}")
        
        return summary

# Usage Example
if __name__ == "__main__":
    
    # Initialize processor
    processor = UnifiedBiomedicalDataProcessor()
    
    # Process all datasets - UPDATE THESE PATHS
    dataset_paths = {
        'ppg_dalia_path': '/Users/HP/Desktop/University/Thesis/Code/data/ppg+dalia/PPG_FieldStudy',
        'mit_bih_path': '/Users/HP/Desktop/University/Thesis/Code/data/mit-bih-arrhythmia-database-1.0.0', 
        'wesad_path': '/Users/HP/Desktop/University/Thesis/Code/data/WESAD'
    }
    
    # Create unified dataset
    unified_dataset, summary = processor.combine_all_datasets(**dataset_paths)
    
    print("\n🎉 UNIFIED DATASET CREATED!")
    print(f"Format: [{processor.N_CHANNELS} channels × {processor.WINDOW_SAMPLES} samples]")
    print(f"Sampling Rate: {processor.TARGET_FS} Hz")
    print(f"Total Windows: {len(unified_dataset)}")
    
    # Example: Access first window
    if unified_dataset:
        first_window = unified_dataset[0]
        print(f"\nExample Window:")
        print(f"Shape: {first_window['window_data'].shape}")
        print(f"Dataset: {first_window['dataset']}")
        print(f"Subject: {first_window['subject_id']}")
        print(f"Activity Label: {first_window['labels']['activity']}")
        print(f"Stress Label: {first_window['labels']['stress']}")  
        print(f"Arrhythmia Label: {first_window['labels']['arrhythmia']}")