"""
#1
Data Loader and Analyzer for Processed Multimodal Biomedical Datasets
For Edge Intelligence Wearable Sensor-Fusion System
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalBiomedicalDataset(Dataset):
    """PyTorch Dataset for multimodal biomedical data"""
    
    def __init__(self, processed_data, signal_types=['ecg', 'ppg', 'accel_x', 'accel_y', 'accel_z'], 
                 label_type='stress', normalize_per_sample=True):
        self.data = processed_data
        self.signal_types = signal_types
        self.label_type = label_type
        self.normalize_per_sample = normalize_per_sample
        
        # Filter samples that have the required signals and labels
        self.valid_samples = []
        for sample in processed_data:
            has_required_signals = all(
                sample['signals'].get(sig_type) is not None 
                for sig_type in signal_types
            )
            has_label = sample['labels'].get(label_type) is not None
            
            if has_required_signals and has_label:
                self.valid_samples.append(sample)
        
        print(f"Valid samples for {label_type} task: {len(self.valid_samples)}/{len(processed_data)}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels = [sample['labels'][label_type] for sample in self.valid_samples]
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Stack signals into multi-channel tensor
        signals = []
        for sig_type in self.signal_types:
            signal_data = sample['signals'][sig_type]
            if self.normalize_per_sample:
                # Additional per-sample normalization
                signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
            signals.append(signal_data)
        
        signals = np.stack(signals, axis=0)  # Shape: [channels, sequence_length]
        label = self.encoded_labels[idx]
        
        # Convert to tensors
        signals = torch.FloatTensor(signals)
        label = torch.LongTensor([label])
        
        return {
            'signals': signals,
            'label': label,
            'subject_id': sample['subject_id'],
            'dataset': sample['dataset']
        }

class DatasetAnalyzer:
    """Analyze processed dataset characteristics"""
    
    def __init__(self, data_file_path):
        self.data_file_path = Path(data_file_path)
        self.load_data()
    
    def load_data(self):
        """Load processed data"""
        with open(self.data_file_path, 'rb') as f:
            data = pickle.load(f)

        # Support two formats:
        # 1) Legacy dict with keys 'processed_data' and 'summary'
        # 2) Unified dataset list saved by dataset_integration.py (with metadata in sibling file)
        if isinstance(data, dict) and 'processed_data' in data and 'summary' in data:
            self.processed_data = data['processed_data']
            self.summary = data['summary']
            print(f"Loaded {len(self.processed_data)} processed samples")
            return

        if isinstance(data, list):
            # Attempt to adapt unified dataset format to the analyzer's expected schema
            unified_samples = data

            # Try to load metadata for channel mapping and label encodings
            metadata_path = self.data_file_path.parent / 'dataset_metadata.pkl'
            channel_mapping = {'ecg': 0, 'ppg': 1, 'accel_x': 2, 'accel_y': 3, 'accel_z': 4}
            label_encodings = {
                'activity': {'sitting': 0, 'walking': 1, 'cycling': 2, 'driving': 3, 'working': 4, 'stairs': 5, 'table_soccer': 6, 'lunch': 7},
                'stress': {'baseline': 0, 'stress': 1, 'amusement': 2, 'meditation': 3},
                'arrhythmia': {'normal': 0, 'abnormal': 1}
            }
            sampling_rate_hz = 100
            window_length_sec = 10

            if metadata_path.exists():
                with open(metadata_path, 'rb') as mf:
                    meta = pickle.load(mf)
                channel_mapping = meta.get('channel_mapping', channel_mapping)
                label_encodings = meta.get('label_encodings', label_encodings)
                fmt = meta.get('format', {})
                sampling_rate_hz = fmt.get('sampling_rate_hz', sampling_rate_hz)
                window_length_sec = fmt.get('window_length_sec', window_length_sec)

            # Invert label encodings for decoding one-hot
            inverse_label_maps = {
                task: {idx: name for name, idx in enc_map.items()}
                for task, enc_map in label_encodings.items()
            }

            # Build processed_data compatible with analyzer
            processed = []
            for sample in unified_samples:
                window_data = sample.get('window_data')  # [C, T]
                if window_data is None:
                    continue

                # Extract per-signal arrays using channel mapping
                signals = {}
                try:
                    for sig_name, ch_idx in channel_mapping.items():
                        if 0 <= ch_idx < window_data.shape[0]:
                            sig_array = window_data[ch_idx, :]
                            # Keep as 1D numpy array
                            signals[sig_name] = sig_array
                        else:
                            signals[sig_name] = None
                except Exception:
                    # If shape unexpected, skip sample
                    continue

                # Decode one-hot labels to class names when possible
                labels_out = {}
                labels_in = sample.get('labels', {})
                for task, one_hot in labels_in.items():
                    try:
                        if one_hot is None:
                            labels_out[task] = None
                        else:
                            one_hot_arr = np.array(one_hot)
                            if np.sum(one_hot_arr) > 0:
                                cls_idx = int(np.argmax(one_hot_arr))
                                labels_out[task] = inverse_label_maps.get(task, {}).get(cls_idx, str(cls_idx))
                            else:
                                labels_out[task] = None
                    except Exception:
                        labels_out[task] = None

                processed.append({
                    'signals': signals,
                    'labels': labels_out,
                    'subject_id': sample.get('subject_id', 'unknown'),
                    'dataset': sample.get('dataset', 'unknown'),
                    'sampling_rates': {
                        'ecg': sampling_rate_hz,
                        'ppg': sampling_rate_hz,
                        'accel': sampling_rate_hz,
                        'accel_x': sampling_rate_hz,
                        'accel_y': sampling_rate_hz,
                        'accel_z': sampling_rate_hz
                    },
                    'window_index': sample.get('window_index', 0),
                    'start_time': sample.get('start_time', 0.0),
                })

            # Create a minimal summary compatible with analyzer expectations
            self.processed_data = processed
            self.summary = {
                'label_types': list(label_encodings.keys()),
                'signal_types': list(channel_mapping.keys()),
                'sampling_rates': {'ecg': sampling_rate_hz, 'ppg': sampling_rate_hz, 'accel': sampling_rate_hz},
                'processing_params': {
                    'window_length_sec': window_length_sec,
                    'sampling_rate_hz': sampling_rate_hz
                }
            }
            print(f"Loaded {len(self.processed_data)} processed samples (adapted from unified format)")
            return

        # If neither format matched, raise a helpful error
        raise TypeError(
            "Unsupported dataset file format. Expected a dict with 'processed_data'/'summary' or a list of unified samples."
        )
    
    def analyze_signal_quality(self, max_samples=10):
        """Analyze signal quality metrics"""
        print("\n=== Signal Quality Analysis ===")
        
        quality_metrics = {
            'signal_type': [],
            'dataset': [],
            'snr_db': [],
            'signal_length': [],
            'zero_crossings': [],
            'rms_value': []
        }
        
        for i, sample in enumerate(self.processed_data[:max_samples]):
            for signal_name, signal_data in sample['signals'].items():
                if signal_data is not None:
                    # Signal-to-noise ratio estimation
                    signal_power = np.mean(signal_data ** 2)
                    noise_estimate = np.var(np.diff(signal_data))  # High-freq noise estimate
                    snr_db = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
                    
                    # Zero crossings (indicator of signal activity)
                    zero_crossings = len(np.where(np.diff(np.sign(signal_data)))[0])
                    
                    # RMS value
                    rms_value = np.sqrt(np.mean(signal_data ** 2))
                    
                    quality_metrics['signal_type'].append(signal_name)
                    quality_metrics['dataset'].append(sample['dataset'])
                    quality_metrics['snr_db'].append(snr_db)
                    quality_metrics['signal_length'].append(len(signal_data))
                    quality_metrics['zero_crossings'].append(zero_crossings)
                    quality_metrics['rms_value'].append(rms_value)
        
        df_quality = pd.DataFrame(quality_metrics)
        
        # Display summary statistics
        print(df_quality.groupby(['dataset', 'signal_type']).agg({
            'snr_db': ['mean', 'std'],
            'signal_length': ['mean', 'std'],
            'zero_crossings': ['mean', 'std']
        }).round(2))
        
        return df_quality
    
    def analyze_label_distribution(self):
        """Analyze label distribution across datasets"""
        print("\n=== Label Distribution Analysis ===")
        
        label_stats = {}
        
        for label_type in self.summary['label_types']:
            print(f"\n{label_type.upper()} Labels:")
            
            label_counts = {}
            dataset_counts = {}
            
            for sample in self.processed_data:
                dataset = sample['dataset']
                label_value = sample['labels'].get(label_type)
                
                if label_value is not None:
                    if dataset not in dataset_counts:
                        dataset_counts[dataset] = {}
                    if label_value not in dataset_counts[dataset]:
                        dataset_counts[dataset][label_value] = 0
                    dataset_counts[dataset][label_value] += 1
                    
                    if label_value not in label_counts:
                        label_counts[label_value] = 0
                    label_counts[label_value] += 1
            
            print(f"Overall distribution: {label_counts}")
            print(f"Per dataset: {dataset_counts}")
            label_stats[label_type] = {'overall': label_counts, 'per_dataset': dataset_counts}
        
        return label_stats
    
    def plot_sample_signals(self, sample_idx=0, duration_sec=10):
        """Plot sample signals from a specific sample"""
        if sample_idx >= len(self.processed_data):
            print(f"Sample index {sample_idx} out of range")
            return
        
        sample = self.processed_data[sample_idx]
        
        # Count available signals
        available_signals = [(name, data) for name, data in sample['signals'].items() 
                           if data is not None]
        
        if not available_signals:
            print("No signals available in this sample")
            return
        
        # Create subplots
        fig, axes = plt.subplots(len(available_signals), 1, figsize=(12, 2*len(available_signals)))
        if len(available_signals) == 1:
            axes = [axes]
        
        fig.suptitle(f'Sample Signals: {sample["subject_id"]} ({sample["dataset"]})', fontsize=14)
        
        for i, (signal_name, signal_data) in enumerate(available_signals):
            # Get sampling rate
            fs = sample['sampling_rates'].get(signal_name.split('_')[0], 250)
            
            # Create time axis
            time_points = len(signal_data)
            max_points = int(duration_sec * fs) if duration_sec * fs < time_points else time_points
            
            time_axis = np.linspace(0, max_points/fs, max_points)
            
            axes[i].plot(time_axis, signal_data[:max_points], linewidth=0.8)
            axes[i].set_title(f'{signal_name.upper()} (fs={fs}Hz)')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Save plot instead of showing to avoid blocking in headless/non-GUI envs
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'sample_signals_{sample_idx}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to: {out_path}")
    
    def create_dataset_splits(self, test_size=0.2, val_size=0.1, random_state=42):
        """Create train/validation/test splits with subject-wise splitting"""
        print("\n=== Creating Dataset Splits ===")
        
        # Group samples by subject to avoid data leakage
        subject_groups = {}
        for sample in self.processed_data:
            subject_id = sample['subject_id']
            if subject_id not in subject_groups:
                subject_groups[subject_id] = []
            subject_groups[subject_id].append(sample)
        
        subjects = list(subject_groups.keys())
        
        # Split subjects (not individual samples)
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=test_size, random_state=random_state
        )
        
        train_subjects, val_subjects = train_test_split(
            train_subjects, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        # Create sample splits
        train_samples = []
        val_samples = []
        test_samples = []
        
        for subject_id, samples in subject_groups.items():
            if subject_id in train_subjects:
                train_samples.extend(samples)
            elif subject_id in val_subjects:
                val_samples.extend(samples)
            else:
                test_samples.extend(samples)
        
        print(f"Train: {len(train_samples)} samples from {len(train_subjects)} subjects")
        print(f"Val: {len(val_samples)} samples from {len(val_subjects)} subjects")
        print(f"Test: {len(test_samples)} samples from {len(test_subjects)} subjects")
        
        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }, {
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects
        }
    
    def export_for_edge_deployment(self, output_dir="edge_deployment"):
        """Export processed data in formats suitable for edge deployment"""
        print("\n=== Exporting for Edge Deployment ===")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export signal statistics for normalization on device
        signal_stats = {}
        
        for signal_type in ['ecg', 'ppg', 'accel_x', 'accel_y', 'accel_z']:
            values = []
            for sample in self.processed_data:
                signal_data = sample['signals'].get(signal_type)
                if signal_data is not None:
                    values.extend(signal_data)
            
            if values:
                signal_stats[signal_type] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'p5': float(np.percentile(values, 5)),
                    'p95': float(np.percentile(values, 95))
                }
        
        # Save signal statistics
        import json
        with open(output_path / 'signal_normalization_stats.json', 'w') as f:
            json.dump(signal_stats, f, indent=2)
        
        # Export sampling rates and processing parameters
        deployment_config = {
            'target_sampling_rates': self.summary['sampling_rates'],
            'filter_parameters': self.summary['processing_params'],
            'signal_normalization': signal_stats,
            'available_signals': self.summary['signal_types'],
            'available_labels': self.summary['label_types']
        }
        
        with open(output_path / 'deployment_config.json', 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        # Export compact sample for testing (C-friendly format)
        test_sample = self.processed_data[0]
        compact_sample = {}
        
        for signal_type in ['ecg', 'ppg', 'accel_x', 'accel_y', 'accel_z']:
            signal_data = test_sample['signals'].get(signal_type)
            if signal_data is not None:
                # Take first 1000 samples (about 4-16 seconds depending on signal)
                compact_signal = signal_data[:1000]
                # Convert to int16 to save space (typical for embedded systems)
                compact_signal = (compact_signal * 32767).astype(np.int16)
                compact_sample[signal_type] = compact_signal.tolist()
        
        with open(output_path / 'test_sample.json', 'w') as f:
            json.dump(compact_sample, f, indent=2)
        
        print(f"Edge deployment files exported to: {output_path}")
        
        return deployment_config

def create_data_loaders(processed_data, batch_size=32, num_workers=4):
    """Create PyTorch DataLoaders for different tasks"""
    
    data_loaders = {}
    
    # Create different datasets based on available labels
    label_types = set()
    for sample in processed_data:
        for label_name, label_value in sample['labels'].items():
            if label_value is not None:
                label_types.add(label_name)
    
    for label_type in label_types:
        print(f"\nCreating DataLoader for {label_type} classification...")
        
        try:
            dataset = MultimodalBiomedicalDataset(
                processed_data, 
                label_type=label_type
            )
            
            if len(dataset) > 0:
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=num_workers,
                    drop_last=True
                )
                data_loaders[label_type] = {
                    'dataset': dataset,
                    'dataloader': dataloader,
                    'num_classes': len(dataset.label_encoder.classes_),
                    'classes': dataset.label_encoder.classes_.tolist()
                }
                
                print(f"  - {len(dataset)} samples")
                print(f"  - {len(dataset.label_encoder.classes_)} classes: {dataset.label_encoder.classes_}")
            
        except Exception as e:
            print(f"  Error creating dataset for {label_type}: {e}")
    
    return data_loaders

# Usage examples and main execution
if __name__ == "__main__":
    
    # Load and analyze processed data
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve()
    _candidates = [
        _Path('processed_unified_dataset/unified_dataset.pkl'),
        _here.parent.parent.parent / 'processed_unified_dataset' / 'unified_dataset.pkl',  # /Code/processed_unified_dataset
        _here.parent.parent.parent.parent / 'processed_unified_dataset' / 'unified_dataset.pkl',
        _Path('/Users/HP/Desktop/University/Thesis/Code/processed_unified_dataset/unified_dataset.pkl'),
        _Path('/Users/HP/Desktop/University/Thesis/processed_unified_dataset/unified_dataset.pkl'),
        _Path('/Users/HP/Desktop/University/Thesis/Code/multimodal-biomedical-monitoring/processed_unified_dataset/unified_dataset.pkl')
    ]
    _data_path = next((p for p in _candidates if p.exists()), _candidates[-1])
    analyzer = DatasetAnalyzer(str(_data_path))
    
    # Analyze signal quality
    quality_df = analyzer.analyze_signal_quality()
    
    # Analyze label distributions
    label_stats = analyzer.analyze_label_distribution()
    
    # Plot sample signals
    analyzer.plot_sample_signals(sample_idx=0, duration_sec=10)
    
    # Create dataset splits
    splits, subject_splits = analyzer.create_dataset_splits()
    
    # Export for edge deployment
    deployment_config = analyzer.export_for_edge_deployment()
    
    # Create PyTorch DataLoaders
    data_loaders = create_data_loaders(splits['train'])
    
    # Example: Test a data loader
    if 'stress' in data_loaders:
        stress_loader = data_loaders['stress']['dataloader']
        
        print(f"\nTesting stress classification DataLoader...")
        for batch_idx, batch in enumerate(stress_loader):
            signals = batch['signals']  # Shape: [batch_size, channels, sequence_length]
            labels = batch['label']     # Shape: [batch_size, 1]
            
            print(f"Batch {batch_idx}: signals={signals.shape}, labels={labels.shape}")
            print(f"  Unique labels in batch: {torch.unique(labels)}")
            
            if batch_idx >= 2:  # Just show first 3 batches
                break
    
    print("\n=== Dataset Processing Complete! ===")
    print("Files created:")
    print("  - processed_datasets/combined_raw_data.pkl")
    print("  - processed_datasets/combined_windowed_data.pkl") 
    print("  - edge_deployment/deployment_config.json")
    print("  - edge_deployment/signal_normalization_stats.json")
    print("  - edge_deployment/test_sample.json")
