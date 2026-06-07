"""
#3
Unified Dataset Class
PyTorch Dataset class for the unified multimodal biomedical data

Format: [5 channels × 1000 samples] @ 100 Hz
- Channel 0: ECG
- Channel 1: PPG  
- Channel 2: Accel X
- Channel 3: Accel Y
- Channel 4: Accel Z

Labels: One-hot encoded vectors for multi-task learning
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class UnifiedBiomedicalDataset(Dataset):
    """
    Unified PyTorch Dataset for multimodal biomedical signals
    
    Returns:
        - signals: Tensor [5, 1000] = [channels, samples] 
        - labels: Dict of one-hot tensors for each task
        - metadata: Subject ID, dataset, etc.
    """
    
    def __init__(self, data_file_path, task_types=['activity', 'stress', 'arrhythmia'], 
                 missing_channel_strategy='zero'):
        """
        Args:
            data_file_path: Path to unified_dataset.pkl
            task_types: List of tasks to include ['activity', 'stress', 'arrhythmia']
            missing_channel_strategy: 'zero', 'interpolate', or 'mask'
        """
        
        self.data_file_path = Path(data_file_path)
        self.task_types = task_types
        self.missing_strategy = missing_channel_strategy
        
        # Load data
        self._load_data()
        
        # Filter samples that have labels for requested tasks
        self._filter_samples()
        
        print(f"✅ Loaded {len(self.valid_samples)} samples")
        print(f"- Tasks: {self.task_types}")
        self._print_task_statistics()
    
    def _load_data(self):
        """Load unified dataset and metadata"""
        
        with open(self.data_file_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        
        # Load metadata
        metadata_path = self.data_file_path.parent / 'dataset_metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            # Create default metadata
            self.metadata = {
                'format': {
                    'channels': 5,
                    'samples_per_window': 1000,
                    'sampling_rate_hz': 100,
                    'window_length_sec': 10,
                    'overlap': 0.5
                },
                'channel_mapping': {
                    'ecg': 0, 'ppg': 1, 'accel_x': 2, 'accel_y': 3, 'accel_z': 4
                },
                'label_encodings': {
                    'activity': {'sitting': 0, 'walking': 1, 'cycling': 2, 'driving': 3, 
                                'working': 4, 'stairs': 5, 'table_soccer': 6, 'lunch': 7},
                    'stress': {'baseline': 0, 'stress': 1, 'amusement': 2, 'meditation': 3},
                    'arrhythmia': {'normal': 0, 'abnormal': 1}
                }
            }
        
        self.channel_names = ['ECG', 'PPG', 'Accel_X', 'Accel_Y', 'Accel_Z']
        self.n_channels = self.metadata['format']['channels']
        self.n_samples = self.metadata['format']['samples_per_window']
        
    def _filter_samples(self):
        """Filter samples that have valid labels for requested tasks"""
        
        self.valid_samples = []
        
        for sample in self.raw_data:
            # Check if sample has valid labels for at least one requested task
            has_valid_label = False
            
            for task in self.task_types:
                if task in sample['labels']:
                    one_hot = sample['labels'][task]
                    if np.sum(one_hot) > 0:  # Has valid label (not all zeros)
                        has_valid_label = True
                        break
            
            if has_valid_label:
                self.valid_samples.append(sample)
    
    def _print_task_statistics(self):
        """Print statistics for each task"""
        
        for task in self.task_types:
            print(f"\n {task.upper()} Task Statistics:")
            
            task_counts = {}
            valid_samples = 0
            
            for sample in self.valid_samples:
                if task in sample['labels']:
                    one_hot = sample['labels'][task]
                    if np.sum(one_hot) > 0:
                        valid_samples += 1
                        class_idx = np.argmax(one_hot)
                        
                        # Find class name
                        class_name = f"class_{class_idx}"  # fallback
                        if task in self.metadata['label_encodings']:
                            for name, idx in self.metadata['label_encodings'][task].items():
                                if idx == class_idx:
                                    class_name = name
                                    break
                        
                        task_counts[class_name] = task_counts.get(class_name, 0) + 1
            
            print(f"  Valid samples: {valid_samples}")
            print(f"  Class distribution: {task_counts}")
    
    def _handle_missing_channels(self, window_data):
        """Handle missing channels (NaN values)"""
        
        if self.missing_strategy == 'zero':
            # Replace NaN with zeros
            window_data = np.nan_to_num(window_data, nan=0.0)
            
        elif self.missing_strategy == 'interpolate':
            # Simple linear interpolation for missing channels
            for ch in range(window_data.shape[0]):
                channel_data = window_data[ch, :]
                if np.any(np.isnan(channel_data)):
                    # If entire channel is NaN, fill with zeros
                    if np.all(np.isnan(channel_data)):
                        window_data[ch, :] = 0.0
                    else:
                        # Interpolate missing values
                        valid_indices = ~np.isnan(channel_data)
                        if np.sum(valid_indices) > 1:
                            from scipy.interpolate import interp1d
                            valid_samples = np.where(valid_indices)[0]
                            valid_values = channel_data[valid_indices]
                            
                            f = interp1d(valid_samples, valid_values, 
                                       kind='linear', fill_value='extrapolate')
                            all_indices = np.arange(len(channel_data))
                            window_data[ch, :] = f(all_indices)
                        else:
                            window_data[ch, :] = 0.0
        
        elif self.missing_strategy == 'mask':
            # Keep NaN for masking in model
            pass
        
        return window_data
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Extract signal data [5, 1000]
        window_data = sample['window_data'].copy()
        
        # Handle missing channels
        window_data = self._handle_missing_channels(window_data)
        
        # Convert to tensor
        signals = torch.FloatTensor(window_data)
        
        # Extract labels for requested tasks
        labels = {}
        for task in self.task_types:
            if task in sample['labels']:
                one_hot = sample['labels'][task]
                labels[task] = torch.FloatTensor(one_hot)
            else:
                # Create zero vector if task not available
                n_classes = len(self.metadata['label_encodings'].get(task, {'none': 0}))
                labels[task] = torch.zeros(n_classes)
        
        # Create channel availability mask
        channel_mask = ~torch.isnan(signals).all(dim=1)  # [5] - True if channel has data
        
        return {
            'signals': signals,           # [5, 1000] - signal data
            'labels': labels,             # Dict of one-hot tensors
            'channel_mask': channel_mask, # [5] - channel availability
            'subject_id': sample['subject_id'],
            'dataset': sample['dataset'],
            'window_index': sample['window_index']
        }
    
    def get_task_info(self, task):
        """Get information about a specific task"""
        if task not in self.metadata['label_encodings']:
            return None
        
        return {
            'n_classes': len(self.metadata['label_encodings'][task]),
            'class_names': list(self.metadata['label_encodings'][task].keys()),
            'class_indices': list(self.metadata['label_encodings'][task].values())
        }
    
    def plot_sample(self, idx, duration_sec=10):
        """Plot a sample with all available channels"""
        
        if idx >= len(self):
            print(f"Index {idx} out of range")
            return
        
        sample_data = self[idx]
        signals = sample_data['signals'].numpy()  # [5, 1000]
        channel_mask = sample_data['channel_mask'].numpy()
        
        # Count available channels
        available_channels = [(i, name) for i, name in enumerate(self.channel_names) 
                            if channel_mask[i]]
        
        if not available_channels:
            print("No channels available")
            return
        
        # Create time axis
        time_axis = np.linspace(0, duration_sec, self.n_samples)
        
        # Create subplots
        fig, axes = plt.subplots(len(available_channels), 1, 
                               figsize=(12, 2*len(available_channels)))
        if len(available_channels) == 1:
            axes = [axes]
        
        subject_id = sample_data['subject_id']
        dataset = sample_data['dataset']
        fig.suptitle(f'Sample {idx}: {subject_id} ({dataset})', fontsize=14)
        
        for i, (ch_idx, ch_name) in enumerate(available_channels):
            signal_data = signals[ch_idx, :]
            
            axes[i].plot(time_axis, signal_data, linewidth=0.8, color=f'C{ch_idx}')
            axes[i].set_title(f'{ch_name} (Channel {ch_idx})')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Normalized Amplitude')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, duration_sec)
        
        # Add label information
        label_text = []
        for task, label_tensor in sample_data['labels'].items():
            if torch.sum(label_tensor) > 0:
                class_idx = torch.argmax(label_tensor).item()
                task_info = self.get_task_info(task)
                if task_info and class_idx < len(task_info['class_names']):
                    class_name = task_info['class_names'][class_idx]
                    label_text.append(f"{task.title()}: {class_name}")
        
        if label_text:
            fig.text(0.02, 0.02, " | ".join(label_text), fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.show()
        
    def create_subject_splits(self, test_size=0.2, val_size=0.1, random_state=42):
        """Create train/validation/test splits by subject (avoid data leakage)"""
        
        # Group samples by subject
        subject_samples = {}
        for idx, sample in enumerate(self.valid_samples):
            subject_key = f"{sample['dataset']}_{sample['subject_id']}"
            if subject_key not in subject_samples:
                subject_samples[subject_key] = []
            subject_samples[subject_key].append(idx)
        
        subjects = list(subject_samples.keys())
        
        # Split subjects
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=test_size, random_state=random_state
        )
        
        train_subjects, val_subjects = train_test_split(
            train_subjects, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        # Create sample indices for each split
        train_indices = []
        val_indices = []
        test_indices = []
        
        for subject in train_subjects:
            train_indices.extend(subject_samples[subject])
        for subject in val_subjects:
            val_indices.extend(subject_samples[subject])
        for subject in test_subjects:
            test_indices.extend(subject_samples[subject])
        
        print(f"- Subject-wise splits:")
        print(f"  Train: {len(train_indices)} samples from {len(train_subjects)} subjects")
        print(f"  Val:   {len(val_indices)} samples from {len(val_subjects)} subjects") 
        print(f"  Test:  {len(test_indices)} samples from {len(test_subjects)} subjects")
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }

def create_data_loaders(dataset, split_indices, batch_size=32, num_workers=4):
    """Create PyTorch DataLoaders for train/val/test splits"""
    
    from torch.utils.data import Subset
    
    data_loaders = {}
    
    for split_name, indices in split_indices.items():
        subset = Subset(dataset, indices)
        
        shuffle = (split_name == 'train')  # Only shuffle training data
        
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == 'train')  # Drop last for training only
        )
        
        data_loaders[split_name] = loader
        print(f"✅ {split_name.title()} DataLoader: {len(subset)} samples, {len(loader)} batches")
    
    return data_loaders

def analyze_batch(batch, dataset):
    """Analyze a batch from the DataLoader"""
    
    signals = batch['signals']      # [batch_size, 5, 1000]
    channel_mask = batch['channel_mask']  # [batch_size, 5]
    
    print(f"- Batch Analysis:")
    print(f"  Signals shape: {signals.shape}")
    print(f"  Channel availability: {channel_mask.float().mean(dim=0)}")  # Per-channel availability
    
    # Analyze labels
    for task, labels in batch['labels'].items():
        if torch.sum(labels) > 0:  # Has valid labels
            print(f"  {task.title()} labels shape: {labels.shape}")
            
            # Get class distribution in batch
            class_counts = {}
            for i in range(labels.shape[0]):
                if torch.sum(labels[i]) > 0:
                    class_idx = torch.argmax(labels[i]).item()
                    task_info = dataset.get_task_info(task)
                    if task_info and class_idx < len(task_info['class_names']):
                        class_name = task_info['class_names'][class_idx]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"    Class distribution: {class_counts}")

# Usage Example and Testing
if __name__ == "__main__":
    
    # Load unified dataset
    dataset = UnifiedBiomedicalDataset(
        data_file_path='processed_unified_dataset/unified_dataset.pkl',
        task_types=['activity', 'stress', 'arrhythmia'],
        missing_channel_strategy='zero'
    )
    
    # Create subject-wise splits
    split_indices = dataset.create_subject_splits(test_size=0.2, val_size=0.1)
    
    # Create DataLoaders
    data_loaders = create_data_loaders(dataset, split_indices, batch_size=16)
    
    # Test the DataLoaders
    print(f"\n Testing DataLoaders...")
    
    for split_name, loader in data_loaders.items():
        print(f"\n--- {split_name.upper()} SET ---")
        
        for batch_idx, batch in enumerate(loader):
            analyze_batch(batch, dataset)
            
            if batch_idx >= 1:  # Show only first 2 batches
                break
    
    # Plot sample
    print(f"\n Plotting sample...")
    dataset.plot_sample(0)
    
    # Show task information
    print(f"\n Task Information:")
    for task in dataset.task_types:
        info = dataset.get_task_info(task)
        if info:
            print(f"  {task.title()}: {info['n_classes']} classes - {info['class_names']}")
    
    print(f"\n✅ Unified Dataset Ready for Training!")
    print(f"   Format: [5 channels × 1000 samples] @ 100 Hz")
    print(f"   Tasks: {dataset.task_types}")
    print(f"   Total samples: {len(dataset)}")