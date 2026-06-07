#!/usr/bin/env python3
"""
Export PyTorch model weights to C header files for ESP32 deployment.

This script extracts weights from the trained CNNTransformerLite model
and exports them as C arrays that can be directly included in the firmware.

The ESP32 inference engine uses a simplified CNN without transformer layers,
so we extract:
  - CNN conv layers (fused with batch norm)
  - Channel projection layer
  - Task-specific classification heads

Usage:
    python scripts/export_weights_esp32.py --model_path training_results/model.pth

Output:
    firmware/esp32/src/inference/weights/*.inc files
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_transformer_lite import CNNTransformerLite


def load_model(model_path: str) -> torch.nn.Module:
    """Load the trained PyTorch model."""
    print(f"Loading model from {model_path}...")

    model = CNNTransformerLite(
        n_channels=11,
        n_samples=1000,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def fuse_conv_bn(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """
    Fuse convolution and batch normalization weights.

    This allows us to skip batch norm at inference time by folding the
    normalization into the convolution weights.

    Args:
        conv_weight: Convolution weights [out_ch, in_ch, kernel]
        conv_bias: Convolution bias [out_ch] or None
        bn_weight: BatchNorm gamma (scale) [out_ch]
        bn_bias: BatchNorm beta (shift) [out_ch]
        bn_mean: BatchNorm running mean [out_ch]
        bn_var: BatchNorm running variance [out_ch]
        eps: BatchNorm epsilon

    Returns:
        fused_weight, fused_bias
    """
    if conv_bias is None:
        conv_bias = torch.zeros(conv_weight.shape[0])

    # Calculate scale factor
    bn_std = torch.sqrt(bn_var + eps)
    scale = bn_weight / bn_std

    # Fuse weights
    fused_weight = conv_weight * scale.view(-1, 1, 1)

    # Fuse bias
    fused_bias = (conv_bias - bn_mean) * scale + bn_bias

    return fused_weight.numpy(), fused_bias.numpy()


def format_array_as_c(arr: np.ndarray, precision: int = 6) -> str:
    """Format numpy array as C array initializer."""
    flat = arr.flatten()
    lines = []
    values_per_line = 8

    for i in range(0, len(flat), values_per_line):
        chunk = flat[i:i+values_per_line]
        line = ', '.join(f'{v:.{precision}f}f' for v in chunk)
        lines.append(line)

    return ',\n'.join(lines)


def export_layer_weights(weights: np.ndarray, bias: np.ndarray,
                         weight_path: str, bias_path: str,
                         layer_name: str):
    """Export a layer's weights and biases."""
    print(f"  Exporting {layer_name}: weights {weights.shape}, bias {bias.shape}")

    # Export weights
    with open(weight_path, 'w') as f:
        f.write(f"// {layer_name} weights: shape {list(weights.shape)}\n")
        f.write(format_array_as_c(weights))

    # Export bias
    with open(bias_path, 'w') as f:
        f.write(f"// {layer_name} bias: shape {list(bias.shape)}\n")
        f.write(format_array_as_c(bias))


def export_weights(model: torch.nn.Module, output_dir: str):
    """Export all model weights to C include files."""
    print(f"\nExporting weights to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    state_dict = model.state_dict()

    # Print available keys for debugging
    print("\nModel state dict keys:")
    for key in sorted(state_dict.keys()):
        print(f"  {key}: {state_dict[key].shape}")

    total_params = 0

    # ========================================
    # Export CNN layers (with BatchNorm fusion)
    # ========================================

    # Conv1: cnn_layers.0 (Conv1d) + cnn_layers.1 (BatchNorm1d)
    # Shape: [16, 11, 7]
    conv1_weight, conv1_bias = fuse_conv_bn(
        state_dict['cnn_layers.0.weight'],
        state_dict.get('cnn_layers.0.bias'),
        state_dict['cnn_layers.1.weight'],
        state_dict['cnn_layers.1.bias'],
        state_dict['cnn_layers.1.running_mean'],
        state_dict['cnn_layers.1.running_var']
    )
    export_layer_weights(
        conv1_weight, conv1_bias,
        os.path.join(output_dir, 'conv1_weight.inc'),
        os.path.join(output_dir, 'conv1_bias.inc'),
        'conv1 (fused with bn)'
    )
    total_params += conv1_weight.size + conv1_bias.size

    # Conv2: cnn_layers.4 (Conv1d) + cnn_layers.5 (BatchNorm1d)
    # Shape: [32, 16, 5]
    conv2_weight, conv2_bias = fuse_conv_bn(
        state_dict['cnn_layers.4.weight'],
        state_dict.get('cnn_layers.4.bias'),
        state_dict['cnn_layers.5.weight'],
        state_dict['cnn_layers.5.bias'],
        state_dict['cnn_layers.5.running_mean'],
        state_dict['cnn_layers.5.running_var']
    )
    export_layer_weights(
        conv2_weight, conv2_bias,
        os.path.join(output_dir, 'conv2_weight.inc'),
        os.path.join(output_dir, 'conv2_bias.inc'),
        'conv2 (fused with bn)'
    )
    total_params += conv2_weight.size + conv2_bias.size

    # Conv3: cnn_layers.8 (Conv1d) + cnn_layers.9 (BatchNorm1d)
    # Shape: [32, 32, 3]
    conv3_weight, conv3_bias = fuse_conv_bn(
        state_dict['cnn_layers.8.weight'],
        state_dict.get('cnn_layers.8.bias'),
        state_dict['cnn_layers.9.weight'],
        state_dict['cnn_layers.9.bias'],
        state_dict['cnn_layers.9.running_mean'],
        state_dict['cnn_layers.9.running_var']
    )
    export_layer_weights(
        conv3_weight, conv3_bias,
        os.path.join(output_dir, 'conv3_weight.inc'),
        os.path.join(output_dir, 'conv3_bias.inc'),
        'conv3 (fused with bn)'
    )
    total_params += conv3_weight.size + conv3_bias.size

    # ========================================
    # Export channel projection layer
    # ========================================
    # Shape: [64, 32]
    proj_weight = state_dict['channel_projection.weight'].numpy()
    proj_bias = state_dict['channel_projection.bias'].numpy()
    export_layer_weights(
        proj_weight, proj_bias,
        os.path.join(output_dir, 'projection_weight.inc'),
        os.path.join(output_dir, 'projection_bias.inc'),
        'channel_projection'
    )
    total_params += proj_weight.size + proj_bias.size

    # ========================================
    # Export task heads
    # ========================================
    # Note: Task heads have structure: Linear(64, 32) -> ReLU -> Dropout -> Linear(32, n_classes)
    # We need to export both layers for each head

    # Activity head
    # First layer: [32, 64], Second layer: [8, 32]
    act_fc1_weight = state_dict['task_heads.activity.0.weight'].numpy()
    act_fc1_bias = state_dict['task_heads.activity.0.bias'].numpy()
    act_fc2_weight = state_dict['task_heads.activity.3.weight'].numpy()
    act_fc2_bias = state_dict['task_heads.activity.3.bias'].numpy()

    export_layer_weights(
        act_fc1_weight, act_fc1_bias,
        os.path.join(output_dir, 'activity_head_fc1_weight.inc'),
        os.path.join(output_dir, 'activity_head_fc1_bias.inc'),
        'activity_head_fc1'
    )
    export_layer_weights(
        act_fc2_weight, act_fc2_bias,
        os.path.join(output_dir, 'activity_head_fc2_weight.inc'),
        os.path.join(output_dir, 'activity_head_fc2_bias.inc'),
        'activity_head_fc2'
    )
    total_params += act_fc1_weight.size + act_fc1_bias.size
    total_params += act_fc2_weight.size + act_fc2_bias.size

    # Stress head
    # First layer: [32, 64], Second layer: [4, 32]
    stress_fc1_weight = state_dict['task_heads.stress.0.weight'].numpy()
    stress_fc1_bias = state_dict['task_heads.stress.0.bias'].numpy()
    stress_fc2_weight = state_dict['task_heads.stress.3.weight'].numpy()
    stress_fc2_bias = state_dict['task_heads.stress.3.bias'].numpy()

    export_layer_weights(
        stress_fc1_weight, stress_fc1_bias,
        os.path.join(output_dir, 'stress_head_fc1_weight.inc'),
        os.path.join(output_dir, 'stress_head_fc1_bias.inc'),
        'stress_head_fc1'
    )
    export_layer_weights(
        stress_fc2_weight, stress_fc2_bias,
        os.path.join(output_dir, 'stress_head_fc2_weight.inc'),
        os.path.join(output_dir, 'stress_head_fc2_bias.inc'),
        'stress_head_fc2'
    )
    total_params += stress_fc1_weight.size + stress_fc1_bias.size
    total_params += stress_fc2_weight.size + stress_fc2_bias.size

    # Arrhythmia head
    # First layer: [32, 64], Second layer: [2, 32]
    arr_fc1_weight = state_dict['task_heads.arrhythmia.0.weight'].numpy()
    arr_fc1_bias = state_dict['task_heads.arrhythmia.0.bias'].numpy()
    arr_fc2_weight = state_dict['task_heads.arrhythmia.3.weight'].numpy()
    arr_fc2_bias = state_dict['task_heads.arrhythmia.3.bias'].numpy()

    export_layer_weights(
        arr_fc1_weight, arr_fc1_bias,
        os.path.join(output_dir, 'arrhythmia_head_fc1_weight.inc'),
        os.path.join(output_dir, 'arrhythmia_head_fc1_bias.inc'),
        'arrhythmia_head_fc1'
    )
    export_layer_weights(
        arr_fc2_weight, arr_fc2_bias,
        os.path.join(output_dir, 'arrhythmia_head_fc2_weight.inc'),
        os.path.join(output_dir, 'arrhythmia_head_fc2_bias.inc'),
        'arrhythmia_head_fc2'
    )
    total_params += arr_fc1_weight.size + arr_fc1_bias.size
    total_params += arr_fc2_weight.size + arr_fc2_bias.size

    print(f"\nTotal parameters exported: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024:.1f} KB (float32)")
    return total_params


def create_placeholder_weights(layer_name: str, output_dir: str, shape: tuple, output_dim: int):
    """Create placeholder weights for a layer."""
    # Xavier-like initialization
    fan_in = np.prod(shape[1:]) if len(shape) > 1 else shape[0]
    std = np.sqrt(2.0 / fan_in)
    weights = np.random.randn(*shape).astype(np.float32) * std
    bias = np.zeros(output_dim, dtype=np.float32)

    export_layer_weights(
        weights, bias,
        os.path.join(output_dir, f'{layer_name}_weight.inc'),
        os.path.join(output_dir, f'{layer_name}_bias.inc'),
        f'{layer_name} (placeholder)'
    )


def create_all_placeholder_weights(output_dir: str):
    """Create placeholder weights for all layers."""
    print("Creating placeholder weights for testing...")
    os.makedirs(output_dir, exist_ok=True)

    # CNN layers (with correct kernel sizes from the actual model)
    create_placeholder_weights('conv1', output_dir, (16, 11, 7), 16)
    create_placeholder_weights('conv2', output_dir, (32, 16, 5), 32)
    create_placeholder_weights('conv3', output_dir, (32, 32, 3), 32)

    # Projection layer
    create_placeholder_weights('projection', output_dir, (64, 32), 64)

    # Task heads (two layers each)
    create_placeholder_weights('activity_head_fc1', output_dir, (32, 64), 32)
    create_placeholder_weights('activity_head_fc2', output_dir, (8, 32), 8)

    create_placeholder_weights('stress_head_fc1', output_dir, (32, 64), 32)
    create_placeholder_weights('stress_head_fc2', output_dir, (4, 32), 4)

    create_placeholder_weights('arrhythmia_head_fc1', output_dir, (32, 64), 32)
    create_placeholder_weights('arrhythmia_head_fc2', output_dir, (2, 32), 2)

    print("Placeholder weights created successfully!")


def main():
    parser = argparse.ArgumentParser(description='Export model weights for ESP32')
    parser.add_argument('--model_path', type=str, default='training_results/model.pth',
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str,
                        default='firmware/esp32/src/inference/weights',
                        help='Output directory for weight files')
    parser.add_argument('--placeholder', action='store_true',
                        help='Create placeholder weights instead of loading model')
    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / args.model_path
    output_dir = base_dir / args.output_dir

    if args.placeholder:
        create_all_placeholder_weights(str(output_dir))
    elif not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Available models:")
        for pth in base_dir.rglob("*.pth"):
            print(f"  {pth.relative_to(base_dir)}")
        print("\nCreating placeholder weights instead...")
        create_all_placeholder_weights(str(output_dir))
    else:
        model = load_model(str(model_path))
        export_weights(model, str(output_dir))

    print(f"\nWeight files exported to: {output_dir}")
    print("\nTo use these weights:")
    print("1. Rebuild the firmware: cd firmware/esp32 && pio run")
    print("2. Flash to ESP32: pio run -t upload")


if __name__ == '__main__':
    main()
