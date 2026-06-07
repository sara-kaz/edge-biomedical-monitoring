#!/usr/bin/env python3
"""Export Model V5 (CNNTransformerV3) weights to firmware .inc files.

This exports ALL weights including the 2-layer transformer, so the firmware
can run the full V5 model with 94%+ accuracy.
"""

import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / 'firmware' / 'esp32' / 'src' / 'inference' / 'weights'


def fuse_conv_bn(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    if conv_b is None:
        conv_b = torch.zeros(conv_w.shape[0])
    scale = bn_w / torch.sqrt(bn_var + eps)
    return (conv_w * scale.view(-1, 1, 1)).numpy(), ((conv_b - bn_mean) * scale + bn_b).numpy()


def fuse_input_bn_into_conv1(sd, eps=1e-5):
    """Fuse input_bn + conv1 + bn1 into single conv+bias."""
    in_scale = sd['input_bn.weight'] / torch.sqrt(sd['input_bn.running_var'] + eps)
    in_shift = sd['input_bn.bias'] - sd['input_bn.running_mean'] * in_scale

    conv_w = sd['conv1.weight'].clone()  # [32, 5, 7]
    n_out, n_in, k = conv_w.shape

    # Fold input_bn scale into conv weights
    for ic in range(n_in):
        conv_w[:, ic, :] *= in_scale[ic]

    # Bias from input_bn shift through conv
    conv_b = torch.zeros(n_out)
    for oc in range(n_out):
        for ic in range(n_in):
            conv_b[oc] += conv_w[oc, ic, :].sum() / in_scale[ic] * sd['conv1.weight'][oc, ic, :].sum() * in_shift[ic] / conv_w[oc, ic, :].sum() if conv_w[oc, ic, :].sum() != 0 else 0

    # Simpler: just compute bias = sum over ic of (sum_k w_new[oc,ic,k]) * in_shift[ic] / in_scale[ic] * in_scale[ic]
    # Actually let me do this correctly:
    # After input_bn: x' = x * in_scale + in_shift
    # Conv1(x') = sum_ic sum_k w[oc,ic,k] * x'[ic, t-pad+k]
    #           = sum_ic sum_k (w[oc,ic,k] * in_scale[ic]) * x[ic, t-pad+k] + sum_ic sum_k w[oc,ic,k] * in_shift[ic]
    # The new conv weight: w_new[oc,ic,k] = w[oc,ic,k] * in_scale[ic]  (already done above)
    # The conv bias from shift: b_shift[oc] = sum_ic (sum_k w[oc,ic,k]) * in_shift[ic]
    conv_b = torch.zeros(n_out)
    orig_w = sd['conv1.weight']
    for oc in range(n_out):
        for ic in range(n_in):
            conv_b[oc] += orig_w[oc, ic, :].sum() * in_shift[ic]

    # Now fuse with bn1
    return fuse_conv_bn(conv_w, conv_b,
                        sd['bn1.weight'], sd['bn1.bias'],
                        sd['bn1.running_mean'], sd['bn1.running_var'], eps)


def write_inc(arr, path, name):
    flat = arr.flatten()
    lines = []
    for i in range(0, len(flat), 8):
        lines.append(', '.join(f'{v:.6f}f' for v in flat[i:i+8]))
    with open(path, 'w') as f:
        f.write(f'// {name}: shape {list(arr.shape)}\n')
        f.write(',\n'.join(lines))
    print(f'  {name}: {list(arr.shape)} ({arr.size} params)')
    return arr.size


def main():
    model_path = PROJECT_ROOT / 'training_results' / 'model_v5_stress_finetune.pth'
    print(f"Loading V5 model from {model_path}")
    checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
    sd = checkpoint['model_state_dict']
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0

    # Conv1 (fused input_bn + conv1 + bn1)
    w, b = fuse_input_bn_into_conv1(sd)
    total += write_inc(w, OUTPUT_DIR / 'conv1_weight.inc', 'conv1 (fused input_bn+conv1+bn1)')
    total += write_inc(b, OUTPUT_DIR / 'conv1_bias.inc', 'conv1 bias')

    # Conv2 (fused with bn2)
    w, b = fuse_conv_bn(sd['conv2.weight'], None,
                        sd['bn2.weight'], sd['bn2.bias'],
                        sd['bn2.running_mean'], sd['bn2.running_var'])
    total += write_inc(w, OUTPUT_DIR / 'conv2_weight.inc', 'conv2 (fused)')
    total += write_inc(b, OUTPUT_DIR / 'conv2_bias.inc', 'conv2 bias')

    # Conv3 (fused with bn3)
    w, b = fuse_conv_bn(sd['conv3.weight'], None,
                        sd['bn3.weight'], sd['bn3.bias'],
                        sd['bn3.running_mean'], sd['bn3.running_var'])
    total += write_inc(w, OUTPUT_DIR / 'conv3_weight.inc', 'conv3 (fused)')
    total += write_inc(b, OUTPUT_DIR / 'conv3_bias.inc', 'conv3 bias')

    # SE attention
    for name, key in [('se_fc1', 'se_fc1'), ('se_fc2', 'se_fc2')]:
        total += write_inc(sd[f'{key}.weight'].numpy(), OUTPUT_DIR / f'{name}_weight.inc', name)
        total += write_inc(sd[f'{key}.bias'].numpy(), OUTPUT_DIR / f'{name}_bias.inc', f'{name} bias')

    # Projection
    total += write_inc(sd['channel_projection.weight'].numpy(), OUTPUT_DIR / 'projection_weight.inc', 'projection')
    total += write_inc(sd['channel_projection.bias'].numpy(), OUTPUT_DIR / 'projection_bias.inc', 'projection bias')

    # Projection LayerNorm
    total += write_inc(sd['proj_ln.weight'].numpy(), OUTPUT_DIR / 'proj_ln_weight.inc', 'proj_ln')
    total += write_inc(sd['proj_ln.bias'].numpy(), OUTPUT_DIR / 'proj_ln_bias.inc', 'proj_ln bias')

    # Transformer layers (2 layers)
    for layer_idx in range(2):
        prefix = f'transformer.layers.{layer_idx}'
        lname = f'transformer_L{layer_idx}'

        # Self-attention: in_proj (Q,K,V concatenated) [192, 64]
        total += write_inc(sd[f'{prefix}.self_attn.in_proj_weight'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_attn_in_proj_weight.inc',
                           f'{lname}_attn_in_proj')
        total += write_inc(sd[f'{prefix}.self_attn.in_proj_bias'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_attn_in_proj_bias.inc',
                           f'{lname}_attn_in_proj bias')

        # Self-attention: out_proj [64, 64]
        total += write_inc(sd[f'{prefix}.self_attn.out_proj.weight'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_attn_out_proj_weight.inc',
                           f'{lname}_attn_out_proj')
        total += write_inc(sd[f'{prefix}.self_attn.out_proj.bias'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_attn_out_proj_bias.inc',
                           f'{lname}_attn_out_proj bias')

        # norm1 (pre-attention norm)
        total += write_inc(sd[f'{prefix}.norm1.weight'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_norm1_weight.inc', f'{lname}_norm1')
        total += write_inc(sd[f'{prefix}.norm1.bias'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_norm1_bias.inc', f'{lname}_norm1 bias')

        # FFN: linear1 [128, 64], linear2 [64, 128]
        total += write_inc(sd[f'{prefix}.linear1.weight'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_ffn1_weight.inc', f'{lname}_ffn1')
        total += write_inc(sd[f'{prefix}.linear1.bias'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_ffn1_bias.inc', f'{lname}_ffn1 bias')
        total += write_inc(sd[f'{prefix}.linear2.weight'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_ffn2_weight.inc', f'{lname}_ffn2')
        total += write_inc(sd[f'{prefix}.linear2.bias'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_ffn2_bias.inc', f'{lname}_ffn2 bias')

        # norm2 (pre-FFN norm)
        total += write_inc(sd[f'{prefix}.norm2.weight'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_norm2_weight.inc', f'{lname}_norm2')
        total += write_inc(sd[f'{prefix}.norm2.bias'].numpy(),
                           OUTPUT_DIR / f'tf{layer_idx}_norm2_bias.inc', f'{lname}_norm2 bias')

    # Task heads (3-layer each: idx 0=FC1, 1=LN, 4=FC2, 7=FC3)
    for task, prefix in [('activity', 'activity'), ('stress', 'stress'), ('arrhythmia', 'arrhythmia')]:
        for idx, name in [(0, 'fc1'), (1, 'ln'), (4, 'fc2'), (7, 'fc3')]:
            w = sd[f'task_heads.{task}.{idx}.weight'].numpy()
            b = sd[f'task_heads.{task}.{idx}.bias'].numpy()
            total += write_inc(w, OUTPUT_DIR / f'{prefix}_head_{name}_weight.inc', f'{prefix}_{name}')
            total += write_inc(b, OUTPUT_DIR / f'{prefix}_head_{name}_bias.inc', f'{prefix}_{name} bias')

    print(f"\nTotal: {total:,} params ({total*4/1024:.1f} KB)")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
