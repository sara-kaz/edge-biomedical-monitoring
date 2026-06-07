#!/usr/bin/env python3
"""
Train V3 model WITHOUT transformer for firmware deployment.

V3 architecture (CNN + SE-Attention + Projection) but no transformer.
This matches what the firmware can implement while keeping the strong
SE-attention that boosted V5 accuracy.

Firmware inference path will be:
  InputBN → Conv1/BN/Pool → Conv2/BN/Pool → Conv3/BN/Pool →
  SE-Attention → GlobalAvgPool → Projection → LayerNorm → TaskHeads
"""

import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Activity merging
ACTIVITY_MERGE_MAP = {0:0, 1:1, 2:2, 3:0, 4:0, 5:3, 6:3, 7:0}
MERGED_ACTIVITY_NAMES = {0:'sedentary', 1:'walking', 2:'cycling', 3:'high_intensity'}
N_MERGED_ACTIVITY = 4


# ─── Dataset (same as train_v3.py) ───
class BiomedicalDatasetV3(Dataset):
    def __init__(self, data_path, subject_filter=None, binary_stress=True,
                 augment=False, normalize=True, oversample=False, merge_activity=True):
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        self.binary_stress = binary_stress
        self.augment = augment
        self.normalize = normalize
        self.merge_activity = merge_activity
        self.n_channels = None
        self.samples, self.activity_labels, self.stress_labels, self.arrhythmia_labels = [], [], [], []

        if normalize:
            all_data = [s.get('window_data') for s in raw_data
                        if (not subject_filter or s.get('subject_id') in subject_filter)
                        and s.get('window_data') is not None]
            if all_data:
                stacked = np.stack(all_data[:5000])
                self.ch_mean = np.nanmean(stacked, axis=(0,2), keepdims=True).squeeze(0)
                self.ch_std = np.nanstd(stacked, axis=(0,2), keepdims=True).squeeze(0)
                self.ch_std[self.ch_std < 1e-6] = 1.0
            else:
                self.ch_mean = self.ch_std = None

        for sample in raw_data:
            sid = sample.get('subject_id', 'unknown')
            if subject_filter and sid not in subject_filter:
                continue
            wd = sample.get('window_data')
            if wd is None: continue
            wd = np.nan_to_num(wd, nan=0.0, posinf=0.0, neginf=0.0)
            if self.n_channels is None: self.n_channels = wd.shape[0]
            labels = sample.get('labels', {})
            a = self._dec(labels.get('activity'))
            s = self._dec(labels.get('stress'))
            ar = self._dec(labels.get('arrhythmia'))
            if merge_activity and a >= 0: a = ACTIVITY_MERGE_MAP.get(a, -1)
            if binary_stress and s >= 0: s = 1 if s == 1 else 0
            self.samples.append(wd.astype(np.float32))
            self.activity_labels.append(a)
            self.stress_labels.append(s)
            self.arrhythmia_labels.append(ar)

        if oversample: self._oversample()
        print(f"  {len(self.samples)} samples, {self.n_channels}ch")
        self._stats()

    def _dec(self, d):
        if d is None: return -1
        if isinstance(d, np.ndarray): return int(np.argmax(d)) if d.sum()>0 else -1
        return int(d) if isinstance(d, (int, float)) else -1

    def _oversample(self):
        n_act = N_MERGED_ACTIVITY if self.merge_activity else 8
        va = [(i, self.activity_labels[i]) for i in range(len(self.samples)) if self.activity_labels[i]>=0]
        if va:
            ac = Counter(a for _,a in va)
            sc = sorted(ac.values())
            tgt = min(int(np.sqrt(sc[len(sc)//2]*sc[-1])), sc[-1]//2)
            for cls in range(n_act):
                ci = [i for i,a in va if a==cls]
                if len(ci)<2: continue
                need = min(tgt-len(ci), len(ci)*10)
                if need<=0: continue
                for _ in range(need):
                    i1,i2 = random.sample(ci,2); lam=random.uniform(0.3,0.7)
                    syn = self.samples[i1]*lam + self.samples[i2]*(1-lam) + np.random.randn(*self.samples[i1].shape).astype(np.float32)*0.02
                    self.samples.append(syn); self.activity_labels.append(cls)
                    self.stress_labels.append(self.stress_labels[i1]); self.arrhythmia_labels.append(self.arrhythmia_labels[i1])
        # Arrhythmia
        av = [(i,self.arrhythmia_labels[i]) for i in range(len(self.samples)) if self.arrhythmia_labels[i]>=0]
        if av:
            ac2 = Counter(a for _,a in av)
            if 1 in ac2 and 0 in ac2:
                ci = [i for i,a in av if a==1]; need = int(ac2[0]*0.4)-len(ci)
                if need>0 and len(ci)>=2:
                    for _ in range(need):
                        i1,i2=random.sample(ci,2); lam=random.uniform(0.3,0.7)
                        syn=self.samples[i1]*lam+self.samples[i2]*(1-lam)+np.random.randn(*self.samples[i1].shape).astype(np.float32)*0.02
                        self.samples.append(syn); self.activity_labels.append(self.activity_labels[i1])
                        self.stress_labels.append(self.stress_labels[i1]); self.arrhythmia_labels.append(1)
        # Stress
        sv = [(i,self.stress_labels[i]) for i in range(len(self.samples)) if self.stress_labels[i]>=0]
        if sv:
            sc2 = Counter(s for _,s in sv)
            if 1 in sc2 and 0 in sc2:
                ci=[i for i,s in sv if s==1]; need=int(sc2[0]*0.8)-len(ci)
                if need>0 and len(ci)>=2:
                    for _ in range(need):
                        i1,i2=random.sample(ci,2); lam=random.uniform(0.3,0.7)
                        syn=self.samples[i1]*lam+self.samples[i2]*(1-lam)+np.random.randn(*self.samples[i1].shape).astype(np.float32)*0.02
                        self.samples.append(syn); self.activity_labels.append(self.activity_labels[i1])
                        self.stress_labels.append(1); self.arrhythmia_labels.append(self.arrhythmia_labels[i1])

    def _stats(self):
        for name, labels in [('Act', self.activity_labels), ('Str', self.stress_labels), ('Arr', self.arrhythmia_labels)]:
            v = [l for l in labels if l>=0]
            print(f"    {name}({len(v)}): {dict(Counter(v))}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].copy()
        if self.normalize and self.ch_mean is not None:
            x = (x - self.ch_mean) / self.ch_std
        x = torch.from_numpy(x.astype(np.float32))
        if self.augment:
            if random.random()<0.5: x = x + torch.randn_like(x)*0.05
            if random.random()<0.3: x = torch.roll(x, random.randint(-50,50), dims=-1)
            if random.random()<0.4:
                for ch in range(x.size(0)): x[ch] *= random.uniform(0.8,1.2)
            if random.random()<0.3:
                ml=random.randint(20,100); st=random.randint(0,x.size(-1)-ml); x[...,st:st+ml]=0
            if random.random()<0.15: x[random.randint(0,x.size(0)-1)]=0
        return {'window_data': x, 'activity': self.activity_labels[idx],
                'stress': self.stress_labels[idx], 'arrhythmia': self.arrhythmia_labels[idx]}

    def get_sample_weights(self):
        weights = torch.ones(len(self.samples))
        ac=Counter(a for a in self.activity_labels if a>=0)
        sc=Counter(s for s in self.stress_labels if s>=0)
        rc=Counter(a for a in self.arrhythmia_labels if a>=0)
        for i in range(len(self.samples)):
            w=1.0; a,s,ar = self.activity_labels[i],self.stress_labels[i],self.arrhythmia_labels[i]
            if a>=0 and a in ac: w=max(w, max(ac.values())/ac[a])
            if s>=0 and s in sc: w=max(w, max(sc.values())/sc[s])
            if ar>=0 and ar in rc: w=max(w, max(rc.values())/rc[ar])
            weights[i]=w
        return weights

    def get_class_weights(self):
        result = {}
        n_act = N_MERGED_ACTIVITY if self.merge_activity else 8
        n_str = 2 if self.binary_stress else 4
        for task, n, labels in [('activity',n_act,self.activity_labels),('stress',n_str,self.stress_labels),('arrhythmia',2,self.arrhythmia_labels)]:
            counts = np.zeros(n)
            for l in labels:
                if 0<=l<n: counts[l]+=1
            total = counts.sum()
            if total>0:
                w = np.sqrt(total/(n*np.maximum(counts,1))); w = np.clip(w,0.5,5.0)
                result[task] = torch.FloatTensor(w)
            else: result[task] = torch.ones(n)
        return result


# ─── V3 Model (CNN + SE, NO transformer) ───
class CNNSE_ForFirmware(nn.Module):
    """
    V3-style model without transformer. Matches firmware inference path exactly.

    Architecture:
      InputBN → Conv1(5→32,k7)/BN/ReLU/Pool → Conv2(32→64,k5)/BN/ReLU/Pool →
      Conv3(64→64,k3)/BN/ReLU/Pool → SE(64→16→64) → GlobalAvgPool →
      Projection(64→64)/LN → TaskHeads
    """
    def __init__(self, n_channels=5, d_model=64, activity_classes=4,
                 stress_classes=2, arrhythmia_classes=2, dropout=0.3):
        super().__init__()

        # Input batch norm (will be fused into conv1 at export)
        self.input_bn = nn.BatchNorm1d(n_channels)

        # CNN backbone — same as V3/V5
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.cnn_dropout = nn.Dropout(dropout)

        # Squeeze-and-Excitation attention
        self.se_fc1 = nn.Linear(64, 16)
        self.se_fc2 = nn.Linear(16, 64)

        # Projection + LayerNorm
        self.channel_projection = nn.Linear(64, d_model)
        self.proj_ln = nn.LayerNorm(d_model)
        self.proj_drop = nn.Dropout(dropout)

        # Task heads (same structure as V3 — 3-layer MLPs)
        self.task_heads = nn.ModuleDict({
            "activity": self._make_head(d_model, 64, activity_classes, dropout),
            "stress": self._make_head(d_model, 48, stress_classes, dropout),
            "arrhythmia": self._make_head(d_model, 48, arrhythmia_classes, dropout),
        })

        self._init_weights()

    def _make_head(self, in_dim, hidden, n_classes, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, hidden),       # idx 0
            nn.LayerNorm(hidden),            # idx 1
            nn.GELU(),                       # idx 2
            nn.Dropout(dropout),             # idx 3
            nn.Linear(hidden, hidden // 2),  # idx 4
            nn.GELU(),                       # idx 5
            nn.Dropout(dropout * 0.5),       # idx 6
            nn.Linear(hidden // 2, n_classes), # idx 7
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input normalization
        x = self.input_bn(x)

        # CNN backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.cnn_dropout(x)

        # SE attention
        w = x.mean(dim=-1)           # [B, 64]
        w = F.relu(self.se_fc1(w))   # [B, 16]
        w = torch.sigmoid(self.se_fc2(w))  # [B, 64]
        x = x * w.unsqueeze(-1)      # [B, 64, T] * [B, 64, 1]

        # Global avg pool + projection
        z = x.mean(dim=-1)           # [B, 64]
        z = self.channel_projection(z)  # [B, d_model]
        z = self.proj_ln(z)
        z = self.proj_drop(z)

        return {
            "activity": self.task_heads["activity"](z),
            "stress": self.task_heads["stress"](z),
            "arrhythmia": self.task_heads["arrhythmia"](z),
        }


# ─── Focal Loss ───
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ─── Training ───
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Find dataset
    for p in [PROJECT_ROOT / 'processed_unified_dataset' / 'unified_dataset.pkl',
              PROJECT_ROOT.parent / 'multimodal-biomedical-monitoring-improved' / 'processed_unified_dataset' / 'unified_dataset.pkl']:
        if p.exists():
            dataset_path = str(p); break
    else:
        print("ERROR: Cannot find dataset"); sys.exit(1)

    # Splits
    sp = PROJECT_ROOT / 'training_results' / 'subject_splits_v3.json'
    if not sp.exists(): sp = PROJECT_ROOT / 'training_results' / 'subject_splits.json'
    with open(sp) as f: splits = json.load(f)

    print(f"\nDataset: {dataset_path}")
    print(f"Splits: {len(splits['train_subjects'])}T/{len(splits['val_subjects'])}V/{len(splits['test_subjects'])}Te")

    # Datasets
    print("\n--- Train ---")
    train_ds = BiomedicalDatasetV3(dataset_path, splits['train_subjects'], augment=True, oversample=True)
    print("--- Val ---")
    val_ds = BiomedicalDatasetV3(dataset_path, splits['val_subjects'])
    print("--- Test ---")
    test_ds = BiomedicalDatasetV3(dataset_path, splits['test_subjects'])

    n_ch = train_ds.n_channels
    act_cls, str_cls = N_MERGED_ACTIVITY, 2

    sw = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(sw, len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = CNNSE_ForFirmware(n_channels=n_ch, d_model=64, activity_classes=act_cls,
                               stress_classes=str_cls, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: CNNSE_ForFirmware — {n_params:,} params ({n_params*4/1024:.1f} KB)")

    # Loss
    cw = train_ds.get_class_weights()
    loss_fns = {
        'activity': FocalLoss(gamma=2.5, weight=cw['activity'].to(device)),
        'stress': FocalLoss(gamma=1.0, weight=cw['stress'].to(device)),
        'arrhythmia': FocalLoss(gamma=2.5, weight=cw['arrhythmia'].to(device)),
    }
    tw = {'activity': 1.0, 'stress': 1.5, 'arrhythmia': 2.0}

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                           steps_per_epoch=len(train_loader), pct_start=0.1)

    best_score, patience_ctr, best_state = 0, 0, None

    for epoch in range(args.epochs):
        model.train()
        total_loss, nb = 0, 0
        for batch in train_loader:
            x = batch['window_data'].to(device)
            out = model(x)
            loss = torch.tensor(0.0, device=device)
            for task in ['activity', 'stress', 'arrhythmia']:
                lbl = batch[task]; valid = lbl >= 0
                if valid.sum() == 0: continue
                loss += tw[task] * loss_fns[task](out[task][valid], lbl[valid].to(device))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item(); nb += 1

        # Validate
        vm = evaluate(model, val_loader, device)
        score = (vm['activity_f1'] + vm['stress_f1'] + vm['arrhythmia_f1']) / 3

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Ep {epoch+1:3d}/{args.epochs} | L:{total_loss/max(nb,1):.4f} | "
                  f"Act:{vm['activity_f1']:.3f} Str:{vm['stress_f1']:.3f} Arr:{vm['arrhythmia_f1']:.3f} | S:{score:.3f}")

        if score > best_score:
            best_score = score; patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= args.patience:
            print(f"\nEarly stop at epoch {epoch+1}")
            break

    # Load best & test
    if best_state: model.load_state_dict(best_state)
    model.to(device)
    print("\n" + "="*60 + "\nTEST RESULTS\n" + "="*60)
    tm = evaluate(model, test_loader, device)
    for task in ['activity', 'stress', 'arrhythmia']:
        print(f"  {task:12s} F1={tm[f'{task}_f1']:.4f}  Acc={tm[f'{task}_acc']:.4f}")

    # Save
    out_path = PROJECT_ROOT / 'training_results' / 'model_v3_no_transformer.pth'
    torch.save({'model_state_dict': {k:v.cpu() for k,v in model.state_dict().items()},
                'test_metrics': tm, 'best_val_score': best_score,
                'n_channels': n_ch, 'activity_classes': act_cls, 'stress_classes': str_cls},
               out_path)
    print(f"\nSaved: {out_path}")

    # Export weights for firmware
    print("\n" + "="*60 + "\nEXPORTING WEIGHTS\n" + "="*60)
    export_v3_weights(model, n_ch, act_cls, str_cls)

    return tm


def evaluate(model, loader, device):
    from sklearn.metrics import f1_score, accuracy_score
    model.eval()
    preds = {t: [] for t in ['activity','stress','arrhythmia']}
    labels = {t: [] for t in ['activity','stress','arrhythmia']}
    with torch.no_grad():
        for batch in loader:
            x = batch['window_data'].to(device)
            out = model(x)
            for task in ['activity','stress','arrhythmia']:
                lbl = batch[task]; valid = lbl >= 0
                if valid.sum() > 0:
                    preds[task].extend(out[task][valid].argmax(1).cpu().numpy())
                    labels[task].extend(lbl[valid].numpy())
    m = {}
    for task in ['activity','stress','arrhythmia']:
        if preds[task]:
            m[f'{task}_f1'] = f1_score(labels[task], preds[task], average='macro', zero_division=0)
            m[f'{task}_acc'] = accuracy_score(labels[task], preds[task])
        else:
            m[f'{task}_f1'] = m[f'{task}_acc'] = 0
    return m


def export_v3_weights(model, n_channels, act_cls, str_cls):
    """Export V3-no-transformer weights to firmware .inc files."""
    output_dir = PROJECT_ROOT / 'firmware' / 'esp32' / 'src' / 'inference' / 'weights'
    output_dir.mkdir(parents=True, exist_ok=True)

    sd = {k: v.cpu() for k, v in model.state_dict().items()}

    def fuse_conv_bn(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
        if conv_b is None: conv_b = torch.zeros(conv_w.shape[0])
        scale = bn_w / torch.sqrt(bn_var + eps)
        return (conv_w * scale.view(-1,1,1)).numpy(), ((conv_b - bn_mean)*scale + bn_b).numpy()

    def fuse_input_bn_into_conv1(input_bn_w, input_bn_b, input_bn_mean, input_bn_var,
                                  conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
        """Fuse input_bn + conv1 + bn1 into a single conv+bias."""
        # First: input_bn normalizes input channels: x_norm = (x - in_mean) * in_scale + in_bias
        # where in_scale = in_weight / sqrt(in_var + eps)
        in_scale = input_bn_w / torch.sqrt(input_bn_var + eps)  # [n_channels]
        in_shift = input_bn_b - input_bn_mean * in_scale  # [n_channels]

        # Fold input_bn into conv1 weights:
        # conv(input_bn(x)) = conv(x * in_scale + in_shift)
        # = sum_ic (w[oc,ic,:] * (x[ic,:] * in_scale[ic] + in_shift[ic]))
        # = sum_ic (w[oc,ic,:] * in_scale[ic]) * x[ic,:] + sum_ic (w[oc,ic,:] * in_shift[ic])
        # New conv weight: w_new[oc, ic, k] = w[oc, ic, k] * in_scale[ic]
        # New conv bias contribution: sum over ic of (sum over k of w[oc,ic,k]) * in_shift[ic]

        n_out, n_in, k = conv_w.shape
        conv_w_new = conv_w.clone()
        for ic in range(n_in):
            conv_w_new[:, ic, :] *= in_scale[ic]

        # Bias from input_bn folding
        conv_b_from_input = torch.zeros(n_out)
        for oc in range(n_out):
            for ic in range(n_in):
                conv_b_from_input[oc] += conv_w[:, ic, :].sum() / k * in_shift[ic]
                # Actually, for "same" padding convolution, each output position sees
                # the full kernel, so the bias from the shift is:
                # sum_ic (sum_k w[oc,ic,k]) * in_shift[ic] for each output position
                # But this is per-position... for simplicity, let's handle it differently

        # Actually the cleaner approach: just compute the bias contribution as
        # the conv applied to a constant input of in_shift repeated across time
        # For padding='same' conv, the bias is: sum_ic sum_k w[oc,ic,k] * in_shift[ic]
        conv_b_from_input = torch.zeros(n_out)
        for oc in range(n_out):
            for ic in range(n_in):
                conv_b_from_input[oc] += conv_w[oc, ic, :].sum() * in_shift[ic]

        if conv_b is None:
            conv_b = conv_b_from_input
        else:
            conv_b = conv_b + conv_b_from_input

        # Now fuse with BN1
        return fuse_conv_bn(conv_w_new, conv_b, bn_w, bn_b, bn_mean, bn_var, eps)

    def write_inc(arr, path, name):
        flat = arr.flatten()
        lines = []
        for i in range(0, len(flat), 8):
            lines.append(', '.join(f'{v:.6f}f' for v in flat[i:i+8]))
        with open(path, 'w') as f:
            f.write(f'// {name}: shape {list(arr.shape)}\n')
            f.write(',\n'.join(lines))
        print(f"  {name}: {list(arr.shape)} ({arr.size} params)")

    total = 0

    # Conv1 (fused with input_bn + bn1)
    w, b = fuse_input_bn_into_conv1(
        sd['input_bn.weight'], sd['input_bn.bias'],
        sd['input_bn.running_mean'], sd['input_bn.running_var'],
        sd['conv1.weight'], sd.get('conv1.bias'),
        sd['bn1.weight'], sd['bn1.bias'],
        sd['bn1.running_mean'], sd['bn1.running_var'])
    write_inc(w, output_dir/'conv1_weight.inc', 'conv1 (fused input_bn+conv1+bn1)')
    write_inc(b, output_dir/'conv1_bias.inc', 'conv1 bias')
    total += w.size + b.size

    # Conv2 (fused with bn2)
    w, b = fuse_conv_bn(sd['conv2.weight'], sd.get('conv2.bias'),
        sd['bn2.weight'], sd['bn2.bias'], sd['bn2.running_mean'], sd['bn2.running_var'])
    write_inc(w, output_dir/'conv2_weight.inc', 'conv2 (fused)')
    write_inc(b, output_dir/'conv2_bias.inc', 'conv2 bias')
    total += w.size + b.size

    # Conv3 (fused with bn3)
    w, b = fuse_conv_bn(sd['conv3.weight'], sd.get('conv3.bias'),
        sd['bn3.weight'], sd['bn3.bias'], sd['bn3.running_mean'], sd['bn3.running_var'])
    write_inc(w, output_dir/'conv3_weight.inc', 'conv3 (fused)')
    write_inc(b, output_dir/'conv3_bias.inc', 'conv3 bias')
    total += w.size + b.size

    # SE attention weights
    write_inc(sd['se_fc1.weight'].numpy(), output_dir/'se_fc1_weight.inc', 'se_fc1')
    write_inc(sd['se_fc1.bias'].numpy(), output_dir/'se_fc1_bias.inc', 'se_fc1 bias')
    write_inc(sd['se_fc2.weight'].numpy(), output_dir/'se_fc2_weight.inc', 'se_fc2')
    write_inc(sd['se_fc2.bias'].numpy(), output_dir/'se_fc2_bias.inc', 'se_fc2 bias')
    total += sd['se_fc1.weight'].numel() + sd['se_fc1.bias'].numel()
    total += sd['se_fc2.weight'].numel() + sd['se_fc2.bias'].numel()

    # Projection
    write_inc(sd['channel_projection.weight'].numpy(), output_dir/'projection_weight.inc', 'projection')
    write_inc(sd['channel_projection.bias'].numpy(), output_dir/'projection_bias.inc', 'projection bias')
    total += sd['channel_projection.weight'].numel() + sd['channel_projection.bias'].numel()

    # LayerNorm (export as separate weight+bias for firmware)
    write_inc(sd['proj_ln.weight'].numpy(), output_dir/'proj_ln_weight.inc', 'proj_layernorm')
    write_inc(sd['proj_ln.bias'].numpy(), output_dir/'proj_ln_bias.inc', 'proj_layernorm bias')
    total += sd['proj_ln.weight'].numel() + sd['proj_ln.bias'].numel()

    # Task heads — 3 FC layers each (idx 0, 4, 7)
    for task, prefix in [('activity','activity'), ('stress','stress'), ('arrhythmia','arrhythmia')]:
        # FC1 (idx 0) + LayerNorm (idx 1)
        w = sd[f'task_heads.{task}.0.weight'].numpy()
        b = sd[f'task_heads.{task}.0.bias'].numpy()
        write_inc(w, output_dir/f'{prefix}_head_fc1_weight.inc', f'{prefix}_fc1')
        write_inc(b, output_dir/f'{prefix}_head_fc1_bias.inc', f'{prefix}_fc1 bias')
        total += w.size + b.size
        # LayerNorm (idx 1)
        w = sd[f'task_heads.{task}.1.weight'].numpy()
        b = sd[f'task_heads.{task}.1.bias'].numpy()
        write_inc(w, output_dir/f'{prefix}_head_ln_weight.inc', f'{prefix}_ln')
        write_inc(b, output_dir/f'{prefix}_head_ln_bias.inc', f'{prefix}_ln bias')
        total += w.size + b.size
        # FC2 (idx 4)
        w = sd[f'task_heads.{task}.4.weight'].numpy()
        b = sd[f'task_heads.{task}.4.bias'].numpy()
        write_inc(w, output_dir/f'{prefix}_head_fc2_weight.inc', f'{prefix}_fc2')
        write_inc(b, output_dir/f'{prefix}_head_fc2_bias.inc', f'{prefix}_fc2 bias')
        total += w.size + b.size
        # FC3 (idx 7 — output layer)
        w = sd[f'task_heads.{task}.7.weight'].numpy()
        b = sd[f'task_heads.{task}.7.bias'].numpy()
        write_inc(w, output_dir/f'{prefix}_head_fc3_weight.inc', f'{prefix}_fc3')
        write_inc(b, output_dir/f'{prefix}_head_fc3_bias.inc', f'{prefix}_fc3 bias')
        total += w.size + b.size

    print(f"\n  Total: {total:,} params ({total*4/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
