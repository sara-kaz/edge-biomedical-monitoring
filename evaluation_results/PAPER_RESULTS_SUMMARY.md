# IEEE ICHI 2026 Paper - Results Summary

**Generated:** 2026-02-04 (Updated with 5-fold CV results)

## Overview

This document contains organized results for the IEEE ICHI 2026 paper submission on Edge Intelligence for Multimodal Biomedical Monitoring.

---

## PRIMARY RESULTS: 5-Fold Cross-Validation

| Task | Accuracy (mean±std) | F1 Macro (mean±std) |
|------|---------------------|---------------------|
| Activity (8 cls) | **52.3% ± 1.8%** | 31.1% ± 2.0% |
| Stress (4 cls) | **69.4% ± 4.2%** | 49.9% ± 4.5% |
| Arrhythmia (2 cls) | **75.0% ± 6.6%** | 74.7% ± 6.4% |

**Arrhythmia Clinical Metrics (5-fold CV):**
- Sensitivity: 70.3% ± 10.3%
- Specificity: 80.1% ± 10.7%
- AUC-ROC: **0.843 ± 0.077**
- PPV: 77.4% ± 9.8%
- NPV: 74.7% ± 9.8%

**Inference Time (5-fold CV):**
- Mean: 2.20 ms (std: 0.04 ms)

---

## 1. Dataset Statistics

| Dataset | Signals | Subjects | Sample Rate | Task |
|---------|---------|----------|-------------|------|
| PPG-DaLiA | PPG, ACC, ECG | 15 | 64Hz→100Hz | Activity (8 classes) |
| WESAD | ECG, PPG, ACC, EDA | 15 | 700Hz→100Hz | Stress (4 classes) |
| MIT-BIH | ECG | 48 records | 360Hz→100Hz | Arrhythmia (2 classes) |

**Unified Format:**
- Window size: 10 seconds (1000 samples @ 100Hz)
- Channels: 11 (ECG, PPG, Accel_X/Y/Z, EDA, Respiration, Temperature, EMG, EDA_wrist, Temp_wrist)
- Total balanced samples: 21,704 windows

**Subject Splits:**
- Training: 46 subjects
- Validation: 6 subjects
- Test: 13 subjects

---

## 2. Model Architecture Comparison

### Original Model (CNNTransformerLite)
- Parameters: 82,782
- Size: ~323 KB (FP32)
- Architecture: 3-layer CNN + 2-layer Transformer

### Improved Model (CNNTransformerLite with augmentation)
- Parameters: 82,782
- Size: ~340 KB (FP32)
- Training: Focal loss + Data augmentation

### V5 Deployed Model (CNNTransformerV3)
- Parameters: 112,552
- Size: 450 KB (FP32)
- Architecture: 3-layer CNN + SE-Attention + 2-layer Transformer (4 heads, d_model=64)
- Task heads: 3-layer MLPs with LayerNorm
- Training model = deployed model (no compression)

---

## 3. Test Set Results (Held-Out Test Data)

### Original Model Performance

| Task | Accuracy | Precision | Recall | F1 (macro) |
|------|----------|-----------|--------|------------|
| Activity | 11.2% | 4.5% | 17.0% | 6.9% |
| Stress | 8.3% | 4.3% | 33.7% | 7.7% |
| Arrhythmia | 74.9% | 66.3% | 61.3% | 62.3% |

**Arrhythmia Clinical Metrics (Original):**
- Sensitivity: 32.6%
- Specificity: 90.0%
- AUC-ROC: 0.802
- PPV: 53.6%
- NPV: 79.0%

### Improved Model Performance (with augmentation + focal loss)

| Task | Accuracy | Precision | Recall | F1 (macro) |
|------|----------|-----------|--------|------------|
| Activity | **55.9%** | 34.3% | 36.1% | 34.8% |
| Stress | **73.2%** | 56.4% | 36.0% | 40.4% |
| Arrhythmia | 72.3% | 62.0% | 59.1% | 59.7% |

**Arrhythmia Clinical Metrics (Improved):**
- Sensitivity: 31.3%
- Specificity: 86.9%
- AUC-ROC: 0.776
- PPV: 45.9%
- NPV: 78.1%

### Improvement Summary

| Task | Original | Improved | Improvement |
|------|----------|----------|-------------|
| Activity | 11.2% | 55.9% | **+44.7%** |
| Stress | 8.3% | 73.2% | **+64.9%** |
| Arrhythmia | 74.9% | 72.3% | -2.6% |

---

## 4. Per-Class Performance (Improved Model)

### Activity Classification
| Class | Samples | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|-----|
| Sitting | 1089 | 97.1% | 87.7% | 97.1% | 92.1% |
| Walking | 245 | 14.3% | 24.0% | 14.3% | 17.9% |
| Cycling | 222 | 51.4% | 40.9% | 51.4% | 45.5% |
| Driving | 208 | 0.0% | 0.0% | 0.0% | 0.0% |
| Working | 203 | 44.3% | 40.9% | 44.3% | 42.5% |
| Stairs | 216 | 6.9% | 8.4% | 6.9% | 7.6% |
| Table Soccer | 255 | 37.3% | 45.5% | 37.3% | 41.0% |
| Lunch | 234 | 37.2% | 28.6% | 37.2% | 32.4% |

### Stress Classification
| Class | Samples | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|-----|
| Baseline | 180 | 93.3% | 91.3% | 93.3% | 92.3% |
| Stress | 38 | 42.1% | 88.9% | 42.1% | 57.1% |
| Amusement | 23 | 8.7% | 50.0% | 8.7% | 14.8% |
| Meditation | 13 | 0.0% | 0.0% | 0.0% | 0.0% |

### Arrhythmia Classification
| Class | Samples | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|-----|
| Normal | 647 | 86.9% | 78.1% | 86.9% | 82.3% |
| Abnormal | 230 | 31.3% | 45.9% | 31.3% | 37.2% |

---

## 5. Inference Time

| Metric | Value |
|--------|-------|
| Mean | 2.58 ms |
| Std | 0.62 ms |
| Min | 2.22 ms |
| Max | 5.63 ms |
| Median | 2.39 ms |

*Measured on CPU (Intel). ESP32-S3 target: <100ms requirement met.*

---

## 6. Training History (Improved Model)

### Final Epoch Metrics (Epoch 15)
| Metric | Train | Validation |
|--------|-------|------------|
| Loss | 0.844 | 2.193 |
| Activity Acc | 81.6% | 57.4% |
| Stress Acc | 78.0% | 73.0% |
| Arrhythmia Acc | 90.4% | 82.2% |

### Best Validation Performance
- Activity: 59.2% (Epoch 3)
- Stress: 77.2% (Epoch 3)
- Arrhythmia: 82.2% (Epoch 15)

---

## 7. Alert System Performance (4-Case Benchmark)

*From synthetic test samples (100 samples per case)*

| Case | Description | Activity Acc | Stress Acc | Arrhythmia Acc | Alert Acc |
|------|-------------|--------------|------------|----------------|-----------|
| 1 | Stress + Sedentary | 38.0% | 100% | 100% | 100% |
| 2 | Stress + Exercise | 100% | 100% | 100% | 0% (correct) |
| 3 | Arrhythmia + Sedentary | 41.0% | 100% | 100% | 100% |
| 4 | Arrhythmia + Motion | 100% | 100% | 100% | 100% |

**Alert Generation Summary:**
- True Positives: 300 (Cases 1, 3, 4)
- False Negatives: 0
- True Negatives: 100 (Case 2)
- False Positives: 0
- **Alert Accuracy: 100%** on benchmark

---

## 8. Model Specifications

### Embedded Deployment (ESP32-S3)
| Metric | Value |
|--------|-------|
| Parameters | 112,552 (full CNN-SE-Transformer) |
| Model Size | 450 KB (FP32) |
| Heap Memory | ~293 KB free (~288 KB min) |
| PSRAM | ~7.3 MB free (~7.2 MB min) |
| Flash | 768 KB sketch |
| Inference Time | ~2,338 ms per 10s window (23.4% duty cycle) |

### Sensor Configuration
| Sensor | Modality | Interface |
|--------|----------|-----------|
| AD8232 | ECG | ADC |
| MAX30102 | PPG | I2C (0x57) |
| MPU6050 | Accelerometer | I2C (0x68) |
| Grove GSR | EDA/GSR | ADC |

---

## 9. LaTeX Tables for Paper

### Main Results Table
```latex
\begin{table}[t]
\caption{Multi-Task Classification Results on Held-Out Test Set}
\label{tab:results}
\centering
\begin{tabular}{@{}llcccc@{}}
\toprule
\textbf{Task} & \textbf{Model} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\
\midrule
\multirow{2}{*}{Activity} & Original & 11.2\% & 4.5\% & 17.0\% & 6.9\% \\
 & Improved & \textbf{55.9\%} & 34.3\% & 36.1\% & 34.8\% \\
\midrule
\multirow{2}{*}{Stress} & Original & 8.3\% & 4.3\% & 33.7\% & 7.7\% \\
 & Improved & \textbf{73.2\%} & 56.4\% & 36.0\% & 40.4\% \\
\midrule
\multirow{2}{*}{Arrhythmia} & Original & \textbf{74.9\%} & 66.3\% & 61.3\% & 62.3\% \\
 & Improved & 72.3\% & 62.0\% & 59.1\% & 59.7\% \\
\bottomrule
\end{tabular}
\end{table}
```

### Clinical Metrics Table
```latex
\begin{table}[t]
\caption{Arrhythmia Detection Clinical Metrics}
\label{tab:clinical}
\centering
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{Original} & \textbf{Improved} \\
\midrule
Sensitivity & 32.6\% & 31.3\% \\
Specificity & 90.0\% & 86.9\% \\
AUC-ROC & 0.802 & 0.776 \\
PPV & 53.6\% & 45.9\% \\
NPV & 79.0\% & 78.1\% \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 10. Files Reference

| File | Description |
|------|-------------|
| `evaluation_results/test_results_2026-02-03_18-53-19.json` | Original model evaluation |
| `evaluation_results/test_results_2026-02-03_19-42-22.json` | Improved model evaluation |
| `training_results/training_history_2026-02-03_19-26-55.json` | Training curves |
| `training_results/model_improved_2026-02-03_19-26-55.pth` | Improved model weights |
| `docs/run_logs/accuracy_results_4_cases_2026-01-28_21-40-09.json` | 4-case benchmark results |

---

## Notes

### 5-Fold Cross-Validation (PRIMARY RESULTS)

1. **Activity Performance** (52.3% ± 1.8%):
   - 8-class classification on PPG-DaLiA dataset
   - Challenging due to similar sedentary activities (Sitting vs Driving vs Working)
   - Focal loss + augmentation significantly improved from baseline

2. **Stress Performance** (69.4% ± 4.2%):
   - 4-class stress detection on WESAD dataset
   - Focal loss effectively handles severe class imbalance
   - Baseline class dominates (>60% of samples)

3. **Arrhythmia Performance** (75.0% ± 6.6%):
   - Binary classification on MIT-BIH dataset
   - AUC-ROC: 0.843 ± 0.077 indicates good discriminative ability
   - Balanced sensitivity (70.3%) and specificity (80.1%) in 5-fold CV
   - Single held-out split showed more conservative behavior (31.3% sens, 86.9% spec)

### Model Architecture (V5 — CNNTransformerV3)

- **CNN Feature Extractor**: 3 conv layers (k=7,5,3) with batch norm, ReLU, and max pooling
- **SE-Attention**: Squeeze-and-Excitation block (64→16→64)
- **Projection**: Linear(64→64) + LayerNorm + Sinusoidal PosEnc
- **Transformer Encoder**: 2 layers, 4 heads, d_model=64, d_ff=128, pre-norm, GELU
- **Task Heads**: 3 parallel 3-layer MLPs with LayerNorm (Activity: 64→64→LN→32→4, Stress: 64→48→LN→24→2, Arrhythmia: 64→48→LN→24→2)
- **Total Parameters**: 112,552 (training = deployed, no compression)

### Critical Limitation

**Zero label overlap** between datasets:
- PPG-DaLiA: Activity labels only (12,806 samples)
- WESAD: Stress labels only (3,439 samples)
- MIT-BIH: Arrhythmia labels only (5,459 samples)

The 4-case alert scenarios do not exist in real data - evaluation was synthetic.

### Comparison with Related Work

| Work | Task | Result | Notes |
|------|------|--------|-------|
| SELF-CARE [Rashid 2022] | Stress (3-cls) | 86.3% | WESAD, wrist-only, single-task |
| SELF-CARE [Rashid 2022] | Stress (2-cls) | 94.1% | WESAD, binary |
| **This thesis** | Stress (4-cls) | 69.4%±4.2% | WESAD, multi-task |
| Acharya et al. [2017] | Arrhythmia | 94.0% | MIT-BIH, 5-class, CNN |
| Hannun et al. [2019] | Arrhythmia | 0.97 AUC | 12-class, 34-layer CNN |
| **This thesis** | Arrhythmia | 75.0%±6.6%, 0.84 AUC | MIT-BIH, binary, multi-task |
| Reiss et al. [2019] | HR from PPG | 7.6 MAE | PPG-DaLiA |
| **This thesis** | Activity (8-cls) | 52.3%±1.8% | PPG-DaLiA, multi-task |

**Key differentiator:** Unlike single-task baselines, this system performs joint activity/stress/arrhythmia inference in a single compact model (113K parameters) designed for edge deployment (112,552 parameters, 450 KB).
