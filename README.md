# Edge Intelligence for Multimodal Biomedical Monitoring

**IEEE ICHI 2026** — Code and resources for the paper:

> **"Edge Intelligence for Multimodal Biomedical Monitoring: A Wearable Sensor-Fusion System for Simultaneous Activity, Stress, and Arrhythmia Detection on ESP32-S3"**
> *Sara Khaled, [Co-authors] — IEEE International Conference on Healthcare Informatics (ICHI) 2026*
>
> 📄 Paper link: _to be added upon publication_
> 📎 DOI: _to be added upon publication_

---

## Overview

This repository contains the full implementation of a **multi-task, edge-deployable neural network** that simultaneously classifies:

- **Physical Activity** (4 classes: Sedentary, Walking, Cycling, High-Intensity) — from PPG + Accelerometer
- **Stress Level** (2 classes: Baseline, Stress) — from ECG + PPG + EDA
- **Arrhythmia** (2 classes: Normal, Abnormal) — from ECG

The model runs directly on an **ESP32-S3** microcontroller (no cloud, no compression), processing 10-second windows of multimodal biosignals at 100 Hz.

---

## System Architecture

```
Sensors (AD8232 ECG · MAX30102 PPG · MPU6050 IMU · Grove GSR)
        │
        ▼
ESP32-S3 Firmware  ──►  Signal Buffering & Normalization
        │
        ▼
CNNTransformerV3 (112,552 params · 450 KB · FP32)
 ├── CNN Backbone   Conv1D × 3  (k = 7, 5, 3)
 ├── SE-Attention   Squeeze-and-Excitation (64→16→64)
 ├── Transformer    2 layers · 4 heads · d_model = 64
 └── Task Heads     3 parallel MLPs (Activity · Stress · Arrhythmia)
        │
        ▼
  Contextual Alert Logic
  (e.g. suppress arrhythmia alert during high-intensity exercise)
```

### Hardware

| Sensor | Modality | Interface |
|--------|----------|-----------|
| AD8232 | ECG | ADC |
| MAX30102 | PPG (SpO₂) | I²C (0x57) |
| MPU6050 | Accelerometer / Gyro | I²C (0x68) |
| Grove GSR | EDA / GSR | ADC |

Wiring diagram: [`figures/hardware_wiring_diagram.png`](figures/hardware_wiring_diagram.png)

---

## Results

### 5-Fold Subject-Wise Cross-Validation (Primary)

| Task | Accuracy | F1 Macro | AUC-ROC |
|------|----------|----------|---------|
| Activity (4-class) | 59.2% ± 15.0% | — | 0.707 ± 0.064 |
| Stress (2-class) | 74.5% ± 6.0% | — | 0.586 ± 0.095 |
| Arrhythmia (2-class) | 80.0% ± 5.7% | — | **0.832 ± 0.057** |

### ESP32-S3 Deployment

| Metric | Value |
|--------|-------|
| Inference time | 2,338 ms / 10 s window |
| Duty cycle | 23.4% |
| Power draw | 0.54 W (5.23 V · 0.10 A) |
| Flash footprint | 768 KB |
| Free PSRAM | 7.3 MB |

Full results and clinical metrics: [`evaluation_results/PAPER_RESULTS_SUMMARY.md`](evaluation_results/PAPER_RESULTS_SUMMARY.md)

---

## Repository Structure

```
├── src/                         # Core Python modules
│   ├── data_loader.py           # Unified dataset loading (PPG-DaLiA · WESAD · MIT-BIH)
│   ├── dataset_integration.py   # Cross-dataset normalisation & windowing
│   ├── unify_data.py            # 100 Hz resampling pipeline
│   └── models/
│       ├── cnn_transformer_lite.py   # CNNTransformerV3 (main model)
│       └── legacy_model.py
│
├── scripts/                     # Training & evaluation scripts
│   ├── train_model.py           # Baseline training
│   ├── train_improved.py        # Focal loss + augmentation
│   ├── cv_model_c.py            # 5-fold cross-validation
│   ├── comprehensive_evaluation.py
│   ├── ablation_study.py
│   ├── convert_to_tflite.py     # TFLite export
│   ├── export_weights_esp32.py  # C header generation for firmware
│   └── ...
│
├── firmware/esp32/              # PlatformIO ESP32-S3 firmware (C++)
│   ├── platformio.ini
│   └── src/
│       ├── main.cpp
│       ├── inference/           # On-device NN inference engine
│       ├── sensors/             # AD8232 · MAX30102 · MPU6050 · GSR drivers
│       ├── filters/             # ECG bandpass filter
│       ├── buffering/           # Sliding-window buffer
│       └── transport/           # Serial streaming / console
│
├── edge_deployment/             # Deployment configs & normalisation stats
│   ├── deployment_config.json
│   ├── signal_normalization_stats.json
│   ├── training_norm_stats.json
│   └── test_sample.json
│
├── evaluation_results/          # Saved metrics & paper numbers
│   ├── PAPER_RESULTS_SUMMARY.md
│   ├── cv_results_5fold_*.json
│   └── paper_numbers.json
│
├── training_results/
│   └── subject_splits.json      # Train / val / test subject IDs
│
├── figures/                     # Pipeline diagrams & result plots
│   ├── fig1_pipeline.jpg
│   ├── fig2_components.jpg
│   ├── hardware_wiring_diagram.png
│   ├── results_case_metrics.pdf
│   └── results_score_proxies.pdf
│
└── requirements.txt
```

---

## Datasets

The model is trained on three publicly available datasets unified into a common 100 Hz, 5-channel format (ECG · PPG · AccX · AccY · AccZ):

| Dataset | Task | Subjects | Original Rate |
|---------|------|----------|---------------|
| [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) | Activity (4-class) | 15 | 64 Hz |
| [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) | Stress (2-class) | 15 | 700 Hz |
| [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/) | Arrhythmia (2-class) | 48 records | 360 Hz |

**Total:** 21,704 balanced windows · 65 subjects · 46 train / 6 val / 13 test

> **Note:** The three datasets have zero label overlap — each window carries only the label from its source dataset. The 4-case alert benchmark uses synthetic signal compositions to evaluate the alert logic.

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Requires **Python 3.9** and **PyTorch ≥ 1.12**.

---

## Usage

### 1 · Reproduce cross-validation results
```bash
python scripts/cv_model_c.py
```

### 2 · Train from scratch
```bash
python scripts/train_improved.py        # focal loss + augmentation
```

### 3 · Run comprehensive evaluation
```bash
python scripts/comprehensive_evaluation.py
```

### 4 · Export weights for ESP32-S3 firmware
```bash
python scripts/export_weights_esp32.py  # generates firmware/esp32/src/inference/model_weights.h
```

### 5 · Flash firmware
```bash
cd firmware/esp32
pio run --target upload
```

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{khaled2026edge,
  title     = {Edge Intelligence for Multimodal Biomedical Monitoring:
               A Wearable Sensor-Fusion System for Simultaneous Activity,
               Stress, and Arrhythmia Detection on {ESP32-S3}},
  author    = {Khaled, Sara and {others}},
  booktitle = {2026 IEEE International Conference on Healthcare Informatics (ICHI)},
  year      = {2026},
  note      = {DOI to be added upon publication}
}
```

---

## License

This code is released for academic use. See [LICENSE](LICENSE) for details.
