# Edge Intelligence for Multimodal Biomedical Monitoring

**IEEE ICHI 2026** — Can a $5 microcontroller detect stress, arrhythmia, and physical activity — simultaneously — from raw wrist signals, with no cloud connection?

This paper says yes.

> **"Edge Intelligence for Multimodal Biomedical Monitoring: A Wearable Sensor-Fusion System for Simultaneous Activity, Stress, and Arrhythmia Detection on ESP32-S3"**
> *Sara Khaled et al. — IEEE International Conference on Healthcare Informatics (ICHI) 2026*
>
> 📄 **Paper:** _link coming upon publication_
> 📎 **DOI:** _coming upon publication_

---

## What it does

A single compact neural network (113K parameters, 450 KB) runs **entirely on an ESP32-S3** and jointly predicts three health signals every 10 seconds from raw biosensor data:

| Output | Task | Classes |
|--------|------|---------|
| 🏃 Activity | What are you doing? | Sedentary · Walking · Cycling · High-Intensity |
| 😰 Stress | Are you stressed? | Baseline · Stress |
| 💓 Arrhythmia | Is your heart rhythm normal? | Normal · Abnormal |

The system also applies **contextual alert logic** — for example, it suppresses an arrhythmia alert during a high-intensity workout, reducing false alarms.

---

## Hardware

Four sensors feed into a single ESP32-S3 board over ADC and I²C:

![Hardware wiring diagram](figures/hardware_wiring_diagram.png)

| Sensor | Signal | Interface |
|--------|--------|-----------|
| AD8232 | ECG | ADC |
| MAX30102 | PPG (pulse oximetry) | I²C (0x57) |
| MPU6050 / GY-521 | Accelerometer + Gyro | I²C (0x68) |
| Grove GSR | EDA / Galvanic Skin Response | ADC |

---

## Model Architecture

The model is called **CNNTransformerV3**. The design philosophy: squeeze maximum accuracy into minimum flash.

![Model pipeline](figures/fig1_pipeline.jpg)

The pipeline takes a 10-second window (5 channels × 1000 samples) and passes it through four stages:

![Architecture components](figures/fig2_components.jpg)

**(a) Multi-Scale CNN** — four parallel Conv1D branches (k = 3, 5, 7, 11) capture features at different temporal resolutions and concatenate into a 32-channel representation.

**(b) Residual-Conv + SE Block** — a Squeeze-and-Excitation block recalibrates channel importance, helping the model focus on the most informative sensor at each moment.

**(c) Transformer Encoder** — 2 layers, 4 heads, d_model = 64. Self-attention learns temporal dependencies across the 10-second window.

**(d) Task Heads** — three parallel MLPs share the same embedding and each produce one prediction.

**The training model is the deployed model — no quantization or compression needed.**

---

## Results

### 5-Fold Subject-Wise Cross-Validation

| Task | Accuracy | AUC-ROC |
|------|----------|---------|
| Activity (4-class) | 59.2% ± 15.0% | 0.707 ± 0.064 |
| Stress (2-class) | 74.5% ± 6.0% | 0.586 ± 0.095 |
| Arrhythmia (2-class) | **80.0% ± 5.7%** | **0.832 ± 0.057** |

### On-Device Performance (ESP32-S3)

| Metric | Value |
|--------|-------|
| Inference time | 2,338 ms per 10 s window |
| Duty cycle | 23.4% |
| Power draw | 0.54 W |
| Flash footprint | 768 KB |
| Free PSRAM | 7.3 MB |

### Alert Logic Benchmark (4 Synthetic Scenarios)

| Scenario | Expected alert | Correct? |
|----------|---------------|---------|
| Stress + sedentary | Fire alert | 96% accuracy |
| Stress + intense exercise | Suppress alert | 97% suppression |
| Arrhythmia + sedentary | Fire alert | 82% accuracy |
| Arrhythmia + motion | Fire alert | 88% accuracy |

> Full metrics and per-class breakdowns: [`evaluation_results/PAPER_RESULTS_SUMMARY.md`](evaluation_results/PAPER_RESULTS_SUMMARY.md)

---

## Datasets

Three public datasets are unified into a common format (100 Hz, 5 channels: ECG · PPG · AccX · AccY · AccZ):

| Dataset | Task | Subjects | Original Rate | Samples |
|---------|------|----------|---------------|---------|
| [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) | Activity | 15 | 64 Hz | 12,806 |
| [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) | Stress | 15 | 700 Hz | 3,439 |
| [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/) | Arrhythmia | 48 records | 360 Hz | 5,459 |

**Total: 21,704 balanced windows · 65 subjects · split 46 train / 6 val / 13 test**

> Note: Each dataset only provides labels for its own task (zero overlap). The 4-case alert benchmark uses synthetic signal compositions.

---

## Repository Structure

```
├── src/                    # Core Python modules
│   ├── data_loader.py      # Loads PPG-DaLiA, WESAD, MIT-BIH
│   ├── dataset_integration.py
│   ├── unify_data.py       # Resamples all signals to 100 Hz
│   └── models/
│       └── cnn_transformer_lite.py   # CNNTransformerV3
│
├── scripts/                # Training, evaluation, export
│   ├── train_improved.py   # Focal loss + augmentation
│   ├── cv_model_c.py       # 5-fold cross-validation
│   ├── comprehensive_evaluation.py
│   ├── ablation_study.py
│   └── export_weights_esp32.py  # Generates model_weights.h for firmware
│
├── firmware/esp32/         # PlatformIO C++ firmware
│   └── src/
│       ├── main.cpp
│       ├── inference/      # On-device NN engine
│       ├── sensors/        # Drivers for AD8232, MAX30102, MPU6050, GSR
│       ├── filters/        # ECG bandpass filter
│       └── buffering/      # Sliding-window buffer
│
├── edge_deployment/        # Normalisation stats + deployment config
├── evaluation_results/     # Paper metrics (JSON + Markdown)
├── figures/                # Pipeline diagrams and hardware wiring
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/sara-kaz/edge-biomedical-monitoring.git
cd edge-biomedical-monitoring
pip install -r requirements.txt
```

**Reproduce cross-validation results:**
```bash
python scripts/cv_model_c.py
```

**Train from scratch:**
```bash
python scripts/train_improved.py
```

**Export model weights to C header for ESP32:**
```bash
python scripts/export_weights_esp32.py
# → writes firmware/esp32/src/inference/model_weights.h
```

**Flash the firmware:**
```bash
cd firmware/esp32
pio run --target upload
```

---

## Citation

```bibtex
@inproceedings{khaled2026edge,
  title     = {Edge Intelligence for Multimodal Biomedical Monitoring:
               A Wearable Sensor-Fusion System for Simultaneous Activity,
               Stress, and Arrhythmia Detection on {ESP32-S3}},
  author    = {Khaled, Sara and others},
  booktitle = {2026 IEEE International Conference on Healthcare Informatics (ICHI)},
  year      = {2026},
  note      = {DOI to be added upon publication}
}
```

---

## License

Released for academic use. See [LICENSE](LICENSE) for details.
