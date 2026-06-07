#pragma once

#include <stdint.h>

/**
 * Sample data structure for multimodal biomedical monitoring
 *
 * Contains all sensor readings for a single time point.
 * The model expects 11 input channels:
 *   [0] ECG signal
 *   [1] PPG signal (Red LED or analog PPG)
 *   [2-4] Accelerometer (ax, ay, az)
 *   [5-7] Gyroscope (gx, gy, gz)
 *   [8] GSR/EDA (skin conductance)
 *   [9] PPG IR channel (from MAX30102)
 *   [10] Temperature or reserved
 *
 * Note: Not all channels need to be populated. Missing sensors
 * will have their channels set to 0.
 */
struct Sample {
  uint32_t t_us;  // Timestamp in microseconds

  // =============================
  // ECG (AD8232)
  // =============================
  uint16_t ecg_adc_raw;   // Raw ADC reading (0-4095, 12-bit)
  uint16_t ecg_adc;       // Filtered ADC-like value (0-4095) for plotting/inference

  // =============================
  // PPG (MAX30102 or analog)
  // =============================
  // If using MAX30102: ppg_red and ppg_ir are populated
  // If using analog PPG: ppg_red contains analog reading, ppg_ir = 0
  uint32_t ppg_red;  // Red LED reading (18-bit from MAX30102, or 12-bit analog)
  uint32_t ppg_ir;   // IR LED reading (18-bit from MAX30102, 0 if analog)

  // =============================
  // IMU (MPU6050)
  // =============================
  // Accelerometer in g-force (±2g range → ±1.0 in practice)
  float ax_g;
  float ay_g;
  float az_g;

  // Gyroscope in degrees per second (±250 dps range)
  float gx_dps;
  float gy_dps;
  float gz_dps;

  // =============================
  // GSR/EDA (Skin Conductance)
  // =============================
  uint16_t gsr_adc;        // Raw ADC reading (0-4095)
  float gsr_conductance;   // Conductance in microSiemens (calculated)

  // =============================
  // Status flags
  // =============================
  uint8_t flags;  // Bit flags for sensor status
  // Bit 0: ECG lead-off detected
  // Bit 1: MAX30102 data valid
  // Bit 2: IMU data valid
  // Bit 3: GSR electrode attached
  // Bits 4-7: Reserved
};

// Status flag bit definitions
constexpr uint8_t FLAG_ECG_LEAD_OFF   = 0x01;
constexpr uint8_t FLAG_PPG_VALID      = 0x02;
constexpr uint8_t FLAG_IMU_VALID      = 0x04;
constexpr uint8_t FLAG_GSR_ATTACHED   = 0x08;

/**
 * Model output structure
 * Contains prediction results from on-device inference
 */
struct InferenceResult {
  // Activity classification (4 merged classes)
  // 0=Sedentary, 1=Walking, 2=Cycling, 3=HighIntensity
  uint8_t activity_class;
  float activity_confidence;

  // Stress classification (2 classes - binary)
  // 0=Baseline, 1=Stress
  uint8_t stress_class;
  float stress_confidence;

  // Arrhythmia detection (2 classes)
  // 0=Normal, 1=Abnormal
  uint8_t arrhythmia_class;
  float arrhythmia_confidence;

  // Alert status
  bool alert_triggered;
  uint8_t alert_type;  // 0=none, 1=stress, 2=arrhythmia, 3=critical

  // Inference timing
  float inference_time_ms;
};

// Alert type definitions
constexpr uint8_t ALERT_NONE        = 0;
constexpr uint8_t ALERT_STRESS      = 1;
constexpr uint8_t ALERT_ARRHYTHMIA  = 2;
constexpr uint8_t ALERT_CRITICAL    = 3;  // Arrhythmia during activity

