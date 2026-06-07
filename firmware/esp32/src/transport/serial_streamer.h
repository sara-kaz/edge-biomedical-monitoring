#pragma once

#include <Arduino.h>
#include "../types.h"
#include "../../include/config.h"
#include "console.h"

/**
 * Serial data streamer for biomedical sensor data
 *
 * Supports CSV and binary protocols for data transmission
 * to host PC for logging, visualization, and model training.
 */
class SerialStreamer {
 public:
  void begin(uint32_t baud = 115200) {
    consoleBegin(baud);
  }

  /**
   * Print CSV header for all sensor channels
   */
  void printHeader() {
    CONSOLE.println(
        "t_us,ecg_adc_raw,ecg_adc,ppg_red,ppg_ir,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,"
        "gsr_adc,gsr_us,flags");
  }

  /**
   * Print window start marker
   */
  void windowStart(uint32_t idx) {
    CONSOLE.print("WINDOW_START,");
    CONSOLE.println(idx);
  }

  /**
   * Print window end marker
   */
  void windowEnd(uint32_t idx) {
    CONSOLE.print("WINDOW_END,");
    CONSOLE.println(idx);
  }

  /**
   * Print single sample as CSV line
   */
  void printSample(const Sample& s) {
    CONSOLE.print(s.t_us);
    CONSOLE.print(',');
    CONSOLE.print(s.ecg_adc_raw);
    CONSOLE.print(',');
    CONSOLE.print(s.ecg_adc);
    CONSOLE.print(',');
    CONSOLE.print(s.ppg_red);
    CONSOLE.print(',');
    CONSOLE.print(s.ppg_ir);
    CONSOLE.print(',');
    CONSOLE.print(s.ax_g, 4);
    CONSOLE.print(',');
    CONSOLE.print(s.ay_g, 4);
    CONSOLE.print(',');
    CONSOLE.print(s.az_g, 4);
    CONSOLE.print(',');
    CONSOLE.print(s.gx_dps, 4);
    CONSOLE.print(',');
    CONSOLE.print(s.gy_dps, 4);
    CONSOLE.print(',');
    CONSOLE.print(s.gz_dps, 4);
    CONSOLE.print(',');
    CONSOLE.print(s.gsr_adc);
    CONSOLE.print(',');
    CONSOLE.print(s.gsr_conductance, 2);
    CONSOLE.print(',');
    CONSOLE.println(s.flags);
  }

  /**
   * Print inference result
   */
  void printInferenceResult(const InferenceResult& result) {
    CONSOLE.println("=== INFERENCE RESULT ===");
    CONSOLE.print("Activity: ");
    CONSOLE.print(result.activity_class);
    CONSOLE.print(" (");
    CONSOLE.print(result.activity_confidence * 100, 1);
    CONSOLE.println("%)");

    CONSOLE.print("Stress: ");
    CONSOLE.print(result.stress_class);
    CONSOLE.print(" (");
    CONSOLE.print(result.stress_confidence * 100, 1);
    CONSOLE.println("%)");

    CONSOLE.print("Arrhythmia: ");
    CONSOLE.print(result.arrhythmia_class == 0 ? "Normal" : "ABNORMAL");
    CONSOLE.print(" (");
    CONSOLE.print(result.arrhythmia_confidence * 100, 1);
    CONSOLE.println("%)");

    if (result.alert_triggered) {
      CONSOLE.print("*** ALERT: ");
      switch (result.alert_type) {
        case ALERT_STRESS:
          CONSOLE.println("High stress detected ***");
          break;
        case ALERT_ARRHYTHMIA:
          CONSOLE.println("Arrhythmia detected ***");
          break;
        case ALERT_CRITICAL:
          CONSOLE.println("CRITICAL - Arrhythmia during activity ***");
          break;
        default:
          CONSOLE.println("Unknown alert ***");
      }
    }

    CONSOLE.print("Inference time: ");
    CONSOLE.print(result.inference_time_ms, 2);
    CONSOLE.println(" ms");
    CONSOLE.println("========================");
  }

  /**
   * Print sensor status
   */
  void printSensorStatus(bool ecg_ok, bool imu_ok, bool ppg_ok, bool gsr_ok) {
    CONSOLE.println("=== SENSOR STATUS ===");
    CONSOLE.print("ECG (AD8232):   ");
    CONSOLE.println(ecg_ok ? "OK" : "NOT DETECTED");
    CONSOLE.print("IMU (MPU6050):  ");
    CONSOLE.println(imu_ok ? "OK" : "NOT DETECTED");
    CONSOLE.print("PPG (MAX30102): ");
    CONSOLE.println(ppg_ok ? "OK" : "NOT DETECTED");
    CONSOLE.print("GSR:            ");
    CONSOLE.println(gsr_ok ? "OK" : "NOT DETECTED");
    CONSOLE.println("=====================");
  }

  /**
   * Print error message
   */
  void printError(const char* msg) {
    CONSOLE.print("ERROR: ");
    CONSOLE.println(msg);
  }

  /**
   * Print debug message (only if DEBUG_SERIAL is enabled)
   */
  void printDebug(const char* msg) {
    if constexpr (DEBUG_SERIAL) {
      CONSOLE.print("DEBUG: ");
      CONSOLE.println(msg);
    }
  }

#if USE_BINARY_PROTOCOL
  /**
   * Send sample as binary packet (more efficient for high-speed streaming)
   * Format: [SYNC1][SYNC2][LENGTH][DATA...][CHECKSUM]
   */
  void sendBinarySample(const Sample& s) {
    const uint8_t SYNC1 = 0xAA;
    const uint8_t SYNC2 = 0x55;
    const uint8_t len = sizeof(Sample);

    Serial.write(SYNC1);
    Serial.write(SYNC2);
    Serial.write(len);

    const uint8_t* data = reinterpret_cast<const uint8_t*>(&s);
    uint8_t checksum = len;
    for (uint8_t i = 0; i < len; i++) {
      Serial.write(data[i]);
      checksum ^= data[i];
    }
    Serial.write(checksum);
  }
#endif
};
