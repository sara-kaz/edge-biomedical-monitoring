#pragma once

#include <Arduino.h>

/**
 * GSR (Galvanic Skin Response) / EDA (Electrodermal Activity) Sensor Driver
 *
 * GSR measures skin conductance which changes with sweat gland activity,
 * making it useful for stress detection and emotional arousal monitoring.
 *
 * Typical GSR sensor modules output an analog voltage proportional to
 * skin conductance. Common modules include:
 * - Grove GSR Sensor
 * - DIY voltage divider circuits with electrodes
 * - Seeed Studio GSR Sensor
 *
 * Wiring (for typical GSR module):
 *   VCC  -> 3.3V (some modules may need 5V with voltage divider on output)
 *   GND  -> GND
 *   SIG  -> ADC pin (GPIO 3-10 on ESP32-S3 for ADC1)
 *
 * Signal characteristics:
 * - Output voltage typically 0-3.3V (depends on module)
 * - Higher conductance = lower resistance = higher voltage (typically)
 * - Response time: ~0.5-2 seconds for phasic (rapid) changes
 * - Tonic (baseline) level varies significantly between individuals
 *
 * For stress detection, we typically look at:
 * - Skin Conductance Level (SCL): Tonic/baseline level
 * - Skin Conductance Response (SCR): Phasic/rapid changes
 */

class GsrSensor {
 public:
  /**
   * Constructor
   *
   * @param adc_pin ADC-capable GPIO pin (GPIO 1-10 on ESP32-S3 for ADC1)
   * @param ref_voltage Reference voltage (typically 3.3V on ESP32-S3)
   * @param ref_resistor Reference resistor value in ohms (for resistance calculation)
   */
  explicit GsrSensor(int adc_pin, float ref_voltage = 3.3f, float ref_resistor = 10000.0f)
      : _adc_pin(adc_pin),
        _ref_voltage(ref_voltage),
        _ref_resistor(ref_resistor),
        _baseline(0.0f),
        _baseline_samples(0),
        _calibrated(false) {}

  /**
   * Initialize the GSR sensor
   */
  void begin() {
    pinMode(_adc_pin, INPUT);
    // Configure ADC attenuation for full 3.3V range
    analogSetPinAttenuation(_adc_pin, ADC_11db);
    Serial.printf("GSR Sensor: Initialized on pin %d\n", _adc_pin);
  }

  /**
   * Read raw ADC value
   *
   * @return Raw ADC value (0-4095 for 12-bit)
   */
  uint16_t readAdc() const {
    int v = analogRead(_adc_pin);
    if (v < 0) v = 0;
    if (v > 4095) v = 4095;
    return static_cast<uint16_t>(v);
  }

  /**
   * Read ADC scaled to 10-bit (0..1023).
   *
   * Grove GSR reference formulas are typically written for 10-bit ADC boards.
   * We use this helper to keep Grove-compatible conversions stable.
   */
  uint16_t readAdc10() const {
    const uint16_t a12 = readAdc();
    // Scale 12-bit -> 10-bit (rounding)
    return static_cast<uint16_t>((static_cast<uint32_t>(a12) * 1023u + 2047u) / 4095u);
  }

  /**
   * Read voltage from sensor
   *
   * @return Voltage in volts (0.0-3.3V typical)
   */
  float readVoltage() const {
    return (readAdc() / 4095.0f) * _ref_voltage;
  }

  /**
   * Estimate skin resistance (ohms) for Grove-style GSR boards.
   *
   * Grove GSR example code (10-bit ADC) commonly uses:
   *   R = ((1024 + 2*A) * Rref) / (512 - A)
   *
   * where A is the 10-bit ADC value.
   */
  float readResistanceOhms() const {
    const float a = static_cast<float>(readAdc10());
    const float denom = 512.0f - a;
    if (denom <= 1.0f) {
      return 1e9f;  // saturated / invalid -> treat as very high resistance
    }
    const float r = ((1024.0f + 2.0f * a) * _ref_resistor) / denom;
    if (!isfinite(r) || r < 1.0f) return 1.0f;
    return r;
  }

  /**
   * Read skin conductance in microSiemens (µS)
   *
   * For Grove boards we derive conductance from the estimated resistance.
   *
   * @return Conductance in microSiemens (µS)
   */
  float readConductance() const {
    const float r = readResistanceOhms();
    return 1000000.0f / r;
  }

  /**
   * Read normalized GSR value (0.0-1.0)
   * Useful for direct input to ML models
   *
   * Typical human skin conductance ranges from 0.5-50 µS
   * We normalize to this range, clamping outliers
   *
   * @return Normalized conductance (0.0-1.0)
   */
  float readNormalized() const {
    float conductance = readConductance();

    // Typical range: 0.5 to 50 µS
    // Clamp and normalize
    const float MIN_CONDUCTANCE = 0.5f;   // µS
    const float MAX_CONDUCTANCE = 50.0f;  // µS

    conductance = constrain(conductance, MIN_CONDUCTANCE, MAX_CONDUCTANCE);
    return (conductance - MIN_CONDUCTANCE) / (MAX_CONDUCTANCE - MIN_CONDUCTANCE);
  }

  /**
   * Calibrate baseline conductance
   * Call this during a calm, resting state to establish baseline
   *
   * @param num_samples Number of samples to average (default 100)
   * @param delay_ms Delay between samples in ms (default 50ms)
   */
  void calibrateBaseline(int num_samples = 100, int delay_ms = 50) {
    Serial.println("GSR: Calibrating baseline...");

    float sum = 0.0f;
    for (int i = 0; i < num_samples; i++) {
      sum += readConductance();
      delay(delay_ms);
    }

    _baseline = sum / num_samples;
    _baseline_samples = num_samples;
    _calibrated = true;

    Serial.printf("GSR: Baseline calibrated to %.2f µS\n", _baseline);
  }

  /**
   * Update baseline with exponential moving average
   * Call periodically during low-stress periods to adapt to drift
   *
   * @param alpha Smoothing factor (0.0-1.0, lower = slower adaptation)
   */
  void updateBaseline(float alpha = 0.01f) {
    if (!_calibrated) {
      _baseline = readConductance();
      _calibrated = true;
      return;
    }

    float current = readConductance();
    _baseline = alpha * current + (1.0f - alpha) * _baseline;
  }

  /**
   * Read Skin Conductance Response (SCR)
   * This is the phasic component - deviation from baseline
   * Positive values indicate increased arousal/stress
   *
   * @return SCR in microSiemens (can be negative)
   */
  float readSCR() const {
    if (!_calibrated) return 0.0f;
    return readConductance() - _baseline;
  }

  /**
   * Read normalized SCR (Skin Conductance Response)
   * Normalized to typical SCR range (-1.0 to 1.0)
   *
   * @return Normalized SCR (-1.0 to 1.0)
   */
  float readNormalizedSCR() const {
    float scr = readSCR();

    // Typical SCR amplitude: -5 to +10 µS
    const float MAX_SCR = 10.0f;
    const float MIN_SCR = -5.0f;

    scr = constrain(scr, MIN_SCR, MAX_SCR);
    return (scr - MIN_SCR) / (MAX_SCR - MIN_SCR) * 2.0f - 1.0f;
  }

  /**
   * Get baseline conductance level (SCL - Skin Conductance Level)
   *
   * @return Baseline conductance in microSiemens
   */
  float getBaseline() const { return _baseline; }

  /**
   * Check if sensor is calibrated
   */
  bool isCalibrated() const { return _calibrated; }

  /**
   * Detect if electrodes are properly attached
   * Returns false if reading is at extreme low (not touching skin)
   */
  bool isAttached() const {
    // Use a conservative rail-check + a wide conductance window.
    // This works across common modules (including Grove) where absolute scaling varies.
    const uint16_t adc = readAdc();
    if (adc <= 5 || adc >= 4090) {
      return false;  // likely open circuit or saturated/overrange
    }

    const float r = readResistanceOhms();
    // Grove typical skin resistance is often ~10k–1M depending on contact.
    // We accept a wide range to avoid false "not attached".
    return (r > 1000.0f && r < 5e6f);
  }

  /**
   * Detect stress event (simple threshold-based)
   * More sophisticated detection should be done in the ML model
   *
   * @param threshold SCR threshold in µS (default 2.0)
   * @return true if current SCR exceeds threshold
   */
  bool detectStressEvent(float threshold = 2.0f) const {
    return readSCR() > threshold;
  }

 private:
  int _adc_pin;
  float _ref_voltage;
  float _ref_resistor;
  float _baseline;
  int _baseline_samples;
  bool _calibrated;
};
