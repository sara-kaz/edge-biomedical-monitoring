#pragma once

#include <Arduino.h>
#include <Wire.h>

/**
 * MAX30102 Pulse Oximeter and Heart Rate Sensor Driver
 *
 * Features:
 * - PPG (Red and IR LED) measurement for heart rate and SpO2
 * - I2C communication (default address 0x57)
 * - Configurable sample rate, LED current, and pulse width
 * - FIFO-based data acquisition
 *
 * Wiring:
 *   VIN  -> 3.3V
 *   GND  -> GND
 *   SDA  -> I2C SDA (GPIO 21 by default on ESP32-S3)
 *   SCL  -> I2C SCL (GPIO 22 by default on ESP32-S3)
 *   INT  -> Optional interrupt pin (not used in polling mode)
 */

class Max30102 {
 public:
  // I2C address (fixed, cannot be changed)
  static constexpr uint8_t I2C_ADDR = 0x57;

  // Register addresses
  static constexpr uint8_t REG_INTR_STATUS_1   = 0x00;
  static constexpr uint8_t REG_INTR_STATUS_2   = 0x01;
  static constexpr uint8_t REG_INTR_ENABLE_1   = 0x02;
  static constexpr uint8_t REG_INTR_ENABLE_2   = 0x03;
  static constexpr uint8_t REG_FIFO_WR_PTR     = 0x04;
  static constexpr uint8_t REG_OVF_COUNTER     = 0x05;
  static constexpr uint8_t REG_FIFO_RD_PTR     = 0x06;
  static constexpr uint8_t REG_FIFO_DATA       = 0x07;
  static constexpr uint8_t REG_FIFO_CONFIG     = 0x08;
  static constexpr uint8_t REG_MODE_CONFIG     = 0x09;
  static constexpr uint8_t REG_SPO2_CONFIG     = 0x0A;
  static constexpr uint8_t REG_LED1_PA         = 0x0C;  // Red LED
  static constexpr uint8_t REG_LED2_PA         = 0x0D;  // IR LED
  static constexpr uint8_t REG_PILOT_PA        = 0x10;
  static constexpr uint8_t REG_MULTI_LED_CTRL1 = 0x11;
  static constexpr uint8_t REG_MULTI_LED_CTRL2 = 0x12;
  static constexpr uint8_t REG_TEMP_INT        = 0x1F;
  static constexpr uint8_t REG_TEMP_FRAC       = 0x20;
  static constexpr uint8_t REG_TEMP_CONFIG     = 0x21;
  static constexpr uint8_t REG_REV_ID          = 0xFE;
  static constexpr uint8_t REG_PART_ID         = 0xFF;  // Should read 0x15

  // Mode configuration
  static constexpr uint8_t MODE_HR_ONLY        = 0x02;  // Red LED only
  static constexpr uint8_t MODE_SPO2           = 0x03;  // Red + IR LEDs
  static constexpr uint8_t MODE_MULTI_LED      = 0x07;  // Multi-LED mode

  // Sample rate options (in SpO2 mode)
  enum SampleRate {
    RATE_50   = 0x00,
    RATE_100  = 0x01,
    RATE_200  = 0x02,
    RATE_400  = 0x03,
    RATE_800  = 0x04,
    RATE_1000 = 0x05,
    RATE_1600 = 0x06,
    RATE_3200 = 0x07
  };

  // LED pulse width options
  enum PulseWidth {
    WIDTH_69   = 0x00,  // 69 us,  15-bit resolution
    WIDTH_118  = 0x01,  // 118 us, 16-bit resolution
    WIDTH_215  = 0x02,  // 215 us, 17-bit resolution
    WIDTH_411  = 0x03   // 411 us, 18-bit resolution
  };

  // ADC range options
  enum AdcRange {
    RANGE_2048  = 0x00,
    RANGE_4096  = 0x01,
    RANGE_8192  = 0x02,
    RANGE_16384 = 0x03
  };

  Max30102() : _initialized(false) {}

  /**
   * Initialize the MAX30102 sensor
   *
   * @param sda_pin I2C SDA pin (use same as MPU6050 if sharing I2C bus)
   * @param scl_pin I2C SCL pin
   * @param reinit_wire If false, assumes Wire is already initialized (shared bus)
   * @return true if initialization successful
   */
  bool begin(int sda_pin = 21, int scl_pin = 22, bool reinit_wire = false) {
    if (reinit_wire) {
      Wire.begin(sda_pin, scl_pin);
      Wire.setClock(400000);  // 400kHz I2C
    }

    // Check part ID (should be 0x15 for MAX30102)
    uint8_t part_id = readReg(REG_PART_ID);
    if (part_id != 0x15) {
      Serial.printf("MAX30102: Invalid part ID: 0x%02X (expected 0x15)\n", part_id);
      return false;
    }

    // Reset the device
    if (!writeReg(REG_MODE_CONFIG, 0x40)) return false;  // Reset bit
    delay(50);

    // Configure for SpO2 mode (Red + IR LEDs)
    // This gives us both channels needed for PPG analysis

    // Clear FIFO pointers
    if (!writeReg(REG_FIFO_WR_PTR, 0x00)) return false;
    if (!writeReg(REG_OVF_COUNTER, 0x00)) return false;
    if (!writeReg(REG_FIFO_RD_PTR, 0x00)) return false;

    // FIFO configuration: 4 sample averaging, FIFO rollover enabled
    // Bits [7:5] = sample averaging (100 = 4 samples)
    // Bit [4] = FIFO rollover enable
    // Bits [3:0] = FIFO almost full value (not used here)
    if (!writeReg(REG_FIFO_CONFIG, 0x4F)) return false;

    // Mode configuration: SpO2 mode
    if (!writeReg(REG_MODE_CONFIG, MODE_SPO2)) return false;

    // SpO2 configuration
    // Bits [6:5] = ADC range (11 = 16384 nA)
    // Bits [4:2] = Sample rate (001 = 100 Hz)
    // Bits [1:0] = LED pulse width (11 = 411 us, 18-bit)
    uint8_t spo2_config = (RANGE_16384 << 5) | (RATE_100 << 2) | WIDTH_411;
    if (!writeReg(REG_SPO2_CONFIG, spo2_config)) return false;

    // LED current configuration
    // 0x7F = ~25mA per LED (good for finger-based PPG)
    // Range: 0x00 = 0mA, 0xFF = 50mA
    if (!writeReg(REG_LED1_PA, 0x7F)) return false;  // Red LED
    if (!writeReg(REG_LED2_PA, 0x7F)) return false;  // IR LED

    // Enable interrupts (optional, for interrupt-driven operation)
    // For polling mode, we don't need interrupts
    if (!writeReg(REG_INTR_ENABLE_1, 0x00)) return false;
    if (!writeReg(REG_INTR_ENABLE_2, 0x00)) return false;

    _initialized = true;
    Serial.println("MAX30102: Initialized successfully");
    return true;
  }

  /**
   * Read raw PPG data from FIFO
   *
   * @param red_out Red LED reading (output)
   * @param ir_out IR LED reading (output)
   * @return true if data was read successfully
   */
  bool read(uint32_t& red_out, uint32_t& ir_out) {
    if (!_initialized) return false;

    // Check if there's data in FIFO
    uint8_t wr_ptr = readReg(REG_FIFO_WR_PTR);
    uint8_t rd_ptr = readReg(REG_FIFO_RD_PTR);

    int samples_available = wr_ptr - rd_ptr;
    if (samples_available < 0) samples_available += 32;  // Handle wrap-around

    if (samples_available == 0) {
      return false;  // No data available
    }

    // Read 6 bytes from FIFO (3 bytes each for Red and IR)
    uint8_t fifo_data[6];
    if (!readRegs(REG_FIFO_DATA, fifo_data, 6)) return false;

    // Parse Red LED data (18-bit, but stored in 3 bytes with 6 MSBs unused)
    red_out = ((uint32_t)(fifo_data[0] & 0x03) << 16) |
              ((uint32_t)fifo_data[1] << 8) |
              (uint32_t)fifo_data[2];

    // Parse IR LED data
    ir_out = ((uint32_t)(fifo_data[3] & 0x03) << 16) |
             ((uint32_t)fifo_data[4] << 8) |
             (uint32_t)fifo_data[5];

    return true;
  }

  /**
   * Read the most recent sample currently in the FIFO.
   *
   * The firmware sampling loop runs at 100 Hz. If I2C jitter causes the FIFO
   * to accumulate multiple samples, reading a single sample intermittently can
   * produce "spiky/triangular" plots (zeros between successful reads).
   *
   * This helper drains available FIFO samples and returns the latest value.
   */
  bool readLatest(uint32_t& red_out, uint32_t& ir_out) {
    if (!_initialized) return false;
    const int n = samplesAvailable();
    if (n <= 0) return false;
    bool ok = false;
    uint32_t r = 0, i = 0;
    for (int k = 0; k < n; k++) {
      if (!read(r, i)) break;
      ok = true;
    }
    if (!ok) return false;
    red_out = r;
    ir_out = i;
    return true;
  }

  /**
   * Read PPG data as normalized floats
   * Normalizes to 0.0-1.0 range based on 18-bit ADC
   *
   * @param red_norm Red LED normalized reading (output)
   * @param ir_norm IR LED normalized reading (output)
   * @return true if data was read successfully
   */
  bool readNormalized(float& red_norm, float& ir_norm) {
    uint32_t red_raw, ir_raw;
    if (!read(red_raw, ir_raw)) return false;

    // Normalize to 0.0-1.0 range (18-bit max = 262143)
    red_norm = (float)red_raw / 262143.0f;
    ir_norm = (float)ir_raw / 262143.0f;

    return true;
  }

  /**
   * Read die temperature (useful for calibration)
   *
   * @return Temperature in degrees Celsius
   */
  float readTemperature() {
    // Trigger temperature reading
    writeReg(REG_TEMP_CONFIG, 0x01);
    delay(30);  // Wait for conversion

    int8_t temp_int = (int8_t)readReg(REG_TEMP_INT);
    uint8_t temp_frac = readReg(REG_TEMP_FRAC);

    return (float)temp_int + (float)temp_frac * 0.0625f;
  }

  /**
   * Set LED current (brightness)
   *
   * @param red_current Red LED current (0-255, maps to 0-50mA)
   * @param ir_current IR LED current (0-255, maps to 0-50mA)
   */
  void setLedCurrent(uint8_t red_current, uint8_t ir_current) {
    writeReg(REG_LED1_PA, red_current);
    writeReg(REG_LED2_PA, ir_current);
  }

  /**
   * Set sample rate
   *
   * @param rate Sample rate enum value
   */
  void setSampleRate(SampleRate rate) {
    uint8_t current_config = readReg(REG_SPO2_CONFIG);
    current_config = (current_config & 0xE3) | (rate << 2);
    writeReg(REG_SPO2_CONFIG, current_config);
  }

  /**
   * Check if sensor is initialized and responding
   */
  bool isConnected() {
    if (!_initialized) return false;
    return readReg(REG_PART_ID) == 0x15;
  }

  /**
   * Get number of samples available in FIFO
   */
  int samplesAvailable() {
    uint8_t wr_ptr = readReg(REG_FIFO_WR_PTR);
    uint8_t rd_ptr = readReg(REG_FIFO_RD_PTR);
    int samples = wr_ptr - rd_ptr;
    if (samples < 0) samples += 32;
    return samples;
  }

  /**
   * Clear FIFO buffer
   */
  void clearFifo() {
    writeReg(REG_FIFO_WR_PTR, 0x00);
    writeReg(REG_OVF_COUNTER, 0x00);
    writeReg(REG_FIFO_RD_PTR, 0x00);
  }

 private:
  bool _initialized;

  bool writeReg(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(I2C_ADDR);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
  }

  uint8_t readReg(uint8_t reg) {
    Wire.beginTransmission(I2C_ADDR);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return 0;

    if (Wire.requestFrom((uint8_t)I2C_ADDR, (uint8_t)1) != 1) return 0;
    return Wire.read();
  }

  bool readRegs(uint8_t reg, uint8_t* out, size_t n) {
    Wire.beginTransmission(I2C_ADDR);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return false;

    size_t got = Wire.requestFrom((uint8_t)I2C_ADDR, (uint8_t)n);
    if (got != n) return false;

    for (size_t i = 0; i < n; i++) out[i] = Wire.read();
    return true;
  }
};
