#pragma once

#include <Arduino.h>
#include <Wire.h>

// Minimal MPU6050 driver (accel + gyro)
// - Configures:
//   - Accel range: ±2g
//   - Gyro range:  ±250 dps
//   - DLPF:        42 Hz (configurable)
// - Reads raw accel/gyro and converts to g and dps

class Mpu6050 {
 public:
  explicit Mpu6050(uint8_t addr = 0x68) : _addr(addr) {}

  bool begin(int sda_pin, int scl_pin) {
    // NOTE: Wire is initialized in main.cpp (shared I2C bus).
    (void)sda_pin;
    (void)scl_pin;

    // Verify WHO_AM_I register
    // MPU6050 = 0x68, MPU6500 = 0x70, MPU9250 = 0x71, ICM-20948 = 0xEA
    uint8_t who = 0;
    if (!readRegs(0x75, &who, 1)) return false;
    if (who != 0x68 && who != 0x70 && who != 0x71 && who != 0x72 && who != 0x19 && (who & 0x7E) != 0x68) return false;

    // Full device reset via PWR_MGMT_1 bit 7
    if (!writeReg(0x6B, 0x80)) return false;
    delay(100);

    // Wake up device (PWR_MGMT_1 = 0, use internal 8MHz clock)
    if (!writeReg(0x6B, 0x00)) return false;
    delay(50);

    // Sample rate divider: 0 => 1kHz / (1+0) = 1kHz update rate
    if (!writeReg(0x19, 0x00)) return false;

    // Set DLPF (CONFIG register 0x1A). 0x03 -> ~42Hz accel/gyro
    if (!writeReg(0x1A, 0x03)) return false;

    // Gyro full scale ±250 dps (GYRO_CONFIG 0x1B = 0x00)
    if (!writeReg(0x1B, 0x00)) return false;

    // Accel full scale ±2g (ACCEL_CONFIG 0x1C = 0x00)
    if (!writeReg(0x1C, 0x00)) return false;

    return true;
  }

  bool read(float& ax_g, float& ay_g, float& az_g, float& gx_dps, float& gy_dps, float& gz_dps) {
    // ACCEL_XOUT_H starts at 0x3B, 14 bytes: accel(6) + temp(2) + gyro(6)
    uint8_t buf[14];
    if (!readRegs(0x3B, buf, sizeof(buf))) return false;

    int16_t ax = (int16_t)((buf[0] << 8) | buf[1]);
    int16_t ay = (int16_t)((buf[2] << 8) | buf[3]);
    int16_t az = (int16_t)((buf[4] << 8) | buf[5]);
    // int16_t temp = (int16_t)((buf[6] << 8) | buf[7]);
    int16_t gx = (int16_t)((buf[8] << 8) | buf[9]);
    int16_t gy = (int16_t)((buf[10] << 8) | buf[11]);
    int16_t gz = (int16_t)((buf[12] << 8) | buf[13]);

    // Sensitivity scale factors for selected ranges:
    // accel ±2g => 16384 LSB/g
    // gyro  ±250 dps => 131 LSB/(dps)
    ax_g = ((float)ax) / 16384.0f;
    ay_g = ((float)ay) / 16384.0f;
    az_g = ((float)az) / 16384.0f;

    gx_dps = ((float)gx) / 131.0f;
    gy_dps = ((float)gy) / 131.0f;
    gz_dps = ((float)gz) / 131.0f;

    return true;
  }

 private:
  uint8_t _addr;

  bool writeReg(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(_addr);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
  }

  bool readRegs(uint8_t reg, uint8_t* out, size_t n) {
    Wire.beginTransmission(_addr);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return false;

    size_t got = Wire.requestFrom((int)_addr, (int)n, (int)true);
    if (got != n) return false;

    for (size_t i = 0; i < n; i++) out[i] = Wire.read();
    return true;
  }
};

