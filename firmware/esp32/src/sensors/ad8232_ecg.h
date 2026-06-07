#pragma once

#include <Arduino.h>

class AD8232Ecg {
 public:
  AD8232Ecg(int adc_pin, int lo_plus_pin = -1, int lo_minus_pin = -1)
      : _adc_pin(adc_pin), _lo_plus_pin(lo_plus_pin), _lo_minus_pin(lo_minus_pin) {}

  void begin() {
    // Use pulldown so the pins don't float if not connected.
    // AD8232 LO outputs go HIGH when leads are off.
    if (_lo_plus_pin >= 0) pinMode(_lo_plus_pin, INPUT_PULLDOWN);
    if (_lo_minus_pin >= 0) pinMode(_lo_minus_pin, INPUT_PULLDOWN);

    // ADC init handled by Arduino core; you can optionally set attenuation in main.cpp.
    pinMode(_adc_pin, INPUT);
  }

  uint16_t readAdc() const {
    int v = analogRead(_adc_pin);
    if (v < 0) v = 0;
    if (v > 4095) v = 4095;
    return static_cast<uint16_t>(v);
  }

  uint16_t readAdcAveraged(uint8_t n) const {
    if (n <= 1) return readAdc();
    uint32_t acc = 0;
    for (uint8_t i = 0; i < n; i++) {
      acc += readAdc();
    }
    return static_cast<uint16_t>(acc / n);
  }

  bool leadOff() const {
    if (_lo_plus_pin < 0 || _lo_minus_pin < 0) return false;
    return (digitalRead(_lo_plus_pin) == HIGH) || (digitalRead(_lo_minus_pin) == HIGH);
  }

 private:
  int _adc_pin;
  int _lo_plus_pin;
  int _lo_minus_pin;
};

