#pragma once

#include <Arduino.h>

class AnalogPpg {
 public:
  explicit AnalogPpg(int adc_pin) : _adc_pin(adc_pin) {}

  void begin() { pinMode(_adc_pin, INPUT); }

  uint16_t readAdc() const {
    int v = analogRead(_adc_pin);
    if (v < 0) v = 0;
    if (v > 4095) v = 4095;
    return static_cast<uint16_t>(v);
  }

 private:
  int _adc_pin;
};

