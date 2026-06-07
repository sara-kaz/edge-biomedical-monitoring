#pragma once

#include <Arduino.h>

// Console abstraction for ESP32-S3 USB serial variants.
//
// On some ESP32-S3 setups, the host port shows up as:
//   "USB JTAG/serial debug unit" (USB-Serial/JTAG)
// In that case, using USBSerial is the most reliable way to get prints.
//
// If USBSerial isn't enabled/available, we fall back to Serial.

#if defined(ARDUINO_USB_SERIAL_JTAG_ON_BOOT) && ARDUINO_USB_SERIAL_JTAG_ON_BOOT
  #include "USB.h"
  static inline void consoleBegin(uint32_t baud = 115200) {
    USBSerial.begin(baud);
    delay(50);
  }
  #define CONSOLE USBSerial
#else
  static inline void consoleBegin(uint32_t baud = 115200) {
    Serial.begin(baud);
    delay(50);
  }
  #define CONSOLE Serial
#endif

