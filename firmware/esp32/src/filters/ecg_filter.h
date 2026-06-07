#pragma once

#include <math.h>
#include <stdint.h>

// ECG cleanup suitable for plotting and stable inference at 100 Hz:
// - Median-of-3 (tiny impulse cleaner)
// - 2nd order Butterworth high-pass (baseline wander)
// - 2nd order notch (50/60 Hz mains hum)
// - 2nd order Butterworth low-pass (reduce high-frequency noise)
//
// Notes:
// - This does NOT "invent" a textbook ECG; it removes common nuisances so real QRS/T features show.
// - Good electrodes + wiring still matter more than any filter.

class Biquad {
 public:
  Biquad() { reset(); setIdentity(); }

  void reset() { _z1 = 0.0f; _z2 = 0.0f; }

  float process(float x) {
    // Direct Form II Transposed
    const float y = _b0 * x + _z1;
    _z1 = _b1 * x - _a1 * y + _z2;
    _z2 = _b2 * x - _a2 * y;
    return y;
  }

  void setIdentity() {
    _b0 = 1.0f; _b1 = 0.0f; _b2 = 0.0f;
    _a1 = 0.0f; _a2 = 0.0f;
  }

  // RBJ audio EQ cookbook
  void setLowpass(float fs, float fc, float q) {
    const float w0 = 2.0f * 3.1415926f * (fc / fs);
    const float c = cosf(w0);
    const float s = sinf(w0);
    const float alpha = s / (2.0f * q);
    const float b0 = (1.0f - c) * 0.5f;
    const float b1 = (1.0f - c);
    const float b2 = (1.0f - c) * 0.5f;
    const float a0 = 1.0f + alpha;
    const float a1 = -2.0f * c;
    const float a2 = 1.0f - alpha;
    setNormalized(b0, b1, b2, a0, a1, a2);
  }

  void setHighpass(float fs, float fc, float q) {
    const float w0 = 2.0f * 3.1415926f * (fc / fs);
    const float c = cosf(w0);
    const float s = sinf(w0);
    const float alpha = s / (2.0f * q);
    const float b0 = (1.0f + c) * 0.5f;
    const float b1 = -(1.0f + c);
    const float b2 = (1.0f + c) * 0.5f;
    const float a0 = 1.0f + alpha;
    const float a1 = -2.0f * c;
    const float a2 = 1.0f - alpha;
    setNormalized(b0, b1, b2, a0, a1, a2);
  }

  // Notch at f0 with Q (higher Q = narrower notch)
  // Guard: f0 must be below Nyquist (fs/2) or the filter is unstable.
  void setNotch(float fs, float f0, float q) {
    if (f0 <= 0.0f || f0 >= fs * 0.5f) {
      setIdentity();  // can't place a notch at/above Nyquist
      return;
    }
    const float w0 = 2.0f * 3.1415926f * (f0 / fs);
    const float c = cosf(w0);
    const float s = sinf(w0);
    const float alpha = s / (2.0f * q);
    const float b0 = 1.0f;
    const float b1 = -2.0f * c;
    const float b2 = 1.0f;
    const float a0 = 1.0f + alpha;
    const float a1 = -2.0f * c;
    const float a2 = 1.0f - alpha;
    setNormalized(b0, b1, b2, a0, a1, a2);
  }

 private:
  float _b0, _b1, _b2;
  float _a1, _a2;
  float _z1, _z2;

  void setNormalized(float b0, float b1, float b2, float a0, float a1, float a2) {
    if (fabsf(a0) < 1e-9f) a0 = 1.0f;
    _b0 = b0 / a0;
    _b1 = b1 / a0;
    _b2 = b2 / a0;
    _a1 = a1 / a0;
    _a2 = a2 / a0;
  }
};

class EcgFilter {
 public:
  EcgFilter() { reset(2048.0f); }

  void configure(float fs_hz, float hp_hz, float lp_hz, bool enable_notch, float mains_hz) {
    // Butterworth-ish Q for 2nd order sections
    constexpr float BUTTER_Q = 0.70710678f;
    _hp.setHighpass(fs_hz, hp_hz, BUTTER_Q);
    _lp.setLowpass(fs_hz, lp_hz, BUTTER_Q);

    _enable_notch = enable_notch;
    if (_enable_notch) {
      // Narrow-ish notch so it doesn't chew morphology
      _notch.setNotch(fs_hz, mains_hz, 20.0f);
    } else {
      _notch.setIdentity();
    }
  }

  float process(float x) {
    // Median-of-3 to suppress single-sample spikes before IIR
    const float m = median3(x, _x1, _x2);
    _x2 = _x1;
    _x1 = x;

    float y = _hp.process(m);
    y = _notch.process(y);
    y = _lp.process(y);
    return y;
  }

  void reset(float initial) {
    _x1 = initial;
    _x2 = initial;
    _hp.reset();
    _notch.reset();
    _lp.reset();
    _warmup = 0;
  }

  // Feed N samples at 'dc' to settle IIR transients before real data arrives.
  void warmup(float dc, int n = 200) {
    for (int i = 0; i < n; i++) {
      process(dc);
    }
    _warmup = n;
  }

  bool warmedUp() const { return _warmup > 0; }

 private:
  Biquad _hp;
  Biquad _notch;
  Biquad _lp;
  bool _enable_notch = false;
  float _x1 = 0.0f;
  float _x2 = 0.0f;
  int _warmup = 0;

  static float median3(float a, float b, float c) {
    // return median of a,b,c
    if (a > b) { const float t = a; a = b; b = t; }
    if (b > c) { const float t = b; b = c; c = t; }
    if (a > b) { const float t = a; a = b; b = t; }
    return b;
  }
};

