#pragma once

#include <stdint.h>
#include "../types.h"

template <uint16_t N>
class WindowBuffer {
 public:
  WindowBuffer() : _count(0), _window_index(0) {}

  void reset() { _count = 0; }

  uint16_t size() const { return _count; }
  uint16_t capacity() const { return N; }
  bool full() const { return _count >= N; }

  uint32_t windowIndex() const { return _window_index; }

  void push(const Sample& s) {
    if (_count < N) {
      _buf[_count] = s;
      _count++;
    }
  }

  const Sample* data() const { return _buf; }

  // Call after processing a full window
  void nextWindow() {
    _window_index++;
    _count = 0;
  }

 private:
  Sample _buf[N];
  uint16_t _count;
  uint32_t _window_index;
};

