/**
 * ESP32-S3 Multimodal Biomedical Monitoring Firmware
 *
 * Supported sensors:
 *   - AD8232 ECG (analog)
 *   - MPU6050 6-axis IMU (I2C)
 *   - MAX30102 Pulse Oximeter (I2C) - optional
 *   - GSR/EDA Skin Conductance (analog) - optional
 *
 * Features:
 *   - Real-time data acquisition at 100Hz
 *   - On-device neural network inference for:
 *     - Activity classification (8 classes)
 *     - Stress detection (4 classes)
 *     - Arrhythmia detection (2 classes)
 *   - Context-aware alerting system
 *
 * Data is streamed via USB Serial at 115200 baud in CSV format.
 * Each 10-second window (1000 samples @ 100Hz) triggers inference.
 */

#include <Arduino.h>

#include "types.h"
#include "buffering/window_buffer.h"
#include "transport/serial_streamer.h"
#include "transport/console.h"
#include "sensors/ad8232_ecg.h"
#include "sensors/mpu6050.h"
#include "filters/ecg_filter.h"

// Profiling / memory stats (ESP32 Arduino core)
#if ENABLE_PROFILING
#include <esp_heap_caps.h>
#endif

// Optional sensors
#if ENABLE_MAX30102
#include "sensors/max30102.h"
#endif

#if ENABLE_GSR
#include "sensors/gsr_sensor.h"
#endif

#if ENABLE_PPG_ANALOG
#include "sensors/analog_ppg.h"
#endif

// Neural network inference
#if ENABLE_INFERENCE
#include "inference/nn_inference.h"
#endif

// On-device adaptive training
#if ENABLE_ADAPTIVE_TRAINING && ENABLE_INFERENCE
#include "training/adaptive_trainer.h"
#endif

#include "../include/config.h"
#include "../include/pins_esp32s3_devkitc1.h"

// =============================
// Global objects
// =============================

// Window buffer (10 seconds @ 100 Hz = 1000 samples)
static WindowBuffer<WINDOW_SAMPLES> g_window;

// Serial streamer for data output
static SerialStreamer g_stream;

// Sensors
static AD8232Ecg g_ecg(ECG_ADC_PIN, AD8232_LO_PLUS_PIN, AD8232_LO_MINUS_PIN);
static Mpu6050 g_imu(MPU6050_I2C_ADDR);

#if ENABLE_MAX30102
static Max30102 g_ppg;
#endif

#if ENABLE_GSR
static GsrSensor g_gsr(GSR_ADC_PIN);
#endif

#if ENABLE_PPG_ANALOG
static AnalogPpg g_analog_ppg(PPG_ADC_PIN);
#endif

// Sensor status flags
static bool g_imu_ok = false;
static bool g_ppg_ok = false;
static bool g_gsr_ok = false;

// Timing
static uint32_t g_next_sample_us = 0;

// Alert state
static uint8_t g_last_alert_type = ALERT_NONE;
static uint32_t g_alert_count = 0;

// Latest sample for live telemetry
static Sample g_last_sample = {};

// ECG filter state (kept global to preserve continuity across samples)
static EcgFilter g_ecg_filter;

// =============================
// Profiling (paper metrics)
// =============================
#if ENABLE_PROFILING
// Sampling timing accumulation (per 10s window)
static uint64_t g_prof_acq_us_sum = 0;
static uint32_t g_prof_acq_us_max = 0;
static uint32_t g_prof_acq_count = 0;
static uint32_t g_prof_window_start_us = 0;

static void profResetPerWindow() {
    g_prof_acq_us_sum = 0;
    g_prof_acq_us_max = 0;
    g_prof_acq_count = 0;
    g_prof_window_start_us = micros();
}

static void profMaybePrintHeader() {
    // One-line CSV header for log parsers
    CONSOLE.println(
        "PROF_HDR,t_ms,win_idx,window_fill_ms,acq_mean_us,acq_max_us,"
        "infer_compute_ms,postproc_ms,window_total_ms,"
        "heap_free,heap_min,psram_free,psram_min,sketch_size,sketch_free"
    );
}

static void profPrintWindowLine(uint32_t win_idx,
                                float window_fill_ms,
                                float infer_compute_ms,
                                float postproc_ms,
                                float window_total_ms) {
    if (PROFILING_WINDOW_INTERVAL == 0) return;
    if ((win_idx % PROFILING_WINDOW_INTERVAL) != 0) return;

    uint32_t acq_mean_us = 0;
    if (PROFILING_TRACK_SAMPLE_TIMING && g_prof_acq_count > 0) {
        acq_mean_us = static_cast<uint32_t>(g_prof_acq_us_sum / g_prof_acq_count);
    }
    const uint32_t acq_max_us = PROFILING_TRACK_SAMPLE_TIMING ? g_prof_acq_us_max : 0;

    const uint32_t heap_free = ESP.getFreeHeap();
    const uint32_t heap_min = ESP.getMinFreeHeap();
#ifdef BOARD_HAS_PSRAM
    const uint32_t psram_free = ESP.getFreePsram();
    const uint32_t psram_min = ESP.getMinFreePsram();
#else
    const uint32_t psram_free = 0;
    const uint32_t psram_min = 0;
#endif
    const uint32_t sketch_size = ESP.getSketchSize();
    const uint32_t sketch_free = ESP.getFreeSketchSpace();

    CONSOLE.print("PROF,");
    CONSOLE.print(millis());
    CONSOLE.print(',');
    CONSOLE.print(win_idx);
    CONSOLE.print(',');
    CONSOLE.print(window_fill_ms, 2);
    CONSOLE.print(',');
    CONSOLE.print(acq_mean_us);
    CONSOLE.print(',');
    CONSOLE.print(acq_max_us);
    CONSOLE.print(',');
    CONSOLE.print(infer_compute_ms, 2);
    CONSOLE.print(',');
    CONSOLE.print(postproc_ms, 2);
    CONSOLE.print(',');
    CONSOLE.print(window_total_ms, 2);
    CONSOLE.print(',');
    CONSOLE.print(heap_free);
    CONSOLE.print(',');
    CONSOLE.print(heap_min);
    CONSOLE.print(',');
    CONSOLE.print(psram_free);
    CONSOLE.print(',');
    CONSOLE.print(psram_min);
    CONSOLE.print(',');
    CONSOLE.print(sketch_size);
    CONSOLE.print(',');
    CONSOLE.println(sketch_free);
}
#endif

// =============================
// Activity names for display
// =============================
static const char* ACTIVITY_NAMES[] = {
    "Sedentary", "Walking", "Cycling", "HighIntensity"
};

static const char* STRESS_NAMES[] = {
    "Baseline", "Stress"
};

static const char* ALERT_NAMES[] = {
    "None", "STRESS", "ARRHYTHMIA", "CRITICAL"
};

// Forward declaration (used by demo command handler)
static void handleAlert(uint8_t alert_type);

// =============================
// Box-drawing helpers
// =============================
static constexpr int BOX_INNER = 44;

static void printBoxTop() {
    CONSOLE.print("\xe2\x94\x8c"); // ┌
    for (int i = 0; i < BOX_INNER; i++) CONSOLE.print("\xe2\x94\x80"); // ─
    CONSOLE.println("\xe2\x94\x90"); // ┐
}
static void printBoxBottom() {
    CONSOLE.print("\xe2\x94\x94"); // └
    for (int i = 0; i < BOX_INNER; i++) CONSOLE.print("\xe2\x94\x80"); // ─
    CONSOLE.println("\xe2\x94\x98"); // ┘
}
static void printBoxDivider() {
    CONSOLE.print("\xe2\x94\x9c"); // ├
    for (int i = 0; i < BOX_INNER; i++) CONSOLE.print("\xe2\x94\x80"); // ─
    CONSOLE.println("\xe2\x94\xa4"); // ┤
}
static void printBoxLine(const char* content) {
    char buf[80];
    int len = snprintf(buf, sizeof(buf), "%s", content);
    if (len < 0) len = 0;
    if (len > BOX_INNER - 2) len = BOX_INNER - 2;
    CONSOLE.print("\xe2\x94\x82 "); // │
    CONSOLE.print(buf);
    for (int i = len; i < BOX_INNER - 2; i++) CONSOLE.print(' ');
    CONSOLE.println(" \xe2\x94\x82"); //  │
}

static void printLiveTelemetry(const Sample& s) {
    // Low-bandwidth live stream for plotting.
    // LIVE,t_ms,ecg_adc,ppg_red,ppg_ir,gsr_adc,gsr_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,flags
    CONSOLE.print("LIVE,");
    CONSOLE.print(millis());
    CONSOLE.print(',');
    CONSOLE.print(s.ecg_adc);
    CONSOLE.print(',');
    CONSOLE.print(s.ppg_red);
    CONSOLE.print(',');
    CONSOLE.print(s.ppg_ir);
    CONSOLE.print(',');
    CONSOLE.print(s.gsr_adc);
    CONSOLE.print(',');
    CONSOLE.print(s.gsr_conductance, 2);
    CONSOLE.print(',');
    CONSOLE.print(s.ax_g, 4);
    CONSOLE.print(',');
    CONSOLE.print(s.ay_g, 4);
    CONSOLE.print(',');
    CONSOLE.print(s.az_g, 4);
    CONSOLE.print(',');
    CONSOLE.print(s.gx_dps, 2);
    CONSOLE.print(',');
    CONSOLE.print(s.gy_dps, 2);
    CONSOLE.print(',');
    CONSOLE.print(s.gz_dps, 2);
    CONSOLE.print(',');
    CONSOLE.println(s.flags);
}

static void printEcgTelemetry(const Sample& s) {
    // High-rate ECG stream for plotting (100 Hz) without flooding serial.
    // ECG,t_ms,ecg_adc_raw,ecg_adc,flags
    CONSOLE.print("ECG,");
    CONSOLE.print(millis());
    CONSOLE.print(',');
    CONSOLE.print(s.ecg_adc_raw);
    CONSOLE.print(',');
    CONSOLE.print(s.ecg_adc);
    CONSOLE.print(',');
    CONSOLE.println(s.flags);
}

static void printPpgTelemetry(const Sample& s) {
    // High-rate PPG stream for plotting (100 Hz) without flooding serial.
    // PPG,t_ms,ppg_red,ppg_ir,flags
    CONSOLE.print("PPG,");
    CONSOLE.print(millis());
    CONSOLE.print(',');
    CONSOLE.print(s.ppg_red);
    CONSOLE.print(',');
    CONSOLE.print(s.ppg_ir);
    CONSOLE.print(',');
    CONSOLE.println(s.flags);
}

static void handleDemoCommand(char c) {
    if (!ENABLE_DEMO_ALERT_COMMANDS) return;

    switch (c) {
        case '1':
        case 's':
        case 'S':
            CONSOLE.println("DEMO: forcing STRESS alert");
            handleAlert(ALERT_STRESS);
            break;
        case '3':
        case 'a':
        case 'A':
            CONSOLE.println("DEMO: forcing ARRHYTHMIA alert");
            handleAlert(ALERT_ARRHYTHMIA);
            break;
        case '4':
        case 'c':
        case 'C':
            CONSOLE.println("DEMO: forcing CRITICAL alert");
            handleAlert(ALERT_CRITICAL);
            break;
        case '0':
        case 'n':
        case 'N':
            CONSOLE.println("DEMO: cleared (no alert)");
            break;
        default:
            break;
    }
}

static void printTop2Activity(const NNResult& result) {
    // Print the top-2 activity classes to make low-confidence behavior obvious.
    int best = 0;
    int second = 1;
    if (result.activity_probs[second] > result.activity_probs[best]) {
        const int tmp = best;
        best = second;
        second = tmp;
    }
    for (int i = 2; i < NNConfig::ACTIVITY_CLASSES; i++) {
        const float p = result.activity_probs[i];
        if (p > result.activity_probs[best]) {
            second = best;
            best = i;
        } else if (p > result.activity_probs[second]) {
            second = i;
        }
    }

    CONSOLE.printf("Activity (top2): %s %.1f%%, %s %.1f%%\n",
                  ACTIVITY_NAMES[best], result.activity_probs[best] * 100.0f,
                  ACTIVITY_NAMES[second], result.activity_probs[second] * 100.0f);
}

// =============================
// ADC Configuration
// =============================

static void configureAdc() {
    analogReadResolution(12);
    analogSetPinAttenuation(ECG_ADC_PIN, ADC_11db);

#if ENABLE_PPG_ANALOG
    analogSetPinAttenuation(PPG_ADC_PIN, ADC_11db);
#endif

#if ENABLE_GSR
    analogSetPinAttenuation(GSR_ADC_PIN, ADC_11db);
#endif
}

// =============================
// Sensor Initialization
// =============================

static void initializeSensors() {
    g_ecg.begin();
    CONSOLE.println("AD8232 ECG: Initialized");

    // I2C bring-up (shared by MPU6050 + MAX30102).
    // If the pins are invalid for this ESP32-S3 package/board, esp-idf logs:
    //   i2c_set_pin(...): scl gpio number error
    int used_sda = I2C_SDA_PIN;
    int used_scl = I2C_SCL_PIN;
    bool i2c_ok = Wire.begin(used_sda, used_scl);
    if (!i2c_ok) {
        CONSOLE.printf("I2C init FAILED on SDA=%d SCL=%d; trying fallback SDA=8 SCL=9\n",
                       used_sda, used_scl);
        used_sda = 8;
        used_scl = 9;
        i2c_ok = Wire.begin(used_sda, used_scl);
    }
    if (i2c_ok) {
        Wire.setClock(400000);
        CONSOLE.printf("I2C init OK (SDA=%d SCL=%d)\n", used_sda, used_scl);
    } else {
        CONSOLE.println("I2C init FAILED. IMU/PPG will be unavailable until pins/wiring are fixed.");
    }

    // I2C bus scan for debugging
    CONSOLE.println("I2C scan:");
    for (uint8_t a = 1; a < 127; a++) {
        Wire.beginTransmission(a);
        if (Wire.endTransmission() == 0) {
            CONSOLE.printf("  Found device at 0x%02X\n", a);
        }
    }

    g_imu_ok = g_imu.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    if (!g_imu_ok) {
        // Try reading WHO_AM_I raw for debug
        Wire.beginTransmission(0x68);
        Wire.write(0x75);
        Wire.endTransmission(false);
        uint8_t n = Wire.requestFrom(0x68, 1);
        if (n > 0) {
            uint8_t who = Wire.read();
            CONSOLE.printf("MPU6050 WHO_AM_I raw = 0x%02X (expected 0x68 or 0x70)\n", who);
        } else {
            CONSOLE.println("MPU6050: No response at 0x68");
        }
    }
    CONSOLE.println(g_imu_ok ? "MPU6050 IMU: Initialized" : "MPU6050 IMU: NOT DETECTED");

    // IMU diagnostic: 20 samples over 4 seconds with raw+converted values and min/max tracking
    if (g_imu_ok) {
        CONSOLE.println("IMU_DIAG: *** SHAKE THE BOARD NOW *** Reading 20 samples over 4 seconds...");
        float ax_min=999, ay_min=999, az_min=999, ax_max=-999, ay_max=-999, az_max=-999;
        float gx_max_abs=0, gy_max_abs=0, gz_max_abs=0;
        int read_ok = 0;
        for (int i = 0; i < 20; i++) {
            // Read raw registers directly for debugging
            uint8_t raw[14];
            Wire.beginTransmission(0x68);
            Wire.write(0x3B);
            Wire.endTransmission(false);
            Wire.requestFrom(0x68, 14, 1);
            for (int j = 0; j < 14; j++) raw[j] = Wire.read();
            int16_t rax = (int16_t)((raw[0]<<8)|raw[1]);
            int16_t ray = (int16_t)((raw[2]<<8)|raw[3]);
            int16_t raz = (int16_t)((raw[4]<<8)|raw[5]);
            int16_t rgx = (int16_t)((raw[8]<<8)|raw[9]);
            int16_t rgy = (int16_t)((raw[10]<<8)|raw[11]);
            int16_t rgz = (int16_t)((raw[12]<<8)|raw[13]);

            float ax = rax/16384.0f, ay = ray/16384.0f, az = raz/16384.0f;
            float gx = rgx/131.0f, gy = rgy/131.0f, gz = rgz/131.0f;

            if (i % 5 == 0) {  // Print every 5th sample
                CONSOLE.printf("IMU_DIAG: [%2d] raw=(%6d,%6d,%6d) g=(%.4f,%.4f,%.4f) gyro=(%.1f,%.1f,%.1f)\n",
                               i, rax, ray, raz, ax, ay, az, gx, gy, gz);
            }
            if (ax < ax_min) ax_min = ax; if (ax > ax_max) ax_max = ax;
            if (ay < ay_min) ay_min = ay; if (ay > ay_max) ay_max = ay;
            if (az < az_min) az_min = az; if (az > az_max) az_max = az;
            if (fabsf(gx) > gx_max_abs) gx_max_abs = fabsf(gx);
            if (fabsf(gy) > gy_max_abs) gy_max_abs = fabsf(gy);
            if (fabsf(gz) > gz_max_abs) gz_max_abs = fabsf(gz);
            read_ok++;
            delay(200);
        }
        float dx = ax_max-ax_min, dy = ay_max-ay_min, dz = az_max-az_min;
        CONSOLE.printf("IMU_DIAG: DELTA ax=%.4f ay=%.4f az=%.4f (should be >0.1 if shaking)\n", dx, dy, dz);
        CONSOLE.printf("IMU_DIAG: GYRO_MAX gx=%.1f gy=%.1f gz=%.1f dps (should be >10 if rotating)\n",
                       gx_max_abs, gy_max_abs, gz_max_abs);
        if (dx < 0.05f && dy < 0.05f && dz < 0.05f && gx_max_abs < 5.0f) {
            CONSOLE.println("IMU_DIAG: *** WARNING: IMU NOT RESPONDING TO MOTION ***");
            CONSOLE.println("IMU_DIAG: Check: 1) SDA/SCL wiring 2) MPU6050 soldering 3) Try different MPU6050");
        } else {
            CONSOLE.println("IMU_DIAG: Motion detected - IMU working correctly!");
        }
    }

#if ENABLE_MAX30102
    g_ppg_ok = g_ppg.begin(I2C_SDA_PIN, I2C_SCL_PIN, false);
    CONSOLE.println(g_ppg_ok ? "MAX30102 PPG: Initialized" : "MAX30102 PPG: NOT DETECTED");
#endif

#if ENABLE_PPG_ANALOG
    g_analog_ppg.begin();
    CONSOLE.println("Analog PPG: Initialized");
#endif

#if ENABLE_GSR
    g_gsr.begin();
    // Bring-up diagnostics for Grove GSR (and similar analog modules).
    // If adc is stuck near 0 or 4095, wiring/power/voltage-level is wrong.
    const uint16_t gsr_adc = g_gsr.readAdc();
    const uint16_t gsr_adc10 = g_gsr.readAdc10();
    const float gsr_v = g_gsr.readVoltage();
    const float gsr_r = g_gsr.readResistanceOhms();
    const float gsr_us = g_gsr.readConductance();
    CONSOLE.printf("GSR debug: adc12=%u adc10=%u voltage=%.3fV R=%.0f ohm G=%.2f uS\n",
                  static_cast<unsigned>(gsr_adc),
                  static_cast<unsigned>(gsr_adc10),
                  gsr_v, gsr_r, gsr_us);
    // IMPORTANT:
    // "Detected" vs "electrodes attached" are different.
    // The GSR module is analog; if it's wired to the ADC pin, we can always read it.
    // `isAttached()` is about skin contact / not railed readings.
    g_gsr_ok = true;
    const bool gsr_attached = g_gsr.isAttached();
    CONSOLE.println("GSR Sensor: Initialized");
    CONSOLE.printf("GSR electrodes: %s\n", gsr_attached ? "attached" : "NOT attached");
#endif

    g_stream.printSensorStatus(true, g_imu_ok, g_ppg_ok, g_gsr_ok);

    // Configure ECG filter after ADC + sampling settings are known
    if constexpr (ENABLE_ECG_FILTER) {
        g_ecg_filter.configure(
            static_cast<float>(SAMPLE_RATE_HZ),
            ECG_HIGHPASS_HZ,
            ECG_LOWPASS_HZ,
            ENABLE_ECG_NOTCH,
            ECG_MAINS_HZ
        );
        // Don't warmup here — we warmup on first real ADC sample in the loop
        // to use the actual DC level, not a guess.
        CONSOLE.printf("ECG filter: enabled (HP=%.2fHz LP=%.1fHz notch=%s@%.0fHz oversample=%u)\n",
                      ECG_HIGHPASS_HZ, ECG_LOWPASS_HZ,
                      ENABLE_ECG_NOTCH ? "on" : "off",
                      ECG_MAINS_HZ,
                      static_cast<unsigned>(ECG_ADC_OVERSAMPLE));
    } else {
        CONSOLE.println("ECG filter: disabled");
    }

    // Quick ECG bring-up diagnostic: if you see min/max near 0 or 4095, you're reading
    // the wrong pin or the OUT line is shorted / not grounded.
    {
        uint16_t mn = 4095, mx = 0;
        uint32_t acc = 0;
        const int n = 200;  // ~2 seconds at 100Hz equivalent
        for (int i = 0; i < n; i++) {
            const uint16_t v = g_ecg.readAdcAveraged(ECG_ADC_OVERSAMPLE);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            acc += v;
            delay(2);
        }
        const float mean = (float)acc / (float)n;
        CONSOLE.printf("ECG debug: raw_adc mean=%.1f min=%u max=%u on GPIO%d\n",
                      mean,
                      static_cast<unsigned>(mn),
                      static_cast<unsigned>(mx),
                      ECG_ADC_PIN);
        if (mn < 20 || mx > 4075 || (mx - mn) < 5) {
            CONSOLE.println("ECG debug: WARNING - signal looks railed/stuck. Check:");
            CONSOLE.println("  - AD8232 OUT must go to an ESP32-S3 ADC pin (GPIO 1-20).");
            CONSOLE.println("  - Common GND between AD8232 and ESP32-S3.");
            CONSOLE.println("  - Do NOT use GPIO34/35/36/39 (not ADC on ESP32-S3).");
            CONSOLE.println("  - LO+ / LO- are outputs; don't tie them to 3.3V/GND.");
        }
    }
}

// =============================
// Sample Acquisition
// =============================

static Sample readSample() {
    Sample s = {};
    s.t_us = micros();
    s.flags = 0;

    // ECG
    s.ecg_adc_raw = g_ecg.readAdcAveraged(ECG_ADC_OVERSAMPLE);

    // Optional spike clamp (helps electrode pops / cable hits)
    if constexpr (ENABLE_ECG_SPIKE_CLAMP) {
        static uint16_t last_raw = 2048;
        const int d = (int)s.ecg_adc_raw - (int)last_raw;
        if (d > (int)ECG_SPIKE_DELTA_ADC) {
            s.ecg_adc_raw = (uint16_t)(last_raw + ECG_SPIKE_DELTA_ADC);
        } else if (d < -(int)ECG_SPIKE_DELTA_ADC) {
            s.ecg_adc_raw = (uint16_t)(last_raw - ECG_SPIKE_DELTA_ADC);
        }
        last_raw = s.ecg_adc_raw;
    }

    if constexpr (ENABLE_ECG_FILTER) {
        // Warmup on first sample to settle IIR with actual DC level
        if (!g_ecg_filter.warmedUp()) {
            float dc = static_cast<float>(s.ecg_adc_raw);
            g_ecg_filter.reset(dc);
            g_ecg_filter.warmup(dc, 500);
        }
        const float filt = g_ecg_filter.process(static_cast<float>(s.ecg_adc_raw));
        // HP filter output is centered around 0 (DC removed).
        // Re-center at the running DC level so it stays within 12-bit range.
        static float dc_est = 2048.0f;
        static bool dc_init = false;
        if (!dc_init) { dc_est = static_cast<float>(s.ecg_adc_raw); dc_init = true; }
        dc_est += 0.001f * (static_cast<float>(s.ecg_adc_raw) - dc_est);
        int v = static_cast<int>(lroundf(filt + dc_est));
        if (v < 0) v = 0;
        if (v > 4095) v = 4095;
        s.ecg_adc = static_cast<uint16_t>(v);
    } else {
        s.ecg_adc = s.ecg_adc_raw;
    }
    if (g_ecg.leadOff()) {
        s.flags |= FLAG_ECG_LEAD_OFF;
    }

    // PPG
#if ENABLE_MAX30102
    // Hold-last valid PPG so plots don't become "triangles" when reads are intermittent.
    static uint32_t last_ppg_red = 0;
    static uint32_t last_ppg_ir = 0;
    bool ppg_fresh = false;
    if (g_ppg_ok) {
        uint32_t red, ir;
        if (g_ppg.readLatest(red, ir)) {
            last_ppg_red = red;
            last_ppg_ir = ir;
            ppg_fresh = true;
        }
    }
    s.ppg_red = last_ppg_red;
    s.ppg_ir = last_ppg_ir;
    if (ppg_fresh) {
        s.flags |= FLAG_PPG_VALID;
    }
#elif ENABLE_PPG_ANALOG
    s.ppg_red = g_analog_ppg.readAdc();
    s.ppg_ir = 0;
    s.flags |= FLAG_PPG_VALID;
#else
    s.ppg_red = 0;
    s.ppg_ir = 0;
#endif

    // IMU
    if (g_imu_ok) {
        float ax, ay, az, gx, gy, gz;
        if (g_imu.read(ax, ay, az, gx, gy, gz)) {
            s.ax_g = ax;
            s.ay_g = ay;
            s.az_g = az;
            s.gx_dps = gx;
            s.gy_dps = gy;
            s.gz_dps = gz;
            s.flags |= FLAG_IMU_VALID;
        }
    }

    // GSR
#if ENABLE_GSR
    s.gsr_adc = g_gsr.readAdc();
    s.gsr_conductance = g_gsr.readConductance();
    if (g_gsr.isAttached()) {
        s.flags |= FLAG_GSR_ATTACHED;
    }
#else
    s.gsr_adc = 0;
    s.gsr_conductance = 0.0f;
#endif

    return s;
}

// =============================
// Alert Handling
// =============================

static void handleAlert(uint8_t alert_type) {
    if (alert_type == ALERT_NONE) {
        return;
    }

    g_alert_count++;

    char line[64];
    CONSOLE.println();
    printBoxTop();

    switch (alert_type) {
        case ALERT_STRESS:
            printBoxLine("** STRESS ALERT DETECTED **");
            printBoxLine("Elevated stress at rest");
            break;

        case ALERT_ARRHYTHMIA:
            printBoxLine("!! ARRHYTHMIA ALERT DETECTED !!");
            printBoxLine("Abnormal heart rhythm at rest");
            break;

        case ALERT_CRITICAL:
            printBoxLine("!! CRITICAL ALERT !!");
            printBoxLine("ARRHYTHMIA DURING ACTIVITY");
            printBoxLine("Seek immediate medical attention");
            break;
    }

    printBoxDivider();
    snprintf(line, sizeof(line), "Alert count: %lu",
             static_cast<unsigned long>(g_alert_count));
    printBoxLine(line);
    printBoxBottom();
    CONSOLE.println();

    // TODO: Add buzzer, LED, or wireless notification here
}

// =============================
// Window Processing
// =============================

static void processFullWindow() {
    const uint32_t idx = g_window.windowIndex();
    g_stream.windowStart(idx);

#if ENABLE_PROFILING
    const uint32_t t_window_proc_start_us = micros();
    const float window_fill_ms = (g_prof_window_start_us > 0)
        ? (static_cast<float>(micros() - g_prof_window_start_us) / 1000.0f)
        : 0.0f;
#endif

    // Stream samples if enabled
    if constexpr (STREAM_EACH_SAMPLE) {
        const Sample* buf = g_window.data();
        for (uint16_t i = 0; i < g_window.size(); i++) {
            g_stream.printSample(buf[i]);
        }
    }

    g_stream.windowEnd(idx);

    // Run inference if enabled
#if ENABLE_INFERENCE
    if (!PROFILING_QUIET_MODE) {
        CONSOLE.println("\n--- Running Inference ---");
    }
    const unsigned long inference_start = micros();

    // Get sample buffer
    const Sample* samples = g_window.data();

    // Run inference
    NNResult result = nnInference.predictFromSamples(samples, g_window.size());

#if ENABLE_PROFILING
    const float infer_compute_ms = (static_cast<float>(micros() - inference_start) / 1000.0f);
    const uint32_t t_post_start_us = micros();
#endif

    if (result.valid) {
        // Feed calibration if active
        if (nnInference.isCalibrating()) {
            nnInference.feedCalibrationWindow(result);
        }

        // Machine-parseable inference line (always emitted, even in quiet mode)
        // INFER,t_ms,activity,act_conf,stress,str_conf,arrhythmia,arr_conf,is_moving,alert,str_logit0,str_logit1,arr_logit0,arr_logit1,calibrated
        CONSOLE.print("INFER,");
        CONSOLE.print(millis());           CONSOLE.print(',');
        CONSOLE.print(result.activity_class); CONSOLE.print(',');
        CONSOLE.print(result.activity_confidence, 3); CONSOLE.print(',');
        CONSOLE.print(result.stress_class);   CONSOLE.print(',');
        CONSOLE.print(result.stress_confidence, 3); CONSOLE.print(',');
        CONSOLE.print(result.arrhythmia_class); CONSOLE.print(',');
        CONSOLE.print(result.arrhythmia_confidence, 3); CONSOLE.print(',');
        CONSOLE.print(result.is_moving ? 1 : 0); CONSOLE.print(',');
        CONSOLE.print(result.alert_type);     CONSOLE.print(',');
        CONSOLE.print(result.stress_logits[0], 3); CONSOLE.print(',');
        CONSOLE.print(result.stress_logits[1], 3); CONSOLE.print(',');
        CONSOLE.print(result.arrhythmia_logits[0], 3); CONSOLE.print(',');
        CONSOLE.print(result.arrhythmia_logits[1], 3); CONSOLE.print(',');
        CONSOLE.println(nnInference.isCalibrated() ? 1 : 0);

        if (!PROFILING_QUIET_MODE) {
            char line[64];
            const bool activity_confident = (result.activity_confidence >= 0.25f);
            const char* activity_label = activity_confident
                ? ACTIVITY_NAMES[result.activity_class] : "Uncertain";

            CONSOLE.println();
            printBoxTop();
            printBoxLine("INFERENCE RESULTS");
            printBoxDivider();

            snprintf(line, sizeof(line), "Activity: %-12s (%.1f%%)",
                     activity_label, result.activity_confidence * 100.0f);
            printBoxLine(line);

            snprintf(line, sizeof(line), "Motion:   %-12s (a=%.3fg g=%.1f)",
                     result.is_moving ? "MOVING" : "REST",
                     result.accel_std_g, result.gyro_rms_dps);
            printBoxLine(line);

            snprintf(line, sizeof(line), "Stress:   %-12s (%.1f%%)",
                     STRESS_NAMES[result.stress_class],
                     result.stress_confidence * 100.0f);
            printBoxLine(line);

            snprintf(line, sizeof(line), "Heart:    %-12s (%.1f%%)",
                     result.arrhythmia_class == 0 ? "Normal" : "ABNORMAL",
                     result.arrhythmia_confidence * 100.0f);
            printBoxLine(line);

            printBoxDivider();
            snprintf(line, sizeof(line), "Inference time: %.2f ms",
                     result.inference_time_ms);
            printBoxLine(line);
            printBoxBottom();

            if (!activity_confident) {
                CONSOLE.printf("  Motion context: %s\n",
                              result.is_moving ? "Moving" : "Sedentary");
                printTop2Activity(result);
            }

            if (result.alert_triggered) {
                handleAlert(result.alert_type);
            }
        }

        g_last_alert_type = result.alert_type;

        // ---- On-Device Adaptive Training Hook ----
        // Feed inference results + CNN features to the adaptive trainer.
        // This enables drift tracking, trigger evaluation, and
        // automatic training episodes when conditions are met.
#if ENABLE_ADAPTIVE_TRAINING
        if (adaptiveTrainer.isReady()) {
            const float* features = nnInference.getProjectedFeatures();
            adaptiveTrainer.onInferenceComplete(result, features,
                                                 result.inference_time_ms);
        }
#endif

    } else {
        if (!PROFILING_QUIET_MODE) {
            CONSOLE.println("Inference failed!");
        }
    }

#if ENABLE_PROFILING
    const float postproc_ms = static_cast<float>(micros() - t_post_start_us) / 1000.0f;
    const float window_total_ms = static_cast<float>(micros() - t_window_proc_start_us) / 1000.0f;
    profPrintWindowLine(idx, window_fill_ms, infer_compute_ms, postproc_ms, window_total_ms);
    profResetPerWindow();
#endif
#endif

    g_window.nextWindow();
}

// =============================
// Arduino Setup
// =============================

void setup() {
    g_stream.begin(115200);
    delay(500);

    // Short boot message so the host sees output even if the monitor
    // attaches slightly after reset.
    for (int i = 0; i < 2; i++) {
        CONSOLE.printf("BOOT: ESP32-S3 Biomedical Monitor starting... (%d/2)\n", i + 1);
        delay(200);
    }

    CONSOLE.println();
    CONSOLE.println("╔═══════════════════════════════════════════╗");
    CONSOLE.println("║  ESP32-S3 Biomedical Monitor v1.0         ║");
    CONSOLE.println("║  Multimodal Health Monitoring System      ║");
    CONSOLE.println("╚═══════════════════════════════════════════╝");
    CONSOLE.println();

    // Configure ADC
    configureAdc();

    // Initialize sensors
    initializeSensors();

#if ENABLE_INFERENCE
    CONSOLE.println("\nInitializing Neural Network...");
    if (nnInference.begin()) {
        const uint32_t model_params = static_cast<uint32_t>(nnInference.getModelSize() / sizeof(float));
        CONSOLE.printf("Model loaded: %lu parameters (%.1f KB)\n",
                      static_cast<unsigned long>(model_params),
                      nnInference.getModelSize() / 1024.0f);
    } else {
        CONSOLE.println("ERROR: Failed to initialize inference engine!");
    }
#if ENABLE_ADAPTIVE_TRAINING
    CONSOLE.println("\nInitializing Adaptive Training System...");
    if (adaptiveTrainer.begin()) {
        CONSOLE.println("Adaptive Training System ready!");
    } else {
        CONSOLE.println("WARNING: Adaptive Training System init failed");
        CONSOLE.println("  On-device adaptation will be unavailable");
    }
#else
    CONSOLE.println("\nAdaptive training disabled (ENABLE_ADAPTIVE_TRAINING=false)");
#endif

#else
    CONSOLE.println("\nInference disabled (ENABLE_INFERENCE=false)");
    CONSOLE.println("Set ENABLE_INFERENCE=true in config.h to enable on-device ML");
#endif

    // Print CSV header
    CONSOLE.println();
    g_stream.printHeader();

#if ENABLE_PROFILING
    CONSOLE.println();
    profMaybePrintHeader();
    profResetPerWindow();
#endif

    // Initialize timing
    g_next_sample_us = micros();

    CONSOLE.println("\nStarting data acquisition...");
    CONSOLE.println("Each window = 10 seconds (1000 samples @ 100Hz)\n");
}

// =============================
// Arduino Loop
// =============================

void loop() {
    const uint32_t now = micros();

    // Serial command processing (demo alerts + adaptive training commands)
    {
        static char cmd_buf[64];
        static uint8_t cmd_len = 0;

        while (CONSOLE.available() > 0) {
            const char c = static_cast<char>(CONSOLE.read());

            // Line-based commands (for adaptive training: CORRECT, TRAIN, etc.)
            if (c == '\n' || c == '\r') {
                if (cmd_len > 0) {
                    cmd_buf[cmd_len] = '\0';

#if ENABLE_ADAPTIVE_TRAINING && ENABLE_INFERENCE
                    // Try adaptive training commands first
                    if (!adaptiveTrainer.handleCommand(cmd_buf)) {
                        // Not a training command — try single-char demo
                        if (cmd_len == 1 && ENABLE_DEMO_ALERT_COMMANDS) {
                            handleDemoCommand(cmd_buf[0]);
                        }
                    }
#else
                    // Handle CALIBRATE command
                    if (strncmp(cmd_buf, "CALIBRATE", 9) == 0) {
                        int n = 5;
                        if (cmd_len > 10) n = atoi(cmd_buf + 10);
                        if (n < 1) n = 1;
                        if (n > 20) n = 20;
                        nnInference.startCalibration(n);
                    } else if (strcmp(cmd_buf, "CALRESET") == 0) {
                        nnInference.resetCalibration();
                    } else if (ENABLE_DEMO_ALERT_COMMANDS && cmd_len == 1) {
                        handleDemoCommand(cmd_buf[0]);
                    }
#endif
                    cmd_len = 0;
                }
            } else if (cmd_len < sizeof(cmd_buf) - 1) {
                cmd_buf[cmd_len++] = c;
            }

            // Single-character demo commands (immediate, no newline needed)
            if (ENABLE_DEMO_ALERT_COMMANDS && cmd_len == 0 &&
                (c == '0' || c == '1' || c == '3' || c == '4')) {
                handleDemoCommand(c);
            }
        }
    }

    // Heartbeat: proves the firmware is running. Frequent before first sample,
    // sparse once telemetry is flowing.
    static uint32_t last_heartbeat_ms = 0;
    static bool first_sample_seen = false;
    const uint32_t now_ms = millis();
    const uint32_t hb_interval = first_sample_seen ? 30000 : 5000;
    if (DEBUG_SERIAL && (now_ms - last_heartbeat_ms) >= hb_interval) {
        last_heartbeat_ms = now_ms;
        CONSOLE.printf("DEBUG: alive t=%lums win=%u/%u\n",
                       (unsigned long)now_ms,
                       g_window.size(), WINDOW_SAMPLES);
    }

    // Live telemetry at a low rate (default 10 Hz)
    static uint32_t last_live_ms = 0;
    if (ENABLE_LIVE_TELEMETRY) {
        const uint32_t period_ms = (LIVE_TELEMETRY_HZ > 0) ? (1000u / LIVE_TELEMETRY_HZ) : 100u;
        if ((now_ms - last_live_ms) >= period_ms) {
            last_live_ms = now_ms;
            printLiveTelemetry(g_last_sample);
        }
    }

    // High-rate ECG telemetry (default 100 Hz) to avoid "squared" ECG in plots.
    static uint32_t last_ecg_ms = 0;
    if (ENABLE_ECG_TELEMETRY) {
        const uint32_t period_ms = (ECG_TELEMETRY_HZ > 0) ? (1000u / ECG_TELEMETRY_HZ) : 10u;
        if ((now_ms - last_ecg_ms) >= period_ms) {
            last_ecg_ms = now_ms;
            printEcgTelemetry(g_last_sample);
        }
    }

    // High-rate PPG telemetry (default 100 Hz) for a clean pulse waveform in plots.
    static uint32_t last_ppg_ms = 0;
    if (ENABLE_PPG_TELEMETRY) {
        const uint32_t period_ms = (PPG_TELEMETRY_HZ > 0) ? (1000u / PPG_TELEMETRY_HZ) : 10u;
        if ((now_ms - last_ppg_ms) >= period_ms) {
            last_ppg_ms = now_ms;
            printPpgTelemetry(g_last_sample);
        }
    }

    // Sample at precise 100Hz
    if ((int32_t)(now - g_next_sample_us) >= 0) {
        g_next_sample_us += SAMPLE_PERIOD_US;

        // Acquire sample
#if ENABLE_PROFILING
        const uint32_t t0 = micros();
        const Sample s = readSample();
        const uint32_t dt = micros() - t0;
        if (PROFILING_TRACK_SAMPLE_TIMING) {
            g_prof_acq_us_sum += dt;
            g_prof_acq_count++;
            if (dt > g_prof_acq_us_max) g_prof_acq_us_max = dt;
        }
        // Track window start timestamp when the first sample arrives
        if (g_window.size() == 0) {
            g_prof_window_start_us = micros();
        }
#else
        const Sample s = readSample();
#endif
        g_last_sample = s;
        first_sample_seen = true;
        g_window.push(s);

        // Process complete window
        if (g_window.full()) {
            processFullWindow();
        }
    }

    // Yield to RTOS
    delay(0);
}
