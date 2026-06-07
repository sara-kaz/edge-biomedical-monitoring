#pragma once

/**
 * Neural Network Inference Engine for ESP32-S3
 *
 * Full V5 architecture: CNN + SE-Attention + 2-Layer Transformer
 *   Input: [5 ch x 1000 samples]
 *   → Conv1(5→32,k7)/BN/ReLU/Pool → [32,500]
 *   → Conv2(32→64,k5)/BN/ReLU/Pool → [64,250]
 *   → Conv3(64→64,k3)/BN/ReLU/Pool → [64,125]
 *   → SE-Attention(64→16→64)
 *   → Per-timestep Projection(64→64) + LayerNorm + PosEnc
 *   → 2-Layer Transformer (4 heads, d_ff=128, pre-norm, GELU)
 *   → Mean Pool → [64]
 *   → 3-layer Task Heads
 */

#include <Arduino.h>
#include <cmath>
#include "../types.h"

namespace NNConfig {
    constexpr int INPUT_CHANNELS = 5;
    constexpr int INPUT_SAMPLES = 1000;
    constexpr int INPUT_SIZE = INPUT_CHANNELS * INPUT_SAMPLES;

    // CNN
    constexpr int CONV1_OUT_CH = 32;
    constexpr int CONV2_OUT_CH = 64;
    constexpr int CONV3_OUT_CH = 64;
    constexpr int CONV1_KERNEL = 7;
    constexpr int CONV2_KERNEL = 5;
    constexpr int CONV3_KERNEL = 3;
    constexpr int POOL_SIZE = 2;

    // SE
    constexpr int SE_REDUCTION = 16;
    constexpr int CNN_OUT_DIM = 64;

    // Transformer
    constexpr int D_MODEL = 64;
    constexpr int NHEAD = 4;
    constexpr int D_HEAD = D_MODEL / NHEAD;  // 16
    constexpr int D_FF = 128;
    constexpr int TF_LAYERS = 2;

    constexpr int FEATURE_DIM = 64;

    // Head hidden dims
    constexpr int ACT_HIDDEN1 = 64;
    constexpr int ACT_HIDDEN2 = 32;
    constexpr int STR_HIDDEN1 = 48;
    constexpr int STR_HIDDEN2 = 24;
    constexpr int ARR_HIDDEN1 = 48;
    constexpr int ARR_HIDDEN2 = 24;

    // Output classes
    constexpr int ACTIVITY_CLASSES = 4;
    constexpr int STRESS_CLASSES = 2;
    constexpr int ARRHYTHMIA_CLASSES = 2;

    // Legacy compat
    constexpr int HEAD_HIDDEN_DIM = 64;

    // Alerts
    constexpr float ARRHYTHMIA_THRESHOLD = 0.7f;
    constexpr float STRESS_THRESHOLD = 0.6f;

    // Seq lengths after pooling
    constexpr int SEQ_LEN_1 = INPUT_SAMPLES / POOL_SIZE;  // 500
    constexpr int SEQ_LEN_2 = SEQ_LEN_1 / POOL_SIZE;      // 250
    constexpr int SEQ_LEN_3 = SEQ_LEN_2 / POOL_SIZE;      // 125
}

struct NNResult {
    uint8_t activity_class;
    float activity_confidence;
    float activity_probs[NNConfig::ACTIVITY_CLASSES];
    uint8_t stress_class;
    float stress_confidence;
    float stress_probs[NNConfig::STRESS_CLASSES];
    float stress_logits[NNConfig::STRESS_CLASSES];   // raw logits before calibration
    uint8_t arrhythmia_class;
    float arrhythmia_confidence;
    float arrhythmia_probs[NNConfig::ARRHYTHMIA_CLASSES];
    float arrhythmia_logits[NNConfig::ARRHYTHMIA_CLASSES]; // raw logits before calibration
    uint8_t alert_type;
    bool alert_triggered;
    bool is_moving;
    float accel_std_g;
    float gyro_rms_dps;
    float inference_time_ms;
    bool valid;
};

class NNInference {
public:
    NNInference();
    ~NNInference();
    bool begin();
    NNResult predict(const float* input);
    NNResult predictFromSamples(const Sample* samples, int num_samples);
    bool isReady() const { return _initialized; }
    size_t getModelSize() const { return _model_size; }
    const float* getProjectedFeatures() const { return _projected; }

    // Output calibration: corrects domain-shift bias in stress/arrhythmia heads
    void startCalibration(int num_windows = 5);
    void feedCalibrationWindow(const NNResult& result);
    bool isCalibrating() const { return _cal_remaining > 0; }
    bool isCalibrated() const { return _calibrated; }
    void resetCalibration();

private:
    bool _initialized;
    size_t _model_size;

    // Calibration state
    bool _calibrated;
    int _cal_remaining;
    int _cal_total;
    float _cal_stress_logit_sum[NNConfig::STRESS_CLASSES];
    float _cal_arr_logit_sum[NNConfig::ARRHYTHMIA_CLASSES];
    float _stress_bias[NNConfig::STRESS_CLASSES];
    float _arr_bias[NNConfig::ARRHYTHMIA_CLASSES];

    // CNN buffers
    float* _input_normalized;
    float* _conv1_out;
    float* _conv2_out;
    float* _conv3_out;
    float* _pooled_features;
    float* _se_hidden;
    float* _se_scale;

    // Transformer buffers
    float* _seq;         // [SEQ_LEN_3, D_MODEL] = [125, 64]
    float* _seq_tmp;     // temp for residual
    float* _qkv;         // [SEQ_LEN_3, 3*D_MODEL] = [125, 192]
    float* _attn_out;    // [SEQ_LEN_3, D_MODEL]
    float* _attn_scores; // [SEQ_LEN_3, SEQ_LEN_3] (one head at a time)
    float* _ffn_mid;     // [SEQ_LEN_3, D_FF] = [125, 128]

    // Output buffers
    float* _projected;   // [FEATURE_DIM]
    float* _head_buf1;
    float* _head_buf2;

    // Layer ops
    void conv1d_relu(const float* input, float* output, int in_ch, int out_ch,
                     int in_len, int kernel, const float* w, const float* b);
    void maxpool1d(const float* input, float* output, int ch, int in_len, int pool);
    void global_avg_pool(const float* input, float* output, int ch, int len);
    void linear(const float* input, float* output, int in_f, int out_f,
                const float* w, const float* b);
    void linear_seq(const float* seq_in, float* seq_out, int seq_len,
                    int in_f, int out_f, const float* w, const float* b);
    void layernorm(float* data, int size, const float* w, const float* b);
    void layernorm_seq(float* seq, int seq_len, int dim, const float* w, const float* b);
    void softmax(float* data, int size);
    void relu(float* data, int size);
    void gelu(float* data, int size);
    void gelu_seq(float* seq, int seq_len, int dim);
    void sigmoid(float* data, int size);
    uint8_t argmax(const float* data, int size);
    void normalizeInput(const float* raw, float* normalized);
    uint8_t determineAlert(const NNResult& result);

    // Transformer ops
    void transformerLayer(float* seq, int seq_len,
                          const float* in_proj_w, const float* in_proj_b,
                          const float* out_proj_w, const float* out_proj_b,
                          const float* norm1_w, const float* norm1_b,
                          const float* ffn1_w, const float* ffn1_b,
                          const float* ffn2_w, const float* ffn2_b,
                          const float* norm2_w, const float* norm2_b);
    void multiheadAttention(const float* seq_in, float* seq_out, int seq_len,
                            const float* in_proj_w, const float* in_proj_b,
                            const float* out_proj_w, const float* out_proj_b);
    void addPosEncoding(float* seq, int seq_len, int d_model);
};

extern NNInference nnInference;
