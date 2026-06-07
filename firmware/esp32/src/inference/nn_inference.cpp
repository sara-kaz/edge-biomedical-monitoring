/**
 * Neural Network Inference — Full V5 (CNN + SE + 2-Layer Transformer)
 */

#include "nn_inference.h"
#include "model_weights.h"
#include "../transport/console.h"

NNInference nnInference;

NNInference::NNInference()
    : _initialized(false), _model_size(0),
      _calibrated(false), _cal_remaining(0), _cal_total(0),
      _input_normalized(nullptr), _conv1_out(nullptr), _conv2_out(nullptr),
      _conv3_out(nullptr), _pooled_features(nullptr), _se_hidden(nullptr),
      _se_scale(nullptr), _seq(nullptr), _seq_tmp(nullptr), _qkv(nullptr),
      _attn_out(nullptr), _attn_scores(nullptr), _ffn_mid(nullptr),
      _projected(nullptr), _head_buf1(nullptr), _head_buf2(nullptr) {
    memset(_cal_stress_logit_sum, 0, sizeof(_cal_stress_logit_sum));
    memset(_cal_arr_logit_sum, 0, sizeof(_cal_arr_logit_sum));
    memset(_stress_bias, 0, sizeof(_stress_bias));
    memset(_arr_bias, 0, sizeof(_arr_bias));
}

NNInference::~NNInference() {
    float** ptrs[] = {&_input_normalized, &_conv1_out, &_conv2_out, &_conv3_out,
                      &_pooled_features, &_se_hidden, &_se_scale,
                      &_seq, &_seq_tmp, &_qkv, &_attn_out, &_attn_scores, &_ffn_mid,
                      &_projected, &_head_buf1, &_head_buf2};
    for (auto p : ptrs) { if (*p) free(*p); }
}

bool NNInference::begin() {
    if (_initialized) return true;
    CONSOLE.println("Initializing V5 Inference Engine (CNN+SE+Transformer)...");

    const int S = NNConfig::SEQ_LEN_3;  // 125
    const int D = NNConfig::D_MODEL;    // 64

    // Allocate all buffers in PSRAM
    auto alloc = [](size_t bytes) -> float* {
#ifdef BOARD_HAS_PSRAM
        float* p = (float*)ps_malloc(bytes);
        if (p) return p;
#endif
        return (float*)malloc(bytes);
    };

    // _input_normalized is reused as temp buffer after conv1 maxpool
    // Needs to hold max(INPUT_SIZE, CONV1_OUT_CH * SEQ_LEN_1) floats
    const int input_buf_size = max((int)NNConfig::INPUT_SIZE,
                                   NNConfig::CONV1_OUT_CH * NNConfig::SEQ_LEN_1);
    _input_normalized = alloc(input_buf_size * 4);
    _conv1_out = alloc(NNConfig::CONV1_OUT_CH * NNConfig::INPUT_SAMPLES * 4);
    _conv2_out = alloc(NNConfig::CONV2_OUT_CH * NNConfig::SEQ_LEN_1 * 4);
    _conv3_out = alloc(NNConfig::CONV3_OUT_CH * NNConfig::SEQ_LEN_2 * 4);
    _pooled_features = alloc(NNConfig::CNN_OUT_DIM * 4);
    _se_hidden = alloc(NNConfig::SE_REDUCTION * 4);
    _se_scale = alloc(NNConfig::CNN_OUT_DIM * 4);

    // Transformer buffers
    _seq = alloc(S * D * 4);
    _seq_tmp = alloc(S * D * 4);
    _qkv = alloc(S * 3 * D * 4);
    _attn_out = alloc(S * D * 4);
    _attn_scores = alloc(S * S * 4);   // One head at a time
    _ffn_mid = alloc(S * NNConfig::D_FF * 4);

    _projected = alloc(NNConfig::FEATURE_DIM * 4);
    _head_buf1 = alloc(64 * 4);  // max head hidden
    _head_buf2 = alloc(64 * 4);

    // Check all allocations
    if (!_input_normalized || !_conv1_out || !_conv2_out || !_conv3_out ||
        !_pooled_features || !_se_hidden || !_se_scale ||
        !_seq || !_seq_tmp || !_qkv || !_attn_out || !_attn_scores || !_ffn_mid ||
        !_projected || !_head_buf1 || !_head_buf2) {
        CONSOLE.println("ERROR: Buffer allocation failed!");
        return false;
    }

    size_t total_buf = (input_buf_size + NNConfig::CONV1_OUT_CH*NNConfig::INPUT_SAMPLES +
                        NNConfig::CONV2_OUT_CH*NNConfig::SEQ_LEN_1 + NNConfig::CONV3_OUT_CH*NNConfig::SEQ_LEN_2 +
                        NNConfig::CNN_OUT_DIM + NNConfig::SE_REDUCTION + NNConfig::CNN_OUT_DIM +
                        S*D + S*D + S*3*D + S*D + S*S + S*NNConfig::D_FF +
                        NNConfig::FEATURE_DIM + 64 + 64) * 4;
    CONSOLE.printf("Buffers: %.1f KB\n", total_buf / 1024.0f);

    _model_size = MODEL_TOTAL_PARAMS * sizeof(float);
    CONSOLE.printf("Model: %d params (%.1f KB)\n", MODEL_TOTAL_PARAMS, _model_size / 1024.0f);

    _initialized = true;
    CONSOLE.println("V5 Inference Engine ready!");
    return true;
}

// ============================================
// Main Inference
// ============================================

NNResult NNInference::predict(const float* input) {
    NNResult result = {};
    result.valid = false;
    result.is_moving = false;
    if (!_initialized) return result;

    unsigned long start = micros();
    const int S = NNConfig::SEQ_LEN_3;  // 125
    const int D = NNConfig::D_MODEL;    // 64

    // 1. Per-window per-channel z-score normalization.
    //    Training pipeline applied global z-score (mean≈0, std≈1 per channel)
    //    before the model saw data.  Per-window z-score approximates this.
    //    NOTE: Domain shift between hardware sensors (AD8232, MAX30102, MPU6050)
    //    and research-grade training sensors (Empatica E4, RespiBAN) means the
    //    model's CNN features don't activate the same way. This is a known
    //    limitation — see thesis Section 5.3 (Domain Shift Analysis).
    {
        constexpr float MIN_STD = 1e-3f;
        const int N = NNConfig::INPUT_SAMPLES;  // 1000
        memcpy(_input_normalized, input, NNConfig::INPUT_SIZE * sizeof(float));
        for (int ch = 0; ch < NNConfig::INPUT_CHANNELS; ch++) {
            float* ptr = _input_normalized + ch * N;
            float sum = 0.0f, sum2 = 0.0f;
            for (int s = 0; s < N; s++) {
                sum  += ptr[s];
                sum2 += ptr[s] * ptr[s];
            }
            float mean = sum / N;
            float var  = sum2 / N - mean * mean;
            float std  = sqrtf(fmaxf(var, 0.0f));
            if (std < MIN_STD) std = 1.0f;  // dead/constant channel fallback
            float inv_std = 1.0f / std;
            for (int s = 0; s < N; s++) {
                ptr[s] = (ptr[s] - mean) * inv_std;
            }
        }
    }

    // 2. CNN backbone
    conv1d_relu(_input_normalized, _conv1_out,
                NNConfig::INPUT_CHANNELS, NNConfig::CONV1_OUT_CH,
                NNConfig::INPUT_SAMPLES, NNConfig::CONV1_KERNEL, conv1_weight, conv1_bias);
    maxpool1d(_conv1_out, _input_normalized,
              NNConfig::CONV1_OUT_CH, NNConfig::INPUT_SAMPLES, NNConfig::POOL_SIZE);

    conv1d_relu(_input_normalized, _conv2_out,
                NNConfig::CONV1_OUT_CH, NNConfig::CONV2_OUT_CH,
                NNConfig::SEQ_LEN_1, NNConfig::CONV2_KERNEL, conv2_weight, conv2_bias);
    maxpool1d(_conv2_out, _conv1_out,
              NNConfig::CONV2_OUT_CH, NNConfig::SEQ_LEN_1, NNConfig::POOL_SIZE);

    conv1d_relu(_conv1_out, _conv3_out,
                NNConfig::CONV2_OUT_CH, NNConfig::CONV3_OUT_CH,
                NNConfig::SEQ_LEN_2, NNConfig::CONV3_KERNEL, conv3_weight, conv3_bias);
    maxpool1d(_conv3_out, _conv2_out,
              NNConfig::CONV3_OUT_CH, NNConfig::SEQ_LEN_2, NNConfig::POOL_SIZE);
    // _conv2_out now holds [64, 125]

    // 3. SE-Attention
    global_avg_pool(_conv2_out, _pooled_features, NNConfig::CNN_OUT_DIM, S);
    linear(_pooled_features, _se_hidden, NNConfig::CNN_OUT_DIM, NNConfig::SE_REDUCTION,
           se_fc1_weight, se_fc1_bias);
    relu(_se_hidden, NNConfig::SE_REDUCTION);
    linear(_se_hidden, _se_scale, NNConfig::SE_REDUCTION, NNConfig::CNN_OUT_DIM,
           se_fc2_weight, se_fc2_bias);
    sigmoid(_se_scale, NNConfig::CNN_OUT_DIM);
    for (int c = 0; c < NNConfig::CNN_OUT_DIM; c++)
        for (int t = 0; t < S; t++)
            _conv2_out[c * S + t] *= _se_scale[c];

    // 4. Transpose [64, 125] → [125, 64] into _seq
    for (int t = 0; t < S; t++)
        for (int d = 0; d < D; d++)
            _seq[t * D + d] = _conv2_out[d * S + t];

    // 5. Per-timestep projection + LayerNorm
    linear_seq(_seq, _seq_tmp, S, NNConfig::CNN_OUT_DIM, D, projection_weight, projection_bias);
    memcpy(_seq, _seq_tmp, S * D * sizeof(float));
    layernorm_seq(_seq, S, D, proj_ln_weight, proj_ln_bias);

    // 6. Add sinusoidal positional encoding
    addPosEncoding(_seq, S, D);

    // 7. Transformer (2 layers, pre-norm)
    transformerLayer(_seq, S,
                     tf0_attn_in_proj_weight, tf0_attn_in_proj_bias,
                     tf0_attn_out_proj_weight, tf0_attn_out_proj_bias,
                     tf0_norm1_weight, tf0_norm1_bias,
                     tf0_ffn1_weight, tf0_ffn1_bias,
                     tf0_ffn2_weight, tf0_ffn2_bias,
                     tf0_norm2_weight, tf0_norm2_bias);

    transformerLayer(_seq, S,
                     tf1_attn_in_proj_weight, tf1_attn_in_proj_bias,
                     tf1_attn_out_proj_weight, tf1_attn_out_proj_bias,
                     tf1_norm1_weight, tf1_norm1_bias,
                     tf1_ffn1_weight, tf1_ffn1_bias,
                     tf1_ffn2_weight, tf1_ffn2_bias,
                     tf1_norm2_weight, tf1_norm2_bias);

    // 8. Mean pool over sequence → [64]
    for (int d = 0; d < D; d++) {
        float sum = 0;
        for (int t = 0; t < S; t++) sum += _seq[t * D + d];
        _projected[d] = sum / S;
    }

    // 9. Task heads (FC1→LN→GELU→FC2→GELU→FC3→Softmax)
    // Activity: [64]→[64]→LN→GELU→[32]→GELU→[4]
    linear(_projected, _head_buf1, D, NNConfig::ACT_HIDDEN1,
           activity_head_fc1_weight, activity_head_fc1_bias);
    layernorm(_head_buf1, NNConfig::ACT_HIDDEN1, activity_head_ln_weight, activity_head_ln_bias);
    gelu(_head_buf1, NNConfig::ACT_HIDDEN1);
    linear(_head_buf1, _head_buf2, NNConfig::ACT_HIDDEN1, NNConfig::ACT_HIDDEN2,
           activity_head_fc2_weight, activity_head_fc2_bias);
    gelu(_head_buf2, NNConfig::ACT_HIDDEN2);
    linear(_head_buf2, result.activity_probs, NNConfig::ACT_HIDDEN2, NNConfig::ACTIVITY_CLASSES,
           activity_head_fc3_weight, activity_head_fc3_bias);
    softmax(result.activity_probs, NNConfig::ACTIVITY_CLASSES);
    result.activity_class = argmax(result.activity_probs, NNConfig::ACTIVITY_CLASSES);
    result.activity_confidence = result.activity_probs[result.activity_class];

    // Stress: [64]→[48]→LN→GELU→[24]→GELU→[2]
    linear(_projected, _head_buf1, D, NNConfig::STR_HIDDEN1,
           stress_head_fc1_weight, stress_head_fc1_bias);
    layernorm(_head_buf1, NNConfig::STR_HIDDEN1, stress_head_ln_weight, stress_head_ln_bias);
    gelu(_head_buf1, NNConfig::STR_HIDDEN1);
    linear(_head_buf1, _head_buf2, NNConfig::STR_HIDDEN1, NNConfig::STR_HIDDEN2,
           stress_head_fc2_weight, stress_head_fc2_bias);
    gelu(_head_buf2, NNConfig::STR_HIDDEN2);
    linear(_head_buf2, result.stress_probs, NNConfig::STR_HIDDEN2, NNConfig::STRESS_CLASSES,
           stress_head_fc3_weight, stress_head_fc3_bias);
    // Save raw logits before calibration
    memcpy(result.stress_logits, result.stress_probs, sizeof(result.stress_logits));
    // Apply calibration bias if available
    if (_calibrated) {
        for (int i = 0; i < NNConfig::STRESS_CLASSES; i++)
            result.stress_probs[i] -= _stress_bias[i];
    }
    softmax(result.stress_probs, NNConfig::STRESS_CLASSES);
    result.stress_class = argmax(result.stress_probs, NNConfig::STRESS_CLASSES);
    result.stress_confidence = result.stress_probs[result.stress_class];

    // Arrhythmia: [64]→[48]→LN→GELU→[24]→GELU→[2]
    linear(_projected, _head_buf1, D, NNConfig::ARR_HIDDEN1,
           arrhythmia_head_fc1_weight, arrhythmia_head_fc1_bias);
    layernorm(_head_buf1, NNConfig::ARR_HIDDEN1, arrhythmia_head_ln_weight, arrhythmia_head_ln_bias);
    gelu(_head_buf1, NNConfig::ARR_HIDDEN1);
    linear(_head_buf1, _head_buf2, NNConfig::ARR_HIDDEN1, NNConfig::ARR_HIDDEN2,
           arrhythmia_head_fc2_weight, arrhythmia_head_fc2_bias);
    gelu(_head_buf2, NNConfig::ARR_HIDDEN2);
    linear(_head_buf2, result.arrhythmia_probs, NNConfig::ARR_HIDDEN2, NNConfig::ARRHYTHMIA_CLASSES,
           arrhythmia_head_fc3_weight, arrhythmia_head_fc3_bias);
    // Save raw logits before calibration
    memcpy(result.arrhythmia_logits, result.arrhythmia_probs, sizeof(result.arrhythmia_logits));
    // Apply calibration bias if available
    if (_calibrated) {
        for (int i = 0; i < NNConfig::ARRHYTHMIA_CLASSES; i++)
            result.arrhythmia_probs[i] -= _arr_bias[i];
    }
    softmax(result.arrhythmia_probs, NNConfig::ARRHYTHMIA_CLASSES);
    result.arrhythmia_class = argmax(result.arrhythmia_probs, NNConfig::ARRHYTHMIA_CLASSES);
    result.arrhythmia_confidence = result.arrhythmia_probs[result.arrhythmia_class];

    result.alert_type = determineAlert(result);
    result.alert_triggered = (result.alert_type != ALERT_NONE);
    result.inference_time_ms = (micros() - start) / 1000.0f;
    result.valid = true;
    return result;
}

// ============================================
// Transformer Layer (pre-norm)
// ============================================

void NNInference::transformerLayer(float* seq, int seq_len,
    const float* in_proj_w, const float* in_proj_b,
    const float* out_proj_w, const float* out_proj_b,
    const float* norm1_w, const float* norm1_b,
    const float* ffn1_w, const float* ffn1_b,
    const float* ffn2_w, const float* ffn2_b,
    const float* norm2_w, const float* norm2_b) {

    const int D = NNConfig::D_MODEL;

    // --- Self-attention with pre-norm ---
    // Save residual
    memcpy(_seq_tmp, seq, seq_len * D * sizeof(float));
    // Pre-norm
    layernorm_seq(seq, seq_len, D, norm1_w, norm1_b);
    // Multi-head attention
    multiheadAttention(seq, _attn_out, seq_len, in_proj_w, in_proj_b, out_proj_w, out_proj_b);
    // Residual add
    for (int i = 0; i < seq_len * D; i++) seq[i] = _seq_tmp[i] + _attn_out[i];

    // --- FFN with pre-norm ---
    // Save residual
    memcpy(_seq_tmp, seq, seq_len * D * sizeof(float));
    // Pre-norm
    layernorm_seq(seq, seq_len, D, norm2_w, norm2_b);
    // FFN: linear1 → GELU → linear2
    linear_seq(seq, _ffn_mid, seq_len, D, NNConfig::D_FF, ffn1_w, ffn1_b);
    gelu_seq(_ffn_mid, seq_len, NNConfig::D_FF);
    linear_seq(_ffn_mid, seq, seq_len, NNConfig::D_FF, D, ffn2_w, ffn2_b);
    // Residual add
    for (int i = 0; i < seq_len * D; i++) seq[i] += _seq_tmp[i];
}

void NNInference::multiheadAttention(const float* seq_in, float* seq_out, int seq_len,
    const float* in_proj_w, const float* in_proj_b,
    const float* out_proj_w, const float* out_proj_b) {

    const int D = NNConfig::D_MODEL;
    const int H = NNConfig::NHEAD;
    const int Dh = NNConfig::D_HEAD;
    const float scale = 1.0f / sqrtf((float)Dh);

    // Compute Q, K, V: [S, D] → [S, 3*D]
    linear_seq(seq_in, _qkv, seq_len, D, 3 * D, in_proj_w, in_proj_b);

    // Clear output
    memset(seq_out, 0, seq_len * D * sizeof(float));

    // Process each head
    for (int h = 0; h < H; h++) {
        int q_off = h * Dh;
        int k_off = D + h * Dh;
        int v_off = 2 * D + h * Dh;

        // Compute attention scores: [S, S]
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float dot = 0;
                for (int d = 0; d < Dh; d++) {
                    dot += _qkv[i * 3 * D + q_off + d] * _qkv[j * 3 * D + k_off + d];
                }
                _attn_scores[i * seq_len + j] = dot * scale;
            }
        }

        // Softmax per row
        for (int i = 0; i < seq_len; i++) {
            float max_val = _attn_scores[i * seq_len];
            for (int j = 1; j < seq_len; j++)
                if (_attn_scores[i * seq_len + j] > max_val)
                    max_val = _attn_scores[i * seq_len + j];
            float sum = 0;
            for (int j = 0; j < seq_len; j++) {
                _attn_scores[i * seq_len + j] = expf(_attn_scores[i * seq_len + j] - max_val);
                sum += _attn_scores[i * seq_len + j];
            }
            for (int j = 0; j < seq_len; j++)
                _attn_scores[i * seq_len + j] /= sum;
        }

        // Weighted sum of V: attn_scores @ V_h → [S, Dh]
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < Dh; d++) {
                float sum = 0;
                for (int j = 0; j < seq_len; j++) {
                    sum += _attn_scores[i * seq_len + j] * _qkv[j * 3 * D + v_off + d];
                }
                // Write to attn_out at head position
                // We accumulate across heads directly in the concat buffer
                _attn_out[i * D + h * Dh + d] = sum;
            }
        }
    }

    // Output projection: [S, D] → [S, D]
    // Use _seq_tmp as temp
    memcpy(_seq_tmp, _attn_out, seq_len * D * sizeof(float));
    linear_seq(_seq_tmp, _attn_out, seq_len, D, D, out_proj_w, out_proj_b);
}

void NNInference::addPosEncoding(float* seq, int seq_len, int d_model) {
    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < d_model; i += 2) {
            float div = expf((float)i * (-logf(10000.0f) / d_model));
            seq[t * d_model + i] += sinf((float)t * div);
            if (i + 1 < d_model)
                seq[t * d_model + i + 1] += cosf((float)t * div);
        }
    }
}

// ============================================
// Predict from Raw Samples
// ============================================

NNResult NNInference::predictFromSamples(const Sample* samples, int num_samples) {
    NNResult result = {};
    result.valid = false;
    result.is_moving = false;
    if (num_samples != NNConfig::INPUT_SAMPLES) return result;

    float* raw = (float*)malloc(NNConfig::INPUT_SIZE * sizeof(float));
    if (!raw) return result;

    for (int s = 0; s < num_samples; s++) {
        raw[0 * NNConfig::INPUT_SAMPLES + s] = (samples[s].ecg_adc - 2048.0f) / 2048.0f;
        raw[1 * NNConfig::INPUT_SAMPLES + s] = (samples[s].ppg_red - 131072.0f) / 131072.0f;
        raw[2 * NNConfig::INPUT_SAMPLES + s] = samples[s].ax_g;
        raw[3 * NNConfig::INPUT_SAMPLES + s] = samples[s].ay_g;
        raw[4 * NNConfig::INPUT_SAMPLES + s] = samples[s].az_g;
    }

    result = predict(raw);

    // Motion from IMU
    float mean_a = 0, mean_a2 = 0, mean_g2 = 0;
    int n = 0;
    for (int s = 0; s < num_samples; s++) {
        if (!(samples[s].flags & FLAG_IMU_VALID)) continue;
        float am = sqrtf(samples[s].ax_g*samples[s].ax_g + samples[s].ay_g*samples[s].ay_g + samples[s].az_g*samples[s].az_g);
        float gm = sqrtf(samples[s].gx_dps*samples[s].gx_dps + samples[s].gy_dps*samples[s].gy_dps + samples[s].gz_dps*samples[s].gz_dps);
        mean_a += am; mean_a2 += am*am; mean_g2 += gm*gm; n++;
    }
    if (n > 0) {
        mean_a /= n; mean_a2 /= n; mean_g2 /= n;
        result.accel_std_g = sqrtf(fmaxf(0, mean_a2 - mean_a*mean_a));
        result.gyro_rms_dps = sqrtf(mean_g2);
    }
    result.is_moving = (result.accel_std_g > 0.02f) || (result.gyro_rms_dps > 8.0f);
    result.alert_type = determineAlert(result);
    result.alert_triggered = (result.alert_type != ALERT_NONE);

    free(raw);
    return result;
}

// ============================================
// Layer Implementations
// ============================================

void NNInference::conv1d_relu(const float* input, float* output,
    int in_ch, int out_ch, int in_len, int kernel, const float* w, const float* b) {
    int pad = kernel / 2;
    for (int oc = 0; oc < out_ch; oc++) {
        for (int i = 0; i < in_len; i++) {
            float sum = b[oc];
            for (int ic = 0; ic < in_ch; ic++)
                for (int k = 0; k < kernel; k++) {
                    int idx = i - pad + k;
                    if (idx >= 0 && idx < in_len)
                        sum += input[ic*in_len + idx] * w[oc*in_ch*kernel + ic*kernel + k];
                }
            output[oc*in_len + i] = sum > 0 ? sum : 0;
        }
    }
}

void NNInference::maxpool1d(const float* input, float* output, int ch, int in_len, int pool) {
    int out_len = in_len / pool;
    for (int c = 0; c < ch; c++)
        for (int i = 0; i < out_len; i++) {
            float mx = input[c*in_len + i*pool];
            for (int p = 1; p < pool; p++) {
                float v = input[c*in_len + i*pool + p];
                if (v > mx) mx = v;
            }
            output[c*out_len + i] = mx;
        }
}

void NNInference::global_avg_pool(const float* input, float* output, int ch, int len) {
    for (int c = 0; c < ch; c++) {
        float sum = 0;
        for (int i = 0; i < len; i++) sum += input[c*len + i];
        output[c] = sum / len;
    }
}

void NNInference::linear(const float* input, float* output,
    int in_f, int out_f, const float* w, const float* b) {
    for (int o = 0; o < out_f; o++) {
        float sum = b[o];
        for (int i = 0; i < in_f; i++) sum += input[i] * w[o*in_f + i];
        output[o] = sum;
    }
}

void NNInference::linear_seq(const float* seq_in, float* seq_out, int seq_len,
    int in_f, int out_f, const float* w, const float* b) {
    for (int t = 0; t < seq_len; t++) {
        for (int o = 0; o < out_f; o++) {
            float sum = b[o];
            for (int i = 0; i < in_f; i++)
                sum += seq_in[t*in_f + i] * w[o*in_f + i];
            seq_out[t*out_f + o] = sum;
        }
    }
}

void NNInference::layernorm(float* data, int size, const float* w, const float* b) {
    float mean = 0;
    for (int i = 0; i < size; i++) mean += data[i];
    mean /= size;
    float var = 0;
    for (int i = 0; i < size; i++) { float d = data[i]-mean; var += d*d; }
    var /= size;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < size; i++) data[i] = (data[i]-mean)*inv*w[i] + b[i];
}

void NNInference::layernorm_seq(float* seq, int seq_len, int dim, const float* w, const float* b) {
    for (int t = 0; t < seq_len; t++)
        layernorm(seq + t*dim, dim, w, b);
}

void NNInference::softmax(float* data, int size) {
    float mx = data[0];
    for (int i = 1; i < size; i++) if (data[i] > mx) mx = data[i];
    float sum = 0;
    for (int i = 0; i < size; i++) { data[i] = expf(data[i]-mx); sum += data[i]; }
    for (int i = 0; i < size; i++) data[i] /= sum;
}

void NNInference::relu(float* data, int size) {
    for (int i = 0; i < size; i++) if (data[i] < 0) data[i] = 0;
}

void NNInference::gelu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)));
    }
}

void NNInference::gelu_seq(float* seq, int seq_len, int dim) {
    gelu(seq, seq_len * dim);
}

void NNInference::sigmoid(float* data, int size) {
    for (int i = 0; i < size; i++) data[i] = 1.0f / (1.0f + expf(-data[i]));
}

uint8_t NNInference::argmax(const float* data, int size) {
    uint8_t idx = 0; float mx = data[0];
    for (int i = 1; i < size; i++) if (data[i] > mx) { mx = data[i]; idx = i; }
    return idx;
}

void NNInference::normalizeInput(const float* raw, float* normalized) {
    for (int c = 0; c < NNConfig::INPUT_CHANNELS; c++) {
        float mean = 0;
        for (int i = 0; i < NNConfig::INPUT_SAMPLES; i++)
            mean += raw[c*NNConfig::INPUT_SAMPLES + i];
        mean /= NNConfig::INPUT_SAMPLES;
        float var = 0;
        for (int i = 0; i < NNConfig::INPUT_SAMPLES; i++) {
            float d = raw[c*NNConfig::INPUT_SAMPLES + i] - mean;
            var += d*d;
        }
        float std = sqrtf(var / NNConfig::INPUT_SAMPLES + 1e-8f);
        for (int i = 0; i < NNConfig::INPUT_SAMPLES; i++)
            normalized[c*NNConfig::INPUT_SAMPLES + i] =
                (raw[c*NNConfig::INPUT_SAMPLES + i] - mean) / std;
    }
}

uint8_t NNInference::determineAlert(const NNResult& result) {
    float arr = result.arrhythmia_probs[1];
    float str = result.stress_probs[1];
    bool moving = result.is_moving || (result.activity_class >= 1);
    if (arr > NNConfig::ARRHYTHMIA_THRESHOLD && moving) return ALERT_CRITICAL;
    if (arr > NNConfig::ARRHYTHMIA_THRESHOLD) return ALERT_ARRHYTHMIA;
    if (str > NNConfig::STRESS_THRESHOLD && !moving) return ALERT_STRESS;
    return ALERT_NONE;
}

// ============================================
// Output Calibration (Domain-Shift Correction)
// ============================================

void NNInference::startCalibration(int num_windows) {
    _cal_remaining = num_windows;
    _cal_total = num_windows;
    _calibrated = false;
    memset(_cal_stress_logit_sum, 0, sizeof(_cal_stress_logit_sum));
    memset(_cal_arr_logit_sum, 0, sizeof(_cal_arr_logit_sum));
    memset(_stress_bias, 0, sizeof(_stress_bias));
    memset(_arr_bias, 0, sizeof(_arr_bias));
    CONSOLE.printf("CAL: Starting calibration (%d windows). Keep still, relax.\n", num_windows);
}

void NNInference::feedCalibrationWindow(const NNResult& result) {
    if (_cal_remaining <= 0) return;

    for (int i = 0; i < NNConfig::STRESS_CLASSES; i++)
        _cal_stress_logit_sum[i] += result.stress_logits[i];
    for (int i = 0; i < NNConfig::ARRHYTHMIA_CLASSES; i++)
        _cal_arr_logit_sum[i] += result.arrhythmia_logits[i];

    _cal_remaining--;
    CONSOLE.printf("CAL: Window %d/%d collected\n",
                   _cal_total - _cal_remaining, _cal_total);

    if (_cal_remaining == 0) {
        // Compute mean logits during baseline
        float mean_s[2], mean_a[2];
        for (int i = 0; i < NNConfig::STRESS_CLASSES; i++)
            mean_s[i] = _cal_stress_logit_sum[i] / _cal_total;
        for (int i = 0; i < NNConfig::ARRHYTHMIA_CLASSES; i++)
            mean_a[i] = _cal_arr_logit_sum[i] / _cal_total;

        // Bias correction: shift so calibration windows would predict
        // class 0 (Baseline/Normal) with ~80% confidence.
        // Target logits: [+0.7, -0.7] → softmax ≈ [0.80, 0.20]
        // bias[i] = mean[i] - target[i], so corrected = raw - bias = target
        const float MARGIN = 0.7f;
        _stress_bias[0] = mean_s[0] - MARGIN;   // subtract less from class 0
        _stress_bias[1] = mean_s[1] + MARGIN;   // subtract more from class 1
        _arr_bias[0] = mean_a[0] - MARGIN;
        _arr_bias[1] = mean_a[1] + MARGIN;

        _calibrated = true;
        CONSOLE.printf("CAL: Complete! Mean stress=[%.3f,%.3f] arr=[%.3f,%.3f]\n",
                       mean_s[0], mean_s[1], mean_a[0], mean_a[1]);
        CONSOLE.printf("CAL: Bias stress=[%.3f,%.3f] arr=[%.3f,%.3f]\n",
                       _stress_bias[0], _stress_bias[1],
                       _arr_bias[0], _arr_bias[1]);
        CONSOLE.println("CAL: Baseline will predict class 0 with ~80% confidence.");
    }
}

void NNInference::resetCalibration() {
    _calibrated = false;
    _cal_remaining = 0;
    memset(_stress_bias, 0, sizeof(_stress_bias));
    memset(_arr_bias, 0, sizeof(_arr_bias));
    CONSOLE.println("CAL: Calibration reset.");
}
