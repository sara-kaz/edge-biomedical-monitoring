/**
 * Model Weights for ESP32-S3 — Full V5 Model (CNNTransformerV3)
 *
 * Architecture: CNN + SE-Attention + 2-Layer Transformer
 *   Input: [5 ch x 1000 samples]
 *   Conv1(5→32,k7)/BN/ReLU/Pool → Conv2(32→64,k5)/BN/ReLU/Pool →
 *   Conv3(64→64,k3)/BN/ReLU/Pool → SE(64→16→64) →
 *   Per-timestep Projection(64→64) + LN + PosEnc →
 *   Transformer(2 layers, 4 heads, d_model=64, d_ff=128) →
 *   MeanPool → TaskHeads(3-layer each)
 */

#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

// ============================================
// CNN Layer Dimensions
// ============================================

constexpr int CONV1_IN_CH = 5;
constexpr int CONV1_OUT_CH_W = 32;
constexpr int CONV1_KERNEL_W = 7;
constexpr int CONV1_WEIGHT_SIZE = (CONV1_OUT_CH_W * CONV1_IN_CH * CONV1_KERNEL_W);
constexpr int CONV1_BIAS_SIZE = CONV1_OUT_CH_W;

constexpr int CONV2_IN_CH = 32;
constexpr int CONV2_OUT_CH_W = 64;
constexpr int CONV2_KERNEL_W = 5;
constexpr int CONV2_WEIGHT_SIZE = (CONV2_OUT_CH_W * CONV2_IN_CH * CONV2_KERNEL_W);
constexpr int CONV2_BIAS_SIZE = CONV2_OUT_CH_W;

constexpr int CONV3_IN_CH = 64;
constexpr int CONV3_OUT_CH_W = 64;
constexpr int CONV3_KERNEL_W = 3;
constexpr int CONV3_WEIGHT_SIZE = (CONV3_OUT_CH_W * CONV3_IN_CH * CONV3_KERNEL_W);
constexpr int CONV3_BIAS_SIZE = CONV3_OUT_CH_W;

// SE-Attention
constexpr int SE_FC1_WEIGHT_SIZE = (16 * 64);
constexpr int SE_FC1_BIAS_SIZE = 16;
constexpr int SE_FC2_WEIGHT_SIZE = (64 * 16);
constexpr int SE_FC2_BIAS_SIZE = 64;

// Projection + LayerNorm
constexpr int PROJ_WEIGHT_SIZE = (64 * 64);
constexpr int PROJ_BIAS_SIZE = 64;

// ============================================
// Transformer Dimensions (per layer)
// ============================================

constexpr int TF_D_MODEL = 64;
constexpr int TF_NHEAD = 4;
constexpr int TF_D_HEAD = TF_D_MODEL / TF_NHEAD;  // 16
constexpr int TF_D_FF = 128;
constexpr int TF_NUM_LAYERS = 2;

// in_proj: [3*d_model, d_model] = [192, 64]
constexpr int TF_IN_PROJ_WEIGHT_SIZE = (3 * TF_D_MODEL * TF_D_MODEL);
constexpr int TF_IN_PROJ_BIAS_SIZE = (3 * TF_D_MODEL);
// out_proj: [d_model, d_model] = [64, 64]
constexpr int TF_OUT_PROJ_WEIGHT_SIZE = (TF_D_MODEL * TF_D_MODEL);
constexpr int TF_OUT_PROJ_BIAS_SIZE = TF_D_MODEL;
// FFN: linear1 [d_ff, d_model], linear2 [d_model, d_ff]
constexpr int TF_FFN1_WEIGHT_SIZE = (TF_D_FF * TF_D_MODEL);
constexpr int TF_FFN1_BIAS_SIZE = TF_D_FF;
constexpr int TF_FFN2_WEIGHT_SIZE = (TF_D_MODEL * TF_D_FF);
constexpr int TF_FFN2_BIAS_SIZE = TF_D_MODEL;
// LayerNorms: [d_model] each
constexpr int TF_NORM_SIZE = TF_D_MODEL;

// ============================================
// Task Head Dimensions
// ============================================

constexpr int ACT_FC1_IN_DIM = 64;
constexpr int ACT_FC1_OUT_DIM = 64;
constexpr int ACT_LN_DIM = 64;
constexpr int ACT_FC2_IN_DIM = 64;
constexpr int ACT_FC2_OUT_DIM = 32;
constexpr int ACT_FC3_IN_DIM = 32;
constexpr int ACT_FC3_OUT_DIM = 4;
constexpr int ACT_FC1_WEIGHT_SIZE = (ACT_FC1_OUT_DIM * ACT_FC1_IN_DIM);
constexpr int ACT_FC1_BIAS_SIZE = ACT_FC1_OUT_DIM;
constexpr int ACT_FC2_WEIGHT_SIZE = (ACT_FC2_OUT_DIM * ACT_FC2_IN_DIM);
constexpr int ACT_FC2_BIAS_SIZE = ACT_FC2_OUT_DIM;
constexpr int ACT_FC3_WEIGHT_SIZE = (ACT_FC3_OUT_DIM * ACT_FC3_IN_DIM);
constexpr int ACT_FC3_BIAS_SIZE = ACT_FC3_OUT_DIM;

constexpr int STRESS_FC1_IN_DIM = 64;
constexpr int STRESS_FC1_OUT_DIM = 48;
constexpr int STRESS_LN_DIM = 48;
constexpr int STRESS_FC2_IN_DIM = 48;
constexpr int STRESS_FC2_OUT_DIM = 24;
constexpr int STRESS_FC3_IN_DIM = 24;
constexpr int STRESS_FC3_OUT_DIM = 2;
constexpr int STRESS_FC1_WEIGHT_SIZE = (STRESS_FC1_OUT_DIM * STRESS_FC1_IN_DIM);
constexpr int STRESS_FC1_BIAS_SIZE = STRESS_FC1_OUT_DIM;
constexpr int STRESS_FC2_WEIGHT_SIZE = (STRESS_FC2_OUT_DIM * STRESS_FC2_IN_DIM);
constexpr int STRESS_FC2_BIAS_SIZE = STRESS_FC2_OUT_DIM;
constexpr int STRESS_FC3_WEIGHT_SIZE = (STRESS_FC3_OUT_DIM * STRESS_FC3_IN_DIM);
constexpr int STRESS_FC3_BIAS_SIZE = STRESS_FC3_OUT_DIM;

constexpr int ARR_FC1_IN_DIM = 64;
constexpr int ARR_FC1_OUT_DIM = 48;
constexpr int ARR_LN_DIM = 48;
constexpr int ARR_FC2_IN_DIM = 48;
constexpr int ARR_FC2_OUT_DIM = 24;
constexpr int ARR_FC3_IN_DIM = 24;
constexpr int ARR_FC3_OUT_DIM = 2;
constexpr int ARR_FC1_WEIGHT_SIZE = (ARR_FC1_OUT_DIM * ARR_FC1_IN_DIM);
constexpr int ARR_FC1_BIAS_SIZE = ARR_FC1_OUT_DIM;
constexpr int ARR_FC2_WEIGHT_SIZE = (ARR_FC2_OUT_DIM * ARR_FC2_IN_DIM);
constexpr int ARR_FC2_BIAS_SIZE = ARR_FC2_OUT_DIM;
constexpr int ARR_FC3_WEIGHT_SIZE = (ARR_FC3_OUT_DIM * ARR_FC3_IN_DIM);
constexpr int ARR_FC3_BIAS_SIZE = ARR_FC3_OUT_DIM;

// Total params (approximate, for display)
constexpr int MODEL_TOTAL_PARAMS = 112552;

// ============================================
// Weight Arrays
// ============================================

// CNN
static const float conv1_weight[CONV1_WEIGHT_SIZE] = {
#include "weights/conv1_weight.inc"
};
static const float conv1_bias[CONV1_BIAS_SIZE] = {
#include "weights/conv1_bias.inc"
};
static const float conv2_weight[CONV2_WEIGHT_SIZE] = {
#include "weights/conv2_weight.inc"
};
static const float conv2_bias[CONV2_BIAS_SIZE] = {
#include "weights/conv2_bias.inc"
};
static const float conv3_weight[CONV3_WEIGHT_SIZE] = {
#include "weights/conv3_weight.inc"
};
static const float conv3_bias[CONV3_BIAS_SIZE] = {
#include "weights/conv3_bias.inc"
};

// SE
static const float se_fc1_weight[SE_FC1_WEIGHT_SIZE] = {
#include "weights/se_fc1_weight.inc"
};
static const float se_fc1_bias[SE_FC1_BIAS_SIZE] = {
#include "weights/se_fc1_bias.inc"
};
static const float se_fc2_weight[SE_FC2_WEIGHT_SIZE] = {
#include "weights/se_fc2_weight.inc"
};
static const float se_fc2_bias[SE_FC2_BIAS_SIZE] = {
#include "weights/se_fc2_bias.inc"
};

// Projection + LN
static const float projection_weight[PROJ_WEIGHT_SIZE] = {
#include "weights/projection_weight.inc"
};
static const float projection_bias[PROJ_BIAS_SIZE] = {
#include "weights/projection_bias.inc"
};
static const float proj_ln_weight[TF_D_MODEL] = {
#include "weights/proj_ln_weight.inc"
};
static const float proj_ln_bias[TF_D_MODEL] = {
#include "weights/proj_ln_bias.inc"
};

// Transformer Layer 0
static const float tf0_attn_in_proj_weight[TF_IN_PROJ_WEIGHT_SIZE] = {
#include "weights/tf0_attn_in_proj_weight.inc"
};
static const float tf0_attn_in_proj_bias[TF_IN_PROJ_BIAS_SIZE] = {
#include "weights/tf0_attn_in_proj_bias.inc"
};
static const float tf0_attn_out_proj_weight[TF_OUT_PROJ_WEIGHT_SIZE] = {
#include "weights/tf0_attn_out_proj_weight.inc"
};
static const float tf0_attn_out_proj_bias[TF_OUT_PROJ_BIAS_SIZE] = {
#include "weights/tf0_attn_out_proj_bias.inc"
};
static const float tf0_norm1_weight[TF_NORM_SIZE] = {
#include "weights/tf0_norm1_weight.inc"
};
static const float tf0_norm1_bias[TF_NORM_SIZE] = {
#include "weights/tf0_norm1_bias.inc"
};
static const float tf0_ffn1_weight[TF_FFN1_WEIGHT_SIZE] = {
#include "weights/tf0_ffn1_weight.inc"
};
static const float tf0_ffn1_bias[TF_FFN1_BIAS_SIZE] = {
#include "weights/tf0_ffn1_bias.inc"
};
static const float tf0_ffn2_weight[TF_FFN2_WEIGHT_SIZE] = {
#include "weights/tf0_ffn2_weight.inc"
};
static const float tf0_ffn2_bias[TF_FFN2_BIAS_SIZE] = {
#include "weights/tf0_ffn2_bias.inc"
};
static const float tf0_norm2_weight[TF_NORM_SIZE] = {
#include "weights/tf0_norm2_weight.inc"
};
static const float tf0_norm2_bias[TF_NORM_SIZE] = {
#include "weights/tf0_norm2_bias.inc"
};

// Transformer Layer 1
static const float tf1_attn_in_proj_weight[TF_IN_PROJ_WEIGHT_SIZE] = {
#include "weights/tf1_attn_in_proj_weight.inc"
};
static const float tf1_attn_in_proj_bias[TF_IN_PROJ_BIAS_SIZE] = {
#include "weights/tf1_attn_in_proj_bias.inc"
};
static const float tf1_attn_out_proj_weight[TF_OUT_PROJ_WEIGHT_SIZE] = {
#include "weights/tf1_attn_out_proj_weight.inc"
};
static const float tf1_attn_out_proj_bias[TF_OUT_PROJ_BIAS_SIZE] = {
#include "weights/tf1_attn_out_proj_bias.inc"
};
static const float tf1_norm1_weight[TF_NORM_SIZE] = {
#include "weights/tf1_norm1_weight.inc"
};
static const float tf1_norm1_bias[TF_NORM_SIZE] = {
#include "weights/tf1_norm1_bias.inc"
};
static const float tf1_ffn1_weight[TF_FFN1_WEIGHT_SIZE] = {
#include "weights/tf1_ffn1_weight.inc"
};
static const float tf1_ffn1_bias[TF_FFN1_BIAS_SIZE] = {
#include "weights/tf1_ffn1_bias.inc"
};
static const float tf1_ffn2_weight[TF_FFN2_WEIGHT_SIZE] = {
#include "weights/tf1_ffn2_weight.inc"
};
static const float tf1_ffn2_bias[TF_FFN2_BIAS_SIZE] = {
#include "weights/tf1_ffn2_bias.inc"
};
static const float tf1_norm2_weight[TF_NORM_SIZE] = {
#include "weights/tf1_norm2_weight.inc"
};
static const float tf1_norm2_bias[TF_NORM_SIZE] = {
#include "weights/tf1_norm2_bias.inc"
};

// Task heads
static const float activity_head_fc1_weight[ACT_FC1_WEIGHT_SIZE] = {
#include "weights/activity_head_fc1_weight.inc"
};
static const float activity_head_fc1_bias[ACT_FC1_BIAS_SIZE] = {
#include "weights/activity_head_fc1_bias.inc"
};
static const float activity_head_ln_weight[ACT_LN_DIM] = {
#include "weights/activity_head_ln_weight.inc"
};
static const float activity_head_ln_bias[ACT_LN_DIM] = {
#include "weights/activity_head_ln_bias.inc"
};
static const float activity_head_fc2_weight[ACT_FC2_WEIGHT_SIZE] = {
#include "weights/activity_head_fc2_weight.inc"
};
static const float activity_head_fc2_bias[ACT_FC2_BIAS_SIZE] = {
#include "weights/activity_head_fc2_bias.inc"
};
static const float activity_head_fc3_weight[ACT_FC3_WEIGHT_SIZE] = {
#include "weights/activity_head_fc3_weight.inc"
};
static const float activity_head_fc3_bias[ACT_FC3_BIAS_SIZE] = {
#include "weights/activity_head_fc3_bias.inc"
};

static const float stress_head_fc1_weight[STRESS_FC1_WEIGHT_SIZE] = {
#include "weights/stress_head_fc1_weight.inc"
};
static const float stress_head_fc1_bias[STRESS_FC1_BIAS_SIZE] = {
#include "weights/stress_head_fc1_bias.inc"
};
static const float stress_head_ln_weight[STRESS_LN_DIM] = {
#include "weights/stress_head_ln_weight.inc"
};
static const float stress_head_ln_bias[STRESS_LN_DIM] = {
#include "weights/stress_head_ln_bias.inc"
};
static const float stress_head_fc2_weight[STRESS_FC2_WEIGHT_SIZE] = {
#include "weights/stress_head_fc2_weight.inc"
};
static const float stress_head_fc2_bias[STRESS_FC2_BIAS_SIZE] = {
#include "weights/stress_head_fc2_bias.inc"
};
static const float stress_head_fc3_weight[STRESS_FC3_WEIGHT_SIZE] = {
#include "weights/stress_head_fc3_weight.inc"
};
static const float stress_head_fc3_bias[STRESS_FC3_BIAS_SIZE] = {
#include "weights/stress_head_fc3_bias.inc"
};

static const float arrhythmia_head_fc1_weight[ARR_FC1_WEIGHT_SIZE] = {
#include "weights/arrhythmia_head_fc1_weight.inc"
};
static const float arrhythmia_head_fc1_bias[ARR_FC1_BIAS_SIZE] = {
#include "weights/arrhythmia_head_fc1_bias.inc"
};
static const float arrhythmia_head_ln_weight[ARR_LN_DIM] = {
#include "weights/arrhythmia_head_ln_weight.inc"
};
static const float arrhythmia_head_ln_bias[ARR_LN_DIM] = {
#include "weights/arrhythmia_head_ln_bias.inc"
};
static const float arrhythmia_head_fc2_weight[ARR_FC2_WEIGHT_SIZE] = {
#include "weights/arrhythmia_head_fc2_weight.inc"
};
static const float arrhythmia_head_fc2_bias[ARR_FC2_BIAS_SIZE] = {
#include "weights/arrhythmia_head_fc2_bias.inc"
};
static const float arrhythmia_head_fc3_weight[ARR_FC3_WEIGHT_SIZE] = {
#include "weights/arrhythmia_head_fc3_weight.inc"
};
static const float arrhythmia_head_fc3_bias[ARR_FC3_BIAS_SIZE] = {
#include "weights/arrhythmia_head_fc3_bias.inc"
};

#endif // MODEL_WEIGHTS_H
