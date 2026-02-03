#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

namespace llaisys {

// 辅助：将 K/V head 重复以匹配 Q head 数量 (GQA)
// 输入: [total_len, nkvhead, d] -> 输出: [total_len, nhead, d]
template <typename T>
void repeat_kv_heads(T* expanded_kv, const T* kv,
                     size_t total_len, size_t nkvhead, size_t d, size_t nhead) {
    size_t heads_per_group = nhead / nkvhead;
    
    for (size_t t = 0; t < total_len; ++t) {
        for (size_t kv_h = 0; kv_h < nkvhead; ++kv_h) {
            for (size_t r = 0; r < heads_per_group; ++r) {
                size_t target_h = kv_h * heads_per_group + r;
                // 复制当前 kv head 到 target_h
                for (size_t k = 0; k < d; ++k) {
                    expanded_kv[t * nhead * d + target_h * d + k] = 
                        kv[t * nkvhead * d + kv_h * d + k];
                }
            }
        }
    }
}

// 辅助：计算 QK^T 矩阵乘法
// Q: [seqlen, nhead, d], K: [total_len, nhead, d] (已扩展)
// 输出: [seqlen, nhead, total_len]
template <typename T>
void compute_qk_t(float *qk_t, const T *q, const T *k_expanded,
                  size_t seqlen, size_t nhead, size_t d, size_t total_len) {
    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t h = 0; h < nhead; ++h) {
            for (size_t j = 0; j < total_len; ++j) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx < d; ++k_idx) {
                    float q_val = llaisys::utils::cast<float>(q[i * nhead * d + h * d + k_idx]);
                    float k_val = llaisys::utils::cast<float>(k_expanded[j * nhead * d + h * d + k_idx]);
                    sum += q_val * k_val;
                }
                qk_t[i * nhead * total_len + h * total_len + j] = sum;
            }
        }
    }
}

// 辅助：应用因果 Softmax
// 输入: [seqlen, nhead, total_len]
template <typename T>
void apply_causal_softmax(float *attn_weights, size_t seqlen, size_t nhead, size_t total_len) {
    // 注意：total_len 是 kv 长度，seqlen 是 query 长度
    // 对于自回归，假设 query 是最后 seqlen 个位置，kv 是前 total_len 个位置
    // 因果掩码：query 位置 i 只能 attend 到 kv 位置 j，其中 j <= (total_len - seqlen + i)
    
    size_t kv_offset = total_len - seqlen;  // query[0] 对应 kv[kv_offset]
    
    for (size_t i = 0; i < seqlen; ++i) {
        size_t max_j = kv_offset + i;  // 因果边界
        
        for (size_t h = 0; h < nhead; ++h) {
            // 找到最大值（仅考虑有效位置）
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j <= max_j; ++j) {
                max_val = std::max(max_val, attn_weights[i * nhead * total_len + h * total_len + j]);
            }

            // 计算 exp 并求和
            float sum_exp = 0.0f;
            for (size_t j = 0; j <= max_j; ++j) {
                float val = attn_weights[i * nhead * total_len + h * total_len + j];
                sum_exp += std::exp(val - max_val);
            }

            // 归一化并应用掩码
            for (size_t j = 0; j < total_len; ++j) {
                if (j > max_j) {
                    attn_weights[i * nhead * total_len + h * total_len + j] = 0.0f;
                } else {
                    float val = attn_weights[i * nhead * total_len + h * total_len + j];
                    attn_weights[i * nhead * total_len + h * total_len + j] = 
                        std::exp(val - max_val) / sum_exp;
                }
            }
        }
    }
}

// 辅助：计算注意力权重 × V
// attn_weights: [seqlen, nhead, total_len], V: [total_len, nhead, dv] (已扩展)
// 输出: [seqlen, nhead, dv]
template <typename T>
void compute_attn_v(T *attn_val, const float *attn_weights, const T *v_expanded,
                    size_t seqlen, size_t nhead, size_t total_len, size_t dv) {
    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t h = 0; h < nhead; ++h) {
            for (size_t v_idx = 0; v_idx < dv; ++v_idx) {
                float sum = 0.0f;
                for (size_t j = 0; j < total_len; ++j) {
                    float weight = attn_weights[i * nhead * total_len + h * total_len + j];
                    if (weight != 0.0f) {  // 跳过被掩码的位置
                        float v_val = llaisys::utils::cast<float>(v_expanded[j * nhead * dv + h * dv + v_idx]);
                        sum += weight * v_val;
                    }
                }
                attn_val[i * nhead * dv + h * dv + v_idx] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}

} // namespace llaisys

// 模板核心函数
template <typename T>
void self_attention_(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                     size_t seqlen, size_t nhead, size_t d,
                     size_t total_len, size_t nkvhead, size_t dv, float scale) {
    const T *q_ptr = reinterpret_cast<const T*>(q);
    const T *k_ptr = reinterpret_cast<const T*>(k);
    const T *v_ptr = reinterpret_cast<const T*>(v);
    T *attn_val_ptr = reinterpret_cast<T*>(attn_val);

    // 分配临时空间
    size_t expanded_kv_size = total_len * nhead * d * sizeof(T);
    size_t expanded_v_size = total_len * nhead * dv * sizeof(T);
    size_t qk_t_size = seqlen * nhead * total_len * sizeof(float);
    
    T *k_expanded = reinterpret_cast<T*>(malloc(expanded_kv_size));
    T *v_expanded = reinterpret_cast<T*>(malloc(expanded_v_size));
    float *qk_t = reinterpret_cast<float*>(malloc(qk_t_size));
    
    if (!k_expanded || !v_expanded || !qk_t) {
        free(k_expanded);
        free(v_expanded);
        free(qk_t);
        throw std::runtime_error("Failed to allocate memory for self-attention.");
    }

    // 1. 扩展 K 和 V head (GQA: repeat_interleave)
    llaisys::repeat_kv_heads(k_expanded, k_ptr, total_len, nkvhead, d, nhead);
    llaisys::repeat_kv_heads(v_expanded, v_ptr, total_len, nkvhead, dv, nhead);

    // 2. 计算 QK^T
    llaisys::compute_qk_t<T>(qk_t, q_ptr, k_expanded, seqlen, nhead, d, total_len);
    
    // 3. 应用缩放因子
    for (size_t idx = 0; idx < seqlen * nhead * total_len; ++idx) {
        qk_t[idx] *= scale;
    }

    // 4. 应用因果 Softmax
    llaisys::apply_causal_softmax<T>(qk_t, seqlen, nhead, total_len);

    // 5. 计算注意力权重 × V
    llaisys::compute_attn_v<T>(attn_val_ptr, qk_t, v_expanded, seqlen, nhead, total_len, dv);

    // 释放临时空间
    free(k_expanded);
    free(v_expanded);
    free(qk_t);
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t data_type, size_t seqlen, size_t nhead, size_t d,
                    size_t total_len, size_t nkvhead, size_t dv, float scale) {
    // 空值保护
    if (seqlen == 0 || nhead == 0 || d == 0 || total_len == 0 || nkvhead == 0 || dv == 0) {
        throw std::invalid_argument("Self-Attention: all dimensions must be non-zero.");
    }
    if (nhead % nkvhead != 0) {
        throw std::invalid_argument("Self-Attention: nhead must be a multiple of nkvhead.");
    }
    if (total_len < seqlen) {
        throw std::invalid_argument("Self-Attention: total_len must be >= seqlen.");
    }

    // 数据类型分发
    switch (data_type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(attn_val, q, k, v, seqlen, nhead, d, total_len, nkvhead, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(attn_val, q, k, v, seqlen, nhead, d, total_len, nkvhead, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(attn_val, q, k, v, seqlen, nhead, d, total_len, nkvhead, dv, scale);
    default:
        std::string err_msg = "Self-Attention: unsupported data type (" + 
                             std::to_string(static_cast<int>(data_type)) + ").";
        throw std::runtime_error(err_msg);
    }
}
} // namespace llaisys::ops::cpu