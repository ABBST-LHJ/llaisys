#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>

namespace llaisys {
// 辅助：对单个向量执行 RoPE 旋转
template <typename T>
void rotate_vector(T *out_vec, const T *in_vec, size_t pos, size_t d, float theta) {
    size_t half_d = d / 2;
    for (size_t j = 0; j < half_d; ++j) {
        // 计算旋转角度
        float phi = static_cast<float>(pos) / std::pow(theta, 2.0f * j / d);
        float cos_phi = std::cos(phi);
        float sin_phi = std::sin(phi);

        // 取出输入的 a_j 和 b_j
        float a = llaisys::utils::cast<float>(in_vec[j]);
        float b = llaisys::utils::cast<float>(in_vec[j + half_d]);

        // 计算旋转后的 a'_j 和 b'_j
        float a_rot = a * cos_phi - b * sin_phi;
        float b_rot = b * cos_phi + a * sin_phi;

        // 写入输出
        out_vec[j] = llaisys::utils::cast<T>(a_rot);
        out_vec[j + half_d] = llaisys::utils::cast<T>(b_rot);
    }
}
} // namespace llaisys

// 模板核心函数
template <typename T>
void rope_(std::byte *out, const std::byte *in, const std::byte *pos_ids,
           size_t seq_len, size_t n_head, size_t d, float theta) {
    const T *in_ptr = reinterpret_cast<const T*>(in);
    T *out_ptr = reinterpret_cast<T*>(out);
    const int64_t *pos_ids_ptr = reinterpret_cast<const int64_t*>(pos_ids);

    // 遍历每个序列位置和注意力头
    for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
        int64_t pos = pos_ids_ptr[seq_idx];
        for (size_t head_idx = 0; head_idx < n_head; ++head_idx) {
            // 计算当前向量的起始索引
            size_t vec_start = seq_idx * n_head * d + head_idx * d;
            const T *in_vec = in_ptr + vec_start;
            T *out_vec = out_ptr + vec_start;
            // 执行旋转
            llaisys::rotate_vector<T>(out_vec, in_vec, pos, d, theta);
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t data_type, size_t seq_len, size_t n_head, size_t d, float theta) {
    // 空值保护
    if (seq_len == 0 || n_head == 0 || d == 0 || d % 2 != 0) {
        throw std::invalid_argument("RoPE: seq_len/n_head/d cannot be zero, and d must be even.");
    }

    // 数据类型分发
    switch (data_type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(out, in, pos_ids, seq_len, n_head, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(out, in, pos_ids, seq_len, n_head, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(out, in, pos_ids, seq_len, n_head, d, theta);
    default:
        std::string err_msg = "RoPE: unsupported data type (" + std::to_string(static_cast<int>(data_type)) + ").";
        throw std::runtime_error(err_msg);
    }
}
} // namespace llaisys::ops::cpu