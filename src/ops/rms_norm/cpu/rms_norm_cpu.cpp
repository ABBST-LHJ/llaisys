#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>

namespace llaisys {
// 辅助：计算单 RMS 归一化行（适配框架原生低精度类型，移除 CustomFloat16/CustomBFloat16）
template <typename T>
void rms_norm_row(T *out_row, const T *in_row, const T *weight, size_t hidden_dim, float eps) {
    // 1. 计算平方和（统一转为 float 计算，避免低精度溢出）
    float sum_sq = 0.0f;
    for (size_t j = 0; j < hidden_dim; ++j) {
        float val = llaisys::utils::cast<float>(in_row[j]);
        sum_sq += val * val;
    }

    // 2. 计算 RMS（加 eps 防止除零）
    float rms = std::sqrt(sum_sq / hidden_dim + eps);

    // 3. 归一化并乘以权重（结果转回目标类型）
    for (size_t j = 0; j < hidden_dim; ++j) {
        float in_val = llaisys::utils::cast<float>(in_row[j]);
        float weight_val = llaisys::utils::cast<float>(weight[j]);
        out_row[j] = llaisys::utils::cast<T>( (in_val / rms) * weight_val );
    }
}
} // namespace llaisys

// 模板核心函数（通用模板，适配所有框架原生类型）
template <typename T>
void rms_norm_(std::byte *out, const std::byte *in, const std::byte *weight,
               size_t batch_size, size_t hidden_dim, float eps) {
    const T *in_ptr = reinterpret_cast<const T*>(in);
    const T *weight_ptr = reinterpret_cast<const T*>(weight);
    T *out_ptr = reinterpret_cast<T*>(out);

    // 遍历每一批，执行 RMS 归一化
    for (size_t i = 0; i < batch_size; ++i) {
        const T *in_row = in_ptr + i * hidden_dim;
        T *out_row = out_ptr + i * hidden_dim;
        llaisys::rms_norm_row<T>(out_row, in_row, weight_ptr, hidden_dim, eps);
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t data_type, size_t batch_size, size_t hidden_dim, float eps) {
    // 1. 空值保护
    if (batch_size == 0 || hidden_dim == 0) {
        throw std::invalid_argument("RMS Norm: batch_size/hidden_dim cannot be zero.");
    }

    // 2. 数据类型分发（使用框架原生类型名 fp16_t/bf16_t，移除 Custom 前缀）
    switch (data_type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(out, in, weight, batch_size, hidden_dim, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(out, in, weight, batch_size, hidden_dim, eps); // 框架原生 F16 类型
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(out, in, weight, batch_size, hidden_dim, eps); // 框架原生 BF16 类型
    default:
        std::string err_msg = "RMS Norm: unsupported data type (" + std::to_string(static_cast<int>(data_type)) + ").";
        throw std::runtime_error(err_msg);
    }
}
} // namespace llaisys::ops::cpu