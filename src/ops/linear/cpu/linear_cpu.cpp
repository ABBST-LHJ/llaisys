#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llaisys {
// 辅助函数：单个元素矩阵乘法计算（适配自定义低精度类型，移除直接类型转换）
template <typename T>
T matmul_element(const T *in_row, const T *weight, size_t in_features, size_t out_feature_idx) {
    // 关键点1：低精度类型不直接初始化，先通过 float 计算，最后用 utils::cast 转换
    if constexpr (std::is_same_v<T, llaisys::CustomFloat16> || std::is_same_v<T, llaisys::CustomBFloat16>) {
        float sum_float = 0.0f;
        for (size_t k = 0; k < in_features; ++k) {
            // weight 转置映射：weight[out_feature_idx][k] → 对应 weight 原始索引 [out_feature_idx * in_features + k]
            float in_val = llaisys::utils::cast<float>(in_row[k]);
            float weight_val = llaisys::utils::cast<float>(weight[out_feature_idx * in_features + k]);
            sum_float += in_val * weight_val;
        }
        // 用框架工具函数转换为自定义低精度类型，避免直接构造
        return llaisys::utils::cast<T>(sum_float);
    } else {
        // F32 直接计算，正常初始化
        float sum = 0.0f;
        for (size_t k = 0; k < in_features; ++k) {
            sum += llaisys::utils::cast<float>(in_row[k]) * llaisys::utils::cast<float>(weight[out_feature_idx * in_features + k]);
        }
        return llaisys::utils::cast<T>(sum);
    }
}

// 辅助函数：偏置加法（适配自定义低精度类型，移除直接赋值）
template <typename T>
void add_bias(T *out_row, const T *bias, size_t out_features) {
    if constexpr (std::is_same_v<T, llaisys::CustomFloat16> || std::is_same_v<T, llaisys::CustomBFloat16>) {
        // 低精度类型：先转为 float 计算，再转换回原类型
        for (size_t j = 0; j < out_features; ++j) {
            float out_val = llaisys::utils::cast<float>(out_row[j]);
            float bias_val = llaisys::utils::cast<float>(bias[j]);
            // 用框架工具函数赋值，避免直接重载赋值运算符
            out_row[j] = llaisys::utils::cast<T>(out_val + bias_val);
        }
    } else {
        // F32 直接计算
        for (size_t j = 0; j < out_features; ++j) {
            float out_val = llaisys::utils::cast<float>(out_row[j]);
            float bias_val = llaisys::utils::cast<float>(bias[j]);
            out_row[j] = llaisys::utils::cast<T>(out_val + bias_val);
        }
    }
}

// 模板函数：Linear 核心计算逻辑（适配自定义低精度类型，移除 static_cast<T>(0)）
template <typename T>
void linear_(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
             size_t N, size_t in_features, size_t out_features) {
    // 1. 转换指针类型
    const T *in_ptr = reinterpret_cast<const T*>(in);
    const T *weight_ptr = reinterpret_cast<const T*>(weight);
    T *out_ptr = reinterpret_cast<T*>(out);
    const T *bias_ptr = bias ? reinterpret_cast<const T*>(bias) : nullptr;

    // 2. 矩阵乘法（in [N, in_features] * weight.T [in_features, out_features]）
    for (size_t i = 0; i < N; ++i) {
        // 2.1 获取 in 的第 i 行起始地址
        const T *in_row = in_ptr + i * in_features;
        // 2.2 计算 out 的第 i 行每个元素（关键点2：移除直接初始化，依赖 matmul_element 返回合法类型）
        for (size_t j = 0; j < out_features; ++j) {
            // 直接接收 matmul_element 的返回值，无需手动初始化 T 类型变量
            out_ptr[i * out_features + j] = matmul_element<T>(in_row, weight_ptr, in_features, j);
        }
        // 2.3 可选偏置加法（逐行添加）
        if (bias_ptr != nullptr) {
            add_bias<T>(out_ptr + i * out_features, bias_ptr, out_features);
        }
    }
}
} // namespace llaisys

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t data_type, size_t N, size_t in_features, size_t out_features) {
    // 1. 空值保护
    if (N == 0 || in_features == 0 || out_features == 0) {
        throw std::invalid_argument("Linear: N/in_features/out_features cannot be zero.");
    }

    // 2. 数据类型分发（适配框架自定义低精度类型名，替换为实际的 CustomFloat16/CustomBFloat16）
    switch (data_type) {
    case LLAISYS_DTYPE_F32:
        return llaisys::linear_<float>(out, in, weight, bias, N, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return llaisys::linear_<llaisys::CustomFloat16>(out, in, weight, bias, N, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return llaisys::linear_<llaisys::CustomBFloat16>(out, in, weight, bias, N, in_features, out_features);
    default:
        std::string err_msg = "Linear: unsupported data type (" + std::to_string(static_cast<int>(data_type)) + ").";
        throw std::runtime_error(err_msg);
    }
}
} // namespace llaisys::ops::cpu