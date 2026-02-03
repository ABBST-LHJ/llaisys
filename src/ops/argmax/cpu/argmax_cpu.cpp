#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <stdexcept>  // 引入标准异常，保底兜底

namespace llaisys {
// 辅助函数：判断a是否大于b（处理BF16/F16低精度类型，对齐add算子逻辑）
template <typename T>
bool is_greater(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // 低精度先转为float比较，避免精度误差和硬件不支持
        float a_float = llaisys::utils::cast<float>(a);
        float b_float = llaisys::utils::cast<float>(b);
        return a_float > b_float;
    } else {
        // F32直接比较，高效无损耗
        return a > b;
    }
}
} // namespace llaisys

// 模板函数：argmax核心计算逻辑（仅处理1D张量，符合作业要求）
template <typename T>
void argmax_(std::byte *max_idx, std::byte *max_val, const T *vals, size_t numel) {
    // 初始化：默认第一个元素为最大值，索引为0
    T current_max = vals[0];
    size_t current_idx = 0;

    // 遍历所有元素，寻找最大值和对应索引
    for (size_t i = 1; i < numel; ++i) {
        if (llaisys::is_greater(vals[i], current_max)) {
            current_max = vals[i];
            current_idx = i;
        }
    }

    // 写入结果到输出张量
    // 1. 写入最大值到max_val（转换为对应数据类型指针）
    T* max_val_ptr = reinterpret_cast<T*>(max_val);
    max_val_ptr[0] = current_max;

    // 2. 写入索引到max_idx（索引是size_t类型，和数值类型无关）
    size_t* max_idx_ptr = reinterpret_cast<size_t*>(max_idx);
    max_idx_ptr[0] = current_idx;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, 
            llaisysDataType_t val_type, size_t numel) {
    // 空张量保护：避免越界访问，抛出清晰异常（兼容框架宏和标准异常）
    if (numel == 0) {

            throw std::invalid_argument("Argmax: input tensor vals is empty (numel = 0).");
        
    }

    // 数据类型分发（支持F32/BF16/F16，对齐add算子）
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        return argmax_<float>(max_idx, max_val, reinterpret_cast<const float*>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_<llaisys::bf16_t>(max_idx, max_val, reinterpret_cast<const llaisys::bf16_t*>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_<llaisys::fp16_t>(max_idx, max_val, reinterpret_cast<const llaisys::fp16_t*>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu