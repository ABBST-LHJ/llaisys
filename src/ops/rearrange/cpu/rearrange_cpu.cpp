#include "rearrange_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <stdexcept>
#include <cstring>

namespace llaisys {
// 辅助：逐元素复制（适配任意数据类型）
template <typename T>
void copy_elementwise(T *out, const T *in, size_t total_elements) {
    for (size_t idx = 0; idx < total_elements; ++idx) {
        out[idx] = in[idx];
    }
}
} // namespace llaisys

// 模板核心函数
template <typename T>
void rearrange_(std::byte *out, const std::byte *in, size_t total_elements) {
    const T *in_ptr = reinterpret_cast<const T*>(in);
    T *out_ptr = reinterpret_cast<T*>(out);

    llaisys::copy_elementwise<T>(out_ptr, in_ptr, total_elements);
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in,
               llaisysDataType_t data_type, size_t total_elements, size_t elem_size) {
    // 空值保护
    if (total_elements == 0 || elem_size == 0) {
        throw std::invalid_argument("Rearrange: total_elements/elem_size cannot be zero.");
    }

    // 数据类型分发
    switch (data_type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_<float>(out, in, total_elements);
    case LLAISYS_DTYPE_F16:
        return rearrange_<llaisys::fp16_t>(out, in, total_elements);
    case LLAISYS_DTYPE_BF16:
        return rearrange_<llaisys::bf16_t>(out, in, total_elements);
    default:
        std::string err_msg = "Rearrange: unsupported data type (" + std::to_string(static_cast<int>(data_type)) + ").";
        throw std::runtime_error(err_msg);
    }
}
} // namespace llaisys::ops::cpu