#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>

namespace llaisys {
// 辅助：逐元素计算 SwiGLU
// SwiGLU(gate, up) = SiLU(gate) * up = (gate * sigmoid(gate)) * up
template <typename T>
void swiglu_elementwise(T *out, const T *gate, const T *up,
                        size_t seqlen, size_t intermediate_size) {
    size_t total_elements = seqlen * intermediate_size;
    for (size_t idx = 0; idx < total_elements; ++idx) {
        // 读取输入
        float gate_val = llaisys::utils::cast<float>(gate[idx]);
        float up_val = llaisys::utils::cast<float>(up[idx]);

        // 计算 SiLU (Swish) 激活: x * sigmoid(x)
        float sigmoid = 1.0f / (1.0f + std::exp(-gate_val));
        float silu = gate_val * sigmoid;  // SiLU(gate) = gate * sigmoid(gate)

        // 计算 SwiGLU：out = SiLU(gate) * up
        float out_val = silu * up_val;

        // 写入输出
        out[idx] = llaisys::utils::cast<T>(out_val);
    }
}
} // namespace llaisys

// 模板核心函数
template <typename T>
void swiglu_(std::byte *out, const std::byte *gate, const std::byte *up,
             size_t seqlen, size_t intermediate_size) {
    const T *gate_ptr = reinterpret_cast<const T*>(gate);
    const T *up_ptr = reinterpret_cast<const T*>(up);
    T *out_ptr = reinterpret_cast<T*>(out);

    llaisys::swiglu_elementwise<T>(out_ptr, gate_ptr, up_ptr, seqlen, intermediate_size);
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t data_type, size_t seqlen, size_t intermediate_size) {
    // 空值保护
    if (seqlen == 0 || intermediate_size == 0) {
        throw std::invalid_argument("SwiGLU: seqlen/intermediate_size cannot be zero.");
    }

    // 数据类型分发
    switch (data_type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float>(out, gate, up, seqlen, intermediate_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_<llaisys::fp16_t>(out, gate, up, seqlen, intermediate_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<llaisys::bf16_t>(out, gate, up, seqlen, intermediate_size);
    default:
        std::string err_msg = "SwiGLU: unsupported data type (" + std::to_string(static_cast<int>(data_type)) + ").";
        throw std::runtime_error(err_msg);
    }
}
} // namespace llaisys::ops::cpu