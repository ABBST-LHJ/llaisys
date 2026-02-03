#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"
#include <string>
#include <stdexcept>
#include <vector>

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 1. 设备一致性校验
    auto out_device = out->deviceType();
    auto out_device_id = out->deviceId();
    auto gate_device = gate->deviceType();
    auto gate_device_id = gate->deviceId();
    auto up_device = up->deviceType();
    auto up_device_id = up->deviceId();

    if (out_device != gate_device || out_device != up_device ||
        out_device_id != gate_device_id || out_device_id != up_device_id) {
        throw std::invalid_argument("SwiGLU: all tensors must be on the same device.");
    }

    // 2. 维度校验
    std::vector<size_t> out_shape = out->shape();
    std::vector<size_t> gate_shape = gate->shape();
    std::vector<size_t> up_shape = up->shape();

    if (out_shape.size() != 2 || gate_shape.size() != 2 || up_shape.size() != 2) {
        throw std::invalid_argument("SwiGLU: all tensors must be 2D.");
    }

    // 3. 形状匹配校验
    size_t seqlen = out_shape[0];
    size_t intermediate_size = out_shape[1];

    if (gate_shape != out_shape || up_shape != out_shape) {
        throw std::invalid_argument("SwiGLU: gate/up shape must match out shape.");
    }

    // 4. 数据类型校验
    llaisysDataType_t dtype = out->dtype();
    if (gate->dtype() != dtype || up->dtype() != dtype) {
        throw std::invalid_argument("SwiGLU: all tensors must have the same data type.");
    }

    // 5. 连续性校验
    if (!out->isContiguous() || !gate->isContiguous() || !up->isContiguous()) {
        throw std::invalid_argument("SwiGLU: all tensors must be contiguous.");
    }

    // 6. CPU 快速路径
    if (out_device == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                          dtype, seqlen, intermediate_size);
    }

    // 7. 非 CPU 设备
    llaisys::core::context().setDevice(out_device, out_device_id);

    switch (out_device) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                          dtype, seqlen, intermediate_size);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        throw std::runtime_error("SwiGLU: NVIDIA device is not implemented yet.");
#endif
    default:
        throw std::runtime_error("SwiGLU: unsupported device type.");
    }
}
} // namespace llaisys::ops