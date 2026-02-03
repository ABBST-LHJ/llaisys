#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"
#include <string>
#include <stdexcept>
#include <vector>

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 1. 设备一致性校验
    auto out_device = out->deviceType();
    auto out_device_id = out->deviceId();
    auto in_device = in->deviceType();
    auto in_device_id = in->deviceId();
    auto weight_device = weight->deviceType();
    auto weight_device_id = weight->deviceId();

    if (out_device != in_device || out_device != weight_device ||
        out_device_id != in_device_id || out_device_id != weight_device_id) {
        throw std::invalid_argument("RMS Norm: all tensors must be on the same device.");
    }

    // 2. 维度校验
    std::vector<size_t> out_shape = out->shape();
    std::vector<size_t> in_shape = in->shape();
    std::vector<size_t> weight_shape = weight->shape();

    if (out_shape.size() != 2 || in_shape.size() != 2) {
        throw std::invalid_argument("RMS Norm: out/in must be 2D tensors.");
    }
    if (weight_shape.size() != 1) {
        throw std::invalid_argument("RMS Norm: weight must be a 1D tensor.");
    }

    // 3. 形状匹配校验
    size_t batch_size = in_shape[0];
    size_t hidden_dim = in_shape[1];

    if (out_shape[0] != batch_size || out_shape[1] != hidden_dim) {
        throw std::invalid_argument("RMS Norm: out shape must match in shape.");
    }
    if (weight_shape[0] != hidden_dim) {
        throw std::invalid_argument("RMS Norm: weight length must match in hidden dim.");
    }

    // 4. 数据类型校验
    llaisysDataType_t dtype = out->dtype();
    if (in->dtype() != dtype || weight->dtype() != dtype) {
        throw std::invalid_argument("RMS Norm: all tensors must have the same data type.");
    }

    // 5. 连续性校验
    if (!out->isContiguous() || !in->isContiguous() || !weight->isContiguous()) {
        throw std::invalid_argument("RMS Norm: all tensors must be contiguous.");
    }

    // 6. CPU 快速路径
    if (out_device == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                            dtype, batch_size, hidden_dim, eps);
    }

    // 7. 非 CPU 设备
    llaisys::core::context().setDevice(out_device, out_device_id);

    switch (out_device) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                            dtype, batch_size, hidden_dim, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        throw std::runtime_error("RMS Norm: NVIDIA device is not implemented yet.");
#endif
    default:
        throw std::runtime_error("RMS Norm: unsupported device type.");
    }
}
} // namespace llaisys::ops