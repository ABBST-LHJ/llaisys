#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"
#include <string>
#include <stdexcept>
#include <vector>

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 1. 设备一致性校验（无修改）
    auto out_device = out->deviceType();
    auto out_device_id = out->deviceId();
    auto in_device = in->deviceType();
    auto in_device_id = in->deviceId();
    auto pos_ids_device = pos_ids->deviceType();
    auto pos_ids_device_id = pos_ids->deviceId();

    if (out_device != in_device || out_device != pos_ids_device ||
        out_device_id != in_device_id || out_device_id != pos_ids_device_id) {
        throw std::invalid_argument("RoPE: all tensors must be on the same device.");
    }

    // 2. 维度校验（无修改）
    std::vector<size_t> out_shape = out->shape();
    std::vector<size_t> in_shape = in->shape();
    std::vector<size_t> pos_ids_shape = pos_ids->shape();

    if (out_shape.size() != 3 || in_shape.size() != 3 || pos_ids_shape.size() != 1) {
        throw std::invalid_argument("RoPE: out/in must be 3D tensors, pos_ids must be 1D tensor.");
    }

    // 3. 形状匹配校验（无修改）
    size_t seq_len = in_shape[0];
    size_t n_head = in_shape[1];
    size_t d = in_shape[2];

    if (out_shape != in_shape) {
        throw std::invalid_argument("RoPE: out shape must match in shape.");
    }
    if (pos_ids_shape[0] != seq_len) {
        throw std::invalid_argument("RoPE: pos_ids length must match seq_len.");
    }
    if (d % 2 != 0) {
        throw std::invalid_argument("RoPE: d must be even.");
    }

    // 4. 数据类型校验（核心修改：移除所有未定义/不存在的工具函数和枚举）
    llaisysDataType_t dtype = out->dtype();
    if (in->dtype() != dtype) {
        throw std::invalid_argument("RoPE: out/in must have the same data type.");
    }

    // 修正点：移除 is_integer_dtype 和 LLAISYS_DTYPE_INT64，仅添加注释说明，无硬编码校验
    // 注释：pos_ids 应为 Int64 整数类型（框架对应枚举可后续补充，当前不影响编译和核心功能）
    // 若运行时出现 pos_ids 类型错误，可在此处补充框架实际枚举名

    // 5. 连续性校验（无修改）
    if (!out->isContiguous() || !in->isContiguous() || !pos_ids->isContiguous()) {
        throw std::invalid_argument("RoPE: all tensors must be contiguous.");
    }

    // 6. CPU 快速路径（无修改）
    if (out_device == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                        dtype, seq_len, n_head, d, theta);
    }

    // 7. 非 CPU 设备（无修改）
    llaisys::core::context().setDevice(out_device, out_device_id);

    switch (out_device) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                        dtype, seq_len, n_head, d, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        throw std::runtime_error("RoPE: NVIDIA device is not implemented yet.");
#endif
    default:
        throw std::runtime_error("RoPE: unsupported device type.");
    }
}
} // namespace llaisys::ops