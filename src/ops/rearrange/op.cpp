#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rearrange_cpu.hpp"
#include <string>
#include <stdexcept>
#include <vector>

// 新增：辅助函数 - 计算 dtype 对应的字节大小
static size_t get_dtype_size(llaisysDataType_t dtype) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return sizeof(float);
        case LLAISYS_DTYPE_F16:
            return sizeof(llaisys::fp16_t);
        case LLAISYS_DTYPE_BF16:
            return sizeof(llaisys::bf16_t);
        default:
            throw std::runtime_error("get_dtype_size: unsupported data type.");
    }
}

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    // 1. 设备一致性校验
    auto out_device = out->deviceType();
    auto out_device_id = out->deviceId();
    auto in_device = in->deviceType();
    auto in_device_id = in->deviceId();

    if (out_device != in_device || out_device_id != in_device_id) {
        throw std::invalid_argument("Rearrange: out/in must be on the same device.");
    }

    // 2. 形状匹配校验
    std::vector<size_t> out_shape = out->shape();
    std::vector<size_t> in_shape = in->shape();

    if (out_shape != in_shape) {
        throw std::invalid_argument("Rearrange: out shape must match in shape.");
    }

    // 3. 计算总元素数和元素大小
    size_t total_elements = 1;
    for (size_t dim : out_shape) {
        total_elements *= dim;
    }

    // 4. 数据类型校验
    llaisysDataType_t dtype = out->dtype();
    if (in->dtype() != dtype) {
        throw std::invalid_argument("Rearrange: out/in must have the same data type.");
    }

    // 修正：替换 out->dtypeSize()，调用手动实现的 get_dtype_size()
    size_t elem_size = get_dtype_size(dtype);

    // 5. CPU 快速路径
    if (out_device == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(),
                             dtype, total_elements, elem_size);
    }

    // 6. 非 CPU 设备
    llaisys::core::context().setDevice(out_device, out_device_id);

    switch (out_device) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(),
                             dtype, total_elements, elem_size);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        throw std::runtime_error("Rearrange: NVIDIA device is not implemented yet.");
#endif
    default:
        throw std::runtime_error("Rearrange: unsupported device type.");
    }
}
} // namespace llaisys::ops