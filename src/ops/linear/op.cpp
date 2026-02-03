#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"
#include <string>
#include <stdexcept>
#include <vector>

// 辅助函数：判断张量是否为空（框架中通常用 numel() == 0 或 is_null() 表示）


namespace llaisys::ops {
bool is_tensor_null(tensor_t tensor) {
    return !tensor || tensor->numel() == 0;
}
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 步骤 1：输入合法性校验（严格符合 2D 张量约束，无广播）
    // 1.1 非空张量设备一致（bias 可选，可空）
    auto out_device = out->deviceType();
    auto out_device_id = out->deviceId();
    auto in_device = in->deviceType();
    auto in_device_id = in->deviceId();
    auto weight_device = weight->deviceType();
    auto weight_device_id = weight->deviceId();

    if (out_device != in_device || out_device != weight_device ||
        out_device_id != in_device_id || out_device_id != weight_device_id) {
        throw std::invalid_argument("Linear: out/in/weight must be on the same device.");
    }
    // 若 bias 非空，校验设备一致
    if (!is_tensor_null(bias)) {
        auto bias_device = bias->deviceType();
        auto bias_device_id = bias->deviceId();
        if (out_device != bias_device || out_device_id != bias_device_id) {
            throw std::invalid_argument("Linear: bias must be on the same device as other tensors.");
        }
    }

    // 1.2 校验张量维度
    std::vector<size_t> out_shape = out->shape();
    std::vector<size_t> in_shape = in->shape();
    std::vector<size_t> weight_shape = weight->shape();

    if (out_shape.size() != 2) {
        throw std::invalid_argument("Linear: out must be a 2D tensor.");
    }
    if (in_shape.size() != 2) {
        throw std::invalid_argument("Linear: in must be a 2D tensor.");
    }
    if (weight_shape.size() != 2) {
        throw std::invalid_argument("Linear: weight must be a 2D tensor.");
    }
    // 若 bias 非空，校验为 1D 张量
    if (!is_tensor_null(bias)) {
        std::vector<size_t> bias_shape = bias->shape();
        if (bias_shape.size() != 1) {
            throw std::invalid_argument("Linear: bias must be a 1D tensor if provided.");
        }
    }

    // 1.3 提取形状参数
    size_t N = in_shape[0];                  // 批量大小
    size_t in_features = in_shape[1];        // 输入特征数
    size_t out_features = weight_shape[0];   // 输出特征数（weight 第一维）
    size_t weight_in_features = weight_shape[1]; // weight 第二维（需与 in_features 一致）

    // 1.4 校验形状匹配
    if (in_features != weight_in_features) {
        throw std::invalid_argument("Linear: in second dim must match weight second dim.");
    }
    if (out_shape[0] != N) {
        throw std::invalid_argument("Linear: out first dim must match in first dim.");
    }
    if (out_shape[1] != out_features) {
        throw std::invalid_argument("Linear: out second dim must match weight first dim.");
    }
    // 若 bias 非空，校验形状匹配
    if (!is_tensor_null(bias)) {
        size_t bias_features = bias->numel();
        if (bias_features != out_features) {
            throw std::invalid_argument("Linear: bias numel must match out second dim.");
        }
    }

    // 1.5 校验数据类型一致（out/in/weight/bias 需同类型，F32/BF16/F16）
    llaisysDataType_t dtype = out->dtype();
    if (in->dtype() != dtype || weight->dtype() != dtype) {
        throw std::invalid_argument("Linear: out/in/weight must have the same data type.");
    }
    if (!is_tensor_null(bias) && bias->dtype() != dtype) {
        throw std::invalid_argument("Linear: bias must have the same data type as other tensors.");
    }

    // 1.6 校验所有张量都是连续存储
    if (!out->isContiguous() || !in->isContiguous() || !weight->isContiguous()) {
        throw std::invalid_argument("Linear: out/in/weight must be contiguous.");
    }
    if (!is_tensor_null(bias) && !bias->isContiguous()) {
        throw std::invalid_argument("Linear: bias must be contiguous if provided.");
    }

    // 步骤 2：提取指针（bias 为空则传 nullptr）
    std::byte *bias_data = is_tensor_null(bias) ? nullptr : bias->data();

    // 步骤 3：CPU 设备快速路径（使用框架原生设备枚举，无冲突）
    if (out_device == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                           dtype, N, in_features, out_features);
    }

    // 步骤 4：非 CPU 设备处理（框架扩展预留）
    llaisys::core::context().setDevice(out_device, out_device_id);

    switch (out_device) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                           dtype, N, in_features, out_features);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        throw std::runtime_error("Linear: NVIDIA device is not implemented yet.");
#endif
    default:
        throw std::runtime_error("Linear: unsupported device type.");
    }
}
} // namespace llaisys::ops