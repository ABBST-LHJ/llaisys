#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"
#include <string>  // 用于拼接详细错误信息

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 步骤1：输入合法性校验（优化dtype校验，添加详细错误信息）
    // 1. 所有张量必须在同一设备
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // 2. 校验max_val和vals数据类型一致（自定义详细错误，避免模糊报错）
    if (max_val->dtype() != vals->dtype()) {
        std::string err_msg = "Argmax: Datatypes mismatch - max_val dtype (" + 
                              std::to_string(static_cast<int>(max_val->dtype())) + 
                              ") != vals dtype (" + 
                              std::to_string(static_cast<int>(vals->dtype())) + ").";
        throw std::invalid_argument(err_msg);
    }

    // 3. vals必须是1D张量（作业明确要求）
    ASSERT(vals->shape().size() == 1, "Argmax: input tensor vals must be a 1D tensor.");

    // 4. max_idx必须是单个元素的1D张量
    ASSERT(max_idx->shape().size() == 1 && max_idx->numel() == 1, 
           "Argmax: max_idx must be a 1D tensor with a single element.");

    // 5. max_val必须是单个元素的1D张量
    ASSERT(max_val->shape().size() == 1 && max_val->numel() == 1, 
           "Argmax: max_val must be a 1D tensor with a single element.");

    // 6. 所有张量必须连续存储（简化内存访问，对齐add算子）
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), 
           "Argmax: all tensors must be contiguous.");

    // 步骤2：CPU设备快速路径（对齐add算子，提升常用场景效率）
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), 
                           vals->dtype(), vals->numel());
    }

    // 步骤3：非CPU设备处理（框架扩展预留，对齐add算子结构）
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), 
                           vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops