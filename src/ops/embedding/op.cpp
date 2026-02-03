#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"
#include <string>
#include <stdexcept>
#include <vector>

#ifndef LLAISYS_DTYPE_INT64
#define LLAISYS_DTYPE_INT64 static_cast<llaisysDataType_t>(6)
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 1. 校验设备一致
    auto out_device = out->deviceType();
    auto out_device_id = out->deviceId();
    auto index_device = index->deviceType();
    auto index_device_id = index->deviceId();
    auto weight_device = weight->deviceType();
    auto weight_device_id = weight->deviceId();

    if (out_device != index_device || out_device != weight_device ||
        out_device_id != index_device_id || out_device_id != weight_device_id) {
        throw std::invalid_argument("Embedding: all tensors must be on the same device.");
    }

    // 2. 校验维度
    std::vector<size_t> out_shape = out->shape();
    std::vector<size_t> index_shape = index->shape();
    std::vector<size_t> weight_shape = weight->shape();

    if (out_shape.size() != 2) {
        throw std::invalid_argument("Embedding: out must be a 2D tensor.");
    }
    if (index_shape.size() != 1) {
        throw std::invalid_argument("Embedding: index must be a 1D tensor.");
    }
    if (weight_shape.size() != 2) {
        throw std::invalid_argument("Embedding: weight must be a 2D tensor.");
    }

    // 3. 校验形状匹配
    size_t batch_size = index->numel();
    size_t hidden_dim = weight_shape[1];
    size_t vocab_size = weight_shape[0];

    if (out_shape[0] != batch_size) {
        throw std::invalid_argument("Embedding: out first dim must match index numel.");
    }
    if (out_shape[1] != hidden_dim) {
        throw std::invalid_argument("Embedding: out second dim must match weight second dim.");
    }

    // 4. 校验index为Int64
    if (index->dtype() != LLAISYS_DTYPE_INT64) {
        std::string err_msg = "Embedding: index must be Int64 dtype, but got " +
                              std::to_string(static_cast<int>(index->dtype())) + ".";
        throw std::invalid_argument(err_msg);
    }

    // 5. 校验out和weight dtype一致
    if (out->dtype() != weight->dtype()) {
        std::string err_msg = "Embedding: Datatypes mismatch - out dtype (" +
                              std::to_string(static_cast<int>(out->dtype())) +
                              ") != weight dtype (" +
                              std::to_string(static_cast<int>(weight->dtype())) + ").";
        throw std::invalid_argument(err_msg);
    }

    // 6. 校验连续存储
    if (!out->isContiguous() || !index->isContiguous() || !weight->isContiguous()) {
        throw std::invalid_argument("Embedding: all tensors must be contiguous.");
    }

    // 7. CPU快速路径
    if (out_device == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              weight->dtype(), batch_size, hidden_dim, vocab_size);
    }

    // 8. 非CPU设备处理
    llaisys::core::context().setDevice(out_device, out_device_id);

    switch (out_device) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              weight->dtype(), batch_size, hidden_dim, vocab_size);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        throw std::runtime_error("Embedding: NVIDIA device is not implemented yet.");
#endif
    default:
        throw std::runtime_error("Embedding: unsupported device type.");
    }
}
} // namespace llaisys::ops