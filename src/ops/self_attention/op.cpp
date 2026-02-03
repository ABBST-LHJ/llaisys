#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"
#include <string>
#include <stdexcept>
#include <vector>

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 1. 设备一致性校验
    auto out_device = attn_val->deviceType();
    auto out_device_id = attn_val->deviceId();
    auto q_device = q->deviceType();
    auto q_device_id = q->deviceId();
    auto k_device = k->deviceType();
    auto k_device_id = k->deviceId();
    auto v_device = v->deviceType();
    auto v_device_id = v->deviceId();

    if (out_device != q_device || out_device != k_device || out_device != v_device ||
        out_device_id != q_device_id || out_device_id != k_device_id || out_device_id != v_device_id) {
        throw std::invalid_argument("Self-Attention: all tensors must be on the same device.");
    }

    // 2. 维度校验
    std::vector<size_t> attn_val_shape = attn_val->shape();
    std::vector<size_t> q_shape = q->shape();
    std::vector<size_t> k_shape = k->shape();
    std::vector<size_t> v_shape = v->shape();

    if (attn_val_shape.size() != 3 || q_shape.size() != 3 || k_shape.size() != 3 || v_shape.size() != 3) {
        throw std::invalid_argument("Self-Attention: all tensors must be 3D.");
    }

    // 3. 提取形状参数
    size_t seqlen = q_shape[0];
    size_t nhead = q_shape[1];
    size_t d = q_shape[2];
    size_t total_len = k_shape[0];
    size_t nkvhead = k_shape[1];
    size_t dv = v_shape[2];

    // 4. 形状匹配校验
    if (attn_val_shape != std::vector<size_t>{seqlen, nhead, dv}) {
        throw std::invalid_argument("Self-Attention: attn_val shape must be [seqlen, nhead, dv].");
    }
    if (k_shape[2] != d) {
        throw std::invalid_argument("Self-Attention: k last dim must match q last dim.");
    }
    if (v_shape[0] != total_len || v_shape[1] != nkvhead) {
        throw std::invalid_argument("Self-Attention: v shape must be [total_len, nkvhead, dv].");
    }
    if (nhead % nkvhead != 0) {
        throw std::invalid_argument("Self-Attention: nhead must be a multiple of nkvhead.");
    }

    // 5. 数据类型校验
    llaisysDataType_t dtype = attn_val->dtype();
    if (q->dtype() != dtype || k->dtype() != dtype || v->dtype() != dtype) {
        throw std::invalid_argument("Self-Attention: all tensors must have the same data type.");
    }

    // 6. 连续性校验
    if (!attn_val->isContiguous() || !q->isContiguous() || !k->isContiguous() || !v->isContiguous()) {
        throw std::invalid_argument("Self-Attention: all tensors must be contiguous.");
    }

    // 7. CPU 快速路径
    if (out_device == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  dtype, seqlen, nhead, d, total_len, nkvhead, dv, scale);
    }

    // 8. 非 CPU 设备
    llaisys::core::context().setDevice(out_device, out_device_id);

    switch (out_device) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  dtype, seqlen, nhead, d, total_len, nkvhead, dv, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        throw std::runtime_error("Self-Attention: NVIDIA device is not implemented yet.");
#endif
    default:
        throw std::runtime_error("Self-Attention: unsupported device type.");
    }
}
} // namespace llaisys::ops