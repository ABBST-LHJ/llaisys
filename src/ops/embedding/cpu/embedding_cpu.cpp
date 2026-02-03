#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llaisys {
template <typename T>
void copy_embedding_row(T *out_row, const T *weight_row, size_t hidden_dim) {
    std::memcpy(out_row, weight_row, hidden_dim * sizeof(T));
}
} // namespace llaisys

template <typename T>
void embedding_(std::byte *out, const std::byte *index, const std::byte *weight,
                size_t batch_size, size_t hidden_dim, size_t vocab_size) {
    const int64_t *index_ptr = reinterpret_cast<const int64_t*>(index);
    const T *weight_ptr = reinterpret_cast<const T*>(weight);
    T *out_ptr = reinterpret_cast<T*>(out);

    for (size_t i = 0; i < batch_size; ++i) {
        int64_t idx = index_ptr[i];
        if (idx < 0 || static_cast<size_t>(idx) >= vocab_size) {
            std::string err_msg = "Embedding: index " + std::to_string(idx) +
                                  " out of bounds (vocab size: " + std::to_string(vocab_size) + ").";
            throw std::out_of_range(err_msg);
        }

        const T *weight_row = weight_ptr + (static_cast<size_t>(idx) * hidden_dim);
        T *out_row = out_ptr + (i * hidden_dim);
        llaisys::copy_embedding_row(out_row, weight_row, hidden_dim);
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t data_type, size_t batch_size,
               size_t hidden_dim, size_t vocab_size) {
    // 原 EXCEPTION_INVALID_INPUT 替换为 std::invalid_argument
    if (batch_size == 0 || hidden_dim == 0 || vocab_size == 0) {
        throw std::invalid_argument("Embedding: batch_size/hidden_dim/vocab_size cannot be zero.");
    }

    switch (data_type) {
    case LLAISYS_DTYPE_F32:
        return embedding_<float>(out, index, weight, batch_size, hidden_dim, vocab_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_<llaisys::bf16_t>(out, index, weight, batch_size, hidden_dim, vocab_size);
    case LLAISYS_DTYPE_F16:
        return embedding_<llaisys::fp16_t>(out, index, weight, batch_size, hidden_dim, vocab_size);
    default:
        std::string err_msg = "Embedding: unsupported data type (" + std::to_string(static_cast<int>(data_type)) + ").";
        throw std::runtime_error(err_msg);
    }
}
} // namespace llaisys::ops::cpu