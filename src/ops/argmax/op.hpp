#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// argmax 算子上层接口声明：获取vals的最大值（max_val）和索引（max_idx）
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
} // namespace llaisys::ops