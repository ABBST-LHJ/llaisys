#pragma once
#include "llaisys.h"
#include <cstddef>

// 移除所有手动兜底的 LLAISYS_DTYPE_* 定义，直接使用框架原生枚举
// 确保框架 utils/types.hpp 已包含这些枚举，无需手动定义

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t data_type, size_t N, size_t in_features, size_t out_features);
} // namespace llaisys::ops::cpu