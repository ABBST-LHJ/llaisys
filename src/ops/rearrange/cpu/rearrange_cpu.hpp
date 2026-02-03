#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in,
               llaisysDataType_t data_type, size_t total_elements, size_t elem_size);
} // namespace llaisys::ops::cpu