#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t data_type, size_t seqlen, size_t intermediate_size);
} // namespace llaisys::ops::cpu