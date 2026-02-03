#pragma once
#include "llaisys.h"
#include <cstddef>

// 定义缺失的异常宏（对齐框架风格，避免编译报错）
#ifndef EXCEPTION_INVALID_INPUT
#define EXCEPTION_INVALID_INPUT(msg)  EXCEPTION(msg)
#endif

namespace llaisys::ops::cpu {
// CPU 底层 argmax 函数：处理通用字节指针，分发到具体数据类型
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, 
            llaisysDataType_t val_type, size_t numel);
} // namespace llaisys::ops::cpu