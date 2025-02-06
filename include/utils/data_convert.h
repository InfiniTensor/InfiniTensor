#pragma once
#include <cstdint>
#include <iostream>

namespace infini {
union Uf32 {
    float f32;
    uint32_t u32;
};
uint16_t float_to_fp16(const float x);
float fp16_to_float(const uint16_t x);
uint16_t float_to_bfp16(const float x);
float bfp16_to_float(const uint16_t x);
} // namespace infini
