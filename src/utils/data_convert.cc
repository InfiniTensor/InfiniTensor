#include "utils/data_convert.h"

namespace infini {

uint16_t float_to_fp16(const float x) {
    Uf32 u;
    u.f32 = x;
    const uint32_t b = u.u32 + 0x00001000;
    const uint32_t e = (b & 0x7F800000) >> 23;
    const uint32_t m = b & 0x007FFFFF;
    return (b & 0x80000000) >> 16 |
           (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
           ((e < 113) & (e > 101)) *
               ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
           (e > 143) * 0x7FFF;
}

float fp16_to_float(const uint16_t x) {
    Uf32 u;
    const uint32_t e = (x & 0x7C00) >> 10;
    const uint32_t m = (x & 0x03FF) << 13;
    u.f32 = (float)m;
    const uint32_t v = u.u32 >> 23;
    const uint32_t r = (x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
                       ((e == 0) & (m != 0)) *
                           ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000));
    u.u32 = r;
    return u.f32;
}
} // namespace infini
