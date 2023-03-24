#pragma once

namespace infini {
void div_kernel(float *a, float *b, float *c, int a0, int a1, int a2, int a3,
                int b0, int b1, int b2, int b3, int c0, int c1, int c2, int c3);
void pow_kernel(float *a, float *b, float *c, int a0, int a1, int a2, int a3,
                int b0, int b1, int b2, int b3, int c0, int c1, int c2, int c3);
}; // namespace infini
