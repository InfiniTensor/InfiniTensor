#pragma once

namespace infini {
void div_kernel(void *a, void *b, void *c, int a0, int a1, int a2, int a3,
                int b0, int b1, int b2, int b3, int c0, int c1, int c2, int c3);
void add_kernel(void *a, void *b, void *c, int a0, int a1, int a2, int a3,
                int b0, int b1, int b2, int b3, int c0, int c1, int c2, int c3);
void pow_kernel(void *a, void *b, void *c, int a0, int a1, int a2, int a3,
                int b0, int b1, int b2, int b3, int c0, int c1, int c2, int c3);
void less_kernel(void *a, void *b, void *c, int a0, int a1, int a2, int a3,
                 int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                 int c3);
}; // namespace infini
