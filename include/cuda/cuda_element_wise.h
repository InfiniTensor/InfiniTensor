#pragma once

namespace infini {
void div_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void add_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void mul_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void pow_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void less_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                 int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                 int c2, int c3);
void equal_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                  int a2, int a3, int b0, int b1, int b2, int b3, int c0,
                  int c1, int c2, int c3);
void div_special_kernel(int dType, void *a, void *b, void *c, int num);
void add_special_kernel(int dType, void *a, void *b, void *c, int num);
void pow_special_kernel(int dType, void *a, void *b, void *c, int num);
void less_special_kernel(int dType, void *a, void *b, void *c, int num);
void div_const_kernel(int dType, void *a, void *b, void *c, size_t n);

void pow_const_kernel(int dType, void *a, void *b, void *c, size_t n);
}; // namespace infini
