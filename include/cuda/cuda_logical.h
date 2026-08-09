#pragma once
// CUDA kernel entrypoint declarations for logical and bitwise operators.
// These functions dispatch to device kernels implemented in
// `src/kernels/cuda/logical.cu`.
namespace infini {
void And_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void Or_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
               int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
               int c2, int c3);
void Xor_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void Not_kernel(int dtypeIndex, void *a, void *b, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3);
}; // namespace infini
