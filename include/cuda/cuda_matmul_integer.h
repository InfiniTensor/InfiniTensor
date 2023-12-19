#pragma once

namespace infini {
void subA_kernel(int dType, void *a, void *b, int size, int k, int delta);
void subB_kernel(int dType, void *a, void *b, int size, int k, int n,
                 int delta);
}; // namespace infini
