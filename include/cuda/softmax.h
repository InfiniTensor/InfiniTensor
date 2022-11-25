#pragma once

namespace infini {
void softmax_kernel(int max_threadblock_size, int batch_size, float *x,
                    float *y, int dim, int stride);
}
