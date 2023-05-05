#include "cuda/cuda_common.h"
#include <assert.h>
#include <vector>

template <int numSM, int numWarp>
__global__ void kernel_transpose_last(float *ptrA, float *ptrB, int dim0,
                                      int dim1, int dim2) {
    int laneId = threadIdx.x % 32;
    int warpId = blockIdx.x * numWarp + threadIdx.x / 32;
    int n1 = (dim1 + 31) / 32;
    int n2 = (dim2 + 31) / 32;
    float bufA[32];
    for (int i = warpId; i < dim0 * n1 * n2; i += numSM * numWarp) {
        // clock_t ck0 = clock();
        int i0 = i / (n1 * n2);
        int i1 = (i % (n1 * n2)) / n2;
        int i2 = (i % (n1 * n2)) % n2;
        int offsetA = i0 * dim1 * dim2 + i2 * 32 * dim1 + i1 * 32;
        int offsetB = i0 * dim1 * dim2 + i1 * 32 * dim2 + i2 * 32;
        int ld1 = min(32, dim1 - i1 * 32);
        int ld2 = min(32, dim2 - i2 * 32);
        // if (i == 4 && laneId == 0)
        //     printf("%d %d\n", ld1, ld2);

        if (ld2 == 32) {
#pragma unroll
            for (int i = 0; i < 32; i++) {
                if ((laneId + i) % 32 < ld1) {
                    bufA[i] = ptrA[offsetA + i * dim1 + (laneId + i) % 32];
                }
            }
        } else if (ld2 == 17) {
#pragma unroll
            for (int i = 0; i < 17; i++) {
                if ((laneId + i) % 32 < ld1) {
                    bufA[i] = ptrA[offsetA + i * dim1 + (laneId + i) % 32];
                }
            }
        } else if (ld2 == 4) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if ((laneId + i) % 32 < ld1) {
                    bufA[i] = ptrA[offsetA + i * dim1 + (laneId + i) % 32];
                }
            }
        } else {
            for (int i = 0; i < ld2; i++) {
                if ((laneId + i) % 32 < ld1) {
                    bufA[i] = ptrA[offsetA + i * dim1 + (laneId + i) % 32];
                }
            }
        };

        if (ld1 == 32) {
#pragma unroll
            for (int i = 0; i < 32; i++) {
                if ((i + 32 - laneId) % 32 < ld2) {
                    ptrB[offsetB + i * dim2 + (i + 32 - laneId) % 32] =
                        bufA[(i + 32 - laneId) % 32];
                }
            }
        } else if (ld1 == 17) {
#pragma unroll
            for (int i = 0; i < 17; i++) {
                if ((i + 32 - laneId) % 32 < ld2) {
                    ptrB[offsetB + i * dim2 + (i + 32 - laneId) % 32] =
                        bufA[(i + 32 - laneId) % 32];
                }
            }
        } else if (ld1 == 4) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if ((i + 32 - laneId) % 32 < ld2) {
                    ptrB[offsetB + i * dim2 + (i + 32 - laneId) % 32] =
                        bufA[(i + 32 - laneId) % 32];
                }
            }
        } else {
            for (int i = 0; i < ld1; i++) {
                if ((i + 32 - laneId) % 32 < ld2) {
                    ptrB[offsetB + i * dim2 + (i + 32 - laneId) % 32] =
                        bufA[(i + 32 - laneId) % 32];
                }
            }
        };
    }
}

namespace infini {

/// @brief
/// @param ptrA Input tensor of shape [dim0, dim2, dim1]
/// @param ptrB Output tensor  of shape [dim0, dim1, dim2]
/// @param dim0
/// @param dim1
/// @param dim2
void invoke_transpose_last_two_dim(float *ptrA, float *ptrB, int dim0, int dim1,
                                   int dim2, int numSMs) {
    constexpr int numWarps = 4;
    dim3 gridDim(numSMs, 1);
    dim3 blockDim(numWarps * 32, 1);
    if (numSMs == 80) { // V100
        kernel_transpose_last<80, numWarps>
            <<<gridDim, blockDim>>>(ptrA, ptrB, dim0, dim1, dim2);
    } else if (numSMs == 108) { // A100
        kernel_transpose_last<108, numWarps>
            <<<gridDim, blockDim>>>(ptrA, ptrB, dim0, dim1, dim2);
    } else {
        IT_TODO_HALT_MSG(std::string("transpose_last_two_dim with ") +
                         std::to_string(numSMs) + " SMs is not implemented");
    }
    // cudaCheckError();
}

} // namespace infini

// constexpr int numWarm = 128, numEval = 128;
//
// void eval_transpose_last(const std::vector<int> &shape) {
//     assert(shape.size() == 3);
//     int size = shape[0] * shape[1] * shape[2];
//     float *dataA, *dataB;
//     dataA = (float *)malloc(size * sizeof(float));
//     dataB = (float *)malloc(size * sizeof(float));
//     for (int i0 = 0; i0 < shape[0]; i0++) {
//         for (int i2 = 0; i2 < shape[2]; i2++) {
//             for (int i1 = 0; i1 < shape[1]; i1++) {
//                 dataA[i0 * shape[1] * shape[2] + i2 * shape[1] + i1] =
//                     i0 * shape[1] * shape[2] + i2 * shape[1] + i1;
//             }
//         }
//     }
//     float *ptrA, *ptrB;
//     checkCudaError(cudaMalloc(&ptrA, size * sizeof(float)));
//     checkCudaError(cudaMalloc(&ptrB, size * sizeof(float)));
//     checkCudaError(
//         cudaMemcpy(ptrA, dataA, size * sizeof(float),
//         cudaMemcpyHostToDevice));

//     invoke_transpose_last_two_dim(ptrA, ptrB, shape[0], shape[1], shape[2]);
//     checkCudaError(
//         cudaMemcpy(dataB, ptrB, size * sizeof(float),
//         cudaMemcpyDeviceToHost));
//     for (int i0 = 0; i0 < shape[0]; i0++) {
//         for (int i1 = 0; i1 < shape[1]; i1++) {
//             for (int i2 = 0; i2 < shape[2]; i2++) {
//                 if (dataA[i0 * shape[1] * shape[2] + i1 + i2 * shape[1]] !=
//                     dataB[i0 * shape[1] * shape[2] + i1 * shape[2] + i2]) {
//                     std::cout
//                         << i0 << " " << i1 << " " << i2 << " "
//                         << dataA[i0 * shape[1] * shape[2] + i1 + i2 *
//                         shape[1]]
//                         << " "
//                         << dataB[i0 * shape[1] * shape[2] + i1 * shape[2] +
//                         i2]
//                         << std::endl;
//                     exit(-1);
//                 }
//             }
//         }
//     }
//     cudaEvent_t st, ed;
//     checkCudaError(cudaEventCreate(&st));
//     checkCudaError(cudaEventCreate(&ed));
//     for (int i = 0; i < numWarm; i++) {
//         invoke_transpose_last_two_dim(ptrA, ptrB, shape[0], shape[1],
//         shape[2]);
//     }
//     checkCudaError(cudaEventRecord(st));
//     for (int i = 0; i < numEval; i++) {
//         invoke_transpose_last_two_dim(ptrA, ptrB, shape[0], shape[1],
//         shape[2]);
//     }
//     checkCudaError(cudaEventRecord(ed));
//     checkCudaError(cudaEventSynchronize(st));
//     checkCudaError(cudaEventSynchronize(ed));
//     float time;
//     checkCudaError(cudaEventElapsedTime(&time, st, ed));
//     float bandwidth = size * 2 * sizeof(float) * numEval / time / 1e6;
//     std::cout << "transpose_last: " << shape[0] << " " << shape[1] << " "
//               << shape[2] << " time: " << time / numEval
//               << " ms. bandwidth: " << bandwidth << " GB/s" << std::endl;
// }

// Performance evaluation
// int main() {
//     eval_transpose_last({16, 1024, 256});
//     eval_transpose_last({16, 14 * 14, 1024});
//     eval_transpose_last({16, 7 * 7, 2048});
//     eval_transpose_last({16, 7 * 7, 128});
//     eval_transpose_last({1, 14 * 14, 1024});
//     eval_transpose_last({1, 7 * 7, 2048});
//     eval_transpose_last({1, 7 * 7, 128});
// }
