#include "core/common.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_det.h"

template <typename T> __device__ float _sum_diagonal(const T *input, int n) {
    T output = 1;
    for (int i = 0; i < n; ++i) {
        output *= input[i * n + i];
    }
    return output;
}

template <typename T>
__global__ void _det_kernel(const T *input, T *output, int n, int batch_size,
                            int mode) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < batch_size) {
        T det = _sum_diagonal(input + index * n * n, n);

        if (mode == 1) { // mode = LogDet
            if constexpr (std::is_same_v<T, half>) {
                output[index] = __float2half(__logf(__half2float(det)));
            } else if constexpr (std::is_same_v<T, float>) {
                output[index] = __logf(det);
            } else if constexpr (std::is_arithmetic_v<T>) {
                output[index] = std::log(det);
            } else {
                printf("Unsupported data type for det kernel.\n");
            }
        } else {
            output[index] = det;
        }
    }
}

namespace infini {
void det_kernel(const CudaRuntimeObj *context, void *input, void *output,
                const int n, const int batch_size, int mode) {
    size_t workspaceSize =
        batch_size * (sizeof(float *) + sizeof(int) + n * n * sizeof(float));
    CudaPtr workspace = context->getWorkspace(workspaceSize);

    float **a = reinterpret_cast<float **>(workspace);
    int *info = reinterpret_cast<int *>(a + batch_size);
    float *inputF32 =
        reinterpret_cast<float *>(reinterpret_cast<char *>(workspace) +
                                  batch_size * (sizeof(float *) + sizeof(int)));
    float *outputF32 = reinterpret_cast<float *>(output);

    /**
     * If datatype is not Float32, needs to convert to float32.
     * If dataype is double, can directly use the function:
     * cublasDgetrfBatched() instead of cublasSgetrfBatched()
     */
    // if (dataType != DataType::Float32) { ...Perform data conversion... }
    cudaMemcpy(inputF32, input, batch_size * n * n * sizeof(*inputF32),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(a, &inputF32, batch_size * sizeof(*a), cudaMemcpyHostToDevice);

    checkCublasError(cublasSgetrfBatched(context->cublasHandle(), n, a, n,
                                         nullptr, info, batch_size));

    _det_kernel<<<1, batch_size, 0, CUDAStream::getCurrentStream()>>>(
        inputF32, outputF32, n, batch_size, mode);

    cudaDeviceSynchronize();
}

}; // namespace infini
