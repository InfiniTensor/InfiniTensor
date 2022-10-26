#include "cuda.h"
#include "cuda_utils.h"

#include <vector>

void invoke_func_2(float *tensor_ptr_2, float *tensor_ptr_3);
void invoke_func_3(float *tensor_ptr_2, float *tensor_ptr_3,
    float *tensor_ptr_4);

int main() {
    std::vector<int> shape = {1, 1, 512, 512};
    float *t0, *t1, *t2;
    size_t size = 1;
    for (auto x : shape) {
        size *= x;
    }

    cudaSafeCall(cudaMalloc((void **)&t0, size * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&t1, size * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&t2, size * sizeof(float)));

    float duration = 0;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    int cnt = 128;
    for (int t = 0; t < cnt; t++) {
        invoke_func_2(t0, t1);
        invoke_func_3(t0, t1, t2);
    }
    cudaEventRecord(st, 0);
    for (int t = 0; t < cnt; t++) {
        invoke_func_2(t0, t1);
        invoke_func_3(t0, t1, t2);
    }
    cudaEventRecord(ed, 0);
    cudaEventSynchronize(st);
    cudaEventSynchronize(ed);
    cudaEventElapsedTime(&duration, st, ed);
    std::cout << "[INFO] time: " << duration / cnt << std::endl;
    double perf = double(size) * 8.0f * cnt / (duration * 1e-3) / 1024.0f / 1024.0f / 1024.0f;
    std::cout << "[INFO] Perf: " << perf << "GB/s" << std::endl;
    std::cout << "[Exit] successful." << std::endl;
}
