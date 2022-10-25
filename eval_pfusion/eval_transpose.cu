#include "cuda.h"
#include "cuda_utils.h"

#include <vector>

void invoke_func(float *src, float *dst);

int main() {
    std::vector<int> shape = {31, 32, 32, 33};
    std::vector<int> perm = {2, 0, 3, 1};
    float *src, *dst;
    size_t size = 1;
    for (auto x : shape) {
        size *= x;
    }
    std::vector<int> stride_src(4), stride_dst(4);
    stride_dst[0] = 1;
    for (int i = 1; i < 4; i++) {
        stride_dst[i] = stride_dst[i-1] * shape[i-1];
    }
    size_t this_stride = 1;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (perm[j] == i) {
                stride_src[i] = this_stride;
                this_stride *= shape[j];
            }
        }
    }

    cudaSafeCall(cudaMalloc((void **)&src, size * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&dst, size * sizeof(float)));

    float *src_host, *dst_host;
    src_host = (float *)malloc(size * sizeof(float));
    dst_host = (float *)malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++) {
        src_host[i] = i;
    }
    cudaSafeCall(cudaMemcpy(src, src_host, size * sizeof(float), cudaMemcpyHostToDevice));
    invoke_func(src, dst);
    cudaSafeCall(cudaMemcpy(dst_host, dst, size * sizeof(float), cudaMemcpyDeviceToHost));
    bool flag = 0;
    for (size_t i = 0; i < size; i++) {
        size_t base = i;
        size_t offset_src = 0;
        for (int j = 0; j < 4; j++) {
            offset_src += base % shape[j] * stride_src[perm[j]];
            base /= shape[j];
        }
        if (dst_host[i] != src_host[offset_src]) {
            flag = 1;
            std::cout << "[ERROR] at " << i << "," << offset_src << ":" << dst_host[i] << "," << src_host[offset_src] << std::endl;
            break;
        }
    }

    if (!flag) {
        std::cout << "[INFO] transpose correct." << std::endl;
    } else {
        std::cout << "[ERROR] transpose incorrect." << std::endl;
    }

    float duration = 0;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    int cnt = 128;
    for (int t = 0; t < cnt; t++) {
        invoke_func(src, dst);
    }
    cudaEventRecord(st, 0);
    for (int t = 0; t < cnt; t++) {
        invoke_func(src, dst);
    }
    cudaEventRecord(ed, 0);
    cudaEventSynchronize(st);
    cudaEventSynchronize(ed);
    cudaEventElapsedTime(&duration, st, ed);
    std::cout << "[INFO] time: " << duration << std::endl;
    double perf = double(size) * 8.0f * cnt / (duration * 1e-3) / 1024.0f / 1024.0f / 1024.0f;
    std::cout << "[INFO] Perf: " << perf << "GB/s" << std::endl;
    std::cout << "[Exit] successful." << std::endl;
}
