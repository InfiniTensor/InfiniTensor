#pragma once
#include "core/runtime.h"
#include "cuda/cuda_common.h"

namespace infini {

class CudaRuntimeObj : public RuntimeObj {
  private:
    cudaStream_t stream;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    CudaPtr workspace;
    size_t workspaceSize;

    // Memory information
    size_t allocatedGPUMemorySize = 0;
    map<void *, size_t> allocationMap;

    bool cudaGraphStatus; // Whether CUDA graph stream capture is enabled

    // CUDA device properties
    cudaDeviceProp deviceProperties;

    bool enableTF32 = false;

  public:
    CudaRuntimeObj();
    virtual ~CudaRuntimeObj();
    string toString() const override;

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;
    // double runEvaluation(const Graph &graph, int nWarmups,
    //                      int nEvaluations) const;
    void sync() const override;
    CudaPtr alloc(size_t size) override {
        void *ptr;
        // printf("Try to cudaMalloc: %lu bytes\n", size);
        checkCudaError(cudaMalloc(&ptr, size));
        allocatedGPUMemorySize += size;
        allocationMap[ptr] = size;
        // printf("cuda malloc: %p %lu bytes, total %lu bytes (%.2lf GB)\n",
        // ptr,
        //        size, allocatedGPUMemorySize,
        //        double(allocatedGPUMemorySize) / 1024 / 1024 / 1024);
        return ptr;
    }
    void dealloc(void *ptr) override {
        checkCudaError(cudaFree(ptr));
        allocatedGPUMemorySize -= allocationMap.at(ptr);
        allocationMap.erase(ptr);
        // printf("cuda dealloc: %p %lu bytes, total %lu\n", ptr,
        //        allocationMap.at(ptr), allocatedGPUMemorySize);
    }
    cudnnHandle_t cudnnHandle() const { return cudnn; }
    cublasHandle_t cublasHandle() const { return cublas; }
    size_t getWorkspaceSize() const { return workspaceSize; }
    CudaPtr getWorkspace(size_t size) const {
        IT_ASSERT(size <= workspaceSize);
        return workspace;
    }
    pair<int, int> getComputeCapacitiy() const {
        return {deviceProperties.major, deviceProperties.minor};
    }
    int getNumSMs() const { return deviceProperties.multiProcessorCount; }

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override {
        checkCudaError(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    }

    void copyBlobToCPU(void *dst, const void *src,
                       size_t bytes) const override {
        checkCudaError(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    }

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
        checkCudaError(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    }

    void runWithoutSync(const Graph &graph) const;

    bool isInCudaGraph() const { return cudaGraphStatus; }
    cudaStream_t getStream() const { return stream; }

    double timeWithCudaGraph(Graph graph, int rounds = 50);
    double timeWithCudaGraph(vector<std::function<void(void)>> funcs,
                             int rounds = 50);
    void setEnableTF32(bool state);
    bool getEnableTF32() const { return enableTF32; }

  private:
    void tune(const Graph &graph, bool profiling) const;

    void beginCudaGraphStreamCapture();
    tuple<cudaGraphExec_t, size_t> endCudaGraphStreamCapture();
};
} // namespace infini
