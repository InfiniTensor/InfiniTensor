#pragma once
#include "core/runtime.h"
#include "cuda/cuda_common.h"

namespace infini {

class CudaRuntimeObj : public RuntimeObj {
  private:
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    CudaPtr workspace;
    size_t workspaceSize;

  public:
    CudaRuntimeObj() : RuntimeObj(Device::CUDA) {

        checkCudnnError(cudnnCreate(&cudnn));
        checkCublasError(cublasCreate(&cublas));
        // 10GB for Longformer
        // size_t longformerNum = 3lu * (1 << 30);
        workspaceSize = 7ll << 30; // 7 GB
        workspace = alloc(workspaceSize);
    }
    virtual ~CudaRuntimeObj() {
        try {
            dealloc(workspace);
            checkCudnnError(cudnnDestroy(cudnn));
            checkCublasError(cublasDestroy(cublas));
        } catch (const std::exception &e) {
            std::cerr << "Error in ~CudaRuntimeObj: " << e.what() << std::endl;
        }
    }
    string toString() const override;

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;
    // double runEvaluation(const Graph &graph, int nWarmups,
    //                      int nEvaluations) const;
    void sync() const;
    CudaPtr alloc(size_t size) override {
        void *ptr;
        checkCudaError(cudaMalloc(&ptr, size));
        // printf("cuda malloc: %p %lu bytes\n", ptr, size);
        return ptr;
    }
    void dealloc(void *ptr) override { printf("cuda dealloc: %p\n", ptr); checkCudaError(cudaFree(ptr)); }
    cudnnHandle_t cudnnHandle() const { return cudnn; }
    cublasHandle_t cublasHandle() const { return cublas; }
    size_t getWorkspaceSize() const { return workspaceSize; }
    CudaPtr getWorkspace(size_t size) const {
        IT_ASSERT(size <= workspaceSize);
        return workspace;
    }

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

  private:
    void tune(const Graph &graph, bool profiling) const;
};
} // namespace infini
