#pragma once
#include "core/runtime.h"
#include "cuda/cuda_common.h"
#include "nnet/dbg.h"

namespace infini {

class CudaRuntimeObj : public RuntimeObj {
  private:
    cudaStream_t stream;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    CudaPtr workspace;
    size_t workspaceSize;
    bool cudaGraphStatus; // Whether CUDA graph stream capture is enabled

  public:
    CudaRuntimeObj();
    virtual ~CudaRuntimeObj();
    string toString() const override;

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;
    // double runEvaluation(const Graph &graph, int nWarmups,
    //                      int nEvaluations) const;
    void sync() const;
    CudaPtr alloc(size_t size) override {
        void *ptr;
        // dbg(size);
        checkCudaError(cudaMalloc(&ptr, size));
        // printf("cuda malloc: %p %lu bytes\n", ptr, size);
        return ptr;
    }
    void dealloc(void *ptr) override { checkCudaError(cudaFree(ptr)); }
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

    bool isInCudaGraph() const { return cudaGraphStatus; }
    cudaStream_t getStream() const { return stream; }

    double timeWithCudaGraph(Graph graph);

  private:
    void tune(const Graph &graph, bool profiling) const;

    void beginCudaGraphStreamCapture();
    tuple<cudaGraphExec_t, size_t> endCudaGraphStreamCapture();
};
} // namespace infini
