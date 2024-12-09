#pragma once
#include "core/runtime.h"
#include "cuda/cuda_common.h"
#ifdef INFINI_USE_NCCL
#include "cuda/nccl_communicator.h"
#endif

namespace infini {

class CudaRuntimeObj : public RuntimeObj {
  private:
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    std::unique_ptr<CommunicatorObj> comm;
    CudaPtr workspace;
    size_t workspaceSize;
    bool isCudaGraphCreated;
    cudaGraph_t cudaGraph;
    cudaGraphExec_t cudaGraphInstance;

  public:
    explicit CudaRuntimeObj(int deviceId = 0)
        : RuntimeObj(Device::CUDA, deviceId) {

        checkCudaError(cudaSetDevice(deviceId));
        checkCudnnError(cudnnCreate(&cudnn));
        checkCublasError(cublasCreate(&cublas));
        // 10GB for Longformer
        // size_t longformerNum = 3lu * (1 << 30);
        workspaceSize = 7ll << 30; // 7 GB
        workspace = alloc(workspaceSize);
        isCudaGraphCreated = false;
        CUDAStream::Init();
    }
    virtual ~CudaRuntimeObj() {
        try {
            if (isCudaGraphCreated) {
                checkCudaError(cudaGraphExecDestroy(cudaGraphInstance));
                checkCudaError(cudaGraphDestroy(cudaGraph));
                CUDAStream::destroyStream();
            }
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

    void *getCurrentStream() const override {
        return CUDAStream::getCurrentStream();
    }

    // double runEvaluation(const Graph &graph, int nWarmups,
    //                      int nEvaluations) const;
    void sync() const;
    CudaPtr alloc(size_t size) override {
        void *ptr;
        checkCudaError(cudaMalloc(&ptr, size));
        // printf("cuda malloc: %p %lu bytes\n", ptr, size);
        return ptr;
    }
    void dealloc(void *ptr) override { checkCudaError(cudaFree(ptr)); }
    cudnnHandle_t cudnnHandle() const { return cudnn; }
    cublasHandle_t cublasHandle() const { return cublas; }
    size_t getWorkspaceSize() const override { return workspaceSize; }
    CudaPtr getWorkspace(size_t size) const override {
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
        checkCudaError(cudaMemcpyAsync(dst, src, bytes,
                                       cudaMemcpyDeviceToDevice,
                                       CUDAStream::getCurrentStream()));
    }

    void runWithoutSync(const Graph &graph) const;

    void runWithCudaGraph(const Graph &graph);

    // init communicator
    void initComm(const string &name, int worldSize, int rank) final;

    CommunicatorObj &getCommunicator() const final { return *comm; }

  private:
    void tune(const Graph &graph, bool profiling) const;
};
} // namespace infini
