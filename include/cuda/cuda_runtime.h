#pragma once
#include "core/runtime.h"
#include "cuda/cuda_common.h"
#include <list>
#include <mutex>
#include <unordered_map>
#ifdef INFINI_USE_NCCL
#include "cuda/nccl_communicator.h"
#endif

namespace infini {

class CudaRuntimeObj : public RuntimeObj {
  private:
    struct CapturedTensorState {
        const TensorObj *tensor;
        int dtype;
        vector<int> shape;
        size_t tensorBytes;
        uint64_t storageId;
        size_t storageOffset;
        size_t blobBytes;
        const void *address;

        bool operator==(const CapturedTensorState &other) const;
    };

    struct CapturedGraphState {
        uint64_t graphId;
        size_t topologyEpoch;
        vector<CapturedTensorState> tensors;

        bool operator==(const CapturedGraphState &other) const;
    };

    struct CudaGraphCacheEntry {
        WRef<GraphObj> owner;
        CapturedGraphState state;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t instance = nullptr;

        CudaGraphCacheEntry(WRef<GraphObj> owner, CapturedGraphState state)
            : owner(std::move(owner)), state(std::move(state)) {}
        ~CudaGraphCacheEntry() noexcept;
        CudaGraphCacheEntry(const CudaGraphCacheEntry &) = delete;
        CudaGraphCacheEntry &operator=(const CudaGraphCacheEntry &) = delete;
    };

    using CudaGraphCache = std::list<std::unique_ptr<CudaGraphCacheEntry>>;

    struct ActiveCudaGraphState {
        CudaGraphCache::iterator entry;
        size_t generation;
    };

    cudnnHandle_t cudnn = nullptr;
    cublasHandle_t cublas = nullptr;
    std::unique_ptr<CommunicatorObj> comm;
    CudaPtr workspace = nullptr;
    size_t workspaceSize = 7ll << 30;
    mutable cudaStream_t stream = nullptr;
    size_t cudaGraphCacheCapacity;
    size_t cudaGraphCaptureCount = 0;
    CudaGraphCache cudaGraphCache;
    std::unordered_map<uint64_t, ActiveCudaGraphState> activeCudaGraphs;
    mutable std::recursive_mutex executionMutex;
    mutable std::recursive_mutex cacheMutex;

  public:
    explicit CudaRuntimeObj(int deviceId = 0,
                            size_t cudaGraphCacheCapacity = 16);
    ~CudaRuntimeObj() override;
    string toString() const override;

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const override;
    // double runEvaluation(const Graph &graph, int nWarmups,
    //                      int nEvaluations) const;
    void sync() const;
    CudaPtr alloc(size_t size) override;
    void dealloc(void *ptr) override;
    cudnnHandle_t cudnnHandle() const { return cudnn; }
    cublasHandle_t cublasHandle() const { return cublas; }
    size_t getWorkspaceSize() const { return workspaceSize; }
    CudaPtr getWorkspace(size_t size) const {
        IT_ASSERT(size <= workspaceSize);
        return workspace;
    }

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override;

    void copyBlobToCPU(void *dst, const void *src, size_t bytes) const override;

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override;

    void runWithoutSync(const Graph &graph) const;

    void runWithCudaGraph(const Graph &graph);

    void clearCudaGraphCache();
    size_t getCudaGraphCacheSize() const;
    size_t getCudaGraphCaptureCount() const;
    void invalidateGraphCaptureCache(uint64_t graphId) noexcept override;

    // init communicator
    void initComm(const string &name, int worldSize, int rank) final;

    CommunicatorObj &getCommunicator() const final { return *comm; }

  private:
    void tune(const Graph &graph, bool profiling) const;
    void activateDevice() const;
    void bindLibraryHandlesToStream() const;
    void ensureExecutionStream() const;
    void runWithoutSyncImpl(const Graph &graph, bool validate) const;
    void syncImpl() const;
    CapturedGraphState captureStateOf(const Graph &graph) const;
    std::unique_ptr<CudaGraphCacheEntry> captureGraph(const Graph &graph,
                                                      CapturedGraphState state);
    void recoverExecutionStreamAfterFailure() noexcept;
    void clearCudaGraphCacheImpl() noexcept;
    void eraseGraphCacheEntries(uint64_t graphId) noexcept;
    void purgeExpiredGraphCacheEntries() noexcept;
    void markActiveGraph(CudaGraphCache::iterator entry, size_t generation);
    void clearActiveGraph(const CudaGraphCacheEntry *entry) noexcept;
};
} // namespace infini
