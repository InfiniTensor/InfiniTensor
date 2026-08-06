#pragma once

#include "core/runtime.h"

#include <infini/rt.h>

#include <list>
#include <mutex>
#include <unordered_map>

namespace infini {

class InfiniRuntimeObj final : public RuntimeObj {
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

    struct GraphCacheEntry {
        WRef<GraphObj> owner;
        CapturedGraphState state;
        ::infini::rt::runtime::Graph graph{};
        ::infini::rt::runtime::GraphExec instance{};

        GraphCacheEntry(WRef<GraphObj> owner, CapturedGraphState state)
            : owner(std::move(owner)), state(std::move(state)) {}
    };

    using GraphCache = std::list<std::unique_ptr<GraphCacheEntry>>;

    struct ActiveGraphState {
        GraphCache::iterator entry;
        size_t generation;
    };

    ::infini::rt::Device runtimeDevice;
    mutable ::infini::rt::runtime::Stream stream{};
    size_t graphCacheCapacity;
    size_t graphCaptureCount = 0;
    GraphCache graphCache;
    std::unordered_map<uint64_t, ActiveGraphState> activeGraphs;
    std::unique_ptr<CommunicatorObj> communicator;
    mutable std::recursive_mutex executionMutex;
    mutable std::recursive_mutex cacheMutex;

  public:
    explicit InfiniRuntimeObj(const string &deviceType, int deviceId = 0,
                              size_t graphCacheCapacity = 16);
    ~InfiniRuntimeObj() override;

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const override;
    void runWithGraph(const Graph &graph) override;
    void clearGraphCache() override;
    size_t getGraphCacheSize() const override;
    size_t getGraphCaptureCount() const override;
    void invalidateGraphCaptureCache(uint64_t graphId) noexcept override;

    void *alloc(size_t size) override;
    void dealloc(void *ptr) override;
    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override;
    void copyBlobToCPU(void *dst, const void *src, size_t bytes) const override;
    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override;

    void sync() const;
    void *getStream() const { return reinterpret_cast<void *>(stream); }
    const ::infini::rt::Device &getInfiniDevice() const {
        return runtimeDevice;
    }
    string toString() const override;

    void initComm(const string &name, int worldSize, int rank) override;
    CommunicatorObj &getCommunicator() const override;

  private:
    void activateDevice() const;
    void ensureExecutionStream() const;
    void runWithoutSyncImpl(const Graph &graph, bool validate) const;
    void syncImpl() const;
    CapturedGraphState captureStateOf(const Graph &graph) const;
    std::unique_ptr<GraphCacheEntry> captureGraph(const Graph &graph,
                                                  CapturedGraphState state);
    void recoverExecutionStreamAfterFailure() noexcept;
    void destroyGraphEntry(GraphCacheEntry &entry) noexcept;
    void clearGraphCacheImpl() noexcept;
    void eraseGraphCacheEntries(uint64_t graphId) noexcept;
    void purgeExpiredGraphCacheEntries() noexcept;
    void markActiveGraph(GraphCache::iterator entry, size_t generation);
    void clearActiveGraph(const GraphCacheEntry *entry) noexcept;
};

} // namespace infini
