#include "core/infini_runtime.h"

#include "core/blob.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace infini {
namespace {

using RtError = ::infini::rt::runtime::Error;

void checkInfiniRt(RtError error, const char *operation) {
    if (error != ::infini::rt::runtime::kSuccess) {
        throw Exception(string(operation) + " failed with InfiniRT status " +
                        std::to_string(static_cast<int64_t>(error)));
    }
}

void logInfiniRtError(const char *operation, RtError error) noexcept {
    if (error != ::infini::rt::runtime::kSuccess) {
        std::fprintf(stderr, "%s failed with InfiniRT status %lld\n", operation,
                     static_cast<long long>(error));
    }
}

template <auto... devices>
bool containsDevice(::infini::rt::Device::Type device,
                    ::infini::rt::List<devices...>) {
    return (false || ... || (device == devices));
}

bool isRuntimeDeviceEnabled(::infini::rt::Device::Type device) {
    return containsDevice(device,
                          ::infini::rt::ActiveDevices<InfiniRuntimeObj>{});
}

::infini::rt::Device parseRuntimeDevice(const string &deviceType,
                                        int deviceId) {
    IT_ASSERT(deviceId >= 0, "Device index must be non-negative");
    try {
        auto type = ::infini::rt::Device::TypeFromString(deviceType);
        IT_ASSERT(isRuntimeDeviceEnabled(type),
                  "InfiniRT was not built for device type '" + deviceType +
                      "'");
        return {type, deviceId};
    } catch (const std::out_of_range &) {
        IT_ASSERT(false, "Unknown InfiniRT device type '" + deviceType + "'");
    }
    return {};
}

} // namespace

#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
bool InfiniRuntimeObj::CapturedTensorState::operator==(
    const CapturedTensorState &other) const {
    return tensor == other.tensor && dtype == other.dtype &&
           shape == other.shape && tensorBytes == other.tensorBytes &&
           storageId == other.storageId &&
           storageOffset == other.storageOffset &&
           blobBytes == other.blobBytes && address == other.address;
}

bool InfiniRuntimeObj::CapturedGraphState::operator==(
    const CapturedGraphState &other) const {
    return graphId == other.graphId && topologyEpoch == other.topologyEpoch &&
           tensors == other.tensors;
}
#endif

InfiniRuntimeObj::InfiniRuntimeObj(const string &deviceType, int deviceId,
                                   size_t graphCacheCapacity)
    : RuntimeObj(ExecutionProvider::Infini, deviceId),
      runtimeDevice(parseRuntimeDevice(deviceType, deviceId))
#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
      ,
      graphCacheCapacity(graphCacheCapacity)
#endif
{
#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
    IT_ASSERT(graphCacheCapacity > 0,
              "Graph cache capacity must be greater than zero");
#else
    (void)graphCacheCapacity;
#endif
    activateDevice();
    checkInfiniRt(::infini::rt::runtime::StreamCreate(&stream),
                  "InfiniRT StreamCreate");
}

InfiniRuntimeObj::~InfiniRuntimeObj() {
    std::lock_guard<std::recursive_mutex> executionLock(executionMutex);
#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
    std::lock_guard<std::recursive_mutex> cacheLock(cacheMutex);
#endif
    ::infini::rt::set_runtime_device_type(runtimeDevice.type());
    logInfiniRtError("InfiniRT SetDevice during cleanup",
                     ::infini::rt::runtime::SetDevice(deviceId));
    if (stream) {
        logInfiniRtError("InfiniRT StreamSynchronize during cleanup",
                         ::infini::rt::runtime::StreamSynchronize(stream));
    }
#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
    clearGraphCacheImpl();
#endif
    if (stream) {
        logInfiniRtError("InfiniRT StreamDestroy",
                         ::infini::rt::runtime::StreamDestroy(stream));
        stream = {};
    }
}

void InfiniRuntimeObj::activateDevice() const {
    ::infini::rt::set_runtime_device_type(runtimeDevice.type());
    checkInfiniRt(::infini::rt::runtime::SetDevice(deviceId),
                  "InfiniRT SetDevice");
}

void InfiniRuntimeObj::ensureExecutionStream() const {
    if (stream)
        return;
    checkInfiniRt(::infini::rt::runtime::StreamCreate(&stream),
                  "InfiniRT StreamCreate");
}

void *InfiniRuntimeObj::alloc(size_t size) {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    void *ptr = nullptr;
    checkInfiniRt(::infini::rt::runtime::Malloc(&ptr, size), "InfiniRT Malloc");
    return ptr;
}

void InfiniRuntimeObj::dealloc(void *ptr) {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    checkInfiniRt(::infini::rt::runtime::Free(ptr), "InfiniRT Free");
}

std::shared_ptr<void> InfiniRuntimeObj::acquireWorkspace(size_t bytes) const {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
    if (captureWorkspaces && replayCaptureWorkspaces) {
        IT_ASSERT(captureWorkspaceCursor < captureWorkspaces->size(),
                  "Capture requested more workspace buffers than warmup");
        const auto &workspace = (*captureWorkspaces)[captureWorkspaceCursor++];
        IT_ASSERT(workspace.bytes == bytes,
                  "Capture workspace size differs from warmup");
        return workspace.data;
    }
#endif
    activateDevice();
    void *ptr = nullptr;
    checkInfiniRt(
        ::infini::rt::runtime::Malloc(&ptr, std::max(bytes, size_t{1})),
        "InfiniRT workspace Malloc");
    // A Blob holds a strong runtime reference. Putting Blobs inside the
    // runtime's graph cache would create an ownership cycle, so capture the
    // device identity directly in this allocation's non-throwing deleter.
    auto data = std::shared_ptr<void>(ptr, [device = runtimeDevice](void *p) {
        ::infini::rt::set_runtime_device_type(device.type());
        const auto status = ::infini::rt::runtime::SetDevice(device.index());
        logInfiniRtError("InfiniRT SetDevice during workspace cleanup", status);
        if (status == ::infini::rt::runtime::kSuccess)
            logInfiniRtError("InfiniRT workspace Free",
                             ::infini::rt::runtime::Free(p));
    });
#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
    if (captureWorkspaces)
        captureWorkspaces->push_back(CaptureWorkspace{bytes, data});
#endif
    return data;
}

void InfiniRuntimeObj::copyBlobFromCPU(void *dst, const void *src,
                                       size_t bytes) const {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    checkInfiniRt(
        ::infini::rt::runtime::Memcpy(
            dst, src, bytes, ::infini::rt::runtime::kMemcpyHostToDevice),
        "InfiniRT host-to-device Memcpy");
}

void InfiniRuntimeObj::copyBlobToCPU(void *dst, const void *src,
                                     size_t bytes) const {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    checkInfiniRt(
        ::infini::rt::runtime::Memcpy(
            dst, src, bytes, ::infini::rt::runtime::kMemcpyDeviceToHost),
        "InfiniRT device-to-host Memcpy");
}

void InfiniRuntimeObj::copyBlobInsideRuntime(void *dst, const void *src,
                                             size_t bytes) const {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    // The pinned CPU backend does not implement MemcpyAsync.
    if (runtimeDevice.type() == ::infini::rt::Device::Type::kCpu) {
        checkInfiniRt(
            ::infini::rt::runtime::Memcpy(
                dst, src, bytes, ::infini::rt::runtime::kMemcpyDeviceToDevice),
            "InfiniRT CPU device-to-device Memcpy");
        return;
    }
    ensureExecutionStream();
    // Keep graph copies ordered with kernels and visible to stream capture.
    checkInfiniRt(::infini::rt::runtime::MemcpyAsync(
                      dst, src, bytes,
                      ::infini::rt::runtime::kMemcpyDeviceToDevice, stream),
                  "InfiniRT device-to-device MemcpyAsync");
    // Outside capture, callers may immediately read the result or release
    // either allocation. Preserve the synchronous copy contract explicitly;
    // device Free/host copies need not synchronize this execution stream.
#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
    if (!streamCaptureActive)
#endif
        syncImpl();
}

void InfiniRuntimeObj::runWithoutSyncImpl(const Graph &graph,
                                          bool validate) const {
    IT_ASSERT(graph != nullptr, "Cannot run a null graph");
    if (validate)
        graph->validateMemory();
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    for (auto &op : graph->getOperators()) {
        const auto kernelAttrs = KernelAttrs{ExecutionProvider::Infini,
                                             op->getOpType().underlying()};
        auto *kernel = kernelRegistry.getKernel(kernelAttrs);
        const auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        if (perfData) {
            kernel->getComputeFunc(perfKey)(op, perfData, this);
        } else {
            kernel->compute(op, this);
        }
    }
}

#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
InfiniRuntimeObj::CapturedGraphState
InfiniRuntimeObj::captureStateOf(const Graph &graph) const {
    CapturedGraphState state{
        graph->getCaptureStateId(), graph->getTopologyEpoch(), {}};
    state.tensors.reserve(graph->getTensors().size());
    for (const auto &tensor : graph->getTensors()) {
        const auto &blob = tensor->getDataBlob();
        IT_ASSERT(blob != nullptr, "Cannot capture a Tensor without memory");
        state.tensors.emplace_back(CapturedTensorState{
            tensor.get(), tensor->getDTypeIndex(), tensor->getDims(),
            tensor->getBytes(), blob->getStorageId(), blob->getStorageOffset(),
            blob->getBytes(), tensor->getRawDataPtr<const void *>()});
    }
    return state;
}

void InfiniRuntimeObj::recoverExecutionStreamAfterFailure() noexcept {
    if (stream) {
        logInfiniRtError("InfiniRT StreamDestroy after capture failure",
                         ::infini::rt::runtime::StreamDestroy(stream));
    }
    stream = {};
    logInfiniRtError("InfiniRT StreamCreate after capture failure",
                     ::infini::rt::runtime::StreamCreate(&stream));
}

std::unique_ptr<InfiniRuntimeObj::GraphCacheEntry>
InfiniRuntimeObj::captureGraph(const Graph &graph, CapturedGraphState state) {
    auto entry = std::make_unique<GraphCacheEntry>(WRef<GraphObj>(graph),
                                                   std::move(state));
    bool captureStarted = false;
    IT_ASSERT(captureWorkspaces == nullptr, "Nested graph preparation");
    captureWorkspaces = &entry->workspaces;
    captureWorkspaceCursor = 0;
    replayCaptureWorkspaces = false;
    try {
        // Prepare both InfiniOps descriptors and scratch storage outside
        // capture. Each captured graph owns its own workspace allocations.
        runWithoutSyncImpl(graph, false);
        syncImpl();
        replayCaptureWorkspaces = true;
        checkInfiniRt(::infini::rt::runtime::StreamBeginCapture(
                          stream, ::infini::rt::runtime::StreamCaptureMode::
                                      kStreamCaptureModeThreadLocal),
                      "InfiniRT StreamBeginCapture");
        captureStarted = true;
        streamCaptureActive = true;
        runWithoutSyncImpl(graph, false);
        IT_ASSERT(captureWorkspaceCursor == entry->workspaces.size(),
                  "Capture requested fewer workspace buffers than warmup");
        auto endStatus =
            ::infini::rt::runtime::StreamEndCapture(stream, &entry->graph);
        captureStarted = false;
        streamCaptureActive = false;
        checkInfiniRt(endStatus, "InfiniRT StreamEndCapture");
        checkInfiniRt(::infini::rt::runtime::GraphInstantiate(&entry->instance,
                                                              entry->graph),
                      "InfiniRT GraphInstantiate");
    } catch (...) {
        captureWorkspaces = nullptr;
        streamCaptureActive = false;
        replayCaptureWorkspaces = false;
        auto originalError = std::current_exception();
        if (captureStarted) {
            ::infini::rt::runtime::Graph abandonedGraph{};
            auto endStatus = ::infini::rt::runtime::StreamEndCapture(
                stream, &abandonedGraph);
            if (abandonedGraph) {
                logInfiniRtError(
                    "InfiniRT GraphDestroy after capture failure",
                    ::infini::rt::runtime::GraphDestroy(abandonedGraph));
            }
            logInfiniRtError("InfiniRT StreamEndCapture after failure",
                             endStatus);
        }
        if (entry->instance) {
            logInfiniRtError(
                "InfiniRT GraphExecDestroy after failure",
                ::infini::rt::runtime::GraphExecDestroy(entry->instance));
            entry->instance = {};
        }
        if (entry->graph) {
            logInfiniRtError("InfiniRT GraphDestroy after failure",
                             ::infini::rt::runtime::GraphDestroy(entry->graph));
            entry->graph = {};
        }
        recoverExecutionStreamAfterFailure();
        std::rethrow_exception(originalError);
    }
    captureWorkspaces = nullptr;
    replayCaptureWorkspaces = false;
    return entry;
}

void InfiniRuntimeObj::destroyGraphEntry(GraphCacheEntry &entry) noexcept {
    if (entry.instance) {
        logInfiniRtError(
            "InfiniRT GraphExecDestroy",
            ::infini::rt::runtime::GraphExecDestroy(entry.instance));
        entry.instance = {};
    }
    if (entry.graph) {
        logInfiniRtError("InfiniRT GraphDestroy",
                         ::infini::rt::runtime::GraphDestroy(entry.graph));
        entry.graph = {};
    }
}

void InfiniRuntimeObj::markActiveGraph(GraphCache::iterator entry,
                                       size_t generation) {
    activeGraphs.insert_or_assign((*entry)->state.graphId,
                                  ActiveGraphState{entry, generation});
}

void InfiniRuntimeObj::clearActiveGraph(const GraphCacheEntry *entry) noexcept {
    const auto active = activeGraphs.find(entry->state.graphId);
    if (active != activeGraphs.end() && active->second.entry->get() == entry) {
        activeGraphs.erase(active);
    }
}

void InfiniRuntimeObj::purgeExpiredGraphCacheEntries() noexcept {
    for (auto it = graphCache.begin(); it != graphCache.end();) {
        if ((*it)->owner.expired()) {
            clearActiveGraph(it->get());
            destroyGraphEntry(**it);
            it = graphCache.erase(it);
        } else {
            ++it;
        }
    }
}

void InfiniRuntimeObj::eraseGraphCacheEntries(uint64_t graphId) noexcept {
    activeGraphs.erase(graphId);
    for (auto it = graphCache.begin(); it != graphCache.end();) {
        if ((*it)->state.graphId == graphId) {
            destroyGraphEntry(**it);
            it = graphCache.erase(it);
        } else {
            ++it;
        }
    }
}

void InfiniRuntimeObj::clearGraphCacheImpl() noexcept {
    activeGraphs.clear();
    for (auto &entry : graphCache)
        destroyGraphEntry(*entry);
    graphCache.clear();
}

void InfiniRuntimeObj::invalidateGraphCaptureCache(uint64_t graphId) noexcept {
    std::lock_guard<std::recursive_mutex> executionLock(executionMutex);
    std::lock_guard<std::recursive_mutex> cacheLock(cacheMutex);
    ::infini::rt::set_runtime_device_type(runtimeDevice.type());
    logInfiniRtError("InfiniRT SetDevice during graph invalidation",
                     ::infini::rt::runtime::SetDevice(deviceId));
    eraseGraphCacheEntries(graphId);
}

void InfiniRuntimeObj::clearGraphCache() {
    std::lock_guard<std::recursive_mutex> executionLock(executionMutex);
    std::lock_guard<std::recursive_mutex> cacheLock(cacheMutex);
    activateDevice();
    clearGraphCacheImpl();
}

size_t InfiniRuntimeObj::getGraphCacheSize() const {
    std::lock_guard<std::recursive_mutex> lock(cacheMutex);
    return graphCache.size();
}

size_t InfiniRuntimeObj::getGraphCaptureCount() const {
    std::lock_guard<std::recursive_mutex> lock(cacheMutex);
    return graphCaptureCount;
}

void InfiniRuntimeObj::runWithGraph(const Graph &graph) {
    IT_ASSERT(graph != nullptr, "Cannot run a null graph");
    std::lock_guard<std::recursive_mutex> executionLock(executionMutex);
    activateDevice();
    ensureExecutionStream();
    std::lock_guard<std::recursive_mutex> cacheLock(cacheMutex);

    const auto generation = graph->getCaptureGeneration();
    const auto graphId = graph->getCaptureStateId();
    const auto active = activeGraphs.find(graphId);
    if (active != activeGraphs.end() &&
        active->second.generation == generation) {
        auto cacheEntry = active->second.entry;
        auto *entry = cacheEntry->get();
        auto owner = entry->owner.lock();
        if (owner && owner.get() == graph.get()) {
            graphCache.splice(graphCache.begin(), graphCache, cacheEntry);
            try {
                checkInfiniRt(
                    ::infini::rt::runtime::GraphLaunch(entry->instance, stream),
                    "InfiniRT GraphLaunch");
                syncImpl();
            } catch (...) {
                auto originalError = std::current_exception();
                eraseGraphCacheEntries(graphId);
                recoverExecutionStreamAfterFailure();
                std::rethrow_exception(originalError);
            }
            return;
        }
    }

    purgeExpiredGraphCacheEntries();
    graph->validateMemory();
    auto state = captureStateOf(graph);
    auto hit = std::find_if(graphCache.begin(), graphCache.end(),
                            [&state, &graph](const auto &entry) {
                                auto owner = entry->owner.lock();
                                return owner && owner.get() == graph.get() &&
                                       entry->state == state;
                            });
    if (hit != graphCache.end()) {
        auto *entry = hit->get();
        graphCache.splice(graphCache.begin(), graphCache, hit);
        markActiveGraph(hit, generation);
        try {
            checkInfiniRt(
                ::infini::rt::runtime::GraphLaunch(entry->instance, stream),
                "InfiniRT GraphLaunch");
            syncImpl();
        } catch (...) {
            auto originalError = std::current_exception();
            eraseGraphCacheEntries(graphId);
            recoverExecutionStreamAfterFailure();
            std::rethrow_exception(originalError);
        }
        return;
    }

    auto entry = captureGraph(graph, std::move(state));
    if (generation != graph->getCaptureGeneration()) {
        destroyGraphEntry(*entry);
        IT_ASSERT(false, "Graph changed while capture was in progress");
    }
    try {
        checkInfiniRt(
            ::infini::rt::runtime::GraphLaunch(entry->instance, stream),
            "InfiniRT GraphLaunch");
        syncImpl();
    } catch (...) {
        auto originalError = std::current_exception();
        destroyGraphEntry(*entry);
        recoverExecutionStreamAfterFailure();
        std::rethrow_exception(originalError);
    }
    ++graphCaptureCount;
    graphCache.emplace_front(std::move(entry));
    markActiveGraph(graphCache.begin(), generation);
    while (graphCache.size() > graphCacheCapacity) {
        clearActiveGraph(graphCache.back().get());
        destroyGraphEntry(*graphCache.back());
        graphCache.pop_back();
    }
}
#endif

void InfiniRuntimeObj::run(const Graph &graph, bool tune,
                           bool profiling) const {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    ensureExecutionStream();
    IT_ASSERT(!profiling, "Infini runtime profiling is not implemented yet");
    graph->validateMemory();

    if (!tune) {
        runWithoutSyncImpl(graph, false);
        syncImpl();
        return;
    }

    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    for (auto &op : graph->getOperators()) {
        const auto kernelAttrs = KernelAttrs{ExecutionProvider::Infini,
                                             op->getOpType().underlying()};
        auto *kernel = kernelRegistry.getKernel(kernelAttrs);
        const auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        if (!perfData) {
            perfData = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, perfData);
        }
        kernel->computeFuncTune(perfKey, op, perfData, this);
        kernel->getComputeFunc(perfKey)(op, perfData, this);
    }
    syncImpl();
}

void InfiniRuntimeObj::syncImpl() const {
    checkInfiniRt(::infini::rt::runtime::StreamSynchronize(stream),
                  "InfiniRT StreamSynchronize");
}

void InfiniRuntimeObj::sync() const {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    ensureExecutionStream();
    syncImpl();
}

string InfiniRuntimeObj::toString() const {
    return "Infini Runtime (" + runtimeDevice.ToString() + ")";
}

} // namespace infini
