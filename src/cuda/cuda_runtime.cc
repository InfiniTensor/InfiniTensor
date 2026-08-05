#include "cuda/cuda_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#ifdef INFINI_USE_NCCL
#include "cuda/nccl_communicator.h"
#endif
#include "operators/conv.h"
#include "operators/matmul.h"
#include <algorithm>
#include <cstdio>

void CHECK_CUDA_KERNEL_ERROR(infini::Operator op) {
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(kernelError)
                  << std::endl
                  << "Failed Operator: " << op->toString() << std::endl;
        exit(EXIT_FAILURE);
    }
}

namespace infini {

namespace {
void logCudaCleanupError(const char *operation, cudaError_t error) noexcept {
    if (error != cudaSuccess)
        std::fprintf(stderr, "%s failed: %s\n", operation,
                     cudaGetErrorString(error));
}

void logCudnnCleanupError(const char *operation, cudnnStatus_t error) noexcept {
    if (error != CUDNN_STATUS_SUCCESS)
        std::fprintf(stderr, "%s failed: %s\n", operation,
                     cudnnGetErrorString(error));
}

void logCublasCleanupError(const char *operation,
                           cublasStatus_t error) noexcept {
    if (error != CUBLAS_STATUS_SUCCESS)
        std::fprintf(stderr, "%s failed: %s\n", operation,
                     cublasGetErrorString(error));
}

void checkCublasStatus(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw Exception(std::string(operation) +
                        " failed: " + cublasGetErrorString(status));
}
} // namespace

bool CudaRuntimeObj::CapturedTensorState::operator==(
    const CapturedTensorState &other) const {
    return tensor == other.tensor && dtype == other.dtype &&
           shape == other.shape && tensorBytes == other.tensorBytes &&
           storageId == other.storageId &&
           storageOffset == other.storageOffset &&
           blobBytes == other.blobBytes && address == other.address;
}

bool CudaRuntimeObj::CapturedGraphState::operator==(
    const CapturedGraphState &other) const {
    return graphId == other.graphId && topologyEpoch == other.topologyEpoch &&
           tensors == other.tensors;
}

CudaRuntimeObj::CudaGraphCacheEntry::~CudaGraphCacheEntry() noexcept {
    if (instance)
        logCudaCleanupError("cudaGraphExecDestroy",
                            cudaGraphExecDestroy(instance));
    if (graph)
        logCudaCleanupError("cudaGraphDestroy", cudaGraphDestroy(graph));
}

CudaRuntimeObj::CudaRuntimeObj(int deviceId, size_t cudaGraphCacheCapacity)
    : RuntimeObj(Device::CUDA, deviceId),
      cudaGraphCacheCapacity(cudaGraphCacheCapacity) {
    IT_ASSERT(cudaGraphCacheCapacity > 0,
              "CUDA Graph cache capacity must be greater than zero");
    try {
        activateDevice();
        checkCudaError(cudaStreamCreate(&stream));
        checkCudnnError(cudnnCreate(&cudnn));
        checkCublasStatus(cublasCreate(&cublas), "cublasCreate");
        bindLibraryHandlesToStream();
        workspace = alloc(workspaceSize);
    } catch (...) {
        if (workspace)
            logCudaCleanupError("cudaFree(workspace)", cudaFree(workspace));
        if (cublas)
            logCublasCleanupError("cublasDestroy", cublasDestroy(cublas));
        if (cudnn)
            logCudnnCleanupError("cudnnDestroy", cudnnDestroy(cudnn));
        if (stream)
            logCudaCleanupError("cudaStreamDestroy", cudaStreamDestroy(stream));
        throw;
    }
}

CudaRuntimeObj::~CudaRuntimeObj() {
    std::lock_guard<std::recursive_mutex> executionLock(executionMutex);
    std::lock_guard<std::recursive_mutex> cacheLock(cacheMutex);
    logCudaCleanupError("cudaSetDevice", cudaSetDevice(deviceId));
    if (stream)
        logCudaCleanupError("cudaStreamSynchronize",
                            cudaStreamSynchronize(stream));
    clearCudaGraphCacheImpl();
    comm.reset();
    if (workspace) {
        logCudaCleanupError("cudaFree(workspace)", cudaFree(workspace));
        workspace = nullptr;
    }
    if (cublas) {
        logCublasCleanupError("cublasDestroy", cublasDestroy(cublas));
        cublas = nullptr;
    }
    if (cudnn) {
        logCudnnCleanupError("cudnnDestroy", cudnnDestroy(cudnn));
        cudnn = nullptr;
    }
    if (stream) {
        logCudaCleanupError("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void CudaRuntimeObj::activateDevice() const {
    checkCudaError(cudaSetDevice(deviceId));
}

void CudaRuntimeObj::bindLibraryHandlesToStream() const {
    checkCudnnError(cudnnSetStream(cudnn, stream));
    checkCublasStatus(cublasSetStream(cublas, stream), "cublasSetStream");
}

void CudaRuntimeObj::ensureExecutionStream() const {
    if (stream)
        return;
    checkCudaError(cudaStreamCreate(&stream));
    try {
        bindLibraryHandlesToStream();
    } catch (...) {
        logCudaCleanupError("cudaStreamDestroy after binding failure",
                            cudaStreamDestroy(stream));
        stream = nullptr;
        throw;
    }
}

CudaPtr CudaRuntimeObj::alloc(size_t size) {
    activateDevice();
    void *ptr = nullptr;
    checkCudaError(cudaMalloc(&ptr, size));
    return ptr;
}

void CudaRuntimeObj::dealloc(void *ptr) {
    activateDevice();
    checkCudaError(cudaFree(ptr));
}

void CudaRuntimeObj::copyBlobFromCPU(void *dst, const void *src,
                                     size_t bytes) const {
    activateDevice();
    checkCudaError(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void CudaRuntimeObj::copyBlobToCPU(void *dst, const void *src,
                                   size_t bytes) const {
    activateDevice();
    checkCudaError(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

void CudaRuntimeObj::copyBlobInsideRuntime(void *dst, const void *src,
                                           size_t bytes) const {
    activateDevice();
    checkCudaError(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

void CudaRuntimeObj::runWithoutSyncImpl(const Graph &graph,
                                        bool validate) const {
    IT_ASSERT(graph != nullptr, "Cannot run a null graph");
    if (validate)
        graph->validateMemory();
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        if (perfData) {
            ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);
            funcPtr(op, perfData, this);
        } else {
            kernel->compute(op, this);
        }
        checkCudaError(cudaGetLastError()) << op->toString();
    }
}

void CudaRuntimeObj::runWithoutSync(const Graph &graph) const {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    ensureExecutionStream();
    CUDAStream::Guard streamGuard(stream);
    runWithoutSyncImpl(graph, true);
}

CudaRuntimeObj::CapturedGraphState
CudaRuntimeObj::captureStateOf(const Graph &graph) const {
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

void CudaRuntimeObj::recoverExecutionStreamAfterFailure() noexcept {
    cudaGetLastError();
    if (stream)
        logCudaCleanupError("cudaStreamDestroy after capture failure",
                            cudaStreamDestroy(stream));
    stream = nullptr;
    auto status = cudaStreamCreate(&stream);
    if (status != cudaSuccess) {
        logCudaCleanupError("cudaStreamCreate after capture failure", status);
        cudaGetLastError();
        return;
    }
    const auto cudnnStatus = cudnnSetStream(cudnn, stream);
    const auto cublasStatus = cublasSetStream(cublas, stream);
    logCudnnCleanupError("cudnnSetStream after capture failure", cudnnStatus);
    logCublasCleanupError("cublasSetStream after capture failure",
                          cublasStatus);
    if (cudnnStatus != CUDNN_STATUS_SUCCESS ||
        cublasStatus != CUBLAS_STATUS_SUCCESS) {
        logCudaCleanupError("cudaStreamDestroy after rebinding failure",
                            cudaStreamDestroy(stream));
        stream = nullptr;
    }
    cudaGetLastError();
}

std::unique_ptr<CudaRuntimeObj::CudaGraphCacheEntry>
CudaRuntimeObj::captureGraph(const Graph &graph, CapturedGraphState state) {
    auto entry = std::make_unique<CudaGraphCacheEntry>(WRef<GraphObj>(graph),
                                                       std::move(state));
    bool captureStarted = false;
    try {
        checkCudaError(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        captureStarted = true;
        runWithoutSyncImpl(graph, false);
        auto endStatus = cudaStreamEndCapture(stream, &entry->graph);
        captureStarted = false;
        checkCudaError(endStatus);
        checkCudaError(cudaGraphInstantiate(&entry->instance, entry->graph,
                                            nullptr, nullptr, 0));
    } catch (...) {
        auto originalError = std::current_exception();
        if (captureStarted) {
            cudaGraph_t abandonedGraph = nullptr;
            auto endStatus = cudaStreamEndCapture(stream, &abandonedGraph);
            if (abandonedGraph)
                logCudaCleanupError("cudaGraphDestroy after capture failure",
                                    cudaGraphDestroy(abandonedGraph));
            if (endStatus != cudaSuccess)
                logCudaCleanupError("cudaStreamEndCapture after failure",
                                    endStatus);
        }
        recoverExecutionStreamAfterFailure();
        std::rethrow_exception(originalError);
    }
    return entry;
}

void CudaRuntimeObj::markActiveGraph(CudaGraphCache::iterator entry,
                                     size_t generation) {
    activeCudaGraphs.insert_or_assign((*entry)->state.graphId,
                                      ActiveCudaGraphState{entry, generation});
}

void CudaRuntimeObj::clearActiveGraph(
    const CudaGraphCacheEntry *entry) noexcept {
    const auto active = activeCudaGraphs.find(entry->state.graphId);
    if (active != activeCudaGraphs.end() &&
        active->second.entry->get() == entry)
        activeCudaGraphs.erase(active);
}

void CudaRuntimeObj::purgeExpiredGraphCacheEntries() noexcept {
    for (auto it = cudaGraphCache.begin(); it != cudaGraphCache.end();) {
        if ((*it)->owner.expired()) {
            clearActiveGraph(it->get());
            it = cudaGraphCache.erase(it);
        } else {
            ++it;
        }
    }
}

void CudaRuntimeObj::eraseGraphCacheEntries(uint64_t graphId) noexcept {
    activeCudaGraphs.erase(graphId);
    for (auto it = cudaGraphCache.begin(); it != cudaGraphCache.end();) {
        if ((*it)->state.graphId == graphId) {
            it = cudaGraphCache.erase(it);
        } else {
            ++it;
        }
    }
}

void CudaRuntimeObj::clearCudaGraphCacheImpl() noexcept {
    activeCudaGraphs.clear();
    cudaGraphCache.clear();
}

void CudaRuntimeObj::invalidateGraphCaptureCache(uint64_t graphId) noexcept {
    std::lock_guard<std::recursive_mutex> executionLock(executionMutex);
    std::lock_guard<std::recursive_mutex> cacheLock(cacheMutex);
    logCudaCleanupError("cudaSetDevice during CUDA Graph invalidation",
                        cudaSetDevice(deviceId));
    eraseGraphCacheEntries(graphId);
}

void CudaRuntimeObj::clearCudaGraphCache() {
    std::lock_guard<std::recursive_mutex> executionLock(executionMutex);
    std::lock_guard<std::recursive_mutex> cacheLock(cacheMutex);
    activateDevice();
    clearCudaGraphCacheImpl();
}

size_t CudaRuntimeObj::getCudaGraphCacheSize() const {
    std::lock_guard<std::recursive_mutex> lock(cacheMutex);
    return cudaGraphCache.size();
}

size_t CudaRuntimeObj::getCudaGraphCaptureCount() const {
    std::lock_guard<std::recursive_mutex> lock(cacheMutex);
    return cudaGraphCaptureCount;
}

void CudaRuntimeObj::runWithCudaGraph(const Graph &graph) {
    IT_ASSERT(graph != nullptr, "Cannot run a null graph");
    std::lock_guard<std::recursive_mutex> executionLock(executionMutex);
    activateDevice();
    ensureExecutionStream();
    CUDAStream::Guard streamGuard(stream);
    std::lock_guard<std::recursive_mutex> cacheLock(cacheMutex);

    const auto generation = graph->getCaptureGeneration();
    const auto graphId = graph->getCaptureStateId();
    const auto active = activeCudaGraphs.find(graphId);
    if (active != activeCudaGraphs.end() &&
        active->second.generation == generation) {
        auto cacheEntry = active->second.entry;
        auto *entry = cacheEntry->get();
        auto owner = entry->owner.lock();
        if (owner && owner.get() == graph.get()) {
            cudaGraphCache.splice(cudaGraphCache.begin(), cudaGraphCache,
                                  cacheEntry);
            try {
                checkCudaError(cudaGraphLaunch(entry->instance, stream));
                syncImpl();
            } catch (...) {
                auto originalError = std::current_exception();
                eraseGraphCacheEntries(graph->getCaptureStateId());
                recoverExecutionStreamAfterFailure();
                std::rethrow_exception(originalError);
            }
            return;
        }
    }

    purgeExpiredGraphCacheEntries();
    graph->validateMemory();
    auto state = captureStateOf(graph);
    auto hit = std::find_if(cudaGraphCache.begin(), cudaGraphCache.end(),
                            [&state, &graph](const auto &entry) {
                                auto owner = entry->owner.lock();
                                return owner && owner.get() == graph.get() &&
                                       entry->state == state;
                            });
    if (hit != cudaGraphCache.end()) {
        auto *entry = hit->get();
        cudaGraphCache.splice(cudaGraphCache.begin(), cudaGraphCache, hit);
        markActiveGraph(hit, generation);
        try {
            checkCudaError(cudaGraphLaunch(entry->instance, stream));
            syncImpl();
        } catch (...) {
            auto originalError = std::current_exception();
            eraseGraphCacheEntries(graph->getCaptureStateId());
            recoverExecutionStreamAfterFailure();
            std::rethrow_exception(originalError);
        }
        return;
    }

    auto entry = captureGraph(graph, std::move(state));
    IT_ASSERT(generation == graph->getCaptureGeneration(),
              "Graph changed while CUDA Graph capture was in progress");
    try {
        checkCudaError(cudaGraphLaunch(entry->instance, stream));
        syncImpl();
    } catch (...) {
        auto originalError = std::current_exception();
        recoverExecutionStreamAfterFailure();
        std::rethrow_exception(originalError);
    }
    ++cudaGraphCaptureCount;
    cudaGraphCache.emplace_front(std::move(entry));
    markActiveGraph(cudaGraphCache.begin(), generation);
    while (cudaGraphCache.size() > cudaGraphCacheCapacity) {
        clearActiveGraph(cudaGraphCache.back().get());
        cudaGraphCache.pop_back();
    }
}

void CudaRuntimeObj::tune(const Graph &graph, bool profiling) const {
    IT_ASSERT(graph != nullptr, "Cannot tune a null graph");
    graph->validateMemory();
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        PerfRecord record;
        if (!perfData) {
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else {
            record = perfData;
        }
        const double recordedTime = record->time;
        totalTime += recordedTime;
        kernel->computeFuncTune(perfKey, op, record, this);
        if (profiling) {
            ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);
            const double measuredTime =
                timeit([&]() { funcPtr(op, record, this); },
                       [&]() { syncImpl(); }, 1, 1);
            op->print();
            printf(" op_time on cuda %lf\n", measuredTime);
            totalTime += measuredTime;
            opTime[op->getOpType()] += measuredTime;
            opCnt[op->getOpType()]++;
        }
        checkCudaError(cudaGetLastError()) << op->toString();
    }
}

void CudaRuntimeObj::run(const Graph &graph, bool runTune,
                         bool profiling) const {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    ensureExecutionStream();
    CUDAStream::Guard streamGuard(stream);
    if (profiling)
        IT_TODO_HALT();
    if (runTune)
        tune(graph, profiling);
    else
        runWithoutSyncImpl(graph, true);
    syncImpl();
}

void CudaRuntimeObj::syncImpl() const {
    checkCudaError(cudaStreamSynchronize(stream));
}

void CudaRuntimeObj::sync() const {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    ensureExecutionStream();
    CUDAStream::Guard streamGuard(stream);
    syncImpl();
}

string CudaRuntimeObj::toString() const { return "CUDA Runtime"; }

void CudaRuntimeObj::initComm(const string &name, int worldSize, int rank) {
    std::lock_guard<std::recursive_mutex> lock(executionMutex);
    activateDevice();
    ensureExecutionStream();
    CUDAStream::Guard streamGuard(stream);
    IT_ASSERT(worldSize > 0);
    IT_ASSERT(rank >= 0);
    IT_ASSERT(rank < worldSize);
    IT_ASSERT(!comm) << "communicator is already initialized.";
#ifdef INFINI_USE_NCCL
    comm = std::make_unique<NcclCommunicatorObj>(name, worldSize, rank);
#else
    IT_TODO_HALT_MSG("Not compiled with NCCL.");
#endif
}

thread_local cudaStream_t CUDAStream::_stream = nullptr;
} // namespace infini
