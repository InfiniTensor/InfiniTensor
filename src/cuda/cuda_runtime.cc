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
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

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

struct OperatorProfileEntry {
    int index;
    std::string type;
    std::string dtype;
    std::string inputShape;
    std::string outputShape;
    double timeMs;
};

FILE *profileLog() {
    static FILE *logFile = []() {
        const char *path = std::getenv("INFINITENSOR_PROFILE_LOG_PATH");
        if (path == nullptr || path[0] == '\0')
            return stdout;
        FILE *file = std::fopen(path, "w");
        if (file == nullptr) {
            std::fprintf(stderr,
                         "[Profile][warning] failed to open '%s'; using "
                         "stdout\n",
                         path);
            return stdout;
        }
        std::setvbuf(file, nullptr, _IOLBF, 0);
        return file;
    }();
    return logFile;
}

std::string shapeString(const Tensor &tensor) {
    std::string result = "[";
    const auto &dims = tensor->getDims();
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0)
            result += ",";
        result += std::to_string(dims[i]);
    }
    result += "]";
    return result;
}

void printProfileReport(const std::vector<OperatorProfileEntry> &entries,
                        double totalTime) {
    FILE *const log = profileLog();
    std::fprintf(log, "[Profile][meta] op_count=%zu total_ms=%.6f\n",
                 entries.size(), totalTime);

    for (const auto &entry : entries) {
        const double percent = totalTime > 0 ? entry.timeMs / totalTime * 100 : 0;
        std::fprintf(log,
                     "[Profile][op] idx=%d type=%s dtype=%s input=%s "
                     "output=%s time_ms=%.6f percent=%.3f\n",
                     entry.index, entry.type.c_str(), entry.dtype.c_str(),
                     entry.inputShape.c_str(), entry.outputShape.c_str(),
                     entry.timeMs, percent);
    }

    auto hotspots = entries;
    std::sort(hotspots.begin(), hotspots.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.timeMs > rhs.timeMs;
              });
    const size_t hotspotCount = std::min<size_t>(20, hotspots.size());
    for (size_t rank = 0; rank < hotspotCount; ++rank) {
        const auto &entry = hotspots[rank];
        const double percent = totalTime > 0 ? entry.timeMs / totalTime * 100 : 0;
        std::fprintf(log,
                     "[Profile][hotspot] rank=%zu idx=%d type=%s dtype=%s "
                     "time_ms=%.6f percent=%.3f\n",
                     rank + 1, entry.index, entry.type.c_str(),
                     entry.dtype.c_str(), entry.timeMs, percent);
    }

    struct TypeSummary {
        std::string type;
        int count = 0;
        double totalMs = 0;
    };
    std::map<std::string, TypeSummary> summaryByType;
    for (const auto &entry : entries) {
        auto &[type, count, totalMs] = summaryByType[entry.type];
        type = entry.type;
        ++count;
        totalMs += entry.timeMs;
    }
    std::vector<TypeSummary> summaries;
    for (const auto &[_, summary] : summaryByType)
        summaries.emplace_back(summary);
    std::sort(summaries.begin(), summaries.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.totalMs > rhs.totalMs;
              });
    for (const auto &summary : summaries) {
        const double percent = totalTime > 0 ? summary.totalMs / totalTime * 100 : 0;
        std::fprintf(log,
                     "[Profile][summary] type=%s count=%d total_ms=%.6f "
                     "percent=%.3f mean_ms=%.6f\n",
                     summary.type.c_str(), summary.count, summary.totalMs,
                     percent, summary.totalMs / summary.count);
    }
    std::fflush(log);
}

} // namespace

void CudaRuntimeObj::runWithoutSync(const Graph &graph, bool profiling) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::vector<OperatorProfileEntry> profileEntries;
    int opIdx = 0;
    const auto &ops = graph->getOperators();
    for (size_t i = 0; i < ops.size(); ++i) {
        const auto &op = ops[i];
        if (!profiling && !graph->usesNaiveAllocator() &&
            convNhwcRegionEnabled() && op->getOpType() == OpType::Conv) {
            const size_t consumed = getConvNhwcCache().runRegion(ops, i, this);
            if (consumed) {
                checkCudaError(cudaGetLastError()) << op->toString();
                i += consumed - 1;
                continue;
            }
        }
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        if (profiling) {
            double t = timeit(
                [&]() {
                    if (perfData) {
                        ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);
                        funcPtr(op, perfData, this);
                    } else {
                        kernel->compute(op, this);
                    }
                },
                [&]() { sync(); }, 1, 1);
            totalTime += t;
            const auto &inputs = op->getInputs();
            const auto &outputs = op->getOutputs();
            profileEntries.emplace_back(OperatorProfileEntry{
                opIdx++, op->getOpType().toString(),
                op->getDType().toString(),
                inputs.empty() ? "[]" : shapeString(inputs.front()),
                outputs.empty() ? "[]" : shapeString(outputs.front()), t});
        } else {
            if (perfData) {
                ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);
                funcPtr(op, perfData, this);
            } else {
                kernel->compute(op, this);
            }
        }
        checkCudaError(cudaGetLastError()) << op->toString();
    }
    if (profiling) {
        printProfileReport(profileEntries, totalTime);
    }
}

void CudaRuntimeObj::runWithCudaGraph(const Graph &graph) {
    if (!isCudaGraphCreated) {
        CUDAStream::createStream();
        checkCudnnError(cudnnSetStream(cudnn, CUDAStream::getCurrentStream()));
        checkCublasError(
            cublasSetStream(cublas, CUDAStream::getCurrentStream()));
        checkCudaError(cudaStreamBeginCapture(CUDAStream::getCurrentStream(),
                                              cudaStreamCaptureModeGlobal));
        runWithoutSync(graph);
        checkCudaError(
            cudaStreamEndCapture(CUDAStream::getCurrentStream(), &cudaGraph));
        checkCudaError(
            cudaGraphInstantiate(&cudaGraphInstance, cudaGraph, NULL, NULL, 0));
        isCudaGraphCreated = true;
    } else {
        checkCudaError(
            cudaGraphLaunch(cudaGraphInstance, CUDAStream::getCurrentStream()));
    }
    checkCudaError(cudaStreamSynchronize(CUDAStream::getCurrentStream()));
}

void CudaRuntimeObj::tune(const Graph &graph, bool profiling) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        PerfRecord record;
        if (!perfData) {
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;
        double t = record->time;
        totalTime += t;
        json j;

        kernel->computeFuncTune(perfKey, op, record, this);
        if (profiling) {
            ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);
            double t = timeit([&]() { funcPtr(op, record, this); },
                              [&]() { sync(); }, 1, 1);
            op->print();
            printf(" op_time on cuda %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }

        checkCudaError(cudaGetLastError()) << op->toString();
    }
}

void CudaRuntimeObj::run(const Graph &graph, bool runTune,
                         bool profiling) const {
    if (runTune)
        tune(graph, profiling);
    else
        runWithoutSync(graph, profiling);
    sync();
}

void CudaRuntimeObj::sync() const { checkCudaError(cudaDeviceSynchronize()); }

string CudaRuntimeObj::toString() const { return "CUDA Runtime"; }

void CudaRuntimeObj::initComm(const string &name, int worldSize, int rank) {
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

cudaStream_t CUDAStream::_stream = 0;
} // namespace infini
