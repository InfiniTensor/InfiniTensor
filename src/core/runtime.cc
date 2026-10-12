#include "core/runtime.h"
#include "core/blob.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "utils/data_generator.h"
#include <algorithm>
#include <chrono>
#include <cstring>
namespace infini {
void CpuRuntimeObj::run(const Graph &graph, bool tune, bool profiling) const {
    IT_ASSERT(graph != nullptr, "Cannot run a null graph");
    graph->validateMemory();
    if (!tune && profiling)
        IT_TODO_HALT();
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{provider, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);

        // If no record and disable tuning, run with the default argument
        if (!perfData && !tune) {
            kernel->compute(op, this);
            continue;
        }

        // TODO: The copy of record should be eliminated
        PerfRecord record;
        // Tune the kernel if there is no record
        if (!perfData) {
            // TODO: record is not used
            // printf("no record data\n");
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;

        kernel->computeFuncTune(perfKey, op, record, this);
        ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);

        if (!profiling) {
            funcPtr(op, record, this);
            continue;
        } else {
            double t =
                timeit([&]() { funcPtr(op, record, this); }, []() {}, 1, 1);
            op->print();
            printf(" op_time %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
    if (profiling)
        printProfilingData(totalTime, opTime, opCnt);
}

double RuntimeObj::getPerfTime(const Graph &graph, bool profiling) const {
    IT_ASSERT(graph != nullptr, "Cannot profile a null graph");
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{provider, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);

        PerfRecord record;
        // Tune the kernel if there is no record
        if (!perfData) {
            // TODO: should tenosrs automatically allocate when access data?
            // allocate memory for empty tensors and release it after profiling
            TensorVec allocatedTensors;
            for (auto t : op->getInputs())
                if (!t->hasData() ||
                    t->getDataBlob()->getBytes() != t->getBytes())
                    allocatedTensors.emplace_back(t);
            for (auto t : op->getOutputs())
                if (!t->hasData() ||
                    t->getDataBlob()->getBytes() != t->getBytes())
                    allocatedTensors.emplace_back(t);
            try {
                for (auto t : allocatedTensors) {
                    t->dataMalloc();
                    t->setData(IncrementalGenerator());
                }

                // Profile operators and record the results
                record = kernel->tune(op, this);
                perfEngine.setPerfData(perfKey, record);
            } catch (...) {
                for (auto t : allocatedTensors)
                    t->freeData();
                throw;
            }

            // Free allocated memory
            for (auto t : allocatedTensors)
                t->freeData();
        } else
            record = perfData;

        double t = record->time;
        totalTime += t;
        if (profiling) {
            op->print();
            printf(" op_time %lf\n", t);
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
    if (profiling)
        printProfilingData(totalTime, opTime, opCnt);
    return totalTime;
}

void RuntimeObj::printProfilingData(double totalTime,
                                    const std::map<OpType, double> &opTime,
                                    const std::map<OpType, int> &opCnt) const {
    printf("%11s %3s %7s %7s %7s\n", "Op", "Cnt", "T_tot", "Percent", "T_mean");
    for (const auto &[type, t] : opTime) {
        printf("%11s %3d %7.3f %7.1f %7.3f\n", type.toString(), opCnt.at(type),
               t, t / totalTime * 100, t / opCnt.at(type));
    }
}

Blob RuntimeObj::allocBlob(size_t size) {
    const auto allocationSize = std::max<size_t>(size, 1);
    auto ptr = alloc(allocationSize);
    if (ptr == nullptr)
        throw std::bad_alloc();
    try {
        return make_ref<BlobObj>(shared_from_this(), ptr, size);
    } catch (...) {
        dealloc(ptr);
        throw;
    }
}

void RuntimeObj::copyBlob(const TensorObj *dst, const TensorObj *src) const {
    void *dstPtr = dst->getRawDataPtr<void *>();
    void *srcPtr = src->getRawDataPtr<void *>();
    size_t bytes = dst->getBytes();
    auto dstRuntime = dst->getRuntime();
    auto srcRuntime = src->getRuntime();

    if (dstRuntime.get() == srcRuntime.get()) {
        dstRuntime->copyBlobInsideRuntime(dstPtr, srcPtr, bytes);
    } else if (src->getRuntime()->isCpu()) {
        dstRuntime->copyBlobFromCPU(dstPtr, srcPtr, bytes);
    } else if (dst->getRuntime()->isCpu()) {
        srcRuntime->copyBlobToCPU(dstPtr, srcPtr, bytes);
    } else
        IT_TODO_HALT();
}

void CpuRuntimeObj::copyBlobFromCPU(void *dst, const void *src,
                                    size_t bytes) const {
    copyBlobInsideRuntime(dst, src, bytes);
}

void CpuRuntimeObj::copyBlobToCPU(void *dst, const void *src,
                                  size_t bytes) const {
    copyBlobInsideRuntime(dst, src, bytes);
}

void CpuRuntimeObj::copyBlobInsideRuntime(void *dst, const void *src,
                                          size_t bytes) const {
    memcpy(dst, src, bytes);
}

string NativeCpuRuntimeObj::toString() const { return "CPU Runtime"; }

} // namespace infini
