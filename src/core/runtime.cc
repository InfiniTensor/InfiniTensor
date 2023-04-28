#include "core/runtime.h"
#include "core/blob.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "operators/membound.h"
#include "utils/data_generator.h"
#include <chrono>
#include <cstring>

#ifdef USE_CUDA
#include "cuda_profiler_api.h"
#endif

namespace infini {
void CpuRuntimeObj::run(const Graph &graph, bool tune, bool profiling) const {
    if (!tune && profiling)
        IT_TODO_HALT();
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType(), op->getDType()};
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

        if (!profiling) {
            kernel->compute(op, record, this);
            continue;
        } else {
            double t = timeit([&]() { kernel->compute(op, record, this); },
                              []() {}, 1, 1);
            op->print();
            printf(" op_time %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
    // if (profiling)
    //     printProfilingData(totalTime, opTime, opCnt);
}

map<UidBaseType, bool>
RuntimeObj::getCompileTimeComputableAttribute(const Graph &graph) const {
    map<UidBaseType, bool> ctcMap; // compile-time computable
    // Skip static computation
    bool status = graph->topo_sort();
    IT_ASSERT(status, "Topological sort failed");
    for (auto &op : graph->getOperators()) {
        bool compileTimeComputable = true;
        for (auto input : op->getInputs()) {
            // FIXME: propogate the tensor type. Current only the first operator
            // after weights are compile-time computable.
            if (input->getTensorType() != TensorType::Initialized)
                compileTimeComputable = false;
        }
        ctcMap[op->getGuid()] = compileTimeComputable;
    }
    return ctcMap;
}

double RuntimeObj::getPerfTime(const Graph &graph, bool profiling,
                               bool allowEstimation,
                               bool ignoreMemboundOp) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt, opNonCtcCnt;
    // compile-time computable
    map<UidBaseType, bool> ctcMap = getCompileTimeComputableAttribute(graph);

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType(), op->getDType()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);

        double time = -1e9;
        if (ctcMap[op->getGuid()]) { // Compile-time computable operators
            time = 0;
        } else if (op->getOpType() == OpType::MemBound && ignoreMemboundOp) {
            time = 0;
        } else if (op->getOpType() == OpType::MemBound && allowEstimation) {
            time = as<MemBoundObj>(op)->getEstimatedTime();
        } else if (perfData) { // Tune the kernel if there is no record
            time = perfData->time;
        } else {
            // TODO: should tenosrs automatically allocate when access data?
            // allocate memory for empty tensors and release it after
            // profiling
            TensorVec allocatedTensors;
            for (auto t : op->getInputs())
                if (!t->hasData())
                    allocatedTensors.emplace_back(t);
            for (auto t : op->getOutputs())
                if (!t->hasData())
                    allocatedTensors.emplace_back(t);
            for (auto t : allocatedTensors) {
                t->dataMalloc();
                t->setData(IncrementalGenerator());
            }

            // Profile operators and record the results
            PerfRecord record = kernel->tune(op, this);
            time = record->time;
            perfEngine.setPerfData(perfKey, record);

            // Free allocated memory
            for (auto t : allocatedTensors)
                t->freeData();
        }

        // FIXME: ignore trnapose when necessary 
        //     op->getOpType() != OpType::Transpose &&
        //     op->getOpType() != OpType::ReduceMean
        if (op->getOpType() != OpType::Reshape)
            totalTime += time;
        if (profiling) {
            op->print();
            printf("  op_time %lf\n", time);
            opTime[op->getOpType()] += time;
            opCnt[op->getOpType()]++;
            if (!ctcMap[op->getGuid()])
                opNonCtcCnt[op->getOpType()]++;
            else
                opNonCtcCnt[op->getOpType()]; // Create a new entry
        }
    }
    if (profiling)
        printProfilingData(totalTime, opTime, opCnt, opNonCtcCnt);
    return totalTime;
}

void RuntimeObj::printProfilingData(
    double totalTime, const std::map<OpType, double> &opTime,
    const std::map<OpType, int> &opCnt,
    const std::map<OpType, int> &opNonCtcCnt) const {
    printf("%11s %3s %5s %7s %7s %7s\n", "Op", "Cnt", "#NCtc", "T_tot",
           "Percent", "T_mean");
    for (const auto &[type, t] : opTime) {
        printf("%11s %3d %5d %7.3f %7.1f %7.3f\n",
               OpRegistry::getOpName(type).data(), opCnt.at(type),
               opNonCtcCnt.at(type), t, t / totalTime * 100,
               t / opCnt.at(type));
    }
}

Blob RuntimeObj::allocBlob(size_t size) {
    return make_ref<BlobObj>(shared_from_this(), alloc(size));
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

double RuntimeObj::timeNonCtcOperators(const Graph &graph, int warmup,
                                       int repeat) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // compile-time computable
    map<UidBaseType, bool> ctcMap = getCompileTimeComputableAttribute(graph);
    vector<tuple<Operator, Kernel *, PerfRecord>> kernels;
    bool status = graph->topo_sort();
    IT_ASSERT(status, "Topological sort failed");

    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs =
            KernelAttrs{device, op->getOpType(), DataType::Float32};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        if (perfData)
            kernel->compute(op, perfData, this);
        else
            kernel->compute(op, this);
        if (!ctcMap.at(op->getGuid()) && op->getOpType() != OpType::Reshape)
            kernels.emplace_back(op, kernel, perfData);
    }
    for (auto &[op, kernel, perfData] : kernels) {
        dbg(op);
    }
    double ret = timeit(
        [&]() {
            for (auto &[op, kernel, perfData] : kernels) {
                if (perfData)
                    kernel->compute(op, perfData, this);
                else
                    kernel->compute(op, this);
            }
        },
        [&]() { this->sync(); }, warmup, repeat);
    return ret;
}

} // namespace infini
