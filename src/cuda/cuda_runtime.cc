#include "cuda/cuda_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#include "operators/conv.h"
#include "operators/matmul.h"
namespace infini {

void CudaRuntimeObj::runWithoutSync(const Graph &graph) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs = KernelAttrs{device, op->getOpType(), op->getDType()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        // IT_ASSERT(perfData, "No perf data for OP " + op->toString());
        if (perfData) {
            kernel->compute(op, perfData, this);
        } else {
            kernel->compute(op, this);
        }
    }
}

void CudaRuntimeObj::tune(const Graph &graph, bool profiling = false) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs = KernelAttrs{device, op->getOpType(), op->getDType()};
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

        if (profiling) {
            double t = timeit([&]() { kernel->compute(op, record, this); },
                              [&]() { sync(); }, 1, 1);
            op->print();
            printf(" op_time on cuda %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
}

void CudaRuntimeObj::run(const Graph &graph, bool runTune,
                         bool profiling) const {
    if (profiling)
        IT_TODO_HALT();
    if (runTune)
        tune(graph, profiling);
    else
        runWithoutSync(graph);
    sync();
}

void CudaRuntimeObj::sync() const { checkCudaError(cudaDeviceSynchronize()); }

string CudaRuntimeObj::toString() const { return "CUDA Runtime"; }

} // namespace infini
