#include "cuda/cuda_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"

namespace infini {

void CudaRuntimeObj::runWithoutSync(const Graph &graph, bool tune = false) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto perfEngine = PerfEngine::getInstance();
    
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs =
            KernelAttrs{device, op->getOpType(), DataType::Float32};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        std::optional<PerfRecord> perfData = perfEngine.getPerfData(perfKey);
        if (!perfData && !tune) {
            kernel->compute(op, this);
            continue;
        }
        
        PerfRecord record;

        if (!perfData) {
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
           
        } else  
            record = *perfData;
        
        if (!profiling) {
            kernel->compute(op, record, this);
            continue;
        } else {
            double t = timeit([&]() { kernel->compute(op, record, this); }, 1, 1);
            op->print();
            printf(" op_time on cuda %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }

    }
}

void CudaRuntimeObj::run(const Graph &graph, bool tune, bool profiling) const {
    if (profiling)
        IT_TODO_HALT();

    runWithoutSync(graph, tune);
    sync();
}

void CudaRuntimeObj::sync() const { cudaDeviceSynchronize(); }

} // namespace infini