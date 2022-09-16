#include "cuda/cuda_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"

namespace infini {

void CudaRuntimeObj::runWithoutSync(const Graph &graph, bool tune = false,
                                    bool profiling = false) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
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
            // auto record_ = dynamic_cast<const ConvCuDnnPerfRecord &>(tmp);
            perfEngine.setPerfData(perfKey, record);
            // json j;
            // j["233"] = record_.to_json();
            // std::cout << j << std::endl;
        } else
            record = *perfData;

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
    json j;
    j["perfEngine"] = perfEngine;
    std::cout << j << std::endl;
}

void CudaRuntimeObj::run(const Graph &graph, bool tune, bool profiling) const {
    if (profiling)
        IT_TODO_HALT();
    runWithoutSync(graph, tune, profiling);
    sync();
}

void CudaRuntimeObj::sync() const { cudaDeviceSynchronize(); }

} // namespace infini