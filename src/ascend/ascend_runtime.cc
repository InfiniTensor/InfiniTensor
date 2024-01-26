#include "ascend/ascend_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"

namespace infini {

void ASCENDRuntimeObj::runWithoutSync(const Graph &graph, bool tune = false,
                                      bool profiling = false) const {
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
        if (!perfData && !tune) {
            kernel->compute(op, this);
            continue;
        }

        PerfRecord record;
        if (!perfData) {
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;

        double t = record->time;
        totalTime += t;

        if (profiling) {
            double t = timeit([&]() { kernel->compute(op, record, this); },
                              [&]() { sync(); }, 1, 1);
            op->print();
            printf(" op_time on kunlun xpu %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
}

void ASCENDRuntimeObj::run(const Graph &graph, bool tune,
                           bool profiling) const {
    if (profiling)
        IT_TODO_HALT();
    runWithoutSync(graph, tune, profiling);
    sync();
}

void ASCENDRuntimeObj::sync() const { ; }

string ASCENDRuntimeObj::toString() const { return "ASCEND Runtime"; }

} // namespace infini
