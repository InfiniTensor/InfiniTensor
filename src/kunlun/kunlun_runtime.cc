#include "kunlun/kunlun_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include <string>

namespace infini {

void KUNLUNRuntimeObj::runWithoutSync(const Graph &graph, bool tune = false,
                                      bool profiling = false) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
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

void KUNLUNRuntimeObj::run(const Graph &graph, bool tune,
                           bool profiling) const {
    if (profiling)
        IT_TODO_HALT();
    runWithoutSync(graph, tune, profiling);
    sync();
}

void KUNLUNRuntimeObj::sync() const { xpu_wait(); }

string KUNLUNRuntimeObj::toString() const { return "KUNLUN Runtime"; }

void KUNLUNRuntimeObj::initComm(const string &name, int worldSize, int rank) {
    IT_ASSERT(worldSize > 0);
    IT_ASSERT(rank >= 0);
    IT_ASSERT(rank < worldSize);
    IT_ASSERT(!comm) << "communicator is already initialized.";
#ifdef INFINI_USE_XCCL
    comm = std::make_unique<XcclCommunicatorObj>(name, worldSize, rank);
#else
    IT_TODO_HALT_MSG("Not compiled with XCCL");
#endif
}

} // namespace infini
