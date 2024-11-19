#include "musa/musa_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"

namespace infini {
void MusaRuntimeObj::runWithoutSync(const Graph &graph) const {
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
        checkMusaError(musaGetLastError()) << op->toString();
    }
}

void MusaRuntimeObj::run(const Graph &graph, bool tune, bool profiling) const {
    if (profiling)
        IT_TODO_HALT();
    runWithoutSync(graph);
    sync();
}

void MusaRuntimeObj::sync() const { checkMusaError(musaDeviceSynchronize()); }

string MusaRuntimeObj::toString() const { return "MUSA Runtime"; }
} // namespace infini
