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

void MusaRuntimeObj::runWithMusaGraph(const Graph &graph) {
    if (!isMusaGraphCreated) {
        MUSAStream::createStream();
        checkMusaError(musaStreamBeginCapture(MUSAStream::getCurrentStream(),
                                              musaStreamCaptureModeGlobal));
        runWithoutSync(graph);
        checkMusaError(
            musaStreamEndCapture(MUSAStream::getCurrentStream(), &musaGraph));
        checkMusaError(
            musaGraphInstantiate(&musaGraphInstance, musaGraph, NULL, NULL, 0));
        isMusaGraphCreated = true;
    } else {
        checkMusaError(
            musaGraphLaunch(musaGraphInstance, MUSAStream::getCurrentStream()));
    }

    checkMusaError(musaStreamSynchronize(MUSAStream::getCurrentStream()));
}

void MusaRuntimeObj::sync() const { checkMusaError(musaDeviceSynchronize()); }

string MusaRuntimeObj::toString() const { return "MUSA Runtime"; }

musaStream_t MUSAStream::_stream = 0;
} // namespace infini
