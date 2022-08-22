#include "cuda/cuda_runtime.h"

namespace infini {

void CudaRunEngine::runWithoutSync(const Graph &graph) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto perfEngine = PerfEngine::getInstance();

    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs =
            KernelAttrs{device, op->getOpType(), DataType::Float32};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        std::optional<PerfRecord> perfData = perfEngine.getPerfData(perfKey);
        if (perfData)
            kernel->compute(op, *perfData, this);
        else
            kernel->compute(op, this);
    }
}

void CudaRunEngine::run(const Graph &graph) const {
    runWithoutSync(graph);
    sync();
}

void CudaRunEngine::sync() const { cudaDeviceSynchronize(); }

} // namespace infini