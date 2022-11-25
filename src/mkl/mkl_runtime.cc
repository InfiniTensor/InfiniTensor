#include "mkl/mkl_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
namespace infini {
MklRuntimeObj::MklRuntimeObj() : CpuRuntimeObj(Device::MKL) {
    dnnl_engine_create(&engine, dnnl_engine_kind_t::dnnl_cpu, 0);
}

MklRuntimeObj::~MklRuntimeObj() {
    mkl_free_buffers();
    dnnl_engine_destroy(engine);
}

void MklRuntimeObj::preProcess(Graph &graph) {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType(), op->getDType()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);

        kernel->preProcess(op, this, false);
    }
}
} // namespace infini
