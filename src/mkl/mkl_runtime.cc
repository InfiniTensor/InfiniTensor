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
} // namespace infini
