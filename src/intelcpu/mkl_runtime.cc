#include "intelcpu/mkl_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
namespace infini {
MklRuntimeObj::MklRuntimeObj() : CpuRuntimeObj(Device::INTELCPU) {
    dnnl_engine_create(&engine, dnnl_engine_kind_t::dnnl_cpu, 0);
    dnnl_stream_create(
        &stream, engine,
        static_cast<dnnl_stream_flags_t>(dnnl_stream_default_flags));
}

MklRuntimeObj::~MklRuntimeObj() {
    mkl_free_buffers();
    dnnl_stream_destroy(stream);
    dnnl_engine_destroy(engine);
}

void MklRuntimeObj::sync() const { getStream().wait(); }
} // namespace infini
