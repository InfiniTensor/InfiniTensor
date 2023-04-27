#pragma once
#include "core/runtime.h"
#include "dnnl.h"
#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include <dnnl_debug.h>
#include <mkl.h>
namespace infini {
class MklRuntimeObj : public CpuRuntimeObj {
    dnnl_engine_t engine;
    dnnl_stream_t stream;

  public:
    MklRuntimeObj();
    static Ref<MklRuntimeObj> &getInstance() {
        static Ref<MklRuntimeObj> instance = make_ref<MklRuntimeObj>();
        return instance;
    }

    virtual ~MklRuntimeObj();
    void dealloc(void *ptr) override { return mkl_free(ptr); };

    void *alloc(size_t size) override {
        return mkl_calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t),
                          sizeof(uint64_t), 64);
    };

    string toString() const override { return "INTELCPU Runtime"; };
    dnnl::engine getEngine() const { return dnnl::engine(engine, true); }
    dnnl::stream getStream() const { return dnnl::stream(stream, true); }
    void sync() const override;
};

} // namespace infini
