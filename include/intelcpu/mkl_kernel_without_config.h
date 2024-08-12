#pragma once
#include "core/kernel.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {

class MklKernelWithoutConfig : public Kernel {
  public:
    virtual void compute(const Operator &op, const PerfRecord &record,
                         const RuntimeObj *_context) const override {
        compute(op, _context);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);
        context->sync();
    }
    virtual void compute(const Operator &op,
                         const RuntimeObj *context) const = 0;
    // Premise: op is idempotent since it is called multiple times.
    virtual PerfRecord tune(const Operator &op,
                            const RuntimeObj *_context) const override {
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);
        return make_ref<PerfRecordObj>(timeit([&]() { compute(op, _context); },
                                              [&]() { context->sync(); }));
    }

  protected:
    dnnl::memory::format_tag getUserFormatTag(int nDim) const {
        if (nDim == 2)
            return dnnl::memory::format_tag::nc;
        else if (nDim == 3)
            return dnnl::memory::format_tag::ncw;
        else if (nDim == 4)
            return dnnl::memory::format_tag::nchw;
        else if (nDim == 5)
            return dnnl::memory::format_tag::ncdhw;
        else
            IT_TODO_HALT();
    }
};

} // namespace infini
