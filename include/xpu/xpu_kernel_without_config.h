#pragma once
#include "xpu/xpu_runtime.h"
#include "core/kernel.h"

namespace infini {

class XPUKernelWithoutConfig : public Kernel {
  public:
    virtual void compute(const Operator &op, const PerfRecord &record,
                         const RuntimeObj *context) const {
        compute(op, context);
    }
    virtual void compute(const Operator &op,
                         const RuntimeObj *context) const = 0;
    // Premise: op is idempotent since it is called multiple times.
    virtual PerfRecord tune(const Operator &op,
                            const RuntimeObj *_context) const {
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        return make_ref<PerfRecordObj>(timeit([&]() { compute(op, _context); },
                                              [&]() { context->sync(); }));
    }
};

} // namespace infini
