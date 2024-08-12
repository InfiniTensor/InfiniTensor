#pragma once
#include "core/kernel.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {

class KUNLUNKernelWithoutConfig : public Kernel {
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
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        return make_ref<PerfRecordObj>(timeit([&]() { compute(op, _context); },
                                              [&]() { context->sync(); }));
    }
};

} // namespace infini
