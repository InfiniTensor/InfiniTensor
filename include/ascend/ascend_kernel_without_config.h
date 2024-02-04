#pragma once
#include "ascend/ascend_runtime.h"
#include "core/kernel.h"

namespace infini {

class ASCENDKernelWithoutConfig : public Kernel {
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
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        return make_ref<PerfRecordObj>(timeit([&]() { compute(op, _context); },
                                              [&]() { context->sync(); }));
    }
    // transform vector<int> to vector<int64_t>
    std::vector<int64_t> MycastTo64(std::vector<int> const &v32) const {
        std::vector<int64_t> v64(v32.size(), 1);
        for (size_t i = 0; i < v32.size(); ++i) {
            v64[i] = int64_t(v32[i]);
        }
        return v64;
    }
};

} // namespace infini
