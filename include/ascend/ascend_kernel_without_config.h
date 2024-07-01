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
    std::vector<int64_t> castTo64(std::vector<int> const &v32) const {
        if (v32.size() == 0) {
            std::vector<int64_t> v64(1, 1);
            return v64;
        }
        std::vector<int64_t> v64(v32.size(), 1);
        for (size_t i = 0; i < v32.size(); ++i) {
            v64[i] = int64_t(v32[i]);
        }

        return v64;
    }

    Shape getStride(std::vector<int> Dim) const {
        Shape stride(Dim.size());
        ShapeElem p = 1;
        for (auto i = Dim.size(); i > 0; --i) {
            stride[i - 1] = p;
            p = p * Dim[i - 1];
        }
        return stride;
    }
};
} // namespace infini