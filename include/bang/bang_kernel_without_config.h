#pragma once
#include "bang/bang_runtime.h"
#include "core/kernel.h"

namespace infini {

class BangKernelWithoutConfig : public Kernel {
  public:
    virtual void compute(const Operator &op, const PerfRecord &record,
                         const RuntimeObj *context) const {
        compute(op, context);
    }

    virtual void compute2(const Operator &op, const PerfRecord &record,
                         const RuntimeObj *context) const {
        compute2(op, context);
    }
    virtual void compute(const Operator &op,
                         const RuntimeObj *context) const = 0;

    virtual void compute2(const Operator &_op, const RuntimeObj *_context) const = 0;
    // Premise: op is idempotent since it is called multiple times.
    virtual PerfRecord tune(const Operator &op,
                            const RuntimeObj *_context) const {
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        return make_ref<PerfRecordObj>(timeit([&]() { compute(op, _context); },
                                              [&]() { context->sync(); }));
    }

    BangKernelWithoutConfig() {
        funcVec.emplace_back([this]{
            this->compute(Operator{}, PerfRecord{}, nullptr);
        });
    }

    void computeFuncAdd(const Key perfKey, const Operator &op,
                 const PerfRecord &record,
                 const RuntimeObj *context) const override {
        double t = std::numeric_limits<double>::max();
        ComputeFuncPtr funcPtr;
        for (auto& itPtr : funcVec) {
            double tem = timeit(
                    [&]() {funcPtr(op, record, context);},
                    [&]() {}, 1, 1);
            if (tem < t) {
                t = tem;
                funcPtr = itPtr;
            }
        }
        if (funcPtr != nullptr) {
            setComputeFunc(perfKey, funcPtr);
        }
    }

    // Get compute function according to key
    ComputeFuncPtr getComputeFunc(const Key &key) const override {
        auto it = computeMap.find(key);
        if (it != computeMap.end())
            return computeMap.at(key);
        else
            return nullptr;
    }

    void setComputeFunc(const Key &key, ComputeFuncPtr ptr) const override {
        IT_ASSERT(computeMap.find(key) == computeMap.end(), "compute func ptr already exist");
        computeMap.emplace(key, ptr);
    }
};

} // namespace infini
