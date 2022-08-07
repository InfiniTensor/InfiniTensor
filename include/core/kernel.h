#pragma once
#include "core/common.h"
#include "core/operator.h"
#include "core/tensor.h"

namespace infini {

struct PerfRecord {
    double time; // in milliseconds
};

class Kernel {
  public:
    Kernel() {}
    virtual ~Kernel() {}

    virtual void compute(const Operator &op,
                         const PerfRecord &record) const = 0;
    // This function call compute with a default record.
    virtual void compute(const Operator &op) const = 0;
    // Tuning should be idempotent since it is called multiple times.
    virtual PerfRecord tune(const Operator &op) const = 0;
};

class KernelRegistry {
  public:
    ~KernelRegistry() {
        for (auto &[k, v] : kernels)
            delete v;
    }
    static KernelRegistry &getInstance() {
        static KernelRegistry instance;
        return instance;
    }
    bool registerKernel(const KernelAttrs &key, Kernel *kernel) {
        // TODO: kernels with priority
        IT_ASSERT(kernels.find(key) == kernels.end(),
                  "Kernel already registered");
        kernels.emplace(key, kernel);
        return true;
    }
    Kernel *getKernel(const KernelAttrs &kernelAttrs) const {
        return kernels.at(kernelAttrs);
    }

  private:
    std::map<KernelAttrs, Kernel *> kernels;
};

#define _REGISTER_KERNEL_1(device, opType, dataType, kernel, cnt)              \
    namespace infini {                                                         \
    static const bool _CAT(_register_kernel_, cnt) =                           \
        KernelRegistry::getInstance().registerKernel(                          \
            KernelAttrs{device, opType, dataType}, new kernel());              \
    }

#define REGISTER_KERNEL(device, opType, dataType, kernel)                      \
    _REGISTER_KERNEL_1(device, opType, dataType, kernel, __COUNTER__)

} // namespace infini
