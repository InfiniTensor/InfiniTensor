#pragma once
#include "core/common.h"
#include "core/operator.h"
#include "core/tensor.h"

namespace it {

enum class Device { CPU = 1, CUDA };

class Kernel {
  public:
    Kernel() {}
    virtual ~Kernel() {}

    virtual void compute(const Operator &op) const = 0;
};

class KernelRegistry {
  public:
    using Key = std::tuple<Device, OpType, DataType>;

  public:
    ~KernelRegistry() {
        for (auto &[k, v] : kernels)
            delete v;
    }
    static KernelRegistry &getInstance() {
        static KernelRegistry instance;
        return instance;
    }
    bool registerKernel(const Key &key, Kernel *kernel) {
        // TODO: kernels with priority
        IT_ASSERT(kernels.find(key) == kernels.end(),
                  "Kernel already registered");
        kernels.emplace(key, kernel);
        return true;
    }
    Kernel *getKernel(Device device, OpType opType, DataType dataType) const {
        return kernels.at(Key{device, opType, dataType});
    }

  private:
    std::map<Key, Kernel *> kernels;
};

#define _REGISTER_KERNEL_1(device, opType, dataType, kernel, cnt)              \
    namespace it {                                                             \
    static const bool _CAT(_register_kernel_, cnt) =                           \
        KernelRegistry::getInstance().registerKernel(                          \
            KernelRegistry::Key{device, opType, dataType}, new kernel());      \
    }

#define REGISTER_KERNEL(device, opType, dataType, kernel)                      \
    _REGISTER_KERNEL_1(device, opType, dataType, kernel, __COUNTER__)

} // namespace it
