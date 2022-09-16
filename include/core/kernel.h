#pragma once
#include "core/common.h"
#include "core/operator.h"
#include "core/tensor.h"

namespace infini {

class RuntimeObj; // Forward declaration for Kernel::compute

struct PerfRecordObj {
    PerfRecordObj(){};
    PerfRecordObj(double time) : time(time){};
    virtual ~PerfRecordObj() {}
    virtual json to_json() {
        return json {{"time", this->time}};
    }
    /**
     * @brief Dump json to perfrecord.
     * @param j The json to be dumped.
     **/
    virtual void from_json(json j) {
        j.at("time").get_to(this->time);
    }
    double time = 0; // in milliseconds
};
using PerfRecord = Ref<PerfRecordObj>;

class Kernel {
  public:
    Kernel() {}
    virtual ~Kernel() {}

    /**
     * @param op The operator to be executed.
     * @param record The parameters for kernel execution. If extra parameters
     * are required, inherit from PerfRecord and add extra parameters.
     * Otherwire, use PerfRecord directly.
     */
    virtual void compute(const Operator &op, const PerfRecord &record,
                         const RuntimeObj *context) const = 0;
    /**
     * @brief Executes an op with a default parameter.
     */
    virtual void compute(const Operator &op,
                         const RuntimeObj *context) const = 0;
    // Premise: op is idempotent since it is called multiple times.
    virtual PerfRecord* tune(const Operator &op,
                            const RuntimeObj *context) const = 0;
};

class KernelRegistry {
  public:
    using KernelRecord =
        tuple<Kernel *const, const string, const int>; // Kernel, name, ID

  private:
    std::map<KernelAttrs, KernelRecord> kernels;
    int nKernels = 0;

  public:
    ~KernelRegistry() {
        for (auto &[k, v] : kernels)
            delete std::get<0>(v);
    }
    static KernelRegistry &getInstance() {
        static KernelRegistry instance;
        return instance;
    }
    bool registerKernel(const KernelAttrs &key, Kernel *kernel, string name) {
        // TODO: mutliple kernels support: priority and check name
        IT_ASSERT(kernels.find(key) == kernels.end(),
                  "Kernel already registered");
        kernels.emplace(key, KernelRecord{kernel, name, ++nKernels});
        return true;
    }
    Kernel *getKernel(const KernelAttrs &kernelAttrs) const {
        auto it = kernels.find(kernelAttrs);
        IT_ASSERT(it != kernels.end(), "Kernel not found.");
        return std::get<0>(it->second);
    }
    const KernelRecord &getKernelItem(const KernelAttrs &kernelAttrs) const {
        return kernels.at(kernelAttrs);
    }
};

class CpuKernelWithoutConfig : public Kernel {
  public:
    void compute(const Operator &op, const PerfRecord &record,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }
    virtual void compute(const Operator &op,
                         const RuntimeObj *context) const = 0;
    // Premise: op is idempotent since it is called multiple times.
    virtual PerfRecord tune(const Operator &op,
                            const RuntimeObj *context) const override {
        return make_ref<PerfRecordObj>(timeit([&]() { compute(op, context); }));
    }
};

} // namespace infini

#define _REGISTER_KERNEL_1(device, opType, dataType, kernel, name, cnt)        \
    namespace infini {                                                         \
    static const bool _CAT(_register_kernel_, cnt) =                           \
        KernelRegistry::getInstance().registerKernel(                          \
            KernelAttrs{device, opType, dataType}, new kernel(), name);        \
    }

#define REGISTER_KERNEL(device, opType, dataType, kernel, name)                \
    _REGISTER_KERNEL_1(device, opType, dataType, kernel, name, __COUNTER__)
