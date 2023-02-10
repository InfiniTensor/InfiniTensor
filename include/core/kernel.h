#pragma once
#include "core/common.h"
#include "core/operator.h"
#include "core/tensor.h"
#include <functional>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
namespace infini {

class RuntimeObj; // Forward declaration for Kernel::compute

struct PerfRecordObj {
    PerfRecordObj(){};
    PerfRecordObj(double time) : time(time){};
    virtual ~PerfRecordObj(){};
    double time = 0; // in milliseconds
    virtual void to_json(json &j) {
        j["type"] = 0;
        j["data"] = time;
    }
    static Ref<PerfRecordObj> from_json(const json &j) {
        PerfRecordObj tmp;
        tmp.time = j["data"].get<int>();
        return make_ref<PerfRecordObj>(tmp);
    }
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
    virtual PerfRecord tune(const Operator &op,
                            const RuntimeObj *context) const = 0;
};

class PerfRecordRegistry {

  private:
    std::map<int, std::function<PerfRecord(const json &)>> perfrecords;
    int nperfrecord = 0;

  public:
    ~PerfRecordRegistry() = default;
    static PerfRecordRegistry &getInstance() {
        static PerfRecordRegistry instance;
        return instance;
    }
    bool
    registerPerfRecord(const int type,
                       std::function<PerfRecord(const json &)> constructor) {
        IT_ASSERT(perfrecords.find(type) == perfrecords.end(),
                  "Constructor already registered");
        perfrecords.emplace(type, constructor);
        nperfrecord++;
        return true;
    }
    const std::function<PerfRecord(const json &)> &
    getConstructor(const int type) const {
        return perfrecords.at(type);
    }
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
        IT_ASSERT(it != kernels.end(),
                  "Kernel not found for key {" +
                      to_string(enum_to_underlying(std::get<0>(kernelAttrs))) +
                      ", " + OpRegistry::getOpName(std::get<1>(kernelAttrs)) +
                      ", " + std::get<2>(kernelAttrs).toString() + "}");
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

#define _REGISTER_CONSTRUCTOR_1(type, constructor, cnt)                        \
    namespace infini {                                                         \
    static const bool _CAT(_register_constructor_, cnt) =                      \
        PerfRecordRegistry::getInstance().registerPerfRecord(type,             \
                                                             constructor);     \
    }

#define REGISTER_CONSTRUCTOR(type, constructor)                                \
    _REGISTER_CONSTRUCTOR_1(type, constructor, __COUNTER__)
