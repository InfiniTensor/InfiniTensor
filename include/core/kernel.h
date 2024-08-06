#pragma once
#include "core/common.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "utils/operator_utils.h"
#include <functional>
#include <nlohmann/json.hpp>
namespace infini {
using json = nlohmann::json;

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
using ComputeFuncPtr = std::function<void(const Operator &, const PerfRecord &,
                                          const RuntimeObj *)>;

class Kernel {
  public:
    // multiple candiate kernels.
    using Key = std::pair<KernelAttrs, OpPerfKey>;

  protected:
    // Map storing the pairs of perfKey and corresponding optimal function ptr
    std::map<Key, ComputeFuncPtr> computeMap;
    // Vector storing all computing function pointers
    std::vector<ComputeFuncPtr> funcVec;

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

    // Find the optimal computing function by comparing its running time
    virtual void computeFuncAdd(const Key perfKey, const Operator &op,
                                const PerfRecord &record,
                                const RuntimeObj *context) = 0;

    // Get the optimal computing function according to the key
    virtual ComputeFuncPtr getComputeFunc(const Key &key) const = 0;

    // Add perfKey and function as <key, value> to the computeMap
    virtual void setComputeFunc(const Key &key, ComputeFuncPtr ptr) = 0;
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
        IT_ASSERT(it != kernels.end(), "Kernel not found for key {" +
                                           get_kernel_attrs_str(kernelAttrs) +
                                           "}");
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

    void computeFuncAdd(const Key perfKey, const Operator &op,
                        const PerfRecord &record,
                        const RuntimeObj *context) override {}

    ComputeFuncPtr getComputeFunc(const Key &key) const override {
        return nullptr;
    }

    void setComputeFunc(const Key &key, ComputeFuncPtr ptr) override {}
};

} // namespace infini

#define _REGISTER_KERNEL_1(device, opType, kernel, name, cnt)                  \
    namespace infini {                                                         \
    static const bool _CAT(_register_kernel_, cnt) =                           \
        KernelRegistry::getInstance().registerKernel(KernelAttrs{device,       \
                                                                 opType},      \
                                                     new kernel(), name);      \
    }

#define REGISTER_KERNEL(device, opType, kernel, name)                          \
    _REGISTER_KERNEL_1(device, opType, kernel, name, __COUNTER__)

#define _REGISTER_CONSTRUCTOR_1(type, constructor, cnt)                        \
    namespace infini {                                                         \
    static const bool _CAT(_register_constructor_, cnt) =                      \
        PerfRecordRegistry::getInstance().registerPerfRecord(type,             \
                                                             constructor);     \
    }

#define REGISTER_CONSTRUCTOR(type, constructor)                                \
    _REGISTER_CONSTRUCTOR_1(type, constructor, __COUNTER__)
