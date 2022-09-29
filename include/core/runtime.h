#pragma once
#include "core/common.h"
#include "core/ref.h"
#include <memory>

namespace infini {

/***************** Forward declaration begin *****************/
class TensorBaseObj;
class TensorObj;
class OperatorObj;
class GraphObj;
class RuntimeObj;
class BlobObj;

using TensorBase = Ref<TensorBaseObj>;
using Tensor = Ref<TensorObj>;
using Operator = Ref<OperatorObj>;
using Graph = Ref<GraphObj>;
using Runtime = Ref<RuntimeObj>;
using Blob = Ref<BlobObj>;
enum class OpType;

using TensorVec = vector<Tensor>;
using OpVec = vector<Operator>;

using VType = uint32_t;

enum class Device { CPU = 1, CUDA, BANG };
/***************** Forward declaration end *****************/

class RuntimeObj : public std::enable_shared_from_this<RuntimeObj> {
  protected:
    Device device;

  public:
    RuntimeObj(Device device) : device(device) {}
    RuntimeObj(RuntimeObj &other) = delete;
    RuntimeObj &operator=(RuntimeObj const &) = delete;
    virtual ~RuntimeObj() {}

    /**
     * @brief Execute a graph.
     *
     * @param graph
     * @param tune If there is no performance record, whether to tune it. These
     * can be independent method.
     * @param profiling Whether to print breakdown of time
     */
    virtual void run(const Graph &graph, bool tune = false,
                     bool profiling = false) const = 0;
    virtual void *alloc(size_t size) = 0;
    virtual void dealloc(void *ptr) = 0;

    /**
     * @brief Get the execution time of each operator in performance record. No
     * execution happens.
     *
     * @param graph
     * @param profiling Whether to print breakdown of time
     * @return double Return the sum of perf time for each operator
     */
    double getPerfTime(const Graph &graph, bool profiling = false) const;
    Blob allocBlob(size_t size);
    bool isCpu() const { return device == Device::CPU; }
    bool isCuda() const { return device == Device::CUDA; }
    bool isBang() const { return device == Device::BANG; }
    void copyBlob(const TensorObj *dst, const TensorObj *src) const;
    // TODO: unify these copy APIs
    virtual void copyBlobFromCPU(void *dst, const void *src,
                                 size_t bytes) const = 0;
    virtual void copyBlobToCPU(void *dst, const void *src,
                               size_t bytes) const = 0;

  protected:
    void printProfilingData(double totTime,
                            const std::map<OpType, double> &opTime,
                            const std::map<OpType, int> &opCnt) const;
    virtual void copyBlobInsideRuntime(void *dst, const void *src,
                                       size_t bytes) const = 0;
};

class CpuRuntimeObj : public RuntimeObj {
  public:
    CpuRuntimeObj() : RuntimeObj(Device::CPU) {}
    static Ref<CpuRuntimeObj> &getInstance() {
        static Ref<CpuRuntimeObj> instance = make_ref<CpuRuntimeObj>();
        return instance;
    }

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const override;
    void dealloc(void *ptr) override { return free(ptr); };

    void *alloc(size_t size) override {
        return calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t),
                      sizeof(uint64_t));
    };

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override;
    void copyBlobToCPU(void *dst, const void *src, size_t bytes) const override;
    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override;
};

} // namespace infini
