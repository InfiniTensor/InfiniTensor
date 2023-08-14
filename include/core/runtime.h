#pragma once
#include "core/common.h"
#include "core/communicator.h"
#include "core/op_type.h"
#include "core/ref.h"
#include <memory>

namespace infini {

/***************** Forward declaration begin *****************/
class TensorBaseObj;
class TensorObj;
class OperatorObj;
class GraphObj;
class GraphHandlerObj;
class RuntimeObj;
class BlobObj;

using TensorBase = Ref<TensorBaseObj>;
using Tensor = Ref<TensorObj>;
using Operator = Ref<OperatorObj>;
using Graph = Ref<GraphObj>;
using GraphHandler = Ref<GraphHandlerObj>;
using Runtime = Ref<RuntimeObj>;
using Blob = Ref<BlobObj>;

using TensorVec = vector<Tensor>;
using OpVec = vector<Operator>;
using OpLists = list<Operator>;

using VType = uint32_t;

enum class Device { CPU = 1, CUDA, BANG, INTELCPU };
/***************** Forward declaration end *****************/

class RuntimeObj : public std::enable_shared_from_this<RuntimeObj> {
  protected:
    Device device;
    int deviceId;
    std::unique_ptr<CommunicatorObj> comm;

  public:
    explicit RuntimeObj(Device device, int deviceId = 0)
        : device(device), deviceId(deviceId) {}
    RuntimeObj(RuntimeObj &other) = delete;
    RuntimeObj &operator=(RuntimeObj const &) = delete;
    virtual ~RuntimeObj() {}

    CommunicatorObj& getCommunicator() {
      return *comm;
    }

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
    bool isCpu() const {
        return device == Device::CPU || device == Device::INTELCPU;
    }
    bool isCuda() const { return device == Device::CUDA; }
    bool isBang() const { return device == Device::BANG; }
    void copyBlob(const TensorObj *dst, const TensorObj *src) const;
    // TODO: unify these copy APIs
    virtual void copyBlobFromCPU(void *dst, const void *src,
                                 size_t bytes) const = 0;
    virtual void copyBlobToCPU(void *dst, const void *src,
                               size_t bytes) const = 0;
    virtual string toString() const = 0;

    int getDeviceId() const { return deviceId; }

    virtual void initComm(int worldSize, int rank) = 0;

  protected:
    void printProfilingData(double totTime,
                            const std::map<OpType, double> &opTime,
                            const std::map<OpType, int> &opCnt) const;
    virtual void copyBlobInsideRuntime(void *dst, const void *src,
                                       size_t bytes) const = 0;
};

class CpuRuntimeObj : public RuntimeObj {
  public:
    CpuRuntimeObj(Device dev) : RuntimeObj(dev) {}

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const override;

    void copyBlobFromCPU(void *dst, const void *src,
                         size_t bytes) const override;
    void copyBlobToCPU(void *dst, const void *src, size_t bytes) const override;
    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override;
    void initComm(int, int) override { IT_TODO_HALT(); }
};

class NativeCpuRuntimeObj : public CpuRuntimeObj {
  public:
    NativeCpuRuntimeObj() : CpuRuntimeObj(Device::CPU) {}

    static Ref<NativeCpuRuntimeObj> &getInstance() {
        static Ref<NativeCpuRuntimeObj> instance =
            make_ref<NativeCpuRuntimeObj>();
        return instance;
    }
    void dealloc(void *ptr) override { return free(ptr); };

    void *alloc(size_t size) override {
        return calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t),
                      sizeof(uint64_t));
    };
    string toString() const override;
};

} // namespace infini
