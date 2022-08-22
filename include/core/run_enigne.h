#pragma once
#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
namespace infini {

class RuntimeObj : public std::enable_shared_from_this<RuntimeObj> {
  protected:
    Device device;

  public:
    RuntimeObj(Device device) : device(device) {}
    RuntimeObj(RuntimeObj &other) = delete;
    RunEngine &operator=(RuntimeObj const &) = delete;
    virtual ~RuntimeObj() {}

    virtual void run(const Graph &graph, bool tune = false,
                     bool profiling = false) const = 0;
    virtual double getPerfTime(const Graph &graph,
                               bool profiling = false) const = 0;
    virtual void *alloc(size_t size) = 0;
    virtual void dealloc(void *ptr) = 0;
    Blob allocBlob(size_t size);

  protected:
    void printProfilingData(double totTime,
                            const std::map<OpType, double> &opTime,
                            const std::map<OpType, int> &opCnt) const;
};

// TODO: change inheritance relation
class RunEngine : public RuntimeObj {
  public:
    RunEngine(Device device) : RuntimeObj(device) {}
    RunEngine(RunEngine &other) = delete;
    RunEngine &operator=(RunEngine const &) = delete;
    virtual ~RunEngine() {}

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const override;
    double getPerfTime(const Graph &graph,
                       bool profiling = false) const override;
    void dealloc(void *ptr) override { return free(ptr); };

    void *alloc(size_t size) override {
        return calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t),
                      sizeof(uint64_t));
    };
};

} // namespace infini