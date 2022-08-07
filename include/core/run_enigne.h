#pragma once
#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"

namespace infini {

class RunEngine {
  private:
    Device device;

  public:
    RunEngine(Device device) : device(device) {}
    ~RunEngine() {}

    void run(const Graph &graph, bool tune = false,
             bool profiling = false) const;
    double getPerfTime(const Graph &graph, bool profiling = false) const;

  private:
    void printProfilingData(double totTime,
                            const std::map<OpType, double> &opTime,
                            const std::map<OpType, int> &opCnt) const;
};

} // namespace infini