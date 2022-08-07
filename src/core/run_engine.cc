#include "core/run_enigne.h"
#include <chrono>

namespace infini {

void RunEngine::run(const Graph &graph, bool tune, bool profiling) const {
    if (!tune && profiling)
        IT_TODO_HALT();
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
    std::chrono::system_clock::time_point begin, end;

    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs =
            KernelAttrs{device, op->getOpType(), DataType::Int32};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpAttrs()};
        std::optional<PerfRecord> perfData = perfEngine.getPerfData(perfKey);

        // If no record and disable tuning, run with the default argument
        if (!perfData && !tune) {
            kernel->compute(op);
            continue;
        }

        // TODO: The copy of record should be eliminated
        PerfRecord record;
        // Tune the kernel if there is no record
        if (!perfData) {
            record = kernel->tune(op);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = *perfData;

        if (!profiling) {
            kernel->compute(op, *perfData);
            continue;
        } else {
            double t = timeit([&]() { kernel->compute(op, *perfData); });
            op->print();
            printf(" op_time %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
    if (profiling)
        printProfilingData(totalTime, opTime, opCnt);
}

double RunEngine::getPerfTime(const Graph &graph, bool profiling) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;

    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs =
            KernelAttrs{device, op->getOpType(), DataType::Int32};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpAttrs()};
        std::optional<PerfRecord> perfData = perfEngine.getPerfData(perfKey);

        PerfRecord record;
        // Tune the kernel if there is no record
        if (!perfData) {
            record = kernel->tune(op);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = *perfData;

        double t = record.time;
        totalTime += t;
        if (profiling) {
            op->print();
            printf(" op_time %lf\n", t);
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
    if (profiling)
        printProfilingData(totalTime, opTime, opCnt);
    return totalTime;
}

void RunEngine::printProfilingData(double totalTime,
                                   const std::map<OpType, double> &opTime,
                                   const std::map<OpType, int> &opCnt) const {
    printf("%11s %3s %7s %7s %7s\n", "Op", "Cnt", "T_tot", "Percent", "T_mean");
    for (const auto &[type, t] : opTime) {
        printf("%11s %3d %7.3f %7.1f %7.3f\n",
               OpRegistry::getOpName(type).data(), opCnt.at(type), t,
               t / totalTime * 100, t / opCnt.at(type));
    }
}

} // namespace infini