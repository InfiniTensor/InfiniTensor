#include "kunlun/kunlun_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include <string>

namespace infini {

void KUNLUNRuntimeObj::runWithoutSync(const Graph &graph, bool tune = false,
                                      bool profiling = false) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    std::ofstream ofs("./log.txt");
    int i = 0;
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs =
            KernelAttrs{device, op->getOpType().underlying(), op->getDType()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        if (!perfData && !tune) {
            kernel->compute(op, this);
            auto cpuTensor = op->getOutput(0)->clone(cpuRuntime);
            std::cout << i++ << "th Op: " << op->getOpType().underlying() << std::endl;
            std::cout << cpuTensor->getRawDataPtr<float*>()[0] << std::endl;
            if (op->getDType() == OpType::Softmax){
                cpuTensor->dumpData(ofs);
                auto cpuTensorInput = op->getInputs(0)->clone(cpuRuntime);
                cpuTensorInput->dumpData(ofs);
            }
            continue;
        }

        PerfRecord record;
        if (!perfData) {
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;

        double t = record->time;
        totalTime += t;

        if (profiling) {
            double t = timeit([&]() { kernel->compute(op, record, this); },
                              [&]() { sync(); }, 1, 1);
            op->print();
            printf(" op_time on kunlun xpu %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
    ofs.close();
}

void KUNLUNRuntimeObj::run(const Graph &graph, bool tune,
                           bool profiling) const {
    if (profiling)
        IT_TODO_HALT();
    runWithoutSync(graph, tune, profiling);
    sync();
}

void KUNLUNRuntimeObj::sync() const { ; }

string KUNLUNRuntimeObj::toString() const { return "KUNLUN Runtime"; }

void KUNLUNRuntimeObj::initComm(const string &name, int worldSize, int rank) {
    IT_ASSERT(worldSize > 0);
    IT_ASSERT(rank >= 0);
    IT_ASSERT(rank < worldSize);
    IT_ASSERT(!comm) << "communicator is already initialized.";
#ifdef INFINI_USE_XCCL
    comm = std::make_unique<XcclCommunicatorObj>(name, worldSize, rank);
#else
    IT_TODO_HALT_MSG("Not compiled with XCCL");
#endif
}

} // namespace infini
