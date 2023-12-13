#include "bang/bang_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#ifdef INFINI_USE_CNCL
#include "bang/cncl_communicator.h"
#endif

namespace infini {

void BangRuntimeObj::runWithoutSync(const Graph &graph, bool tune = false,
                                    bool profiling = false) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    // std::ofstream ofs("./log.txt");
    int i = 0;
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs =
            KernelAttrs{device, op->getOpType().underlying(), op->getDType()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        if (!perfData && !tune) {
            std::cout << i++ << "th Op: " << op->getOpType().underlying()
                      << std::endl;
            kernel->compute(op, this);
            auto cpuTensor = op->getOutput(0)->clone(cpuRuntime);
            std::cout << cpuTensor->getRawDataPtr<float *>()[0] << std::endl;
            // if (op->getOpType() == OpType::Softmax) {
            //     auto cpuTensorInput = op->getInputs(0)->clone(cpuRuntime);
            //     cpuTensorInput->dumpData(ofs);
            //     ofs.close();
            // }
            this->resetWorkspace();
            continue;
        }

        PerfRecord record;
        if (!perfData) {
            record = kernel->tune(op, this);
            this->resetWorkspace();
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;

        double t = record->time;
        totalTime += t;

        if (profiling) {
            double t = timeit([&]() { kernel->compute(op, record, this); },
                              [&]() { sync(); }, 1, 1);
            this->resetWorkspace();
            op->print();
            printf(" op_time on bang %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
    }
}

void BangRuntimeObj::run(const Graph &graph, bool tune, bool profiling) const {
    if (profiling)
        IT_TODO_HALT();
    runWithoutSync(graph, tune, profiling);
    sync();
}

void BangRuntimeObj::sync() const { cnrtSyncDevice(); }

string BangRuntimeObj::toString() const { return "BANG Runtime"; }

void BangRuntimeObj::initComm(const string &name, int worldSize, int rank) {
    IT_ASSERT(worldSize > 0);
    IT_ASSERT(rank >= 0);
    IT_ASSERT(rank < worldSize);
    IT_ASSERT(!comm) << "communicator is already initialized.";
#ifdef INFINI_USE_CNCL
    std::cout << "Rank: " << rank;
    comm = std::make_unique<CnclCommunicatorObj>(name, worldSize, rank);
    checkBangError(cnrtSetDevice(rank % worldSize));
    checkBangError(cnrtQueueCreate(&(CnclCommManager::getInstance()
                                         ->getCommSet(name, worldSize)
                                         .queues[rank])));
    
#else
    IT_TODO_HALT_MSG("Not compiled with CNCL.");
#endif
}
} // namespace infini
