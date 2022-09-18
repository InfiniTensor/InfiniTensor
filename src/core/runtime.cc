#include "core/runtime.h"
#include "core/blob.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include <chrono>
#include <cstring>
<<<<<<< HEAD
=======
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cudnn.h>
#include <curand.h>
>>>>>>> cf58b99 (clang format)
namespace infini {

void CpuRuntimeObj::run(const Graph &graph, bool tune, bool profiling) const {
    if (!tune && profiling)
        IT_TODO_HALT();
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType(), op->getDType()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);

        // If no record and disable tuning, run with the default argument
        if (!perfData && !tune) {
            kernel->compute(op, this);
            continue;
        }

        // TODO: The copy of record should be eliminated
        PerfRecord record;
        // Tune the kernel if there is no record
        if (!perfData) {
            // TODO: record is not used
            // printf("no record data\n");
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;

        if (!profiling) {
            kernel->compute(op, record, this);
            continue;
        } else {
            double t = timeit([&]() { kernel->compute(op, record, this); },
                              []() {}, 1, 1);
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

double RuntimeObj::getPerfTime(const Graph &graph, bool profiling) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // Statistics
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;

    for (auto &op : graph->getOperators()) {
        auto kernelAttrs = KernelAttrs{device, op->getOpType(), op->getDType()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);

        PerfRecord record;
        // Tune the kernel if there is no record
        if (!perfData) {
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;

        double t = record->time;
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

void RuntimeObj::printProfilingData(double totalTime,
                                    const std::map<OpType, double> &opTime,
                                    const std::map<OpType, int> &opCnt) const {
    printf("%11s %3s %7s %7s %7s\n", "Op", "Cnt", "T_tot", "Percent", "T_mean");
    for (const auto &[type, t] : opTime) {
        printf("%11s %3d %7.3f %7.1f %7.3f\n",
               OpRegistry::getOpName(type).data(), opCnt.at(type), t,
               t / totalTime * 100, t / opCnt.at(type));
    }
}

Blob RuntimeObj::allocBlob(size_t size) {
    return make_ref<BlobObj>(shared_from_this(), alloc(size));
}

void RuntimeObj::copyBlob(const TensorObj *dst, const TensorObj *src) const {
    void *dstPtr = dst->getRawDataPtr<void *>();
    void *srcPtr = src->getRawDataPtr<void *>();
    size_t bytes = dst->getBytes();
    auto dstRuntime = dst->getRuntime();
    auto srcRuntime = src->getRuntime();

    if (dstRuntime.get() == srcRuntime.get()) {
        dstRuntime->copyBlobInsideRuntime(dstPtr, srcPtr, bytes);
    } else if (src->getRuntime()->isCpu()) {
        dstRuntime->copyBlobFromCPU(dstPtr, srcPtr, bytes);
    } else if (dst->getRuntime()->isCpu()) {
        srcRuntime->copyBlobToCPU(dstPtr, srcPtr, bytes);
    } else
        IT_TODO_HALT();
}

void CpuRuntimeObj::copyBlobFromCPU(void *dst, void *src, size_t bytes) const {
    copyBlobInsideRuntime(dst, src, bytes);
}

void CpuRuntimeObj::copyBlobToCPU(void *dst, void *src, size_t bytes) const {
    copyBlobInsideRuntime(dst, src, bytes);
}

void CpuRuntimeObj::copyBlobInsideRuntime(void *dst, void *src,
                                          size_t bytes) const {
    memcpy(dst, src, bytes);
}

void to_json(json &j, const OpPerfKey &p) {
    j = json{{"hashType", p.hash}, {"opType", p.opType}, {"attrs", p.attrs}};
}
void from_json(const json &j, OpPerfKey &p) {
    j.at("hashType").get_to(p.hash);
    j.at("opType").get_to(p.opType);
    j.at("attrs").get_to(p.attrs);
}
void to_json(json &j, const DataType &p) {
    int x = p.toString() == "Float32" ? 0 : 1;
    j = x;
}
void from_json(const json &j, DataType &p) { p = DataType(j.get<int>()); }
void to_json(json &j, const PerfRecord &p) {
    if (as<ConvCuDnnPerfRecordObj>(p) != nullptr) {
        auto tmp = as<ConvCuDnnPerfRecordObj>(p);
        j["type"] = 1;
        j["data"] = std::make_tuple(tmp->algo, tmp->mode, tmp->fuseAct,
                                    tmp->time, tmp->workspaceSize);
    } else if (as<MatmulCudnnPerfRecordObj>(p) != nullptr) {
        auto tmp = as<MatmulCudnnPerfRecordObj>(p);
        j["type"] = 2;
        j["data"] = std::make_pair(tmp->algo, tmp->time);
    } else {
        j["type"] = 0;
        j["data"] = p->time;
    }
}
void from_json(const json &j, PerfRecord &p) {
    int type = j["type"].get<int>();
    if (type == 1) {
        ConvCuDnnPerfRecordObj tmp;
        auto [algo, mode, fuseAct, time, workspaceSize] =
            j["data"].get<tuple<int, int, bool, double, size_t>>();
        tmp.algo = (cublasGemmAlgo_t)algo;
        tmp.mode = mode;
        tmp.fuseAct = fuseAct;
        tmp.time = time;
        tmp.workspaceSize = workspaceSize;
        p = make_ref<ConvCuDnnPerfRecordObj>(tmp);
    } else if (type == 2) {
        MatmulCudnnPerfRecordObj tmp;
        auto pr = j["data"].get<pair<int, double>>();
        tmp.algo = (cublasGemmAlgo_t)pr.first;
        tmp.time = pr.second;
        p = make_ref<MatmulCudnnPerfRecordObj>(tmp);
    } else {
        p->time = j["data"].get<int>();
    }
}

void to_json(json &j, const PerfEngine &p) {
    PerfEngine t = p;
    j["data"] = t.get_data();
}
void from_json(const json &j, PerfEngine &p) {
    // using Key = std::pair<KernelAttrs, OpPerfKey>;
    // map<PerfEngine::Key, PerfRecord> tmp;

    auto tmp = j["data"].get<map<PerfEngine::Key, PerfRecord>>();
    p.set_data(tmp);
}

} // namespace infini
