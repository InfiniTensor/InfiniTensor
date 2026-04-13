#include "cuda/cuda_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#ifdef INFINI_USE_NCCL
#include "cuda/nccl_communicator.h"
#endif
#include "operators/conv.h"
#include "operators/matmul.h"

void CHECK_CUDA_KERNEL_ERROR(infini::Operator op) {
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(kernelError)
                  << std::endl
                  << "Failed Operator: " << op->toString() << std::endl;
        exit(EXIT_FAILURE);
    }
}

namespace infini {
void CudaRuntimeObj::runWithoutSync(const Graph &graph) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()}; //获取内核属性
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs); // 获取内核实现
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        // IT_ASSERT(perfData, "No perf data for OP " + op->toString());
        if (perfData) {
            ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);
            funcPtr(op, perfData, this);
        } else {
            kernel->compute(op, this); //调用内核计算
        }
        checkCudaError(cudaGetLastError()) << op->toString();
    }
}

void CudaRuntimeObj::runWithCudaGraph(const Graph &graph) {
    if (!isCudaGraphCreated) {
        CUDAStream::createStream();
        checkCudnnError(cudnnSetStream(cudnn, CUDAStream::getCurrentStream()));
        checkCublasError(
            cublasSetStream(cublas, CUDAStream::getCurrentStream()));
        checkCudaError(cudaStreamBeginCapture(CUDAStream::getCurrentStream(),
                                              cudaStreamCaptureModeGlobal));
        runWithoutSync(graph);
        checkCudaError(
            cudaStreamEndCapture(CUDAStream::getCurrentStream(), &cudaGraph));
        checkCudaError(
            cudaGraphInstantiate(&cudaGraphInstance, cudaGraph, NULL, NULL, 0));
        isCudaGraphCreated = true;
    } else {
        checkCudaError(
            cudaGraphLaunch(cudaGraphInstance, CUDAStream::getCurrentStream()));
    }
    checkCudaError(cudaStreamSynchronize(CUDAStream::getCurrentStream()));
}

void CudaRuntimeObj::tune(const Graph &graph, bool profiling = false) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        PerfRecord record;
        if (!perfData) {
            record = kernel->tune(op, this);
            perfEngine.setPerfData(perfKey, record);
        } else
            record = perfData;
        double t = record->time;
        totalTime += t;
        json j;

        kernel->computeFuncTune(perfKey, op, record, this);
        if (profiling) {
            ComputeFuncPtr funcPtr = kernel->getComputeFunc(perfKey);
            double t = timeit([&]() { funcPtr(op, record, this); },
                              [&]() { sync(); }, 1, 1);
            op->print();
            printf(" op_time on cuda %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }

        checkCudaError(cudaGetLastError()) << op->toString();
    }
}

void CudaRuntimeObj::run(const Graph &graph, bool runTune,
                         bool profiling) const {
    if (profiling)
        IT_TODO_HALT();
    if (runTune)
        tune(graph, profiling);
    else
        runWithoutSync(graph);
    sync();
}

void CudaRuntimeObj::sync() const { checkCudaError(cudaDeviceSynchronize()); }

string CudaRuntimeObj::toString() const { return "CUDA Runtime"; }

void CudaRuntimeObj::initComm(const string &name, int worldSize, int rank) {
    IT_ASSERT(worldSize > 0);
    IT_ASSERT(rank >= 0);
    IT_ASSERT(rank < worldSize);
    IT_ASSERT(!comm) << "communicator is already initialized.";
#ifdef INFINI_USE_NCCL
    comm = std::make_unique<NcclCommunicatorObj>(name, worldSize, rank);
#else
    IT_TODO_HALT_MSG("Not compiled with NCCL.");
#endif
}

cudaStream_t CUDAStream::_stream = 0;
} // namespace infini
