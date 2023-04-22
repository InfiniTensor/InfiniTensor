#include "cuda/cuda_runtime.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#include "cuda_profiler_api.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#ifdef INFINI_USE_TVM
#include "tvm/runtime/device_api.h"
#endif
namespace infini {

CudaRuntimeObj::CudaRuntimeObj()
    : RuntimeObj(Device::CUDA), stream(cudaStreamPerThread),
      cudaGraphStatus(false) {
    checkCudnnError(cudnnCreate(&cudnn));
    checkCublasError(cublasCreate(&cublas));
    checkCudnnError(cudnnSetStream(cudnn, stream));
    checkCublasError(cublasSetStream(cublas, stream));
    // 10GB for Longformer
    // size_t longformerNum = 3lu * (1 << 30);
    workspaceSize = 7ll << 30; // 7 GB
    workspace = alloc(workspaceSize);
}

CudaRuntimeObj::~CudaRuntimeObj() {
    try {
        dealloc(workspace);
        checkCudnnError(cudnnDestroy(cudnn));
        checkCublasError(cublasDestroy(cublas));
    } catch (const std::exception &e) {
        std::cerr << "Error in ~CudaRuntimeObj: " << e.what() << std::endl;
    }
}

void CudaRuntimeObj::beginCudaGraphStreamCapture() {
    enum cudaStreamCaptureStatus pCaptureStatus;
    checkCudaError(cudaStreamIsCapturing(stream, &pCaptureStatus));
    IT_ASSERT(pCaptureStatus == cudaStreamCaptureStatusNone);
    cudaGraphStatus = true;
    checkCudaError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
}

tuple<cudaGraphExec_t, size_t> CudaRuntimeObj::endCudaGraphStreamCapture() {
    cudaGraph_t cudaGraph;
    cudaGraphExec_t instance;
    checkCudaError(cudaStreamEndCapture(stream, &cudaGraph));
    cudaGraphStatus = false;
    size_t numCudaGraphNodes;
    checkCudaError(cudaGraphGetNodes(cudaGraph, nullptr, &numCudaGraphNodes));
    checkCudaError(cudaGraphInstantiate(&instance, cudaGraph, NULL, NULL, 0));
    return {instance, numCudaGraphNodes};
}

void CudaRuntimeObj::runWithoutSync(const Graph &graph) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs =
            KernelAttrs{device, op->getOpType(), DataType::Float32};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        // IT_ASSERT(perfData, "No perf data for OP " + op->toString());
        if (perfData) {
            kernel->compute(op, perfData, this);
        } else {
            kernel->compute(op, this);
        }
    }
}

void CudaRuntimeObj::tune(const Graph &graph, bool profiling = false) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    double totalTime = 0;
    std::map<OpType, double> opTime;
    std::map<OpType, int> opCnt;
    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs =
            KernelAttrs{device, op->getOpType(), DataType::Float32};
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

        if (profiling) {
            double t = timeit([&]() { kernel->compute(op, record, this); },
                              [&]() { sync(); }, 1, 1);
            op->print();
            printf(" op_time on cuda %lf\n", t);
            totalTime += t;
            opTime[op->getOpType()] += t;
            opCnt[op->getOpType()]++;
        }
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

double CudaRuntimeObj::timeWithCudaGraph(Graph graph) {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    auto &perfEngine = PerfEngine::getInstance();
    // compile-time computable
    map<UidBaseType, bool> ctcMap = getCompileTimeComputableAttribute(graph);
    vector<tuple<Operator, Kernel *, PerfRecord>> kernels;
    bool status = graph->topo_sort();
    IT_ASSERT(status, "Topological sort failed");

    for (auto &op : graph->getOperators()) {
        // HACK: set correct data type
        auto kernelAttrs =
            KernelAttrs{device, op->getOpType(), DataType::Float32};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        auto perfKey = PerfEngine::Key{kernelAttrs, op->getOpPerfKey()};
        auto perfData = perfEngine.getPerfData(perfKey);
        if (perfData)
            kernel->compute(op, perfData, this);
        else
            kernel->compute(op, this);
        // FIXME: transpose
        if (!ctcMap.at(op->getGuid()) && op->getOpType() != OpType::Transpose &&
            op->getOpType() != OpType::Reshape)
            kernels.emplace_back(op, kernel, perfData);
    }
    for (auto &[op, kernel, perfData] : kernels) {
        dbg(op);
    }

// TODO: move this to kernel source?
// Init tvm stream
#ifdef INFINI_USE_TVM
    DLDevice tvm_device_id = {kDLCUDA, 0};
    auto tvm_device = tvm::runtime::DeviceAPI::Get(tvm_device_id);
    tvm_device->SetStream(tvm_device_id, getStream());
#endif

    beginCudaGraphStreamCapture();
    for (auto &[op, kernel, perfData] : kernels) {
        if (perfData)
            kernel->compute(op, perfData, this);
        else
            kernel->compute(op, this);
    }
    auto [cudaGraphInstance, numCudaGraphNodes] = endCudaGraphStreamCapture();
    // Since one TVM packed function may contaion more than one CUDA kernel, the
    // number of captured kernels may exceed the number of operators.
    IT_ASSERT(numCudaGraphNodes >= kernels.size(),
              std::to_string(numCudaGraphNodes) +
                  " != " + std::to_string(kernels.size()));
    printf("numCudaGraphNodes = %lu\n", numCudaGraphNodes);
    return timeit(
        [&, cudaGraphInstance = cudaGraphInstance, stream = getStream()]() {
            checkCudaError(cudaGraphLaunch(cudaGraphInstance, stream));
        },
        [&, stream = getStream()]() { cudaStreamSynchronize(stream); }, 1000,
        1000);
}

} // namespace infini
