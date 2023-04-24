#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "operators/any.h"
#include "operators/conv.h"
#include "operators/matmul.h"

#include "test.h"

namespace infini {

constexpr int PRelu = 0, n = 16, c = 256, h = 2, w = 2, f = 448, r = 4, s = 4,
              oh = 4, ow = 4, ph = 1, pw = 1, sh = 2, sw = 2, dh = 1, dw = 1;
void testOpConvtransOrigin(bool printGraph = false) {
    auto generator = IncrementalGenerator();
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({n, f, h, w}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({f, c, r, s}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit
    // allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv = gCuda->addOp<ConvTransposed2dObj>(i0Cuda, w0Cuda, nullptr, ph,
                                                  pw, sh, sw, dh, dw);
    // allocate CUDA memory
    gCuda->dataMalloc();
    // Execute on CUDA
    cuda->run(gCuda);
    if (printGraph) {
        // print a tensor/operator/graph by print()
        gCuda->print();
    }
}

void testOpConvtransOptimized(bool printGraph = false) {
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);

    auto generator = IncrementalGenerator();

    string kernelName = "convTranspose2dreduce_kernel";
    vector<int> attr{PRelu, n, h, w, f, r, s, oh, ow, ph, pw, sh, sw, dh, dw};

    // Build input data on CPu
    Tensor i0Cpu = gCpu->addTensor({n * h * w, f}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({c * r * s, f}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit
    // allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    Tensor i1Cuda =
        gCuda->addOp<MatmulObj>(w0Cuda, i0Cuda, nullptr, false, true)
            ->getOutput(); // [NHW, CRS]
    Tensor o0Cuda = gCuda->addTensor({n, c, oh, ow});
    auto anyOp = gCuda->addOpWithOutputs<AnyObj>(
        TensorVec{i1Cuda}, TensorVec{o0Cuda}, kernelName, attr);
    // anyOp->print();
    // allocate CUDA memory
    gCuda->dataMalloc();
    // std::cout << "data malloc success..." << std::endl;
    // Execute on CUDA
    cuda->run(gCuda);
    // std::cout << "cuda run success..." << std::endl;
    if (printGraph) {
        // print a tensor/operator/graph by print()
        gCuda->print();
    }
}

constexpr int rounds = 100;

TEST(op_Convtrans, origin) {
    for (int i = 0; i < rounds; ++i) {
        testOpConvtransOrigin();
    }
    cudaProfilerStart();
    testOpConvtransOrigin(true);
    cudaProfilerStop();
}

TEST(op_Convtrans, optimized) {
    for (int i = 0; i < rounds; ++i) {
        testOpConvtransOptimized();
    }
    cudaProfilerStart();
    testOpConvtransOptimized(true);
    cudaProfilerStop();
}

} // namespace infini
