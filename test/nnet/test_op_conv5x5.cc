#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "operators/any.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

constexpr int PRelu = 0, n = 16, c = 32, h = 224, w = 224, f = 1, r = 5, s = 5,
              oh = 224, ow = 224, ph = 2, pw = 2, sh = 1, sw = 1, dh = 1,
              dw = 1;
void testOpConv5x5Origin(bool printGraph = false) {
    auto generator = IncrementalGenerator();
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({n, c, h, w}, DataType::Float32);
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
    auto conv =
        gCuda->addOp<ConvObj>(i0Cuda, w0Cuda, nullptr, ph, pw, sh, sw, dh, dw);
    // allocate CUDA memory
    gCuda->dataMalloc();
    // Execute on CUDA
    cuda->run(gCuda);
    if (printGraph) {
        // print a tensor/operator/graph by print()
        gCuda->print();
    }
}

void testOpConv5x5Optimized(bool printGraph = false) {
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);

    auto generator = IncrementalGenerator();

    string kernelName = "conv5x5ToConv3x3Reduce";
    vector<int> attr{n, f, h, w};

    // Build input data on CPu
    Tensor i0Cpu = gCpu->addTensor({n, c, h, w}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({f * 4, c, 3, 3}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit
    // allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    auto i1Cuda =
        gCuda->addOp<ConvObj>(i0Cuda, w0Cuda, nullptr, ph, pw, sh, sw, dh, dw)
            ->getOutput(); // Conv3x3
    Tensor o0Cuda = gCuda->addTensor({n, f, oh, ow});
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

TEST(op_Conv5x5, origin) {
    for (int i = 0; i < rounds; ++i) {
        testOpConv5x5Origin();
    }
    cudaProfilerStart();
    testOpConv5x5Origin(true);
    cudaProfilerStop();
}

TEST(op_Conv5x5, optimized) {
    for (int i = 0; i < rounds; ++i) {
        testOpConv5x5Optimized();
    }
    cudaProfilerStart();
    testOpConv5x5Optimized(true);
    cudaProfilerStop();
}

} // namespace infini
