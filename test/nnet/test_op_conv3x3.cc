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

constexpr int PRelu = 0, n = 1, c = 512, h = 7, w = 7, f = 512, r = 3, s = 3,
              oh = 7, ow = 7, ph = 1, pw = 1, sh = 1, sw = 1, dh = 1, dw = 1;
void testOpConv3x3Origin(bool printGraph = false) {
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
    cuda->run(gCuda, true);
    std::cout << "Time: " << cuda->getPerfTime(gCuda) << " ms" << std::endl;

    cudaProfilerStart();
    cuda->run(gCuda);
    cudaProfilerStop();
    if (printGraph) {
        // print a tensor/operator/graph by print()
        gCuda->print();
    }
}

void testOpConv3x3Optimized(bool printGraph = false) {
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);

    auto generator = IncrementalGenerator();

    string kernelName = "conv3x3ToReduce";
    vector<int> attr{n, h, w, f};

    // Build input data on CPu
    Tensor i0Cpu = gCpu->addTensor({n * h * w, c}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({f * r * s, c}, DataType::Float32);
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
            ->getOutput(); // [FRS, NHW]
    Tensor o0Cuda = gCuda->addTensor({n, f, oh, ow});
    auto anyOp = gCuda->addOpWithOutputs<AnyObj>(
        TensorVec{i1Cuda}, TensorVec{o0Cuda}, kernelName, attr);
    // anyOp->print();
    // allocate CUDA memory
    gCuda->dataMalloc();
    // std::cout << "data malloc success..." << std::endl;
    // Execute on CUDA
    cuda->run(gCuda, true);
    std::cout << "Time: " << cuda->getPerfTime(gCuda) << " ms" << std::endl;
    cudaProfilerStart();
    cuda->run(gCuda);
    cudaProfilerStop();
    // std::cout << "cuda run success..." << std::endl;
    if (printGraph) {
        // print a tensor/operator/graph by print()
        gCuda->print();
    }
}

constexpr int rounds = 100;

TEST(op_Conv3x3, origin) { testOpConv3x3Origin(); }

TEST(op_Conv3x3, optimized) { testOpConv3x3Optimized(); }

} // namespace infini
