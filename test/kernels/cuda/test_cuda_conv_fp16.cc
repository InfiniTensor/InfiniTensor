#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"
#include <bitset>

#include "test.h"

namespace infini {

void testConvCudnnFP16(
    const std::function<void(void *, size_t, DataType)> &generator,
    vector<float> ansVec) {

    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({1, 3, 4, 4}, DataType::Float16);
    Tensor w0Cpu = gCpu->addTensor({2, 3, 3, 3}, DataType::Float16);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv = gCuda->addOp<ConvObj>(i0Cuda, w0Cuda, nullptr, 1, 1, nullptr, 2,
                                      1, 1, 2);
    // allocate CUDA memory
    gCuda->dataMalloc();
    i0Cuda->setData(generator);
    w0Cuda->setData(generator);
    // Execute on CUDA
    cuda->run(gCuda);
    // copy output from CUDA to CPU
    auto o0Cpu = gCpu->cloneTensor(conv->getOutput());
    // check results on CPU
    EXPECT_TRUE(o0Cpu->equalData(ansVec));
    // print a tensor/operator/graph by print()
    gCuda->print();
}

TEST(cuDNN_Conv_FP16, run) {
    testConvCudnnFP16(IncrementalGenerator(),
                      vector<float>{48, 48, 72, 72, 48, 48, 72, 72});
}

TEST(cuDNN_Conv_FP16, tune) {
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({1, 3, 224, 224}, DataType::Float16);
    Tensor w0Cpu = gCpu->addTensor({2, 3, 3, 3}, DataType::Float16);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv = gCuda->addOp<ConvObj>(i0Cuda, w0Cuda, nullptr, 1, 1, nullptr, 1,
                                      1, 1, 1);
    // allocate CUDA memory
    gCuda->dataMalloc();
    i0Cuda->setData(IncrementalGenerator());
    w0Cuda->setData(IncrementalGenerator());
    // Execute on CUDA
    bool tune = true;
    cuda->run(gCuda, tune);
}
} // namespace infini
