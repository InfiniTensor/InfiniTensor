#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

void testConv3dCudnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    vector<float> ansVec) {
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({1, 3, 4, 4, 4}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({2, 3, 3, 3, 3}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv3d = gCuda->addOp<Conv3dObj>(i0Cuda, w0Cuda, nullptr, 1, 1, 1, 1,
                                          2, 1, 1, 1, 2);
    // allocate CUDA memory
    gCuda->dataMalloc();
    i0Cuda->setData(generator);
    w0Cuda->setData(generator);
    // Execute on CUDA
    cuda->run(gCuda);
    // copy output from CUDA to CPU
    auto o0Cpu = gCpu->cloneTensor(conv3d->getOutput());
    // check results on CPU
    EXPECT_TRUE(o0Cpu->equalData(ansVec));
    // print a tensor/operator/graph by print()
    gCuda->print();
}

TEST(cuDNN_Conv3d, run) {
    testConv3dCudnn(OneGenerator(),
                    vector<float>{24, 24, 36, 36, 36, 36, 54, 54, 36, 36, 54,
                                  54, 24, 24, 36, 36, 24, 24, 36, 36, 36, 36,
                                  54, 54, 36, 36, 54, 54, 24, 24, 36, 36});
    testConv3dCudnn(
        IncrementalGenerator(),
        vector<float>{113412, 110496, 175914, 171378, 172062, 167562, 264681,
                      257688, 196254, 191178, 299673, 291816, 125796, 122328,
                      190818, 185454, 261156, 256296, 415026, 407574, 417006,
                      409590, 658341, 646974, 487854, 479862, 763317, 751086,
                      335748, 330336, 523242, 514962});
}

TEST(cuDNN_Conv3d, tune) {
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({1, 3, 224, 224, 224}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({2, 3, 3, 3, 3}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv3d = gCuda->addOp<Conv3dObj>(i0Cuda, w0Cuda, nullptr, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1);
    // allocate CUDA memory
    gCuda->dataMalloc();
    i0Cuda->setData(IncrementalGenerator());
    w0Cuda->setData(IncrementalGenerator());
    // Execute on CUDA
    bool tune = true;
    cuda->run(gCuda, tune);
}
} // namespace infini
