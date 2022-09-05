#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

TEST(Conv, ShapeInference) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    { // No pad: InfoGAN ConvTranspose_0
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 228, 1, 1});
        Tensor w0 = g->addTensor({228, 448, 2, 2});
        auto conv = g->addOp<ConvTransposed2dObj>(i0, w0, nullptr, 0, 0);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 448, 2, 2}));
    }
    { // Padded, Strided: InfoGAN ConvTranspose_3
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 448, 2, 2});
        Tensor w0 = g->addTensor({448, 256, 4, 4});
        auto conv = g->addOp<ConvTransposed2dObj>(i0, w0, nullptr, 1, 1, 2, 2);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 256, 4, 4}));
    }
    { // With output padding: GCN ConvTranspose_224
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 21, 7, 7});
        Tensor w0 = g->addTensor({21, 21, 3, 3});
        auto conv = g->addOp<ConvTransposed2dObj>(i0, w0, nullptr, 1, 1, 2, 2,
                                                  1, 1, 1, 1);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 21, 14, 14}));
    }
}

// void testConvCudnn(
//     const std::function<void(void *, size_t, DataType)> &generator,
//     vector<float> ansVec) {
//     // Construct Runtime and graph for CPU and CUDA
//     Runtime cpu = CpuRuntimeObj::getInstance(); // CPUruntime is singleton
//     Graph gCpu = make_ref<GraphObj>(cpu);
//     Runtime cuda = make_ref<CudaRuntimeObj>();
//     Graph gCuda = make_ref<GraphObj>(cuda);
//     // Set input data on CPU in a CPU Graph
//     Tensor i0Cpu = gCpu->addTensor({1, 3, 4, 4}, DataType::Float32);
//     Tensor w0Cpu = gCpu->addTensor({2, 3, 3, 3}, DataType::Float32);
//     // Malloc data for all tensors in a graph. Do we need implicit
//     allocation? gCpu->dataMalloc(); i0Cpu->setData(generator);
//     w0Cpu->setData(generator);

//     // Copy input tensors from CPU to CUDA
//     Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
//     Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
//     // Build CUDA graph
//     auto conv =
//         gCuda->addOp<ConvObj>(i0Cuda, w0Cuda, nullptr, 1, 1, 2, 1, 1, 2);
//     // allocate CUDA memory
//     gCuda->dataMalloc();
//     // Execute on CUDA
//     cuda->run(gCuda);
//     // copy output from CUDA to CPU
//     auto o0Cpu = gCpu->cloneTensor(conv->getOutput());
//     // check results on CPU
//     EXPECT_TRUE(o0Cpu->equalData(ansVec));
//     // print a tensor/operator/graph by print()
//     gCuda->print();
// }

// TEST(Conv, cuDNN) {
//     testConvCudnn(OneGenerator(),
//                   vector<float>{12, 12, 18, 18, 12, 12, 18, 18});
//     testConvCudnn(
//         IncrementalGenerator(),
//         vector<float>{4794, 4386, 8199, 7506, 11274, 10542, 20835, 19656});
// }

// TEST(Conv, tune) {
//     Runtime cpu = CpuRuntimeObj::getInstance(); // CPUruntime is singleton
//     Graph gCpu = make_ref<GraphObj>(cpu);
//     Runtime cuda = make_ref<CudaRuntimeObj>();
//     Graph gCuda = make_ref<GraphObj>(cuda);
//     // Set input data on CPU in a CPU Graph
//     Tensor i0Cpu = gCpu->addTensor({1, 3, 224, 224}, DataType::Float32);
//     Tensor w0Cpu = gCpu->addTensor({2, 3, 3, 3}, DataType::Float32);
//     // Malloc data for all tensors in a graph. Do we need implicit
//     allocation? gCpu->dataMalloc(); i0Cpu->setData(IncrementalGenerator());
//     w0Cpu->setData(IncrementalGenerator());

//     // Copy input tensors from CPU to CUDA
//     Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
//     Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
//     // Build CUDA graph
//     auto conv =
//         gCuda->addOp<ConvObj>(i0Cuda, w0Cuda, nullptr, 1, 1, 1, 1, 1, 1);
//     // allocate CUDA memory
//     gCuda->dataMalloc();
//     // Execute on CUDA
//     bool tune = true;
//     cuda->run(gCuda, tune);
// }
} // namespace infini