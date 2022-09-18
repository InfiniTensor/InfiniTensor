#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "test.h"
#include "utils/dataloader.h"

namespace infini {

TEST(PerfEngine, save_and_load) {
    Runtime cpu = CpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({1, 3, 224, 224}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({2, 3, 3, 3}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv =
        gCuda->addOp<ConvObj>(i0Cuda, w0Cuda, nullptr, 1, 1, 1, 1, 1, 1);

    auto ACpu = gCpu->addTensor(Shape{1, 3, 5}, DataType::Float32);
    auto BCpu = gCpu->addTensor(Shape{1, 5, 2}, DataType::Float32);
    gCpu->dataMalloc();
    ACpu->setData(IncrementalGenerator());
    BCpu->setData(IncrementalGenerator());

    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    auto ACuda = gCuda->cloneTensor(ACpu);
    auto BCuda = gCuda->cloneTensor(BCpu);
    auto matmul = gCuda->addOp<MatmulObj>(ACuda, BCuda, nullptr);

    gCuda->dataMalloc();
    cudaRuntime->run(gCuda, true);
    auto perfEngine = PerfEngine::getInstance();
    json j0 = perfEngine;
    std::cout << "Origin PerfEngine:" << std::endl;
    std::cout << j0 << std::endl;
    savePerfEngineData(perfEngine, "test.json");
    loadPerfEngineData(perfEngine, "test.json");
    json j1 = perfEngine;
    std::cout << "PerfEngine loaded:" << std::endl;
    std::cout << j1 << std::endl;
    EXPECT_TRUE(j0 == j1);
}
} // namespace infini