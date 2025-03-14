#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "test.h"

namespace infini {

TEST(PerfEngine, save_and_load) {
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    { // Conv
        Graph gCuda = make_ref<GraphObj>(cuda);

        // Copy input tensors from CPU to CUDA
        Tensor i0Cuda = gCuda->addTensor({1, 3, 224, 224}, DataType::Float32);
        Tensor w0Cuda = gCuda->addTensor({2, 3, 3, 3}, DataType::Float32);
        // Build CUDA graph
        auto conv = gCuda->addOp<ConvObj>(i0Cuda, w0Cuda, nullptr, 1, 1,
                                          nullptr, 1, 1, 1, 1);
        gCuda->dataMalloc();
        cuda->run(gCuda, true);
    }

    { // Matmul
        Graph gCuda = make_ref<GraphObj>(cuda);
        auto ACuda = gCuda->addTensor(Shape{1, 3, 5}, DataType::Float32);
        auto BCuda = gCuda->addTensor(Shape{1, 5, 2}, DataType::Float32);
        auto matmul = gCuda->addOp<MatmulObj>(ACuda, BCuda, nullptr);
        gCuda->dataMalloc();
        cuda->run(gCuda, true);
    }
    auto &perfEngine = PerfEngine::getInstance();

    json j0 = perfEngine;
    std::cout << "PerfEngine saveed:" << std::endl;
    std::cout << j0 << std::endl;
    perfEngine.savePerfEngineData("test.json");
    perfEngine.loadPerfEngineData("test.json");
    json j1 = perfEngine;
    std::cout << "PerfEngine loaded:" << std::endl;
    std::cout << j1 << std::endl;
    EXPECT_TRUE(j0 == j1);
}
} // namespace infini
