#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {
TEST(LeakyRelu, Cuda_WithAlpha) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({2, 2, 3, 1}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, -1.5, -1.0, 1.0, 2.0, 3.0});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputGpu = gCuda->cloneTensor(input);

    float alpha = 0.01;
    auto op = gCuda->addOp<LeakyReluObj>(inputGpu, nullptr, alpha);
    gCuda->dataMalloc();
    inputGpu->copyin(vector<float>{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, -1.5, -1.0, 1.0, 2.0, 3.0});
    cudaRuntime->run(gCuda);

    auto oCpu = gCpu->cloneTensor(op->getOutput());
    oCpu->printData();
    EXPECT_TRUE(oCpu->equalData(vector<float>{-0.01, -0.005, 0.0, 0.5, 1.0, 1.5, -0.02, -0.015, -0.01, 1.0, 2.0, 3.0}));
}

// TEST(LeakyRelu, Cuda_DefaultAlpha) {
//     Runtime runtime = NativeCpuRuntimeObj::getInstance();
//     Graph gCpu = make_ref<GraphObj>(runtime);

//     auto input = gCpu->addTensor({2, 2, 3, 1}, DataType::Float32);
//     gCpu->dataMalloc();
//     input->copyin(vector<float>{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, -1.5, -1.0, 1.0, 2.0, 3.0});

//     auto cudaRuntime = make_ref<CudaRuntimeObj>();
//     Graph gCuda = make_ref<GraphObj>(cudaRuntime);

//     auto inputGpu = gCuda->cloneTensor(input);

//     auto op = gCuda->addOp<LeakyReluObj>(inputGpu, nullptr, std::nullopt);
//     gCuda->dataMalloc();
//     inputGpu->copyin(vector<float>{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, -1.5, -1.0, 1.0, 2.0, 3.0});
//     cudaRuntime->run(gCuda);

//     auto oCpu = gCpu->cloneTensor(op->getOutput());
//     oCpu->printData();
//     EXPECT_TRUE(oCpu->equalData(vector<float>{-0.01, -0.005, 0.0, 0.5, 1.0, 1.5, -0.02, -0.015, -0.01, 1.0, 2.0, 3.0}));
// }

} // namespace infini
