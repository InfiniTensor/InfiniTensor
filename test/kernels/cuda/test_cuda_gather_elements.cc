#include "core/graph.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "cuda/gather.h"
#include "operators/gather.h"

#include "test.h"

namespace infini {
TEST(GatherElements, intDataLongIndices) {
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputCuda = gCuda->addTensor({3, 3}, DataType::Int32);
    auto indexCuda = gCuda->addTensor({2, 3}, DataType::Int64);
    auto op = gCuda->addOp<GatherElementsObj>(inputCuda, indexCuda, nullptr, 0);
    gCuda->dataMalloc();
    inputCuda->copyin(vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    indexCuda->copyin(vector<int64_t>{1, 2, 0, 2, 0, 0});
    cudaRuntime->run(gCuda);
    auto result = op->getOutput()->clone(cpuRuntime);
    EXPECT_TRUE(result->equalData<int>({4, 8, 3, 7, 2, 3}));
}
TEST(GatherElements, floatDataIntIndices) {
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputCuda = gCuda->addTensor({2, 2}, DataType::Float32);
    auto indexCuda = gCuda->addTensor({2, 2}, DataType::Int32);
    auto op = gCuda->addOp<GatherElementsObj>(inputCuda, indexCuda, nullptr, 1);
    gCuda->dataMalloc();
    inputCuda->copyin(vector<float>{1., 2., 3., 4.});
    indexCuda->copyin(vector<int>{0, 0, 1, 0});
    cudaRuntime->run(gCuda);
    auto result = op->getOutput()->clone(cpuRuntime);
    EXPECT_TRUE(result->equalData<float>({1., 1., 4., 3.}));
}
} // namespace infini
