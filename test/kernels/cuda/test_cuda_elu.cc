#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/elu.h"

#include "test.h"

namespace infini {

TEST(Elu, Cuda) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({2, 2, 3, 1}, DataType::Float32);
    gCpu->dataMalloc();
    input->setData(IncrementalGenerator());

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputGpu = gCuda->cloneTensor(input);
    auto op = gCuda->addOp<EluObj>(inputGpu, nullptr, 1.0f);
    gCuda->dataMalloc();
    inputGpu->setData(IncrementalGenerator());
    cudaRuntime->run(gCuda);

    auto oCpu = gCpu->cloneTensor(op->getOutput());
    oCpu->printData();
    EXPECT_TRUE(oCpu->equalData(vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.}));
}

} // namespace infini
