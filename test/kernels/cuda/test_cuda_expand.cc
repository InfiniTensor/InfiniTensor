#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/expand.h"

#include "test.h"

namespace infini {

TEST(Expand, Cuda) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto t1 = gCpu->addTensor({2, 1, 2, 1}, DataType::Float32);

    gCpu->dataMalloc();
    t1->setData(IncrementalGenerator());
    t1->printData();

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto t1Gpu = gCuda->cloneTensor(t1);

    auto op = gCuda->addOp<ExpandObj>(t1Gpu, nullptr, Shape{2, 2, 2, 3});
    gCuda->dataMalloc();
    t1Gpu->setData(IncrementalGenerator());

    cudaRuntime->run(gCuda);

    // cudaPrintTensor(op->getOutput());
    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput());
    oCpu->printData();
    EXPECT_TRUE(
        oCpu->equalData(vector<float>{0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                      2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3}));
}

} // namespace infini
