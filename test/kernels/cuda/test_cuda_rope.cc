#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/rope.h"

#include "test.h"

namespace infini {
TEST(RoPE, Cuda) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();

    Graph gCpu = make_ref<GraphObj>(runtime);

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);
    auto input = gCuda->addTensor({1, 1, 32}, DataType::Float32);
    auto position_id_d = gCuda->addTensor({1, 1}, DataType::UInt32);
    auto output = gCuda->addTensor({1, 1, 32}, DataType::Float32);

    auto op = gCuda->addOpWithOutputs<RoPEObj>(position_id_d, input, output);
    gCuda->dataMalloc();

    input->setData(OneGenerator());
    position_id_d->setData(OneGenerator());
    cudaRuntime->run(gCuda);

    auto oCpu = gCpu->cloneTensor(op->getOutputs()[0]);
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1.381773, 1.381773, 1.381773, 1.381773, 1.381773, 1.381773, 1.381773,
        1.381773, 1.381773, 1.381773, 1.381773, 1.381773, 1.381773, 1.381773,
        1.381773, 1.381773, 1.381773, 1.381773, 1.381773, 1.381773, 1.381773,
        1.381773, 1.381773, 1.381773, 1.381773, 1.381773, 1.381773, 1.381773,
        1.381773, 1.381773, 1.381773, 1.381773}));
}
} // namespace infini
