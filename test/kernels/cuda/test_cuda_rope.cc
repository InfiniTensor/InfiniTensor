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
        0.540302, 0.647906, 0.731761, 0.796458, 0.846009, 0.883756, 0.912396,
        0.934062, 0.950415, 0.962739, 0.972014, 0.978989, 0.98423,  0.988167,
        0.991122, 0.99334,  0.995004, 0.996253, 0.99719,  0.997892, 0.998419,
        0.998815, 0.999111, 0.999333, 0.9995,   0.999625, 0.999719, 0.999789,
        0.999842, 0.999881, 0.999911, 0.999933}));
}
} // namespace infini
