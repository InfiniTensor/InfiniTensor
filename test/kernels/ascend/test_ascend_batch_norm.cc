#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/batch_norm.h"

#include "test.h"

namespace infini {

TEST(ascend_BatchNorm, run) {
    aclInit(nullptr);
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build cpu graph
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto iCpu = gCpu->addTensor(Shape{1, 3, 2, 2}, DataType::Float32);
    auto meanCpu = gCpu->addTensor(Shape{3}, DataType::Float32);
    auto varCpu = gCpu->addTensor(Shape{3}, DataType::Float32);
    auto scaleCpu = gCpu->addTensor(Shape{3}, DataType::Float32);
    auto biasCpu = gCpu->addTensor(Shape{3}, DataType::Float32);

    // Build input data on CPU
    gCpu->dataMalloc();
    iCpu->setData(IncrementalGenerator());
    meanCpu->copyin(vector<float>{1, 6, 9});
    varCpu->copyin(vector<float>{4, 1, 9});
    scaleCpu->setData(OneGenerator());
    biasCpu->setData(ZeroGenerator());

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(npuRuntime);
    auto i = g->cloneTensor(iCpu);
    auto mean = g->cloneTensor(meanCpu);
    auto var = g->cloneTensor(varCpu);
    auto scale = g->cloneTensor(scaleCpu);
    auto bias = g->cloneTensor(biasCpu);
    auto op =
        g->addOp<BatchNormObj>(i, nullptr, mean, var, scale, bias, 0.9, 0);

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    npuRuntime->run(g);

    // clone CUDA output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    // check results on CPU
    EXPECT_TRUE(ocpu->equalData(vector<float>{
        -0.5, 0, 0.5, 1, -2, -1, 0, 1, -0.333333, 0, 0.333333, 0.666667}));

    aclFinalize();
}
} // namespace infini
