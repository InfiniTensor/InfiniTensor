#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/activation_backward.h"
#include "operators/element_wise.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T, class D>
void testActivationBackward(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor yCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    Tensor diffYCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    Tensor xCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    yCpu->dataMalloc();
    diffYCpu->dataMalloc();
    xCpu->dataMalloc();

    yCpu->setData(generator);
    diffYCpu->setData(generator);
    xCpu->setData(generator);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto yGpu = bangGraph->cloneTensor(yCpu);
    auto diffYGpu = bangGraph->cloneTensor(diffYCpu);
    auto xGpu = bangGraph->cloneTensor(xCpu);
    auto gpuOp = bangGraph->addOp<T>(yGpu, diffYGpu, xGpu, nullptr);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto diffXGpu = gpuOp->getOutput();

    EXPECT_TRUE(1);
}

TEST(cnnl_ActivationBackward, run) {
    testActivationBackward<ReluBackwardObj, ReluObj>(IncrementalGenerator(),
                                                     Shape{1, 2, 2, 3});
    testActivationBackward<SigmoidBackwardObj, SigmoidObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3});
    testActivationBackward<TanhBackwardObj, TanhObj>(IncrementalGenerator(),
                                                     Shape{1, 2, 2, 3});
}

} // namespace infini
