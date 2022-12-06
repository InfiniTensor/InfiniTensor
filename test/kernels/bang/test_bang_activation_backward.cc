#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/activation_backward.h"
#include "operators/unary.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

template <class T, class D>
void testActivationBackward(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
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
    std::cout<< "123" << std::endl;

    Graph checkGraph = make_ref<GraphObj>(bangRuntime);
    auto checkOp1 = checkGraph->addOp<AddObj>(xGpu, diffXGpu, nullptr);
    auto checkOp2 = checkGraph->addOp<AddObj>(yGpu, diffYGpu, nullptr);
    checkGraph->dataMalloc();
    bangRuntime->run(checkGraph);
    auto xSum = checkOp1->getOutput();
    auto ySum = checkOp2->getOutput();
    std::cout<< "123" << std::endl;

    Graph checkGraph2 = make_ref<GraphObj>(bangRuntime);
    auto checkOp3 = checkGraph2->addOp<D>(xSum, nullptr);
    checkGraph2->dataMalloc();
    bangRuntime->run(checkGraph2);
    auto yRes = checkOp3->getOutput();
    std::cout<< "123" << std::endl;

    auto ySumCpu = ySum->clone(cpuRuntime);
    auto yResCpu = yRes->clone(cpuRuntime);
    std::cout<< "123" << std::endl;

    ySumCpu->printData();
    yResCpu->printData();
    
    EXPECT_TRUE(ySumCpu->equalData(yResCpu));

    // Check
    EXPECT_TRUE(1);
}

TEST(cnnl_ActivationBackward, run) {
    testActivationBackward<ReluBackwardObj, ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testActivationBackward<SigmoidBackwardObj, SigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testActivationBackward<TanhBackwardObj, TanhObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
