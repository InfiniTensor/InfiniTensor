#include "xpu/xpu_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
template <class T>
void testElementWiseXdnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const ExpectOutput &ansVec) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<XPURuntimeObj>();

    // Build input data on CPU
    Tensor acpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    acpu->dataMalloc();
    acpu->setData(generator);

    Tensor bcpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    bcpu->dataMalloc();
    bcpu->setData(generator);

    // Build XPU graph
    Graph g = make_ref<GraphObj>(xpuRuntime);
    auto a = g->cloneTensor(acpu);
    auto b = g->cloneTensor(bcpu);
    auto op = g->addOp<T>(a, b, nullptr);

    // allocate XPU memory
    g->dataMalloc();
    a->setData(generator);
    b->setData(generator);

    // Execute on XPU
    xpuRuntime->run(g);

    // clone XPU output to CPU
    auto c = op->getOutput();
    auto ccpu = c->clone(cpuRuntime);
    //  check results on CPU
    EXPECT_TRUE(ccpu->equalData(ansVec));
}

TEST(xdnn_ElementWise, run) {
    testElementWiseXdnn<AddObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22});
    testElementWiseXdnn<SubObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    testElementWiseXdnn<MulObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121});
    testElementWiseXdnn<DivObj>(
        OneGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    testElementWiseXdnn<PowObj>(IncrementalGenerator(), Shape{1, 2, 2, 1},
                                 ExpectOutput{1, 1, 4, 27});
}

} // namespace infini
