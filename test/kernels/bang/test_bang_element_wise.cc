#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
template <class T>
void testElementWiseCnnl(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const ExpectOutput &ansVec) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor acpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    acpu->dataMalloc();
    acpu->setData(generator);

    Tensor bcpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    bcpu->dataMalloc();
    bcpu->setData(generator);

    // Build BANG graph
    Graph g = make_ref<GraphObj>(bangRuntime);
    auto a = g->cloneTensor(acpu);
    auto b = g->cloneTensor(bcpu);
    auto op = g->addOp<T>(a, b, nullptr);

    // allocate BANG memory
    g->dataMalloc();

    // Execute on BANG
    bangRuntime->run(g);

    // clone BANG output to CPU
    auto c = op->getOutput();
    auto ccpu = c->clone(cpuRuntime);
    //  check results on CPU
    EXPECT_TRUE(ccpu->equalData(ansVec));
}

TEST(cnnl_ElementWise, run) {
    testElementWiseCnnl<AddObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22});
    testElementWiseCnnl<SubObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    testElementWiseCnnl<MulObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121});
}

} // namespace infini
