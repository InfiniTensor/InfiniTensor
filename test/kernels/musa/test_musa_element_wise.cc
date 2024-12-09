#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "musa/musa_runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
template <class T>
void testElementWise(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const ExpectOutput &ansVec) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto musaRuntime = make_ref<MusaRuntimeObj>();

    // Build input data on CPU
    Tensor acpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    Tensor bcpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // Build Musa graph
    Graph g = make_ref<GraphObj>(musaRuntime);
    auto a = g->cloneTensor(acpu);
    auto b = g->cloneTensor(bcpu);
    auto op = g->addOp<T>(a, b, nullptr);

    // allocate MUSA memory
    g->dataMalloc();
    a->setData(generator);
    b->setData(generator);

    // Execute on MUSA
    musaRuntime->run(g);

    // clone MUSA output to CPU
    auto c = op->getOutput();
    auto ccpu = c->clone(cpuRuntime);
    // musaPrintTensor(c);
    //  check results on CPU
    EXPECT_TRUE(ccpu->equalData(ansVec));
}

TEST(ElementWise, run) {
    testElementWise<AddObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22});
    // testElementWise<SubObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
    //                         ExpectOutput{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //                         0});
    // testElementWise<MulObj>(
    //     IncrementalGenerator(), Shape{1, 2, 2, 3},
    //     ExpectOutput{0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121});

    // testElementWise<DivObj>(OneGenerator(), Shape{1, 2, 2, 3},
    //                         ExpectOutput{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    //                         1});
    // testElementWise<MinimumObj>(
    //     IncrementalGenerator(), Shape{1, 2, 2, 3},
    //     ExpectOutput{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // testElementWise<MaximumObj>(
    //     IncrementalGenerator(), Shape{1, 2, 2, 3},
    //     ExpectOutput{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // testElementWise<PowObj>(IncrementalGenerator(), Shape{1, 2, 2, 1},
    //                         ExpectOutput{1, 1, 4, 27});
}

} // namespace infini
