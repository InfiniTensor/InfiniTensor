#include "core/graph.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
template <class T>
void testElementWiseNativeCpu(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const Shape &shape1, const Shape &shape2, const ExpectOutput &ansVec) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto t1 = g->addTensor(shape1, DataType::Float32);
    auto t2 = g->addTensor(shape2, DataType::Float32);

    auto op = g->addOp<T>(t1, t2, nullptr);
    g->dataMalloc();
    t1->setData(generator1);
    t2->setData(generator2);

    runtime->run(g);
    EXPECT_TRUE(op->getOutput()->equalData(ansVec));
}

TEST(ElementWise, NativeCpu) {
    testElementWiseNativeCpu<AddObj>(
        IncrementalGenerator(), IncrementalGenerator(), Shape{1, 2, 2, 3, 1},
        Shape{2, 1, 1}, ExpectOutput{0, 1, 2, 4, 5, 6, 6, 7, 8, 10, 11, 12});
    testElementWiseNativeCpu<MulObj>(
        IncrementalGenerator(), IncrementalGenerator(), Shape{1, 2, 2, 3, 1},
        Shape{2, 1, 1}, ExpectOutput{0, 0, 0, 3, 4, 5, 0, 0, 0, 9, 10, 11});
    testElementWiseNativeCpu<SubObj>(
        IncrementalGenerator(), IncrementalGenerator(), Shape{1, 2, 2, 3, 1},
        Shape{2, 1, 1}, ExpectOutput{0, 1, 2, 2, 3, 4, 6, 7, 8, 8, 9, 10});
    testElementWiseNativeCpu<DivObj>(
        IncrementalGenerator(), OneGenerator(), Shape{1, 2, 2, 3, 1},
        Shape{2, 1, 1}, ExpectOutput{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

} // namespace infini
