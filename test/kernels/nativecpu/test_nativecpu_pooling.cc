#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/pooling.h"

#include "test.h"

namespace infini {

// Every element is negative, so a maximum started at zero would be returned
// untouched and the answer would come back as a block of zeros -- a plausible
// enough looking tensor to pass a test that only weighed shapes.
TEST(MaxPool, NativeCpuNegativeInput) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto op = g->addOp<MaxPoolObj>(input, nullptr, 2, 2, 1, 1, 0, 0, 2, 2, 0);
    g->dataMalloc();
    input->copyin(vector<float>{-8, -7, -6, -5, -4, -3, -2, -1});

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    EXPECT_EQ(o->getDims(), (Shape{1, 1, 1, 2}));
    EXPECT_TRUE(o->equalData(vector<float>{-3, -1}));
}

TEST(AvgPool, NativeCpuNegativeInput) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto op = g->addOp<AvgPoolObj>(input, nullptr, 2, 2, 1, 1, 0, 0, 2, 2, 0);
    g->dataMalloc();
    input->copyin(vector<float>{-8, -7, -6, -5, -4, -3, -2, -1});

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    EXPECT_EQ(o->getDims(), (Shape{1, 1, 1, 2}));
    EXPECT_TRUE(o->equalData(vector<float>{-5.5, -3.5}));
}

// A pooling operator reads the spatial size of its input when it is built. If
// it kept that size, a later shape would be pooled as though it were the first
// one: the output shape would not move, and the kernel would step across rows
// of the new input by the old row length.
TEST(MaxPool, NativeCpuFollowsChangedShape) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto op = g->addOp<MaxPoolObj>(input, nullptr, 2, 2, 1, 1, 0, 0, 2, 2, 0);
    g->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4});
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(op->getOutput(0))->getDims(), (Shape{1, 1, 1, 1}));

    // Twice as tall and twice as wide, which pools to four values rather than
    // one. Both spatial directions move, and they do not move together.
    input->setShape(Shape{1, 1, 4, 6});
    g->shape_infer();
    g->dataMalloc();
    input->copyin(vector<float>{
        1,  2,  3,  4,  5,  6,  //
        7,  8,  9,  10, 11, 12, //
        13, 14, 15, 16, 17, 18, //
        19, 20, 21, 22, 23, 24, //
    });

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    EXPECT_EQ(o->getDims(), (Shape{1, 1, 2, 3}));
    EXPECT_TRUE(o->equalData(vector<float>{8, 10, 12, 20, 22, 24}));
}

} // namespace infini
