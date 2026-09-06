#include "core/graph.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "operators/gather.h"
#include "operators/reshape.h"
#include "operators/squeeze.h"
#include "operators/unary.h"
#include "operators/unsqueeze.h"
#include "test.h"
#include <limits>

namespace infini {

TEST(DynamicShape, RuntimeTensorAndReshapeValidation) {
    auto cpu = NativeCpuRuntimeObj::getInstance();
    auto g = make_ref<GraphObj>(cpu);
    auto x = g->addTensor({2, 3});
    x->setInput();
    auto target = g->addTensor({2}, DataType::Int64);
    target->setInput();
    target->dataMalloc();
    target->copyin(vector<int64_t>{0, -1});
    auto reshape = g->addOp<ReshapeObj>(x, target, nullptr);
    reshape->getOutput()->setOutput();
    EXPECT_EQ(reshape->numInputs(), 2);
    EXPECT_EQ(reshape->getOutput()->getDims(), (Shape{2, 3}));
    g->dataMalloc();
    x->copyin(vector<float>{1, 2, 3, 4, 5, 6});
    target->copyin(vector<int64_t>{3, 2});
    g->shape_infer();
    g->dataMalloc();
    cpu->run(g);
    EXPECT_EQ(reshape->getOutput()->getDims(), (Shape{3, 2}));
    EXPECT_EQ(reshape->getOutput()->copyout<float>(),
              (vector<float>{1, 2, 3, 4, 5, 6}));
    for (auto invalid :
         vector<vector<int64_t>>{{-1, -1},
                                 {4, 2},
                                 {-2, 3},
                                 {std::numeric_limits<int64_t>::max(), 1}}) {
        target->copyin(invalid);
        EXPECT_ANY_THROW(g->shape_infer());
    }
    EXPECT_ANY_THROW(g->addOp<ReshapeObj>(x, nullptr, Shape{1, 1, 0}));
    auto wrong = g->addTensor({2}, DataType::Float32);
    wrong->dataMalloc();
    EXPECT_ANY_THROW(g->addOp<ReshapeObj>(x, wrong, nullptr));
}

TEST(DynamicShape, ShapeSubgraphAndCapacityReuse) {
    auto cpu = NativeCpuRuntimeObj::getInstance();
    auto g = make_ref<GraphObj>(cpu);
    auto x = g->addTensor({1, 2, 3});
    x->setInput();
    auto index = g->addTensor({}, DataType::Int64);
    auto tail = g->addTensor({1}, DataType::Int64);
    for (auto t : {index, tail}) {
        t->setWeight();
        t->dataMalloc();
    }
    index->copyin(vector<int64_t>{0});
    tail->copyin(vector<int64_t>{-1});
    auto shape = g->addOp<ShapeObj>(x, nullptr)->getOutput();
    EXPECT_EQ(shape->getDType(), DataType::Int64);
    auto gather = g->addOp<GatherObj>(shape, index, nullptr, 0)->getOutput();
    auto i32 =
        g->addOp<CastObj>(gather, nullptr, CastType::Int642Int32)->getOutput();
    auto i64 =
        g->addOp<CastObj>(i32, nullptr, CastType::Int322Int64)->getOutput();
    auto unsqueeze =
        g->addOp<UnsqueezeObj>(i64, nullptr, Shape{0})->getOutput();
    auto target = g->addOp<ConcatObj>(TensorVec{unsqueeze, tail}, nullptr, 0)
                      ->getOutput();
    auto y = g->addOp<ReshapeObj>(x, target, nullptr)->getOutput();
    y->setOutput();
    size_t highWaterAllocations = 0;
    for (int batch : {1, 2, 8, 3, 1}) {
        x->setShape({batch, 2, 3});
        g->shape_infer();
        g->dataMalloc();
        vector<float> data(batch * 6);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = static_cast<float>(i + batch);
        x->copyin(data);
        cpu->run(g);
        EXPECT_EQ(y->getDims(), (Shape{batch, 6}));
        EXPECT_EQ(y->copyout<float>(), data);
        EXPECT_EQ(target->copyout<int64_t>(), (vector<int64_t>{batch, -1}));
        EXPECT_EQ(tail->copyout<int64_t>(), (vector<int64_t>{-1}));
        auto allocations = g->getMemoryStats().at("activation_allocations");
        if (batch == 8)
            highWaterAllocations = allocations;
        if (highWaterAllocations) {
            EXPECT_EQ(allocations, highWaterAllocations);
        }
    }
    EXPECT_EQ(g->getShapeComputeCount(), 6);
}

TEST(DynamicShape, NegativeGatherAndBounds) {
    auto cpu = NativeCpuRuntimeObj::getInstance();
    auto g = make_ref<GraphObj>(cpu);
    auto x = g->addTensor({2, 3}, DataType::Int64);
    auto index = g->addTensor({2}, DataType::Int32);
    auto op = g->addOp<GatherObj>(x, index, nullptr, -1);
    g->dataMalloc();
    x->copyin(vector<int64_t>{10, 20, 30, 40, 50, 60});
    index->copyin(vector<int32_t>{-1, 0});
    cpu->run(g);
    EXPECT_EQ(op->getOutput()->copyout<int64_t>(),
              (vector<int64_t>{30, 10, 60, 40}));
    index->copyin(vector<int32_t>{3, 0});
    EXPECT_ANY_THROW(cpu->run(g));
}

TEST(DynamicShape, SqueezeRecomputesImplicitAxes) {
    auto g = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
    auto x = g->addTensor({1, 2, 1});
    auto squeeze = g->addOp<SqueezeObj>(x, nullptr, Shape{});
    EXPECT_EQ(squeeze->getOutput()->getDims(), (Shape{2}));
    x->setShape({3, 1, 4});
    g->shape_infer();
    EXPECT_EQ(squeeze->getOutput()->getDims(), (Shape{3, 4}));
    EXPECT_ANY_THROW(g->addOp<UnsqueezeObj>(x, nullptr, Shape{0, 0}));
    EXPECT_ANY_THROW(g->addOp<SqueezeObj>(x, nullptr, Shape{0}));
}

} // namespace infini
