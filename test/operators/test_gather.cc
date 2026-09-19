#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/gather.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

TEST(Gather, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 3, 4, 4}, DataType::Int32);
        Tensor index = g->addTensor({2, 1, 2}, DataType::Int32);
        auto op = g->addOp<GatherObj>(i, index, nullptr, 1);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 2, 1, 2, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 3, 4, 4}, DataType::Int32);
        Tensor index = g->addTensor({2, 1, 2}, DataType::Int64);
        auto op = g->addOp<GatherObj>(i, index, nullptr, 1);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 2, 1, 2, 4, 4}));
    }
}

TEST(Gather, ValidatesSignedIndexRange) {
    for (auto dtype : {DataType::Int32, DataType::Int64}) {
        SCOPED_TRACE(dtype.getIndex());
        Graph g = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
        auto data = g->addTensor({2, 3, 4}, DataType::Float32);
        auto index = g->addTensor({1}, dtype);
        auto op = g->addOp<GatherObj>(data, index, nullptr, 1);
        index->dataMalloc();
        auto setIndex = [&](int64_t value) {
            if (dtype == DataType::Int32)
                index->copyin(vector<int32_t>{static_cast<int32_t>(value)});
            else
                index->copyin(vector<int64_t>{value});
        };
        // ONNX accepts both endpoints of [-axis_length, axis_length - 1].
        for (int64_t value : {-3, -1, 0, 2}) {
            SCOPED_TRACE(value);
            setIndex(value);
            EXPECT_NO_THROW(g->shape_infer());
        }
        for (int64_t value : {-4, 3}) {
            SCOPED_TRACE(value);
            setIndex(value);
            EXPECT_THROW(g->shape_infer(), Exception);
        }
    }
}

TEST(Gather, ComputedIndicesUseCurrentShape) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto data = g->addTensor({4}, DataType::Float32);
    auto sized = g->addTensor({3}, DataType::Float32);
    auto indices = g->addOp<ShapeObj>(sized, nullptr)->getOutput();
    auto gather = g->addOp<GatherObj>(data, indices, nullptr, 0);

    for (int length : {4, 2, 5, 3}) {
        SCOPED_TRACE(length);
        data->setShape({length});
        sized->setShape({length - 1});
        // The producer's shape value is current, but after the first run its
        // data buffer still contains the index from the previous shape.
        ASSERT_NO_THROW(g->shape_infer());
        g->dataMalloc();
        data->setData(IncrementalGenerator());
        runtime->run(g);
        EXPECT_TRUE(gather->getOutput()->equalData(
            vector<float>{static_cast<float>(length - 1)}));
    }
    // Conversely, a valid old buffer must not hide a new out-of-range value.
    sized->setShape({4});
    EXPECT_THROW(g->shape_infer(), Exception);
}
} // namespace infini
