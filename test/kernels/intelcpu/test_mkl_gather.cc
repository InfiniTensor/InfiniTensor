#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/gather.h"

#include "test.h"

namespace infini {
TEST(Gather, Cuda) {
    {
        Runtime runtime = MklRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        auto input = g->addTensor({3, 2}, DataType::Float32);
        auto index = g->addTensor({2, 2}, DataType::UInt32);
        g->dataMalloc();
        input->copyin(vector<float>{1, 2, 3, 4, 5, 6});
        index->copyin(vector<uint32_t>{0, 1, 1, 2});

        auto op = g->addOp<GatherObj>(input, index, nullptr, 0);
        g->dataMalloc();
        runtime->run(g);

        EXPECT_TRUE(
            op->getOutput()->equalData(vector<float>{1, 2, 3, 4, 3, 4, 5, 6}));
    }
    {
        Runtime runtime = MklRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        auto input = g->addTensor({3, 3}, DataType::Float32);
        auto index = g->addTensor({1, 2}, DataType::UInt32);
        g->dataMalloc();
        input->setData(IncrementalGenerator());
        index->copyin(vector<uint32_t>{0, 2});

        auto op = g->addOp<GatherObj>(input, index, nullptr, 1);
        g->dataMalloc();
        runtime->run(g);

        EXPECT_TRUE(
            op->getOutput()->equalData(vector<float>{0, 2, 3, 5, 6, 8}));
    }
    {
        Runtime runtime = MklRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        auto input = g->addTensor({2, 4, 2}, DataType::Float32);
        auto index = g->addTensor({3, 1}, DataType::UInt32);
        g->dataMalloc();
        input->setData(IncrementalGenerator());
        index->copyin(vector<uint32_t>{0, 3, 1});

        auto op = g->addOp<GatherObj>(input, index, nullptr, 1);
        g->dataMalloc();
        runtime->run(g);

        EXPECT_TRUE(op->getOutput()->equalData(
            vector<float>{0, 1, 6, 7, 2, 3, 8, 9, 14, 15, 10, 11}));
    }
}
} // namespace infini
