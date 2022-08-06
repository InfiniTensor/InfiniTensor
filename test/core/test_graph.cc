#include "core/graph.h"
#include "core/run_enigne.h"
#include "test.h"

namespace it {

TEST(Graph, build) {
    Graph g = make_ref<GraphNode>();
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::Int32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::Int32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::Int32);
    g->dataMalloc();
    i0->copyData(vector<VType>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.data());
    w0->copyData(vector<VType>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.data());
    g->addOp(make_ref<MatmulNode>(i0, w0, o0));
    RunEngine(Device::CPU).run(g);
    // check answer
    auto ans = make_ref<TensorNode>(Shape{1, 2, 4}, DataType::Int32);
    ans->dataMalloc();
    ans->copyData(vector<VType>{38, 44, 50, 56, 83, 98, 113, 128}.data());
    EXPECT_TRUE(o0->equalData(ans));
}

} // namespace it