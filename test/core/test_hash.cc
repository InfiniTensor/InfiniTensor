#include "core/graph.h"
#include "core/run_enigne.h"
#include "operators/matmul.h"
#include "test.h"

namespace infini {

TEST(Hash, OperatorHash) {
    OpPerfKey key1(0, OpType::Unknown), key2(0, OpType::Unknown);
    { // build with addOpWithOutputs
        Graph g = make_ref<GraphNode>();
        Tensor i0 = g->addTensor({1, 2, 3}, DataType::Int32);
        Tensor w0 = g->addTensor({1, 3, 4}, DataType::Int32);
        Tensor o0 = g->addTensor({1, 2, 4}, DataType::Int32);
        auto matmul = g->addOpWithOutputs<MatmulNode>(i0, w0, o0);
        key1 = matmul->getOpPerfKey();
        EXPECT_NE(key1.hash, 0);
        EXPECT_GT(key1.attrs.size(), 5);
    }
    { // build with addOp
        Graph g = make_ref<GraphNode>();
        Tensor i0 = g->addTensor({2, 2, 3}, DataType::Int32);
        Tensor w0 = g->addTensor({2, 3, 4}, DataType::Int32);
        auto matmul = g->addOp<MatmulNode>(i0, w0, nullptr);
        key2 = matmul->getOpPerfKey();
        EXPECT_NE(key2.hash, 0);
    }
    EXPECT_NE(key1.hash, key2.hash);
}

} // namespace infini