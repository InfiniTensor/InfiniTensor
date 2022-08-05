#include "core/graph.h"
#include "test.h"

namespace it {

TEST(Graph, build) {
    Graph g = make_ref<GraphNode>();
    Tensor i0 = g->addTensor({1, 2, 3});
    Tensor w0 = g->addTensor({1, 3, 4});
    Tensor o0 = g->addTensor({1, 2, 4});
    g->addOp(make_ref<MatmulNode>(i0, w0, o0));
    g->print();
}

} // namespace it