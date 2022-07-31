#include "core/graph.h"
#include "test.h"

namespace it {

TEST(Graph, build) {
    Graph g = make_ref<GraphNode>();
    g->addOp(make_ref<OperatorNode>(TensorVec{}, TensorVec{}));
    g->print();
}

} // namespace it