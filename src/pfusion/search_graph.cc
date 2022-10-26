#include "pfusion/search_graph.h"

namespace memb {
std::shared_ptr<MetaGraph> SearchGraph::exportFirstMetaGraph() {
    auto metaGraph = std::make_shared<MetaGraph>();
    for (auto node : nodes) {
        metaGraph->addOp(node.metaOps[0]);
    }
    for (auto edge : edges) {
        metaGraph->addEdge(nodes[edge.first].metaOps[0],
                           nodes[edge.second].metaOps[0]);
    }
    return metaGraph;
}
} // namespace memb