#pragma once

#include "pfusion/meta_graph.h"
#include "pfusion/meta_op.h"

namespace memb {
class SearchGraph {
  private:
    class Node {
      public:
        int id;
        std::vector<std::shared_ptr<MetaOp>> metaOps;
        std::vector<int> pred;
        std::vector<int> succ;
    };
    // each node is a vector of metaOps.
    std::vector<Node> nodes;
    std::vector<std::pair<int, int>> edges;

  public:
    SearchGraph() {}
    ~SearchGraph() {}
    inline void addNode(std::vector<std::shared_ptr<MetaOp>> metaOps) {
        Node node;
        node.id = nodes.size();
        for (auto metaOp : metaOps) {
            node.metaOps.emplace_back(metaOp);
        }
        nodes.emplace_back(node);
    }
    inline void addEdge(int i, int j) {
        edges.emplace_back(i, j);
        nodes[i].succ.emplace_back(j);
        nodes[j].pred.emplace_back(i);
    }
    std::shared_ptr<MetaGraph> exportFirstMetaGraph();
};

} // namespace memb