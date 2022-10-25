#pragma once

#include "pfusion/common.h"
#include "pfusion/micro_op.h"
#include "pfusion/pointer.h"

namespace memb {
class TensorMapping {
  public:
    TensorMapping() {}
    ~TensorMapping() {}
    std::vector<int> shape, map;
};

class MetaOp {
  public:
    int id;
    int main_loop_st, main_loop_ed, numBlocks, numWarps, numReg, numSmem;
    std::vector<std::shared_ptr<MicroOp>> microOps;
    std::vector<std::shared_ptr<Pointer>> ptrs;
    std::shared_ptr<TensorMapping> mappingSrc, mappingDst;
    MetaOp() {
        static int metaOpId = 0;
        id = metaOpId++;
    }
    ~MetaOp() {}

    inline void setLoopSt(int _main_loop_st) { main_loop_st = _main_loop_st; }
    inline void setLoopEd(int _main_loop_ed) { main_loop_ed = _main_loop_ed; }
    inline int getLoopSt() { return main_loop_st; }
    inline int getLoopEd() { return main_loop_ed; }

    inline void print() {
        std::cout << "MetaOp: " << id << std::endl;
        for (auto microOp : microOps) {
            microOp->print();
        }
    }
    bool checkValid();
};

class MetaGraph {
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
    MetaGraph() {}
    ~MetaGraph() {}
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
    inline void print() {
        for (auto node : nodes) {
            std::cout << node.id << "[(";
            if (node.pred.size() > 0) {
                std::cout << node.pred[0];
            }
            for (size_t i = 1; i < node.pred.size(); i++) {
                std::cout << ", " << node.pred[i];
            }
            std::cout << ")(";
            if (node.succ.size() > 0) {
                std::cout << node.succ[0];
            }
            for (size_t i = 1; i < node.succ.size(); i++) {
                std::cout << ", " << node.succ[i];
            }
            std::cout << ")]" << std::endl;
            for (auto metaOp : node.metaOps) {
                metaOp->print();
            }
        }
    }
    std::string genHeader();
    std::string genKernelFunc();
    std::string genInvokeFunc();
    bool checkValid();
};

} // namespace memb