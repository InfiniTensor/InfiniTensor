#pragma once

#include "assert.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace memb {

class Ptr {
  public:
    std::string base_ptr;
    std::string offset;
    Ptr() {}
};

static Ptr buildPtr(std::string _base_ptr, std::string _offset) {
    Ptr ptr;
    ptr.base_ptr = _base_ptr;
    ptr.offset = _offset;
    return ptr;
}

inline Ptr getPtrbyTensorId(int id) {
    return buildPtr("tensor_" + std::to_string(id), "0");
}

class MicroOp {
  public:
    enum MicroOpType {
        memory = 1,
        kernel,
    };

  protected:
    MicroOpType type;
    int id;

  public:
    MicroOp() {
        static int microOpId = 0;
        id = microOpId++;
    }
    virtual ~MicroOp() {}
    inline MicroOpType getType() { return type; }
    // virtual bool checkValid() {}
    virtual void print() = 0;
};

class MicroGraph {
  public:
    MicroGraph() {}
    ~MicroGraph() {}

  private:
    std::vector<std::shared_ptr<MicroOp>> microOps;
    std::vector<std::pair<int, int>> deps;
};

class MetaOp {
  public:
    int id;
    int main_loop_st, main_loop_ed, main_loop_step, parallelism;
    std::vector<std::shared_ptr<MicroOp>> microOps;
    std::vector<Ptr> ptrs;
    MetaOp() {
        static int metaOpId = 0;
        id = metaOpId++;
    }
    ~MetaOp() {}

    inline void setLoopSt(int _main_loop_st) { main_loop_st = _main_loop_st; }
    inline void setLoopEd(int _main_loop_ed) { main_loop_ed = _main_loop_ed; }
    inline void setLoopStep(int _main_loop_step) {
        main_loop_step = _main_loop_step;
    }
    inline int getLoopSt() { return main_loop_st; }
    inline int getLoopEd() { return main_loop_ed; }
    inline int getLoopStep() { return main_loop_step; }

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
    bool checkValid();
};
} // namespace memb
