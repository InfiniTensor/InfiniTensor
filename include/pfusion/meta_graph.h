#pragma once

#include "pfusion/meta_op.h"

namespace memb {
class MetaGraph {
  private:
    std::vector<std::shared_ptr<MetaOp>> metaOps;
    std::vector<std::pair<size_t, size_t>> edges;
    std::unordered_map<size_t, size_t> metaOpMap;

  public:
    MetaGraph() {}
    ~MetaGraph() {}
    inline void addOp(std::shared_ptr<MetaOp> op) {
        IT_ASSERT(metaOpMap.find(op->id) == metaOpMap.end());
        metaOpMap[op->id] = metaOps.size();
        metaOps.emplace_back(op);
    }
    inline void addEdge(std::shared_ptr<MetaOp> op1,
                        std::shared_ptr<MetaOp> op2) {
        IT_ASSERT(metaOpMap.find(op1->id) != metaOpMap.end());
        IT_ASSERT(metaOpMap.find(op2->id) != metaOpMap.end());
        edges.emplace_back(metaOpMap[op1->id], metaOpMap[op2->id]);
    }
    std::string genHeader();
    std::string genKernelFuncs();
    std::string genInvokeFuncs();
};

} // namespace memb
