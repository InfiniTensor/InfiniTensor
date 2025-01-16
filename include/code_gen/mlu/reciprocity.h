#ifndef RECIPROCITY_H
#define RECIPROCITY_H

#include "graph.h"

namespace tpm {
class Reciprocity {
    const int MAX_RECIPROCITY_DETECT_DEPTH = 3;

  public:
    Reciprocity(const std::vector<std::shared_ptr<Operator>> &ops);
    // find reciprocities among given ops
    void search_reciprocity(const std::vector<std::shared_ptr<Operator>> &ops);
    // check whether the latest op and its parents are reciprocal
    bool is_tail_reciprocity(const OpVec &oplist);
    // check whether all the ops are reciprocal
    bool is_reciprocity(const OpVec &oplist);

    // reciprocal ops chains detected by search_reciprocity
    std::vector<std::vector<uint64_t>> reciprocal_op_chains;

    int maxDetectDepth() const { return MAX_RECIPROCITY_DETECT_DEPTH; }
};
} // end of namespace tpm

#endif
