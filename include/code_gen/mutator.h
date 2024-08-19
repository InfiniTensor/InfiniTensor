#pragma once
#include "code_gen/graph.h"

namespace tpm {

class Mutator {
  public:
    enum SGType {
        Empty,
        NormalConv,
        NormalOddConv,
        DilatedConv,
        TransKernelConv,
        GroupConv,
        TransposeGroupConv,
        Conv1X1,
        NormalMatmul,
        BatchMatmul,
        HetConv,
        Others,
    };

    Mutator(){};
    virtual ~Mutator(){};

    virtual void run(SubGraph *in_graph, std::vector<SubGraph *> &out_graphs,
                     int mdepth = -1,
                     std::vector<std::shared_ptr<Operator>> candidate_ops = {},
                     float threshold = 0.7) = 0;

    virtual SGType statGraph(SubGraph *sg) = 0;

    virtual uint64_t computeHashForSingleComputeOp(const Operator *op) = 0;
};

} // namespace tpm