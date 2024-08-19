#ifndef DMUTATOR_H
#define DMUTATOR_H

#include "code_gen/mutator.h"

namespace tpm {

class DMutator : public Mutator {
  public:
    DMutator();
    ~DMutator();

    void run(SubGraph *in_graph, std::vector<SubGraph *> &out_graphs,
             int mdepth = -1,
             std::vector<std::shared_ptr<Operator>> candidate_ops = {},
             float threshold = 0.7);

    SGType statGraph(SubGraph *sg);

    uint64_t computeHashForSingleComputeOp(const Operator *op);
};

} // namespace tpm

#endif
