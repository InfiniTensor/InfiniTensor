#pragma once
#include "core/mutator.h"

namespace infini {

class DummyMutator : public Mutator {
  public:
    DummyMutator(int candidatesLimit) : Mutator(candidatesLimit){};

    virtual vector<Graph> run(const Graph &inGraph) override;
    virtual vector<Graph> mergeMultiBranch(const Graph &inGraph) override;
    virtual bool isMultiBranchMergable(const Graph &inGraph) override;
};

} // namespace infini
