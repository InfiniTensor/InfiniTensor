#pragma once
#include "core/mutator.h"

namespace infini {

class DummyMutator : public Mutator {
  public:
    DummyMutator(int candidatesLimit) : Mutator(candidatesLimit){};

    virtual vector<Graph> run(const Graph &in_graph) override;
    virtual vector<Graph> fusion(const Graph &in_graph) override;
    virtual bool isFusible(const Graph &in_graph) override;
};

} // namespace infini
