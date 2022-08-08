#pragma once
#include "nnet/visitor.h"

namespace nnet {

class MatchReshapeVisitor : public Functor<bool(void)> {
  private:
    PtrMap<Iterator, int> _coefficient;

  public:
    bool visit_(const RangeOp &c) override;
};

} // namespace nnet