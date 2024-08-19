#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

class ReplaceVariable : public Mutator {
    VecExpr patterns, replacements;
    map<HashType, int> patternHash;

  public:
    ReplaceVariable(Expr _pattern, Expr _replacement) : Mutator(false) {
        set({_pattern}, {_replacement});
    }
    ReplaceVariable(const map<string, pair<Expr, Expr>> &mapping)
        : Mutator(false) {
        VecExpr _patterns, _replacements;
        for (const auto &[_, v] : mapping) {
            _patterns.emplace_back(v.first);
            _replacements.emplace_back(v.second);
        }
        set(_patterns, _replacements);
    }
    Expr visit_(const BinaryOp &c) override;
    // NOT recur to the next stage
    Expr visit_(const RangeOp &c) override;
    Expr visit_(const Var &c) override;

  private:
    void set(VecExpr _pattern, VecExpr _replacement);
    Expr match(const Expr &c);
};

} // namespace nnet