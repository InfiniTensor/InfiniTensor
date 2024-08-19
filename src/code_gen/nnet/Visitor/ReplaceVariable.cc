#include "code_gen/nnet/Visitor/ReplaceVariable.h"

namespace nnet {

Expr ReplaceVariable::visit_(const BinaryOp &c) {
    if (verbose)
        dbg(*c);
    if (auto mutate = match(c); mutate)
        return mutate;
    else
        return Mutator::visit_(c);
}

Expr ReplaceVariable::visit_(const Var &c) {
    if (verbose)
        dbg(*c);
    if (auto mutate = match(c); mutate)
        return mutate;
    else
        return Mutator::visit_(c);
}

Expr ReplaceVariable::visit_(const RangeOp &c) {
    if (verbose)
        dbg(*c);
    return nullptr;
}

void ReplaceVariable::set(VecExpr _pattern, VecExpr _replacement) {
    patterns = _pattern;
    replacements = _replacement;
    for (size_t i = 0; i < patterns.size(); ++i) {
        auto hash = patterns[i]->hash();
        assert(patternHash.count(hash) == 0);
        patternHash[hash] = i;
    }
}

Expr ReplaceVariable::match(const Expr &c) {
    auto hash = c->hash();
    if (auto it = patternHash.find(hash); it != patternHash.end()) {
        const auto &i = it->second;
        if (verbose)
            dbg("Match", *c, *patterns[i], c->hash());
        return replacements[i];
    }
    return nullptr;
}

} // namespace nnet