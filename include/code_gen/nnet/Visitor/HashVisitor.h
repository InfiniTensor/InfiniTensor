#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

// Calculate hash for a normal form, starting at a RangeOp
class HashVisitor : public Functor<HashType(void)> {
    inline const static HashType BKDR_SEED[] = {131, 313, 10007, 65599};

    PtrUmap<Iterator, int> varHash;
    int nLoopVars = 0;
    PtrUmap<Iterator, int> name2id;
    vector<int> rootId;
    vector<bool> haveAlias;
    int nVars = 0;
    vector<HashType> power;

  private:
    HashType visit_(const Constant &c) override;
    HashType visit_(const BinaryOp &c) override;
    HashType visit_(const RangeOp &c) override;
    HashType visit_(const Subscript &c) override;
    HashType visit_(const Tensor &c) override;
    HashType visit_(const Var &c) override;

  public:
    HashVisitor(int _verobse = 0) : Functor(_verobse) {}
    HashType getHash(const Expr &c);
};

} // namespace nnet