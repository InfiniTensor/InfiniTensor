#pragma once
#include "common.h"
#include "derivator.h"
#include "expr.h"
#include "routine.h"
#include <iostream>
#include <unordered_map>

namespace nnet {

template <typename FType> class Functor;

template <typename R, typename... Args> class Functor<R(Args...)> {
  protected:
    int verbose;

    // FIXME: scope should be protected
  public:
    Functor(int _verobse = 0) : verbose(_verobse) {}
    virtual ~Functor() = default;
#define DISPATCH(CLASS)                                                        \
    case NodeType::CLASS##Type:                                                \
        return this->visit_(as<CLASS>(c), std::forward<Args>(args)...);        \
        break

#define FUNCTOR_DEFAULT                                                        \
    { return visitDefault(c, std::forward<Args>(args)...); }

    virtual R dispatch(const Expr &c, Args... args) {
        switch (c->getType()) {
            DISPATCH(ConstantNode);
            DISPATCH(BinaryOpNode);
            DISPATCH(RangeOpNode);
            DISPATCH(SubscriptNode);
            DISPATCH(TensorNode);
            DISPATCH(VarNode);
            DISPATCH(FuncNode);
        default:
            nnet_assert(0, "Unknown type");
            return R();
        }
    }

    virtual R visit_(const Constant &c, Args... args) FUNCTOR_DEFAULT;
    virtual R visit_(const BinaryOp &c, Args... args) FUNCTOR_DEFAULT;
    virtual R visit_(const RangeOp &c, Args... args) FUNCTOR_DEFAULT;
    virtual R visit_(const Subscript &c, Args... args) FUNCTOR_DEFAULT;
    virtual R visit_(const Var &c, Args... args) FUNCTOR_DEFAULT;
    virtual R visit_(const Tensor &c, Args... args) FUNCTOR_DEFAULT;
    virtual R visit_(const Func &c, Args... args) FUNCTOR_DEFAULT;
    virtual R visitDefault(const Expr &c, [[maybe_unused]] Args... args) {
        dbg(*c);
        nnet_assert(0, "Reach unimplemented visit function.");
        return R();
    };

    R operator()(const Expr &e, Args... args) {
        return dispatch(e, std::forward<Args>(args)...);
    }
#undef FUNCTOR_DEFAULT
#undef DISPATCH
};

class Mutator : public Functor<Expr()> {
  public:
    Mutator(int _verobse = 0) : Functor(_verobse) {}
    Expr visit_(const Constant &c) override;
    Expr visit_(const BinaryOp &c) override;
    Expr visit_(const RangeOp &c) override;
    Expr visit_(const Subscript &c) override;
    Expr visit_(const Var &c) override;
    Expr visit_(const Tensor &c) override;
    Expr visit_(const Func &c) override;
};

// template <typename... Args>
// class SingleStageVisitor : public Functor<void, Args...> {
//   public:
//     SingleStageVisitor(int _verobse = 0) : Functor<R, Args...>(_verobse) {}
//     // R visit(const Constant &c) override ;
//     R visit_(const BinaryOp &c) override {
//         if (verbose)
//             dbg(*c);
//         this->dispatch(c->getLhs());
//         this->dispatch(c->getRhs());
//     }
//     R visit_(const RangeOp &c) override {
//         if (verbose)
//             dbg(*c);
//         this->dispatch(ret->getSummand());
//         // NOT visit iterators and its ranges
//     }
//     R visit_(const Subscript &c) override {
//         if (verbose)
//             dbg(*c);
//         this->dispatch(ret->getObject());
//         for (size_t i = 0; i < ret->getDims(); ++i)
//             this->dispatch(ret->getIndex(i));
//     }
//     // R visit(const Var &c) override;
//     // R visit(const Tensor &c) override;
// };

// } // namespace nnet
// #include "code_gen/nnet/Visitor/ReplaceVariable.h"
// #include "code_gen/nnet/Visitor/StrideVisitor.h"
// namespace nnet {

class ExprTreeVisitor : public Functor<void(void)> {
  private:
    bool inBinary, inRange, inSub, inTensor;

  public:
    ExprTreeVisitor(bool _inBinary = 1, bool _inRange = 1, bool _inSub = 1,
                    bool _inTensor = 1, int _verobse = 0)
        : Functor(_verobse), inBinary(_inBinary), inRange(_inRange),
          inSub(_inSub), inTensor(_inTensor) {}
    void visit_(const Constant &c) override;
    void visit_(const BinaryOp &c) override;
    void visit_(const RangeOp &c) override;
    void visit_(const Subscript &c) override;
    void visit_(const Var &c) override;
    void visit_(const Tensor &c) override;
    void visit_(const Func &c) override;
};

} // namespace nnet
