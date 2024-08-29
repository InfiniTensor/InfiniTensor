#include "code_gen/nnet/Visitor/HashVisitor.h"
#include "code_gen/nnet/Visitor/FullPrinterVisitor.h"
#include "code_gen/nnet/Visitor/SimplifyExprVisitor.h"
namespace nnet {

constexpr int varPrefix = 11027;
constexpr int binPrefix = 11047;
constexpr int ssPrefix = 11057;
constexpr int addPrefix = 11059;
constexpr int mulPrefix = 11069;
constexpr int vecPrefix = 11071;
constexpr int tensorPrefix = 11083;
constexpr int valSuffix = 6214587;

static inline HashType hash(const HashType a, const HashType b) {
    return (a * 10007 + b + 12345) % 1000000007;
}

static inline HashType hash(const std::string &s) {
    HashType ret = 0;
    for (auto c : s)
        ret = hash(ret, c);
    return ret;
}

static inline HashType hash(const OpType c) { return HashType(c); }

HashType HashVisitor::getHash(const Expr &c) { return dispatch(c); }

HashType HashVisitor::visit_(const Constant &c) {
    auto val = c->getValue();
    return genhash(val, valSuffix);
}

HashType HashVisitor::visit_(const BinaryOp &c) {
    HashType hasha = dispatch(c->getLhs());
    HashType hashb = dispatch(c->getRhs());

    if (c->isSwapable()) {
        if (hasha > hashb) {
            std::swap(hasha, hashb);
        }
    }
    return hash(binPrefix, hash(hash(c->getOpType()), hash(hasha, hashb)));
    return 0;
}

HashType hashLoopVar(const int id, const Range &range) {
    return hash(varPrefix, hash(id, hash(range.first, range.second)));
}

HashType hashSumVar(const Range &range) {
    return hash(varPrefix, hash(range.first, range.second));
}

HashType HashVisitor::visit_(const RangeOp &c) {
    // Identify loop variables
    for (const auto &[var, range] : c->getLoopVarRanges()) {
        nnet_assert(varHash.find(var) == varHash.end(),
                    "In HashVisiter::RangeOp invalid loop var.");
        varHash[var] = hashLoopVar(nLoopVars++, range);
    }

    // Identify sum variables according to range
    for (const auto &[var, range] : c->getSumVarRanges()) {
        nnet_assert(varHash.find(var) == varHash.end(),
                    "In HashVisiter::RangeOp invalid sum var.");
        varHash[var] = hashSumVar(range);
    }

    auto expr = c->getSummand();
    return dispatch(expr);
}

HashType HashVisitor::visit_(const Subscript &c) {
    HashType curHash = ssPrefix;
    auto obj = c->getObject();
    if (obj->getType() == NodeType::RangeOpNodeType) {
        curHash = hash(curHash, HashVisitor().getHash(obj));
    } else if (obj->getType() == NodeType::TensorNodeType) {
        // TODO: hash should based on arguments
        curHash = hash(curHash, dispatch(obj));
    } else {
        nnet_unimplemented_halt();
    }

    for (const auto &expr : c->getIndex()) {
        if (expr->getType() == NodeType::BinaryOpNodeType) {
            HashType tmp = addPrefix;
            std::vector<std::pair<HashType, HashType>> coefficients;
            auto seVisitor = SimplifyExprVisitor();
            auto [c, x] = seVisitor.getStridesConstant(expr);
            for (const auto &[key, value] : c) {
                coefficients.emplace_back(varHash[key], value);
            }
            for (const auto &[iter, value] : seVisitor.getDivStrides()) {
                nnet_assert(iter.second != 1, "invalid div expr");
                coefficients.emplace_back(
                    hash(binPrefix, hash(varHash[iter.first], iter.second)),
                    value);
            }
            sort(coefficients.begin(), coefficients.end());
            tmp = hash(tmp, x);
            for (const auto &[key, value] : coefficients) {
                tmp = hash(tmp, hash(mulPrefix, hash(key, value)));
            }
            curHash = hash(curHash, tmp);
            continue;
        }
        if (expr->getType() == NodeType::ConstantNodeType) {
            curHash = hash(curHash, dispatch(expr));
            continue;
        }
        if (expr->getType() == NodeType::VarNodeType) {
            curHash = hash(curHash, dispatch(expr));
            continue;
        }
        nnet_unimplemented_halt();
    }
    return curHash;
}

HashType hashPadding(const std::vector<int> &pad) {
    HashType cur = hash(vecPrefix, pad.size());
    for (const auto &e : pad) {
        cur = hash(cur, e);
    }
    return cur;
}

HashType HashVisitor::visit_(const Tensor &c) {
    // TODO: remove this
    // TODO: check if hash name includes padding.

    if (c->getSource() == nullptr) {
        return hash(tensorPrefix, genhash(c->getName()));
    }
    // dbg(c, c->getSource()->getExpr(), FullPrinterVisitor().print(c),
    //     FullPrinterVisitor().print(c->getSource()->getExpr()));
    // std::cout << "Tensor: " << int(c->getSource()->getExpr()->getType())
    //           << std::endl;
    // std::cout << "Tensor: " << c->getSource()->getExpr()->toReadable()
    //           << std::endl;
    return hash(tensorPrefix,
                HashVisitor().dispatch(c->getSource()->getExpr()));
}

HashType HashVisitor::visit_(const Var &c) {
    if (varHash.find(c) == varHash.end()) {
        nnet_unimplemented_halt();
        return 0;
    }
    return varHash[c];
}

} // namespace nnet