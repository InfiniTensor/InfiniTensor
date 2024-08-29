#include "code_gen/nnet/Visitor/Interpreter.h"
#include "code_gen/nnet/Visitor/GetTensorsVisitor.h"
#include "code_gen/nnet/expr.h"

namespace nnet {

using ttype = Interpreter::ttype; // Test data type
using rtype = Interpreter::rtype; // Return data type
using Position = Interpreter::Position;
using Inputs = Interpreter::Inputs;
using Iteration = Interpreter::Iteration;

Inputs Interpreter::genInputStartingFromZero(const RangeOp &range) {
    Inputs inputs;
    GetTensorsVisitor getTensorsVisitor;
    auto tensors = getTensorsVisitor.get(range);

    for (const auto &[name, tensor] : tensors) {
        auto data = make_ref<vector<int>>(tensor->getSize());
        for (ssize_t i = 0; i < tensor->getSize(); i++) {
            data->at(i) = i;
        }
        inputs.emplace(name, data);
    }
    return inputs;
}

Interpreter::Interpreter(RangeOp range, int _verbose)
    : Interpreter(genInputStartingFromZero(range), _verbose){};

rtype Interpreter::visit_(const Constant &c) { return c->getValue(); }

rtype Interpreter::visit_(const BinaryOp &c) {
    rtype valueL = dispatch(c->getLhs()), valueR = dispatch(c->getRhs());

    switch (c->getOpType()) {
    case OpType::Add:
        return valueL + valueR;
    case OpType::Mul:
        return valueL * valueR;
    case OpType::Div:
        nnet_assert(valueR > 0, "Negative divisor is ill-defeind");
        return valueL / valueR;
    case OpType::Mod:
        nnet_assert(valueR > 0, "Negative divisor is ill-defeind");
        return valueL % valueR;
    case OpType::Sub:
        return valueL - valueR;
    default:
        nnet_unimplemented_halt();
        return -1;
    }
}

rtype Interpreter::visit_(const RangeOp &c) {
    rtype ret = 0;
    iterations.emplace_back();
    // loop
    auto loopRanges = c->getLoopVarRanges();
    assert(positions.back().size() == loopRanges.size());
    auto paddings = c->getPaddings();
    for (int i = 0, iEnd = loopRanges.size(); i < iEnd; i++) {
        int left = loopRanges[i].second.first;
        int right = loopRanges[i].second.second;
        int padding = paddings[i];
        int element = positions.back()[i];
        if (0 < padding) {
            nnet_assert(left - padding <= element, "Out of range");
            nnet_assert(element < right + padding, "Out of range");
            if (left <= element && element < right) {
                iterations.back()[loopRanges[i].first] = positions.back()[i];
            } else {
                iterations.pop_back();
                return 0;
            }
        } else {
            nnet_assert(left <= element, "Out of range");
            nnet_assert(element < right, "Out of range");
            iterations.back()[loopRanges[i].first] = positions.back()[i];
        }
    }
    // sum
    auto sumVarRanges = c->getSumVarRanges();
    int nSumIters = sumVarRanges.size();
    if (0 < nSumIters) {
        vector<int> sumIterValues(nSumIters);
        for (const auto &[var, range] : sumVarRanges) {
            sumIterValues.emplace_back(range.first);
            nnet_assert(range.first < range.second, "No empty range");
        }
        // Enumerate all values of sum iterator
        do {
            for (int i = 0; i < nSumIters; i++)
                iterations.back()[sumVarRanges[i].first] = sumIterValues[i];
            ret += dispatch(c->getSummand());

            // Increase with carry to enumerate sum iterators
            sumIterValues[nSumIters - 1]++;
            for (int i = nSumIters - 1; 0 < i; i--) {
                if (sumIterValues[i] == sumVarRanges[i].second.second) {
                    sumIterValues[i] = sumVarRanges[i].second.first;
                    sumIterValues[i - 1]++;
                }
            }
        } while (sumIterValues[0] < sumVarRanges[0].second.second);
    } else {
        ret += dispatch(c->getSummand());
    }
    iterations.pop_back();
    return ret;
}

rtype Interpreter::visit_(const Subscript &c) {
    int ret = 0;
    vector<int> idx;
    auto sub = c->getIndex();
    for (int i = 0, iEnd = sub.size(); i < iEnd; i++) {
        idx.emplace_back(dispatch(sub[i]));
    }

    auto obj = c->getObject();
    if (obj->getType() == NodeType::RangeOpNodeType) {
        positions.emplace_back(idx);
        ret = dispatch(obj);
        positions.pop_back();
    } else if (obj->getType() == NodeType::TensorNodeType) {
        auto tensor = as<TensorNode>(obj);
        const auto &data = inputs[tensor->getName()];
        ret = tensor->getData(data, idx);
    } else
        assert(false);
    return ret;
}

rtype Interpreter::visit_(const Var &c) { return iterations.back()[c]; }

rtype Interpreter::visit_(const Tensor &c) {
    nnet_unimplemented_halt();
    return -1;
}

vector<rtype> Interpreter::interpret(const Expr &expr,
                                     const vector<Position> &poses) {
    vector<rtype> ret;
    for (const auto &pos : poses) {
        positions.emplace_back(pos);
        ret.emplace_back(dispatch(expr));
        positions.pop_back();
    }
    return ret;
}

vector<rtype> Interpreter::interpretUniformSample(const RangeOp &range,
                                                  int nPoses) {
    vector<Interpreter::Position> poses;
    const auto &rangeShape = range->getOutputShape();
    for (int i = 0; i < nPoses; ++i) {
        Interpreter::Position pos(range->getNumOutputDims(), 0);
        ssize_t t = range->getOutputSize() / nPoses * i;
        for (int j = range->getNumOutputDims() - 1; j >= 0; --j) {
            int extent = rangeShape[j];
            pos[j] = t % extent;
            t /= extent;
        }
        poses.emplace_back(pos);
    }
    return interpret(range, poses);
}

vector<rtype> Interpreter::interpretAllOutput(const RangeOp &range) {
    return interpretUniformSample(range, range->getOutputSize());
}

} // namespace nnet