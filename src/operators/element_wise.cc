#include "operators/element_wise.h"
#include "utils/operator_utils.h"
#include <algorithm>
#include <limits>

namespace infini {
ElementWiseObj::ElementWiseObj(OpType type, GraphObj *graph, Tensor input0,
                               Tensor input1, Tensor output)
    : OperatorObj(type, {input0, input1}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ElementWiseObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0], B = inputs[1];
    auto res = infer_broadcast(A->getDims(), B->getDims());
    return {{res}};
}

vector<DimSource> ElementWiseObj::dimSources(size_t output, size_t dim) const {
    IT_ASSERT(output == 0);
    const auto rank = outputs[0]->getRank();
    IT_ASSERT(dim < rank);
    // Broadcasting aligns the inputs at the last dimension, so an input of
    // lesser rank does not reach the leading dimensions at all.
    vector<DimSource> sources;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto inputRank = inputs[i]->getRank();
        const auto missing = rank - inputRank;
        if (dim < missing) {
            continue;
        }
        sources.push_back(DimSource{i, dim - missing});
    }
    return sources;
}

namespace {

/// @brief One integer operation of a shape computation, or nothing when this
/// operator is not one of them.
///
/// Only exact operations on whole numbers belong here. A shape value stands for
/// a count of elements, and an answer that had to be rounded or that overflowed
/// would be wrong rather than approximate, so such a case declines to produce a
/// value at all and leaves the operator to run as it always did.
optional<int64_t> applyToDims(OpType::underlying_t type, int64_t a, int64_t b) {
    switch (type) {
    case OpType::Add:
        return a + b;
    case OpType::Sub:
        return a - b;
    case OpType::Mul:
        return a * b;
    case OpType::Max:
        return std::max(a, b);
    case OpType::Min:
        return std::min(a, b);
    case OpType::Div:
        // Division by zero has no answer to give, and the one case where the
        // quotient does not fit is the most negative value over -1.
        if (b == 0 || (a == std::numeric_limits<int64_t>::min() && b == -1)) {
            return std::nullopt;
        }
        // ONNX truncates an integer quotient towards zero, as C++ does.
        return a / b;
    case OpType::FloorDiv:
        if (b == 0 || (a == std::numeric_limits<int64_t>::min() && b == -1)) {
            return std::nullopt;
        }
        return (a / b) - (((a % b != 0) && ((a < 0) != (b < 0))) ? 1 : 0);
    case OpType::FloorMod:
        if (b == 0) {
            return std::nullopt;
        }
        return a -
               b * ((a / b) - (((a % b != 0) && ((a < 0) != (b < 0))) ? 1 : 0));
    default:
        return std::nullopt;
    }
}

} // namespace

void ElementWiseObj::inferShapeValue() {
    if (!beginShapeValueUpdate()) {
        return;
    }
    const auto &left = *inputs[0]->getShapeValue();
    const auto &right = *inputs[1]->getShapeValue();
    // Both operands are at most a single row of numbers, so the only spreading
    // to account for is a lone number standing in for a whole row.
    const auto elements = std::max(left.size(), right.size());
    if (elements != outputs[0]->size()) {
        return;
    }
    if ((left.size() != elements && left.size() != 1) ||
        (right.size() != elements && right.size() != 1)) {
        return;
    }

    vector<int64_t> value;
    vector<bool> fixed;
    value.reserve(elements);
    fixed.reserve(elements);
    for (size_t i = 0; i < elements; ++i) {
        const auto l = left.size() == 1 ? 0 : i;
        const auto r = right.size() == 1 ? 0 : i;
        const auto answer = applyToDims(type.underlying(), left[l], right[r]);
        if (!answer) {
            // No value for one element means no value for the result: a
            // half-filled row would be read as though it were complete.
            return;
        }
        value.push_back(*answer);
        // An answer is as settled as both of the numbers it came from.
        fixed.push_back(inputs[0]->isShapeValueFixed(l) &&
                        inputs[1]->isShapeValueFixed(r));
    }
    outputs[0]->setShapeValue(std::move(value), std::move(fixed));
}

std::string ElementWiseObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// use output dim or inputs dim?
vector<int> ElementWiseObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ElementWiseObj::getOpAttrVector() const {
    return {type.underlying()};
}

MSELossObj::MSELossObj(GraphObj *graph, Tensor input0, Tensor input1,
                       Reduction reduction, Tensor output)
    : OperatorObj(OpType::MSELoss, {input0, input1}, {output}),
      reductionMode(reduction) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> MSELossObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0], B = inputs[1];
    IT_ASSERT(A->getRank() == B->getRank());
    IT_ASSERT(A->getDims() == B->getDims());

    if (reductionMode == None) {
        return {{A->getDims()}};
    } else {
        Shape temp = {1};
        return {{temp}};
    }
}

std::string MSELossObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// use output dim or inputs dim?
vector<int> MSELossObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> MSELossObj::getOpAttrVector() const { return {type.underlying()}; }

}; // namespace infini
