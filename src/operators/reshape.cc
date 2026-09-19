#include "operators/reshape.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini {
namespace {
/// Works out the real output shape from a target holding the two placeholders
/// ONNX allows: 0 keeps the input dimension in that position, and -1 stands for
/// whatever is left over. Both ways of giving a target share this, so they
/// cannot drift apart.
Shape resolveTargetShape(const Shape &dims, const Shape &inputShape, int size) {
    int count = 0;
    for (auto x : dims) {
        if (x == -1) {
            count++;
        }
        IT_ASSERT(x == -1 || x >= 0);
    }
    IT_ASSERT(count == 0 || count == 1);
    int index = -1;
    Shape outputShape = dims;
    for (int i = 0; i < (int)dims.size(); ++i) {
        if (dims[i] == 0) {
            // A zero says to keep whatever the input has in this position, so
            // there has to be one. A target longer than the input reaches past
            // its end here, which read whatever the memory held and produced a
            // dimension that varied from one run to the next.
            //
            // Nothing distinguishes this from a dimension that is genuinely
            // zero, because ONNX gives the two the same spelling. A graph
            // reaching this point asked for a dimension that does not exist
            // either way, and saying so is the only answer available.
            IT_ASSERT(i < (int)inputShape.size(),
                      "this reshape keeps the input dimension at position " +
                          std::to_string(i) + ", but the input has only " +
                          std::to_string(inputShape.size()) +
                          " dimensions; a zero in a target shape means the "
                          "dimension the input already has there");
            outputShape[i] = inputShape[i];
        }
        if (dims[i] == -1) {
            index = i;
        }
    }
    if (index != -1) {
        const int known =
            -std::accumulate(outputShape.begin(), outputShape.end(), 1,
                             [](auto acc, auto x) { return acc * x; });
        // The leftover is what the other dimensions do not account for, which
        // there is no answer to when they account for nothing: every size at
        // all divides into zero elements the same number of times.
        IT_ASSERT(known != 0,
                  "this reshape asks for the dimension left over once the "
                  "others are taken, but one of those is zero, so there is no "
                  "such number");
        outputShape[index] = size / known;
    }
    int outputSize = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                     [](auto acc, auto x) { return acc * x; });
    IT_ASSERT(outputSize == size,
              "Reshape size mismatch: input size=" + std::to_string(size) +
                  ", output size=" + std::to_string(outputSize));
    return outputShape;
}
} // namespace

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Reshape, {input}, {output}), dims(std::move(dims)) {
    IT_ASSERT(checkValid(graph));
}

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor shape,
                       Tensor output)
    : OperatorObj(OpType::Reshape, {input, shape}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ReshapeObj::inferShape(const TensorVec &inputs) {
    if (inputs.size() == 2) {
        IT_ASSERT(inputs[1]->getShapeValue().has_value(),
                  "the target shape of this Reshape is not known while shapes "
                  "are inferred, so it holds data rather than dimensions");
        dims = inputs[1]->getShapeValueAsShape();
    }
    outputShape = resolveTargetShape(dims, inputs[0]->getDims(),
                                     static_cast<int>(inputs[0]->size()));
    return {{outputShape}};
}

vector<DimSource> ReshapeObj::dimSources(size_t output, size_t dim) const {
    IT_ASSERT(output == 0);
    IT_ASSERT(dim < outputs[0]->getRank());
    // `inferShape` leaves `dims` holding the target of either form, so a target
    // spelled out at construction and one read from an edge are read alike. It
    // has one element per output dimension, which is what makes this a matter
    // of asking what that element says.
    IT_ASSERT(dim < dims.size());
    // An element read from an edge is a number only for the shape the graph
    // currently holds. What it will say about this dimension under another
    // shape is not known, so nothing is claimed.
    if (inputs.size() == 2 && !inputs[1]->isShapeValueFixed(dim)) {
        return OperatorObj::dimSources(output, dim);
    }
    const auto element = dims[dim];
    if (element > 0) {
        // The element names the dimension outright, and it cannot change, so
        // no input shape moves it.
        return {};
    }
    if (element == 0) {
        // Zero keeps whatever the input has in this position, so this
        // dimension is exactly as settled as that one is. `resolveTargetShape`
        // has already refused a position the input does not reach.
        IT_ASSERT(dim < inputs[0]->getRank());
        return {DimSource{0, dim}};
    }
    // What is left once the others are taken, which is the number of elements
    // divided by them: every input dimension takes part, and any one of them
    // moving moves this. The other target elements are constants or are
    // themselves settled, so they add nothing to follow.
    vector<DimSource> sources;
    sources.reserve(inputs[0]->getRank());
    for (size_t d = 0; d < inputs[0]->getRank(); ++d) {
        sources.push_back(DimSource{0, d});
    }
    return sources;
}

std::string ReshapeObj::toString() const {
    std::ostringstream os;
    os << "Reshape[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "outputShape=" << vecToString(outputShape) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ReshapeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), outputShape.begin(), outputShape.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}
vector<int> ReshapeObj::getOpAttrVector() const {
    vector<int> ret = outputShape;
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

FlattenObj::FlattenObj(GraphObj *graph, Tensor input, Tensor output, int _axis)
    : OperatorObj(OpType::Flatten, {input}, {output}) {
    int rank = input->getRank();
    axis = get_real_axis(_axis, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> FlattenObj::inferShape(const TensorVec &inputs) {
    int sizeB = 1, sizeE = 1;
    auto dims = getInputs(0)->getDims();
    int rank = getInputs(0)->getRank();
    for (int i = 0; i < rank; ++i) {
        ((i < axis) ? sizeB : sizeE) *= dims.at(i);
    }
    return {{{sizeB, sizeE}}};
}

std::string FlattenObj::toString() const {
    std::ostringstream os;
    os << "Flatten[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "axis=" << axis << ")";
    return os.str();
}

vector<int> FlattenObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), axis);
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> FlattenObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}

IdentityObj::IdentityObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::Identity, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> IdentityObj::inferShape(const TensorVec &inputs) {
    return {{getInputs(0)->getDims()}};
}

void IdentityObj::inferShapeValue() {
    if (!beginShapeValueUpdate()) {
        return;
    }
    // The output is the input: the same elements, each as settled as it was.
    const auto &value = *inputs[0]->getShapeValue();
    vector<bool> fixed;
    fixed.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        fixed.push_back(inputs[0]->isShapeValueFixed(i));
    }
    outputs[0]->setShapeValue(value, std::move(fixed));
}

std::string IdentityObj::toString() const {
    std::ostringstream os;
    os << "Identity[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> IdentityObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}
vector<int> IdentityObj::getOpAttrVector() const { return {type.underlying()}; }
} // namespace infini
