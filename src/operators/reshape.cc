#include "operators/reshape.h"
#include "core/graph.h"
#include "utils/operator_utils.h"
#include <limits>
#include <numeric>

namespace infini {
ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Reshape, {input}, {output}), dims(std::move(dims)) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ReshapeObj::inferShape(const TensorVec &inputs) {
    Shape target = dims;
    if (isDynamic()) {
        const auto &shape = inputs.at(1);
        IT_ASSERT(shape->getRank() == 1 && shape->getDType() == DataType::Int64,
                  "Reshape shape must be a rank-one int64 tensor");
        IT_ASSERT(shape->hasData(), "Reshape shape tensor has no value");
        target.clear();
        for (int64_t dim : shape->copyout<int64_t>()) {
            IT_ASSERT(dim >= -1 && dim <= std::numeric_limits<int>::max(),
                      "Reshape dimension is outside the supported int32 range");
            target.emplace_back(static_cast<int>(dim));
        }
    }
    const auto inputShape = inputs[0]->getDims();
    const size_t size = inputs[0]->size();
    int index = -1;
    size_t knownSize = 1;
    outputShape = target;
    for (size_t i = 0; i < target.size(); ++i) {
        IT_ASSERT(target[i] >= -1, "Reshape dimension must be >= -1");
        if (target[i] == 0 && !allowZero) {
            IT_ASSERT(i < inputShape.size(),
                      "Reshape zero index exceeds input rank");
            outputShape[i] = inputShape[i];
        }
        if (target[i] == -1) {
            IT_ASSERT(index == -1, "Reshape permits only one -1 dimension");
            index = i;
            continue;
        }
        const auto dim = static_cast<size_t>(outputShape[i]);
        IT_ASSERT(dim == 0 ||
                      knownSize <= std::numeric_limits<size_t>::max() / dim,
                  "Reshape element count overflow");
        knownSize *= dim;
    }
    if (index != -1) {
        IT_ASSERT(knownSize != 0,
                  "Reshape cannot infer -1 with a zero product");
        IT_ASSERT(size % knownSize == 0, "Reshape element count mismatch");
        IT_ASSERT(size / knownSize <= std::numeric_limits<int>::max(),
                  "Reshape inferred dimension overflow");
        outputShape[index] = static_cast<int>(size / knownSize);
    } else {
        IT_ASSERT(knownSize == size, "Reshape element count mismatch");
    }
    return {{outputShape}};
}

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor shape,
                       Tensor output, bool allowZero)
    : OperatorObj(OpType::Reshape, {input, shape}, {output}),
      allowZero(allowZero) {
    // Construction can already evaluate a Shape-derived target, before the
    // device activation pool exists. Keep the real edge after validation.
    if (graph)
        inputs[1] = graph->evaluateShapeTensor(shape);
    IT_ASSERT(checkValid(graph));
    inputs[1] = shape;
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
