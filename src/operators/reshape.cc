#include "operators/reshape.h"
#include "utils/operator_utils.h"
#include <numeric>
#include <limits>
#include <stdexcept>

namespace infini {
ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Reshape, {input}, {output}), dims(std::move(dims)) {
    IT_ASSERT(checkValid(graph));
}

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor target, Tensor output)
    : OperatorObj(OpType::Reshape, {input, target}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ReshapeObj::inferShape(const TensorVec &inputs) {
    if (inputs.size() == 2) {
        auto target = inputs[1];
        if (!(target->getDType() == DataType::Int64) || target->getRank() != 1 ||
            !target->hasData())
            throw std::invalid_argument("Reshape target must be a populated int64 vector");
        dims.clear();
        for (auto dim : target->copyout<int64_t>()) {
            if (dim < -1 || dim > std::numeric_limits<int>::max())
                throw std::invalid_argument("Reshape target dimension out of range");
            dims.push_back(static_cast<int>(dim));
        }
    }
    int count = 0;
    for (auto x : dims) {
        if (x == -1) {
            count++;
        }
        if (x < -1)
            throw std::invalid_argument("Reshape dimensions must be >= -1");
    }
    if (count > 1)
        throw std::invalid_argument("Reshape allows at most one -1");
    auto inputShape = inputs[0]->getDims();
    size_t size = inputs[0]->size();
    int index = -1;
    outputShape = dims;
    for (int i = 0; i < (int)dims.size(); ++i) {
        if (dims[i] == 0) {
            if (i >= static_cast<int>(inputShape.size()))
                throw std::invalid_argument("Reshape zero axis exceeds input rank");
            outputShape[i] = inputShape[i];
        }
        if (dims[i] == -1) {
            index = i;
        }
    }
    size_t product = 1;
    for (auto dim : outputShape) {
        if (dim == -1)
            continue;
        if (dim && product > std::numeric_limits<size_t>::max() / dim)
            throw std::invalid_argument("Reshape element count overflow");
        product *= dim;
    }
    if (index != -1) {
        if (!product || size % product ||
            size / product > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Reshape cannot infer integral dimension");
        outputShape[index] = static_cast<int>(size / product);
    } else if (product != size) {
        throw std::invalid_argument("Reshape element count mismatch");
    }

    return {{outputShape}};
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
