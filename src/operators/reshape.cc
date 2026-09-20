#include "operators/reshape.h"
#include "utils/operator_utils.h"
#include <limits>
#include <numeric>

namespace infini {
ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims,
                       bool allowZero)
    : OperatorObj(OpType::Reshape, {input}, {output}), dims(std::move(dims)),
      allowZero(allowZero) {
    IT_ASSERT(checkValid(graph));
}

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor shapeTensor,
                       Tensor output, bool allowZero)
    : OperatorObj(OpType::Reshape, {input, shapeTensor}, {output}),
      dynamicShape(true), allowZero(allowZero) {
    // Until the first shape subgraph execution, keep the input shape as a
    // conservative placeholder so the graph can allocate its shape tensors.
    dims = input->getDims();
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ReshapeObj::inferShape(const TensorVec &inputs) {
    Shape requested = dims;
    if (dynamicShape && inputs.size() > 1 && inputs[1]->hasData()) {
        IT_ASSERT(inputs[1]->getDType() == DataType::Int64 ||
                  inputs[1]->getDType() == DataType::Int32);
        requested.clear();
        if (inputs[1]->getDType() == DataType::Int64) {
            for (auto value : inputs[1]->copyout<int64_t>())
                IT_ASSERT(value >= std::numeric_limits<int>::min() &&
                          value <= std::numeric_limits<int>::max());
            for (auto value : inputs[1]->copyout<int64_t>())
                requested.push_back(static_cast<int>(value));
        } else {
            for (auto value : inputs[1]->copyout<int32_t>())
                requested.push_back(value);
        }
    }
    int count = 0;
    bool hasZero = false;
    for (auto x : requested) {
        if (x == -1) {
            count++;
        }
        if (x == 0)
            hasZero = true;
        IT_ASSERT(x == -1 || x >= 0);
    }
    IT_ASSERT(count == 0 || count == 1);
    IT_ASSERT(!(allowZero && hasZero && count != 0),
              "Reshape allowzero cannot be combined with -1");
    auto inputShape = inputs[0]->getDims();
    const size_t size = inputs[0]->size();
    int index = -1;
    outputShape = requested;
    for (int i = 0; i < (int)requested.size(); ++i) {
        if (requested[i] == 0 && !allowZero) {
            IT_ASSERT(i < static_cast<int>(inputShape.size()),
                      "Reshape zero dimension exceeds input rank");
            outputShape[i] = inputShape[i];
        }
        if (requested[i] == -1) {
            index = i;
        }
    }
    if (index != -1) {
        uint64_t known = 1;
        for (int i = 0; i < static_cast<int>(outputShape.size()); ++i) {
            if (i == index)
                continue;
            const auto dimension = static_cast<uint64_t>(outputShape[i]);
            IT_ASSERT(dimension == 0 || known <= std::numeric_limits<uint64_t>::max() / dimension,
                      "Reshape known product overflows");
            known *= dimension;
        }
        IT_ASSERT(known != 0, "Reshape cannot infer -1 from a zero product");
        IT_ASSERT(size % known == 0, "Reshape element count is not divisible");
        const auto inferred = size / known;
        IT_ASSERT(inferred <= static_cast<size_t>(std::numeric_limits<int>::max()),
                  "Reshape inferred dimension exceeds int range");
        outputShape[index] = static_cast<int>(inferred);
    }
    size_t outputSize = 1;
    for (const auto x : outputShape) {
        IT_ASSERT(x >= 0, "Reshape output dimension must be non-negative");
        if (x == 0) {
            outputSize = 0;
            continue;
        }
        IT_ASSERT(outputSize <= std::numeric_limits<size_t>::max() /
                                       static_cast<size_t>(x),
                  "Reshape output size overflows");
        outputSize *= static_cast<size_t>(x);
    }
    IT_ASSERT(outputSize == size);

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
