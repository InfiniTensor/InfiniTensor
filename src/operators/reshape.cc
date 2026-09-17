#include "operators/reshape.h"
#include "utils/operator_utils.h"
#include <algorithm>
#include <limits>
#include <numeric>

namespace infini {

namespace {

Shape resolveReshapeSpec(const Shape &inputShape, size_t inputSize, const vector<int64_t> &spec, bool allowZero) {
    Shape result(spec.size());
    int inferAxis = -1;
    bool hasLiteralZero = false;
    size_t knownProduct = 1;

    for (size_t i = 0; i < spec.size(); ++i) {
        int64_t value = spec[i];
        IT_ASSERT(value >= -1, "Reshape dimensions must be >= -1");

        if (value == -1) {
            IT_ASSERT(inferAxis == -1, "Reshape can contain at most one -1");
            inferAxis = static_cast<int>(i);
            result[i] = -1;
            continue;
        }

        if (value == 0) {
            if (allowZero){
                hasLiteralZero = true;
            } else {
                IT_ASSERT(i < inputShape.size(), "Reshape 0 dimension exceeds input rank");
                value = inputShape[i];
            }
        }

        IT_ASSERT(value <= std::numeric_limits<int>::max(), "Reshape dimension exceeds InfiniTensor Shape range");
        result[i] = static_cast<int>(value);

        if (value == 0){
            knownProduct = 0;
        } else if (knownProduct != 0 ){
            const size_t dim = static_cast<size_t>(value);
            IT_ASSERT(knownProduct <= std::numeric_limits<size_t>::max() / dim, "Reshape element count overflow");
            knownProduct *= dim;
        }
    }

    if (inferAxis >= 0) {
        IT_ASSERT(!(allowZero && hasLiteralZero),
                  "Reshape allowzero=1 cannot combine 0 and -1");
        IT_ASSERT(knownProduct != 0,
                  "Reshape -1 with a zero known product is ambiguous");
        IT_ASSERT(inputSize % knownProduct == 0,
                  "Reshape -1 dimension cannot be inferred exactly");

        const size_t inferred = inputSize / knownProduct;
        IT_ASSERT(inferred <= static_cast<size_t>(
                                  std::numeric_limits<int>::max()),
                  "Inferred Reshape dimension is too large");
        result[inferAxis] = static_cast<int>(inferred);
    } else {
        IT_ASSERT(knownProduct == inputSize,
                  "Reshape input and output element counts differ");
    }

    return result;
}    

vector<int64_t> toInt64(const Shape & shape){
    return vector<int64_t>(shape.begin(), shape.end());
}

Shape makeRuntimePlaceholder(const Tensor &input, size_t outputRank) {
    if (outputRank == input->getRank())
        return input->getDims();

    if (outputRank == 0) {
        IT_ASSERT(input->size() == 1,
                  "Scalar Reshape placeholder requires one input element");
        return {};
    }

    IT_ASSERT(input->size() <= static_cast<size_t>(
                                   std::numeric_limits<int>::max()),
              "Reshape placeholder dimension is too large");
    Shape placeholder(outputRank, 1);
    placeholder[0] = static_cast<int>(input->size());
    return placeholder;
}

}


ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Reshape, {input}, {output}), dims(std::move(dims)), runtimeShape(false), allowZero(false) {
    IT_ASSERT(checkValid(graph));
}

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor shapeTensor, Tensor output, bool allowZero)
    : OperatorObj(OpType::Reshape, {input, shapeTensor}, {output}),
    outputShape(output ? output->getDims(): makeRuntimePlaceholder(input, shapeTensor->size())),
    runtimeShape(true), allowZero(allowZero){
        IT_ASSERT(shapeTensor->getRank() == 1, "Reshape shape input must be a 1-D Tensor");
        IT_ASSERT(shapeTensor->getDType() == DataType::Int64, "ONNX Reshape shape input must be Int64");
        IT_ASSERT(checkValid(graph));
    }

optional<vector<Shape>> ReshapeObj::inferShape(const TensorVec &inputs) {
    if (runtimeShape) {
        IT_ASSERT(inputs.size() == 2);
        return {{outputShape}};
    }

    IT_ASSERT(inputs.size() == 1);
    outputShape = resolveReshapeSpec(inputs[0]->getDims(), inputs[0]->size(), toInt64(dims), allowZero);

    return {{outputShape}};
    /*
    int count = 0;
    for (auto x : dims) {
        if (x == -1) {
            count++;
        }
        IT_ASSERT(x == -1 || x >= 0);
    }
    IT_ASSERT(count == 0 || count == 1);
    auto inputShape = inputs[0]->getDims();
    int size = inputs[0]->size();
    int index = -1;
    outputShape = dims;
    for (int i = 0; i < (int)dims.size(); ++i) {
        if (dims[i] == 0) {
            outputShape[i] = inputShape[i];
        }
        if (dims[i] == -1) {
            index = i;
        }
    }
    if (index != -1) {
        outputShape[index] =
            size / (-std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                     [](auto acc, auto x) { return acc * x; }));
    }
    int outputSize = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                     [](auto acc, auto x) { return acc * x; });
    IT_ASSERT(outputSize == size);

    return {{outputShape}};*/
}




bool ReshapeObj::resolveRuntimeShape() {
    IT_ASSERT(runtimeShape,
              "resolveRuntimeShape is only valid for dynamic Reshape");

    const Tensor shapeTensor = inputs.at(1);
    IT_ASSERT(shapeTensor->getRank() == 1,
              "Reshape shape input rank changed at runtime");
    IT_ASSERT(shapeTensor->size() == outputShape.size(),
              "Dynamic Reshape output rank cannot change in the first version");
    IT_ASSERT(shapeTensor->hasData(),
              "Runtime Reshape shape Tensor has no data");
    IT_ASSERT(shapeTensor->getDataBlob()->getBytes() ==
                  shapeTensor->getBytes(),
              "Runtime Reshape shape Tensor storage is invalid");

    const vector<int64_t> spec = shapeTensor->copyout<int64_t>();
    Shape resolved = resolveReshapeSpec(inputs[0]->getDims(),
                                        inputs[0]->size(), spec, allowZero);

    const bool changed = resolved != outputShape;
    outputShape = std::move(resolved);
    outputs[0]->setShape(outputShape);
    return changed;
}

std::string ReshapeObj::toString() const {
    std::ostringstream os;
    os << "Reshape[" << getGuid() << "](";
    os << "inputShape=" << vecToString(inputs[0]->getDims()) << ",";
    os << "outputShape=" << vecToString(outputShape) << ",";
    os << "runtimeShape=" << runtimeShape << ",";
    if (runtimeShape)
        os << "shapeTensor=" << inputs[1]->getGuid() << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ReshapeObj::getWorkloadVector() const {
    vector<int> ret{type.underlying(), static_cast<int>(runtimeShape),
                    static_cast<int>(allowZero)};
    const Shape inputShape = inputs[0]->getDims();
    ret.insert(ret.end(), inputShape.begin(), inputShape.end());
    ret.insert(ret.end(), outputShape.begin(), outputShape.end());
    return ret;
}

vector<int> ReshapeObj::getOpAttrVector() const {
    vector<int> ret{type.underlying(), static_cast<int>(runtimeShape),
                    static_cast<int>(allowZero)};
    if (!runtimeShape)
        ret.insert(ret.end(), dims.begin(), dims.end());
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
