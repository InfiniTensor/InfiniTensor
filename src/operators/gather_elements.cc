#include "operators/gather.h"
#include "utils/operator_utils.h"

namespace infini {
GatherElementsObj::GatherElementsObj(GraphObj *graph, Tensor input,
                                     Tensor indices, Tensor output, int axis)
    : GatherBaseObj(OpType::GatherElements, {input, indices}, {output}, axis) {
    int rank = input->getRank();
    this->axis = get_real_axis(axis, rank);
    IT_ASSERT(checkValid(graph));
}

void GatherElementsObj::initInfiniOp(const Runtime context) {
    auto x_dim = inputs[0]->getDims();
    auto indice_dim = inputs[1]->getDims();
    auto y_dim = outputs[0]->getDims();


    auto x_shape = toInfiniopShape(x_dim);
    auto indice_shape = toInfiniopShape(indice_dim);
    auto y_shape = toInfiniopShape(y_dim);

    // create tensor descriptor
    infiniopTensorDescriptor_t x_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &x_tensor, x_dim.size(), x_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    infiniopTensorDescriptor_t indice_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &indice_tensor, indice_dim.size(), indice_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
    infiniopTensorDescriptor_t y_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &y_tensor, y_dim.size(), y_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));
    // create op descriptor
    if (type == OpType::GatherElements) {
        CHECK_ERROR(infiniopCreateGatherElementsDescriptor(
            context->opHandle(), (infiniopGatherElementsDescriptor_t *)&opDesc,
            y_tensor, x_tensor, indice_tensor, this->axis));
    }else {
        opDesc = nullptr;
    }
    // destroy tensor descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(indice_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
}

bool checkShape(Tensor input, Tensor indices, int axis) {
    auto inputDims = input->getDims();
    auto indicesDims = indices->getDims();
    if (input->getRank() != indices->getRank()) {
        return false;
    }
    for (int i = 0; i < static_cast<int>(input->getRank()); ++i) {
        if (i != axis && inputDims[i] != indicesDims[i]) {
            return false;
        }
    }
    return true;
}

optional<vector<Shape>> GatherElementsObj::inferShape(const TensorVec &inputs) {
    IT_ASSERT(checkShape(inputs[0], inputs[1], axis));
    auto indicesDims = inputs[1]->getDims(); // output has same shape as indices
    return {{indicesDims}};
}

vector<DataType>
GatherElementsObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 2);
    auto indexDtype = inputs[1]->getDType();
    IT_ASSERT(indexDtype == DataType::Int32 || indexDtype == DataType::Int64);
    return {inputs[0]->getDType()};
}

std::string GatherElementsObj::toString() const {
    std::ostringstream os;
    os << "GatherElements"
       << "[" << getGuid() << "]";
    os << "(";
    if (inputs.size() == 2) {
        os << vecToString(inputs[0]->getDims()) << ",";
        os << vecToString(inputs[1]->getDims()) << ",";
    }
    os << "axis=" << axis << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> GatherElementsObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    for (auto it : inputs[1]->getDims())
        ret.emplace_back(it);
    ret.emplace_back(axis);
    return ret;
}

vector<int> GatherElementsObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}

} // namespace infini
