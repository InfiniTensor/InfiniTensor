#include "operators/gather.h"
#include "utils/operator_utils.h"

namespace infini {
GatherObj::GatherObj(GraphObj *graph, Tensor input, Tensor indices,
                     Tensor output, int axis)
    : GatherBaseObj(OpType::Gather, {input, indices}, {output}, axis) {
    int rank = input->getRank();
    this->axis = get_real_axis(axis, rank);
    IT_ASSERT(checkValid(graph));
}
void GatherObj::initInfiniOp(const Runtime context) {
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
    if (type == OpType::Gather) {
        CHECK_ERROR(infiniopCreateGatherDescriptor(
            context->opHandle(), (infiniopGatherDescriptor_t *)&opDesc,
            y_tensor, x_tensor, indice_tensor, this->axis));
    }else {
        opDesc = nullptr;
    }
    // destroy tensor descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(indice_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
}
optional<vector<Shape>> GatherObj::inferShape(const TensorVec &inputs) {
    auto dims0 = inputs[0]->getDims();
    auto dims1 = inputs[1]->getDims();

    IT_ASSERT(CheckIndexValid());

    Shape dim = dims0;
    dim.erase(dim.begin() + axis);
    dim.insert(dim.begin() + axis, dims1.begin(), dims1.end());
    return {{dim}};
}

vector<DataType> GatherObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 2);
    auto index_dtype = inputs[1]->getDType();
    IT_ASSERT(index_dtype == DataType::Int32 || index_dtype == DataType::Int64);
    return {inputs[0]->getDType()};
}

// TODO:should check everytime index updated.
bool GatherObj::CheckIndexValid() const {
    auto index = inputs[1];
    if (index->getDataBlob() == nullptr)
        return true;

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    bool ret = true;
    auto value = inputs[0]->getDims()[axis];
    if (index->getDType() == DataType::Int32) {
        int *data = (int *)runtime->alloc(index->getBytes());
        index->getRuntime()->copyBlobToCPU(
            (void *)data, index->getRawDataPtr<void *>(), index->getBytes());
        for (size_t i = 0; i < index->size(); ++i) {
            if (data[i] < 0 || data[i] >= value) {
                ret = false;
                break;
            }
        }
        runtime->dealloc(data);
    } else {
        int64_t *data = (int64_t *)runtime->alloc(index->getBytes());
        index->getRuntime()->copyBlobToCPU(
            (void *)data, index->getRawDataPtr<void *>(), index->getBytes());
        for (size_t i = 0; i < index->size(); ++i) {
            if (data[i] < 0 || data[i] >= value) {
                ret = false;
                break;
            }
        }
        runtime->dealloc(data);
    }
    return ret;
}

std::string GatherObj::toString() const {
    std::ostringstream os;
    os << "Gather"
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

vector<int> GatherObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    for (auto it : inputs[1]->getDims())
        ret.emplace_back(it);
    ret.emplace_back(axis);
    return ret;
}

vector<int> GatherObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}

} // namespace infini
