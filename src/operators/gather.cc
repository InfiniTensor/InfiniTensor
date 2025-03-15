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
    // get dim data
    auto data_dim = inputs[0]->getDims();
    auto index_dim = inputs[1]->getDims();
    auto out_dim = outputs[0]->getDims();

    // convert dim data to infiniop format
    auto data_shape = toInfiniopShape(data_dim);
    auto index_shape = toInfiniopShape(index_dim);
    auto out_shape = toInfiniopShape(out_dim);

    // create tensor descriptor
    infiniopTensorDescriptor_t data_desc, index_desc, out_desc;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &data_desc, data_dim.size(), data_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &index_desc, index_dim.size(), index_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &out_desc, out_dim.size(), out_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));

    // create op descriptor
    CHECK_ERROR(infiniopCreateGatherDescriptor(
        context->opHandle(), (infiniopGatherDescriptor_t *)&opDesc, out_desc,
        data_desc, index_desc, getAxis()));

    // destroy tensor descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(data_desc));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(index_desc));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(out_desc));
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
