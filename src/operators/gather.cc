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
