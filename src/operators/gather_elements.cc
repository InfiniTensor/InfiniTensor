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
