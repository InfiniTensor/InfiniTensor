#include "operators/gather.h"
#include "utils/operator_utils.h"
#include <algorithm>

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

void GatherObj::inferShapeValue() {
    if (!beginShapeValueUpdate()) {
        return;
    }
    // A shape subgraph gathers from the result of `Shape`, which is a list of
    // dimensions, so only picking along that single axis makes sense here.
    if (inputs[0]->getRank() != 1 || axis != 0) {
        return;
    }
    const auto &dims = *inputs[0]->getShapeValue();
    const auto &indices = *inputs[1]->getShapeValue();
    vector<int64_t> picked;
    vector<bool> fixed;
    picked.reserve(indices.size());
    fixed.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto index = indices[i];
        // ONNX counts a negative index back from the end.
        const auto at =
            index < 0 ? index + static_cast<int64_t>(dims.size()) : index;
        if (at < 0 || at >= static_cast<int64_t>(dims.size())) {
            return;
        }
        picked.push_back(dims[at]);
        // Picking a settled element with an index that is itself settled
        // yields a settled element. An index that could change would reach a
        // different element next time, so the result is not settled even where
        // every element it might reach is.
        fixed.push_back(inputs[0]->isShapeValueFixed(static_cast<size_t>(at)) &&
                        inputs[1]->isShapeValueFixed(i));
    }
    outputs[0]->setShapeValue(std::move(picked), std::move(fixed));
}

bool GatherObj::CheckIndexValid() const {
    const auto &index = inputs[1];
    const int64_t length = inputs[0]->getDims()[axis];
    const auto valid = [length](int64_t value) {
        return value >= -length && value < length;
    };
    if (const auto &values = index->getShapeValue(); values.has_value()) {
        return std::all_of(values->begin(), values->end(), valid);
    }
    // Shape inference precedes execution and uploading the next input. A
    // computed index's buffer may still hold the previous shape's result;
    // only its shape value above is current. The CPU kernel checks runtime
    // indices when it executes.
    if (index->getSource() || index->isInput() || !index->getDataBlob())
        return true;

    if (index->getDType() == DataType::Int32) {
        const auto values = index->copyout<int32_t>();
        return std::all_of(values.begin(), values.end(), valid);
    }
    const auto values = index->copyout<int64_t>();
    return std::all_of(values.begin(), values.end(), valid);
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
