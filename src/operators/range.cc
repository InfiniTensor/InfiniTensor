#include "operators/range.h"
#include "utils/operator_utils.h"

namespace infini {


RangeObj::RangeObj(GraphObj *graph, float start, float limit, float delta, Tensor output)
    : OperatorObj(OpType::Range,TensorVec{}, {output}), start(start), limit(limit), delta(delta) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> RangeObj::inferShape(const TensorVec &inputs) {

    return {{{(int)std::max(std::ceil((getLimit() - getStart()) / getDelta()), 0.0f)}}};   

}

vector<DataType> RangeObj::inferDataType(const TensorVec &inputs) const {
    return {{DataType::Float32}};
}

std::string RangeObj::toString() const {
    std::ostringstream os;
    os << "Range[" << getGuid() << "]";
    os << "(";
    os << "start=" << getStart() << ",";
    os << "limit=" << getLimit() << ",";
    os << "delta=" << getDelta() << ","; //
    os << "output=";
    for (auto output : outputs)
        os << output->getGuid() << ",";
    return os.str();
}

vector<int> RangeObj::getWorkloadVector() const {
    return {type.underlying()};

}

vector<int> RangeObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini
