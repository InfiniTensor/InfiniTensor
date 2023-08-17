#include "operators/constant.h"

namespace infini {

ConstantObj::ConstantObj(GraphObj *graph, Tensor output, Tensor value_)
    : OperatorObj(OpType::Constant,
                  {output}, value(std::move(value_))) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConstantObj::inferShape(const TensorVec &inputs) const {
    vector<int> ret = value->getDims();
    return {{ret}};
}

std::string ConstantObj::toString() const {
    std::ostringstream os;
    os << "Constant[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ConstantObj::getWorkloadVector() const {
    vector<int> ret = value->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ConstantObj::getOpAttrVector() const { 
    return {type.underlying()}; 
}
/***
ConstantTensorValue : public ConstantObj (GraphObj *graph, Tensor input, Tensor output, Tensor value)
//ConstantFloatValue : public ConstantObj (GraphObj *graph, Tensor input, Tensor output, float float_value)
optional<vector<Shape>> ConstantObj::inferShape(const TensorVec &inputs) const {
    return getOutput()->getDims();
}

std::string ConstantObj::toString() const {
    std::ostringstream os;
    os << "Constant[" << getGuid() << "]";
    os << "(";
    os << "value=" << value[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ConstantObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    Shape valueDims = value.getDims();
    ret.insert(ret.end(), valueDims.begin(), valueDims.end());
    ret.insert(ret.begin, type.underlying());
    return ret;
}

vector<int> ConstantObj::getOpAttrVector() const {
    vector<int> ret = {};
    Shape valueDims = value.getDims();
    ret.insert(ret.end(), valueDims.begin(), valueDims.end());
    ret.insert(ret.begin, type.underlying());
    return ret;
}
***/
} // namespace infini