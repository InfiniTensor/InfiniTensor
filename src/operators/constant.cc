#include "operators/constant.h"

namespace infini {
std::string ConstantObj::toString() const {
    std::ostringstream os;
    os << "Constant[" << getGuid() << "]";
    os << "output = " << outputs[0]->getGuid() << ",";
    return os.str();
}

vector<int> ConstantObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

// need eps and momentum?
vector<int> ConstantObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}
} // namespace infini
