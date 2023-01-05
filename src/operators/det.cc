#include "operators/det.h"

namespace infini {
DetObj::DetObj(GraphObj *graph, Tensor input, Tensor output, Mode mode)
    : OperatorObj(OpType::Det, {input}, {output}), modeValue(mode) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> DetObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    auto input = A->getDims();
    int length = input.size();
    if (length == 2) {
        std::vector<int> output = {1};
        return {{output}};
    } else {
        std::vector<int> output(input.begin(), input.end() - 2);
        return {{output}};
    }
}

std::string DetObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> DetObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> DetObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

}; // namespace infini
