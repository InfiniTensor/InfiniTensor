#include "operators/element_wise.h"

namespace infini {
ElementWiseObj::ElementWiseObj(OpType type, GraphObj *graph, Tensor input0,
                               Tensor input1, Tensor output)
    : OperatorObj(type, {input0, input1}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ElementWiseObj::inferShape(const TensorVec &inputs) const {
    // For now,we only process the same dims here, broardcast will be considered
    // in the opt layer.
    const auto A = inputs[0], B = inputs[1];
    if (A->getDims().size() != B->getDims().size() ||
        A->getDims() != B->getDims())
        return {};

    return {{A->getDims()}};
    /*
    int n = A->getDims().size();
    Shape shape;
    for (int i = 0; i < n; i++) {
        auto dimA = A->getDims().at(i);
        auto dimB = B->getDims().at(i);
        if (!(dimA == dimB || dimA == 1 || dimB == 1))
            return {};
        auto dimI = dimA > dimB ? dimA : dimB;
        shape.emplace_back(dimI);
    }
    return {{shape}};*/
}

std::string ElementWiseObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// use output dim or inputs dim?
vector<int> ElementWiseObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> ElementWiseObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}


MSELossObj::MSELossObj(GraphObj *graph, Tensor input0, Tensor input1, Reduction reduction, Tensor output)
    : OperatorObj(OpType::MSELoss, {input0, input1}, {output}), reductionMode(reduction) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
MSELossObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0], B = inputs[1];
    if (A->getDims().size() != B->getDims().size() ||
        A->getDims() != B->getDims())
        return {};

    if (reductionMode == None) {
      return {{A->getDims()}};
    } else {
      Shape temp = { 1 };
      return {{temp}};
    }
}

std::string MSELossObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// use output dim or inputs dim?
vector<int> MSELossObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> MSELossObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

AddNObj::AddNObj(GraphObj *graph, int tensorNum, Tensor output, ...)
    : OperatorObj(OpType::AddN), num(tensorNum) {
        TensorVec temp;
        Tensor *start = &output;
        ++start;
        for(int i = 0; i < num; ++i) {
            temp.push_back(*start);
            start++;
        }
        setOutputs({output});
        setInputs(temp);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
AddNObj::inferShape(const TensorVec &inputs) const {
    // For now,we only process the same dims here, broardcast will be considered
    // in the opt layer.
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string AddNObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// use output dim or inputs dim?
vector<int> AddNObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> AddNObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

}; // namespace infini
