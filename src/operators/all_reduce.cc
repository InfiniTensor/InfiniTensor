#include "operators/all_reduce.h"

namespace infini {
AllReduceBaseObj::AllReduceBaseObj(GraphObj *graph, OpType opType, Tensor input,
                                   Tensor output)
    : OperatorObj(opType, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

std::string AllReduceBaseObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    return os.str();
}

vector<int> AllReduceBaseObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> AllReduceBaseObj::getOpAttrVector() const {
    return {type.underlying()};
}

AllReduceSumObj::AllReduceSumObj(GraphObj *graph, Tensor input, Tensor output)
    : AllReduceBaseObj(graph, OpType::AllReduceSum, input, output) {}

AllReduceProdObj::AllReduceProdObj(GraphObj *graph, Tensor input, Tensor output)
    : AllReduceBaseObj(graph, OpType::AllReduceProd, input, output) {}

AllReduceMinObj::AllReduceMinObj(GraphObj *graph, Tensor input, Tensor output)
    : AllReduceBaseObj(graph, OpType::AllReduceMin, input, output) {}

AllReduceMaxObj::AllReduceMaxObj(GraphObj *graph, Tensor input, Tensor output)
    : AllReduceBaseObj(graph, OpType::AllReduceMax, input, output) {}

AllReduceAvgObj::AllReduceAvgObj(GraphObj *graph, Tensor input, Tensor output)
    : AllReduceBaseObj(graph, OpType::AllReduceAvg, input, output) {}
} // namespace infini