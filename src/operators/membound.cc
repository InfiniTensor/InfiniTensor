#include "operators/membound.h"
#include "nnet/Visitor/HashVisitor.h"

namespace infini {

MemBoundObj::MemBoundObj(GraphObj *graph, const TensorVec &input,
                         const TensorVec &output,
                         const std::vector<nnet::Tensor> &nnetInputs,
                         nnet::Expr expr, double exec_time, std::string hint)
    : OperatorObj(OpType::MemBound, input, output), nnetInputs(nnetInputs),
      expr(expr), exec_time(exec_time), hint(hint) {
    IT_ASSERT(checkValid(graph));
}

string MemBoundObj::toString() const {
    std::ostringstream os;
    os << "MemBound[" << getGuid() << "](";
    for (size_t i = 0; i < inputs.size(); ++i) {
        os << "i" << i << "=" << inputs[i]->getGuid();
        if (i != inputs.size() - 1)
            os << " ";
    }
    os << ", ";
    for (size_t i = 0; i < outputs.size(); ++i) {
        os << "o" << i << "=" << outputs[i]->getGuid();
        if (i != outputs.size() - 1)
            os << " ";
    }
    os << ", ";
    os << "exec_time=" << exec_time << ", ";
    os << "NNet Inputs=[";
    for (const auto &tensor : nnetInputs)
        os << tensor->toReadable() << ",";
    os << "])";
    os << "\n" << (expr ? expr->toReadable() : "Empty expression") << "\n";
    return os.str();
}

optional<vector<Shape>> MemBoundObj::inferShape(const TensorVec &inputs) const {
    // inputs have to match nnetInputs excatly
    if (inputs.size() != nnetInputs.size())
        return {};
    for (size_t i = 0; i < inputs.size(); ++i)
        if (inputs[i]->getDims() != nnetInputs[i]->getShape())
            return {};
    return {{nnet::as<nnet::RangeOpNode>(expr)->getOutputShape()}};
}

vector<int> MemBoundObj::getWorkloadVector() const {
    return {enum_to_underlying(type), (int)getHash()};
}

vector<int> MemBoundObj::getOpAttrVector() const { return getWorkloadVector(); }

HashType MemBoundObj::getHash() const {
    return nnet::HashVisitor().dispatch(expr);
}

} // namespace infini