#include "operators/any.h"

namespace infini {

AnyObj::AnyObj(GraphObj *graph, const TensorVec &inputs,
               const TensorVec &outputs, const string &kernelName,
               const vector<int> &attr)
    : OperatorObj(OpType::Any, inputs, outputs), kernelName(kernelName),
      attr(attr) {
    IT_ASSERT(checkValid(graph));
    // Outputs must assigned when constructing AnyObj
    IT_ASSERT(!outputs.empty());
    for (auto &output : outputs)
        IT_ASSERT(output != nullptr && output->size() > 0);
}

string AnyObj::toString() const {
    std::ostringstream os;
    os << "Any[" << getGuid() << "](";
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
    os << "kernel name: " << kernelName << ", ";
    os << "attr = [";
    for (size_t i = 0; i < attr.size(); ++i) {
        os << attr[i];
        if (i != attr.size() - 1)
            os << ", ";
    }
    os << "])\n";
    return os.str();
}

optional<vector<Shape>> AnyObj::inferShape(const TensorVec &inputs) const {
    vector<Shape> ret;
    for (auto output : outputs) {
        ret.emplace_back(output->getDims());
    }
    return ret;
}

const string AnyObj::getKernelName() const { return kernelName; }

vector<int> AnyObj::getOpAttrVector() const { return attr; };

vector<int> AnyObj::getWorkloadVector() const {
    vector<int> ret = {};
    for (auto &input : inputs) {
        auto inputDims = input->getDims();
        ret.insert(ret.end(), inputDims.begin(), inputDims.end());
    }
    for (auto &output : outputs) {
        auto outputDims = output->getDims();
        ret.insert(ret.end(), outputDims.begin(), outputDims.end());
    }
    for (auto c : kernelName) {
        ret.emplace_back(c);
    }
    for (auto at : attr) {
        ret.emplace_back(at);
    }
    return ret;
}

} // namespace infini
