#include "operators/ascend_plugin_sub.h"

namespace infini {

AscendPluginSubObj::AscendPluginSubObj(GraphObj *graph, Tensor input,
                                       Tensor output, int kernel_size,
                                       int stride)
    : OperatorObj(OpType::AscendPluginSub, {input}, {output}),
      kernel_size(kernel_size), stride(stride) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
AscendPluginSubObj::inferShape(const TensorVec &inputs) {
    auto shape_input = inputs[0]->getDims();
    int N = shape_input[0];
    int C = shape_input[1];
    int H = shape_input[2];
    int W = shape_input[3];
    int H_ = (H - kernel_size) / stride + 1;
    int W_ = (W - kernel_size) / stride + 1;
    int sub_dim = 4 * (kernel_size - 1);
    vector<Shape> ret = {{N, C, H_, sub_dim, W_}};
    return ret;
}

std::string AscendPluginSubObj::toString() const {
    std::ostringstream os;
    os << "AscendPluginSub[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "kernel_size=" << kernel_size << ",";
    os << "stride=" << stride << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> AscendPluginSubObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> AscendPluginSubObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini
