#include "operators/global_pool.h"

namespace infini {

GlobalPoolObj::GlobalPoolObj(GraphObj *graph, OpType optype, Tensor input,
                             Tensor output)
    : OperatorObj(optype, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

void GlobalPoolObj::initInfiniOp(const Runtime context) {
    auto x_dim = inputs[0]->getDims();
    auto y_dim = outputs[0]->getDims();

    if (type == OpType::GlobalAveragePool) {
        auto x_shape = toInfiniopShape(x_dim);
        auto y_shape = toInfiniopShape(y_dim);
        // create tensor descriptor
        infiniopTensorDescriptor_t x_tensor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &x_tensor, x_dim.size(), x_shape.data(), nullptr,
            toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
        infiniopTensorDescriptor_t y_tensor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &y_tensor, y_dim.size(), y_shape.data(), nullptr,
            toInfiniopDataLayout(outputs[0]->getDType().getIndex())));
        // create op descriptor
        CHECK_ERROR(infiniopCreateGlobalAvgPoolDescriptor(
            context->opHandle(), (infiniopGlobalAvgPoolDescriptor_t *)&opDesc,
            y_tensor, x_tensor));

        // 销毁
        CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
        CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
    } else {
        opDesc = nullptr;
    }
}

optional<vector<Shape>> GlobalPoolObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0];

    auto ret = input->getDims();
    for (size_t i = 2; i < ret.size(); i++) {
        ret[i] = 1;
    }
    return {{ret}};
}

std::string GlobalPoolObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> GlobalPoolObj::getWorkloadVector() const {
    return {type.underlying()};
}

vector<int> GlobalPoolObj::getOpAttrVector() const {
    return {type.underlying()};
}

}; // namespace infini
