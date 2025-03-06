#include "operators/where.h"
#include "core/runtime.h"
#include "tensor.h"
#include "tensor/tensor_descriptor.h"
#include "utils/infiniop_utils.h"
#include "utils/operator_utils.h"
#include <cstddef>

namespace infini {

WhereObj::WhereObj(GraphObj *graph, Tensor inputX, Tensor inputY,
                   Tensor condition, Tensor output)
    : OperatorObj(OpType::Where, TensorVec{inputX, inputY, condition},
                  {output}) {
    IT_ASSERT(checkValid(graph));
}

// initInfiniOp
void WhereObj::initInfiniOp(const Runtime context) {
    auto x_dim = inputs[0]->getDims();
    auto y_dim = inputs[1]->getDims();
    auto condition_dim = inputs[2]->getDims();
    auto output_dim = outputs[0]->getDims();

    auto x_shape = toInfiniopShape(x_dim);
    auto y_shape = toInfiniopShape(y_dim);
    auto condition_shape = toInfiniopShape(condition_dim);
    auto output_shape = toInfiniopShape(output_dim);

    // create tensor descriptor
    infiniopTensorDescriptor_t x_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &x_tensor, x_dim.size(), x_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    infiniopTensorDescriptor_t y_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &y_tensor, y_dim.size(), y_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
    infiniopTensorDescriptor_t condition_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &condition_tensor, condition_dim.size(), condition_shape.data(),
        nullptr, toInfiniopDataLayout(inputs[2]->getDType().getIndex())));
    infiniopTensorDescriptor_t output_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &output_tensor, output_dim.size(), output_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));

    // create op descriptor
    CHECK_ERROR(infiniopCreateWhereDescriptor(
        context->opHandle(), (infiniopWhereDescriptor_t *)&opDesc,
        output_tensor, condition_tensor, x_tensor, y_tensor));

    // destroy tensor descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(condition_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(output_tensor));
}

optional<vector<Shape>> WhereObj::inferShape(const TensorVec &inputs) {
    auto shapeX = inputs[0]->getDims();
    auto shapeY = inputs[1]->getDims();
    auto shapeCon = inputs[2]->getDims();
    auto retXY = infer_broadcast(shapeX, shapeY);
    auto ret = infer_broadcast(retXY, shapeCon);
    return {{ret}};
}

std::string WhereObj::toString() const {
    std::ostringstream os;
    os << "Where[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[2]->getDims()) << ",";
    os << "inputX=" << inputs[0]->getGuid() << ",";
    os << "inputY=" << inputs[1]->getGuid() << ",";
    os << "condition=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> WhereObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> WhereObj::getOpAttrVector() const { return {type.underlying()}; }

} // namespace infini
