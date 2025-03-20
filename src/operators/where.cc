#include "operators/where.h"
#include "utils/operator_utils.h"

namespace infini {

WhereObj::WhereObj(GraphObj *graph, Tensor inputX, Tensor inputY,
                   Tensor condition, Tensor output)
    : OperatorObj(OpType::Where, TensorVec{inputX, inputY, condition},
                  {output}) {
    IT_ASSERT(checkValid(graph));
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

void WhereObj::initInfiniOp(const Runtime context) {
    auto inputx = inputs[0]->getDims();
    auto inputy = inputs[1]->getDims();
    auto condition = inputs[2]->getDims();
    auto y_dim = inputs[1]->getDims();

    auto inputx_shape = toInfiniopShape(inputx);
    auto inputy_shape = toInfiniopShape(inputy);
    auto condition_shape = toInfiniopShape(condition);
    auto y_shape = toInfiniopShape(y_dim);

    infiniopTensorDescriptor_t inputx_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &inputx_tensor, inputx.size(), inputx_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    infiniopTensorDescriptor_t inputy_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &inputy_tensor, inputy.size(), inputy_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
    infiniopTensorDescriptor_t condition_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &condition_tensor, condition.size(), condition_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[2]->getDType().getIndex())));
    infiniopTensorDescriptor_t y_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &y_tensor, y_dim.size(), y_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));

    CHECK_ERROR(infiniopCreateWhereDescriptor(
        context->opHandle(), (infiniopWhereDescriptor_t *)&opDesc,
        y_tensor, inputx_tensor, inputy_tensor, condition_tensor));

    CHECK_ERROR(infiniopDestroyTensorDescriptor(inputx_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(inputy_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(condition_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
    
}
vector<int> WhereObj::getOpAttrVector() const { return {type.underlying()}; }

} // namespace infini
