#include "operators/where.h"
#include "utils/operator_utils.h"

namespace infini {

WhereObj::WhereObj(GraphObj *graph, Tensor inputX, Tensor inputY,
                   Tensor condition, Tensor output)
    : OperatorObj(OpType::Where, TensorVec{inputX, inputY, condition},
                  {output}) {
    IT_ASSERT(checkValid(graph));
}

void WhereObj::initInfiniOp(const Runtime context) {
    // get dim data
    auto x_dim = inputs[0]->getDims();
    auto y_dim = inputs[1]->getDims();
    auto con_dim = inputs[2]->getDims();
    auto out_dim = outputs[0]->getDims();

    // convert dim data to infiniop format
    auto x_shape = toInfiniopShape(x_dim);
    auto y_shape = toInfiniopShape(y_dim);
    auto con_shape = toInfiniopShape(con_dim);
    auto out_shape = toInfiniopShape(out_dim);

    // create tensor descriptor
    infiniopTensorDescriptor_t x_desc, y_desc, con_desc, out_desc;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &x_desc, x_dim.size(), x_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &y_desc, y_dim.size(), y_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
    // condition tensor is uint8
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &con_desc, con_dim.size(), con_shape.data(), nullptr,
        toInfiniopDataLayout(DataType::UInt8.getIndex())));
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &out_desc, out_dim.size(), out_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));
    
    // create op descriptor
    CHECK_ERROR(infiniopCreateWhereDescriptor(
        context->opHandle(), (infiniopWhereDescriptor_t *)&opDesc,
        out_desc, con_desc, x_desc, y_desc));

    // destroy tensor descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(x_desc));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(y_desc));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(con_desc));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(out_desc));
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
