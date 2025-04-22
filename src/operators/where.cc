#include "operators/where.h"
#include "utils/operator_utils.h"
#include "core/kernel.h"
#include <iostream>  // 包含标准输出头文件
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
    auto x_dim = inputs[0]->getDims();
    auto y_dim = inputs[1]->getDims();
    auto condition_dim = inputs[2]->getDims();
    auto output_dim = outputs[0]->getDims();

    // convert dim data to infiniop format
    auto x_shape = toInfiniopShape(x_dim);
    auto y_shape = toInfiniopShape(y_dim);
    auto cond_shape = toInfiniopShape(condition_dim);
    auto out_shape = toInfiniopShape(output_dim);

    // create tensor descriptors
    infiniopTensorDescriptor_t x_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &x_tensor, x_dim.size(), x_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));

    infiniopTensorDescriptor_t y_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &y_tensor, y_dim.size(), y_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
        #include <iostream>  // 包含标准输出头文件

    // 在你调用到 inputs[2]->getDType().getIndex() 的地方，插入如下代码
    std::cout << "inputs[2]->getDType().getIndex(): " 
                << inputs[2]->getDType().getIndex() << std::endl;
    

    infiniopTensorDescriptor_t cond_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &cond_tensor, condition_dim.size(), cond_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[2]->getDType().getIndex())));

    infiniopTensorDescriptor_t out_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &out_tensor, output_dim.size(), out_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));

    // create op descriptor
    CHECK_ERROR(infiniopCreateWhereDescriptor(
        context->opHandle(), (infiniopWhereDescriptor_t *)&opDesc,
        cond_tensor, x_tensor, y_tensor,out_tensor));

    // destroy tensor descriptors
    CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(cond_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(out_tensor));
}

vector<int> WhereObj::getOpAttrVector() const { return {type.underlying()}; }

} // namespace infini
