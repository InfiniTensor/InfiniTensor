#include "operators/batch_norm.h"

namespace infini {
BatchNormObj::BatchNormObj(GraphObj *graph, Tensor input, Tensor output,
                           Tensor mean, Tensor var, Tensor scale, Tensor bias,
                           float momentum, float eps, bool trainingMode)
    : OperatorObj(OpType::BatchNormalization, {input, mean, var, scale, bias},
                  {output}),
      momentum(momentum), eps(eps), trainingMode(trainingMode) {
    if (trainingMode)
        IT_TODO_HALT();

    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> BatchNormObj::inferShape(const TensorVec &inputs) {
    auto input = inputs[0];
    auto mean = inputs[1];
    auto var = inputs[2];
    auto scale = inputs[3];
    auto bias = inputs[4];
    auto c = std::vector<int>{input->getDims()[1]};
    IT_ASSERT(mean->getRank() == 1 && mean->getDims() == c);
    IT_ASSERT(var->getRank() == 1 && var->getDims() == c);
    IT_ASSERT(scale->getRank() == 1 && scale->getDims() == c);
    IT_ASSERT(bias->getRank() == 1 && bias->getDims() == c);
    return {{input->getDims()}};
}

vector<DataType> BatchNormObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 5);
    auto index = inputs[1];
    IT_ASSERT(inputs[1]->getDType() == DataType::Float32);
    IT_ASSERT(inputs[2]->getDType() == DataType::Float32);
    IT_ASSERT(inputs[3]->getDType() == DataType::Float32);
    IT_ASSERT(inputs[4]->getDType() == DataType::Float32);
    return {inputs[0]->getDType()};
}

std::string BatchNormObj::toString() const {
    std::ostringstream os;
    os << "batchNormalization[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "momentum=" << momentum << ",";
    os << "eps=" << eps << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "mean=" << inputs[1]->getGuid() << ",";
    os << "var=" << inputs[2]->getGuid() << ",";
    os << "scale=" << inputs[3]->getGuid() << ",";
    os << "bias=" << inputs[4]->getGuid() << ",";
    os << "output=";
    for (auto output : outputs)
        os << output->getGuid() << ",";
    return os.str();
}

void BatchNormObj::initInfiniOp(const Runtime context) {
    auto x_dim = inputs[0]->getDims();
    auto mean_dim = inputs[1]->getDims();  
    auto var_dim = inputs[2]->getDims();      
    auto scale_dim = inputs[3]->getDims(); 
    auto bias_dim = inputs[4]->getDims();  
    auto y_dim = outputs[0]->getDims();

    // convert dim data to infiniop format
    auto x_shape = toInfiniopShape(x_dim);
    auto scale_shape = toInfiniopShape(scale_dim);
    auto bias_shape = toInfiniopShape(bias_dim);
    auto mean_shape = toInfiniopShape(mean_dim);
    auto var_shape = toInfiniopShape(var_dim);
    auto y_shape = toInfiniopShape(y_dim);

    // create tensor descriptor
    infiniopTensorDescriptor_t x_tensor, scale_tensor, bias_tensor;
    infiniopTensorDescriptor_t mean_tensor, var_tensor, y_tensor;

    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &x_tensor, x_dim.size(), x_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &scale_tensor, scale_dim.size(), scale_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &bias_tensor, bias_dim.size(), bias_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[2]->getDType().getIndex())));
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &mean_tensor, mean_dim.size(), mean_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[3]->getDType().getIndex())));
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &var_tensor, var_dim.size(), var_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[4]->getDType().getIndex())));
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &y_tensor, y_dim.size(), y_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));

    // create op descriptor
    CHECK_ERROR(infiniopCreateBatchNormDescriptor(
        context->opHandle(), (infiniopBatchNormDescriptor_t *)&opDesc,
        y_tensor, x_tensor, scale_tensor, bias_tensor,
        mean_tensor, var_tensor, eps));

    // destroy tensor descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(scale_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(bias_tensor)); 
    CHECK_ERROR(infiniopDestroyTensorDescriptor(mean_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(var_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
}

// need eps and momentum?
vector<int> BatchNormObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

// need eps and momentum?
vector<int> BatchNormObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini
