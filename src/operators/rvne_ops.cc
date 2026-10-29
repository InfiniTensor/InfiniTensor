#include "operators/rvne_ops.h"
#include "utils/mlp_compute.h"
namespace infini {
LSMSequenceObj::LSMSequenceObj(GraphObj *graph, Tensor spikes,
                               Tensor inputWeights, Tensor recurrentWeights,
                               Tensor stateIn, const vector<int> &params,
                               bool useQemu, bool traceAllSteps)
    : OperatorObj(OpType::LSMSequence,
                  {spikes, inputWeights, recurrentWeights, stateIn},
                  {nullptr, nullptr, nullptr}),
      useQemu(useQemu), traceAllSteps(traceAllSteps) {
    lsm::require(params.size() == lsm::ParameterCount,
                 "LSMSequence: expected 21 parameters");
    std::copy(params.begin(), params.end(), parameters.begin());
    IT_ASSERT(checkValid(graph), "Invalid LSMSequence shape or dtype");
    lsm::validateParameters(parameters, inputs[2]->getDims()[0]);
}
optional<vector<Shape>> LSMSequenceObj::inferShape(const TensorVec &values) {
    for (const auto &value : values)
        if (!(value->getDType() == DataType::Int32))
            return {};
    const auto &spikeShape = values[0]->getDims();
    const auto &inputWeightShape = values[1]->getDims();
    const auto &recurrentWeightShape = values[2]->getDims();
    if (spikeShape.size() != 2 || inputWeightShape.size() != 2 ||
        recurrentWeightShape.size() != 2)
        return {};
    int steps = spikeShape[0], channels = spikeShape[1],
        neurons = recurrentWeightShape[0];
    if (steps < 1 || steps > 6000 || channels < 1 || channels > 1024 ||
        neurons < 1 || neurons > 1024 || recurrentWeightShape[1] != neurons ||
        inputWeightShape != Shape{channels, neurons} ||
        values[3]->getDims() != Shape{lsm::StateRows, neurons})
        return {};
    return vector<Shape>{{1, neurons},
                         {lsm::StateRows, neurons},
                         {traceAllSteps ? steps : 1, lsm::StateRows, neurons}};
}
vector<DataType> LSMSequenceObj::inferDataType(const TensorVec &) const {
    return {DataType::Float32, DataType::Int32, DataType::Int32};
}
vector<int> LSMSequenceObj::getOpAttrVector() const {
    vector<int> result{type.underlying(), int(useQemu), int(traceAllSteps)};
    result.insert(result.end(), parameters.begin(), parameters.end());
    return result;
}
vector<int> LSMSequenceObj::getWorkloadVector() const {
    auto result = getOpAttrVector();
    for (const auto &input : inputs) {
        const auto &shape = input->getDims();
        result.insert(result.end(), shape.begin(), shape.end());
    }
    return result;
}
string LSMSequenceObj::toString() const {
    return "LSMSequence[" + std::to_string(getGuid()) + "]" +
           vecToString(inputs[0]->getDims()) +
           (useQemu ? " qemu" : " integer_reference");
}

MLPObj::MLPObj(GraphObj *graph, Tensor x, Tensor w1, Tensor b1, Tensor w2,
               Tensor b2, bool useQemu)
    : OperatorObj(OpType::MLP, {x, w1, b1, w2, b2}, {nullptr}),
      useQemu(useQemu) {
    if (!checkValid(graph))
        throw std::runtime_error("MLP: invalid shape or dtype");
}
optional<vector<Shape>> MLPObj::inferShape(const TensorVec &values) {
    for (const auto &value : values)
        if (!(value->getDType() == DataType::Float32))
            return {};
    auto inputShape = values[0]->getDims(),
         firstWeightShape = values[1]->getDims(),
         secondWeightShape = values[3]->getDims();
    if (inputShape.size() != 2 || firstWeightShape.size() != 2 ||
        secondWeightShape.size() != 2 || inputShape[1] != firstWeightShape[0] ||
        firstWeightShape[1] != secondWeightShape[0] ||
        values[2]->getDims() != Shape{firstWeightShape[1]} ||
        values[4]->getDims() != Shape{secondWeightShape[1]})
        return {};
    mlp::validateDimensions(inputShape[0], inputShape[1], firstWeightShape[1],
                            secondWeightShape[1]);
    return vector<Shape>{{inputShape[0], secondWeightShape[1]}};
}
vector<DataType> MLPObj::inferDataType(const TensorVec &) const {
    return {DataType::Float32};
}
vector<int> MLPObj::getOpAttrVector() const {
    return {type.underlying(), int(useQemu)};
}
vector<int> MLPObj::getWorkloadVector() const {
    auto result = getOpAttrVector();
    for (const auto &input : inputs) {
        const auto shape = input->getDims();
        result.insert(result.end(), shape.begin(), shape.end());
    }
    return result;
}
string MLPObj::toString() const {
    return "MLP[" + std::to_string(getGuid()) + "]" +
           vecToString(inputs[0]->getDims()) +
           (useQemu ? " qemu" : " reference");
}
} // namespace infini
