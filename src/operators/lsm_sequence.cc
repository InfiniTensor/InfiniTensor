#include "operators/lsm_sequence.h"
#include "utils/mlp_compute.h"
namespace infini {
LSMSequenceObj::LSMSequenceObj(GraphObj *graph, Tensor spikes,
                               Tensor inputWeights, Tensor recurrent,
                               Tensor state, const vector<int> &params,
                               bool useQemu, bool traceAll)
    : OperatorObj(OpType::LSMSequence, {spikes, inputWeights, recurrent, state},
                  {nullptr, nullptr, nullptr}),
      useQemu(useQemu), traceAll(traceAll) {
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
    const auto &s = values[0]->getDims();
    const auto &w = values[1]->getDims();
    const auto &r = values[2]->getDims();
    if (s.size() != 2 || w.size() != 2 || r.size() != 2)
        return {};
    int t = s[0], channels = s[1], n = r[0];
    if (t < 1 || t > 6000 || channels < 1 || channels > 1024 || n < 1 ||
        n > 1024 || r[1] != n || w != Shape{channels, n} ||
        values[3]->getDims() != Shape{lsm::StateRows, n})
        return {};
    return vector<Shape>{
        {1, n}, {lsm::StateRows, n}, {traceAll ? t : 1, lsm::StateRows, n}};
}
vector<DataType> LSMSequenceObj::inferDataType(const TensorVec &) const {
    return {DataType::Float32, DataType::Int32, DataType::Int32};
}
vector<int> LSMSequenceObj::getOpAttrVector() const {
    vector<int> result{type.underlying(), int(useQemu), int(traceAll)};
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
optional<vector<Shape>> MLPObj::inferShape(const TensorVec &v) {
    for (const auto &t : v)
        if (!(t->getDType() == DataType::Float32))
            return {};
    auto x = v[0]->getDims(), w1 = v[1]->getDims(), w2 = v[3]->getDims();
    if (x.size() != 2 || w1.size() != 2 || w2.size() != 2 || x[1] != w1[0] ||
        w1[1] != w2[0] || v[2]->getDims() != Shape{w1[1]} ||
        v[4]->getDims() != Shape{w2[1]})
        return {};
    mlp::validateDimensions(x[0], x[1], w1[1], w2[1]);
    return vector<Shape>{{x[0], w2[1]}};
}
vector<DataType> MLPObj::inferDataType(const TensorVec &) const {
    return {DataType::Float32};
}
vector<int> MLPObj::getOpAttrVector() const {
    return {type.underlying(), int(useQemu)};
}
vector<int> MLPObj::getWorkloadVector() const {
    auto v = getOpAttrVector();
    for (const auto &t : inputs) {
        const auto shape = t->getDims();
        v.insert(v.end(), shape.begin(), shape.end());
    }
    return v;
}
string MLPObj::toString() const {
    return "MLP[" + std::to_string(getGuid()) + "]" +
           vecToString(inputs[0]->getDims()) +
           (useQemu ? " qemu" : " reference");
}
} // namespace infini
