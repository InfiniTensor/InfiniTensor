#include "utils/operator_utils.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {
    if (A.empty() && B.empty()) {
        return {};
    }
    auto A_ = A;
    auto B_ = B;
    int rankA = A.size();
    int rankB = B.size();
    int rank = std::max(rankA, rankB);
    if (rankA < rank) {
        for (int i = 0; i < rank - rankA; ++i) {
            A_.insert(A_.begin(), 1);
        }
    }
    if (rankB < rank) {
        for (int i = 0; i < rank - rankB; ++i) {
            B_.insert(B_.begin(), 1);
        }
    }
    Shape ret;
    for (int i = 0; i < rank; ++i) {
        IT_ASSERT(A_[i] == B_[i] || A_[i] == 1 || B_[i] == 1);
        auto shapeEle = std::max(A_[i], B_[i]);
        ret.emplace_back(shapeEle);
    }
    return ret;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

void addOperatorFromGraphTopo(
    GraphObj &g,
    const GraphTopoSearcher<refactor::graph::NodeInfo,
                            refactor::graph::EdgeInfo>::Node node,
    const std::unordered_map<int32_t, Tensor> edgeIdxToTensor,
    const std::unordered_map<int32_t, infini::Shape> edgeIdxToShape) {
    vector<int> inputs, outputs;
    std::vector<infini::Shape> shapes;
    for (auto edge : node.inputs()) {
        if (edge.info().isTensor()) {
            IT_ASSERT(edgeIdxToTensor.count(edge.index()) > 0);
            inputs.emplace_back(edge.index());
        } else if (edge.info().isShapeVariable()) {
            IT_ASSERT(edgeIdxToShape.count(edge.index()) > 0);
            shapes.emplace_back(edgeIdxToShape.at(edge.index()));
        }
    }
    for (auto edge : node.outputs()) {
        IT_ASSERT(edge.info().isTensor());
        IT_ASSERT(edgeIdxToTensor.count(edge.index()) > 0);
        outputs.emplace_back(edge.index());
    }
    auto attr = node.info().attributes;
    if (node.info().opType == refactor::common::OpType::Conv) {
        auto p = attr["pads"].ints();
        auto s = attr["strides"].ints();
        auto d = attr["dilations"].ints();
        g.addOpWithOutputs<ConvObj>(
            edgeIdxToTensor.at(inputs[0]), edgeIdxToTensor.at(inputs[1]),
            edgeIdxToTensor.at(outputs[0]), p[0], p[1], s[0], s[1], d[0], d[1]);
    } else if (node.info().opType == refactor::common::OpType::Relu) {
        g.addOpWithOutputs<ReluObj>(edgeIdxToTensor.at(inputs[0]),
                                    edgeIdxToTensor.at(outputs[0]));
    } else if (node.info().opType == refactor::common::OpType::Add) {
        g.addOpWithOutputs<AddObj>(edgeIdxToTensor.at(inputs[0]),
                                   edgeIdxToTensor.at(inputs[1]),
                                   edgeIdxToTensor.at(outputs[0]));
    } else if (node.info().opType == refactor::common::OpType::Identity) {
        g.addOpWithOutputs<IdentityObj>(edgeIdxToTensor.at(inputs[0]),
                                        edgeIdxToTensor.at(outputs[0]));
    } else if (node.info().opType == refactor::common::OpType::AveragePool) {
        auto p = attr["pads"].ints();
        auto s = attr["strides"].ints();
        auto d = attr["dilations"].ints();
        int h = edgeIdxToTensor.at(inputs[0])->getDims()[2];
        int w = edgeIdxToTensor.at(inputs[0])->getDims()[3];
        g.addOpWithOutputs<AvgPoolObj>(edgeIdxToTensor.at(inputs[0]),
                                       edgeIdxToTensor.at(outputs[0]), h, w,
                                       d[0], d[1], p[0], p[1], s[0], s[1]);
    } else if (node.info().opType == refactor::common::OpType::Reshape) {
        g.addOpWithOutputs<ReshapeObj>(edgeIdxToTensor.at(inputs[0]),
                                       edgeIdxToTensor.at(outputs[0]),
                                       shapes[0]);
    } else if (node.info().opType == refactor::common::OpType::Gemm) {
        // FIXME unsupport attributes: `alpha` `beta`
        auto alpha = attr["alpha"].float_();
        auto beta = attr["beta"].float_();
        auto transA = attr["transA"].int_();
        auto transB = attr["transB"].int_();
        IT_ASSERT(alpha == 1.0);
        IT_ASSERT(beta == 1.0);
        g.addOpWithOutputs<MatmulObj>(
            edgeIdxToTensor.at(inputs[0]), edgeIdxToTensor.at(inputs[1]),
            edgeIdxToTensor.at(outputs[0]), transA, transB,
            inputs.size() > 2 ? edgeIdxToTensor.at(inputs[2]) : nullptr,
            ActType::None);
    } else if (node.info().opType ==
               refactor::common::OpType::BatchNormalization) {
        auto epsilon = attr["epsilon"].float_();
        auto momentum = attr["momentum"].float_();
        auto training_mode = attr["training_mode"].int_();
        g.addOpWithOutputs<BatchNormObj>(
            edgeIdxToTensor.at(inputs[0]), edgeIdxToTensor.at(outputs[0]),
            edgeIdxToTensor.at(inputs[3]), edgeIdxToTensor.at(inputs[4]),
            edgeIdxToTensor.at(inputs[1]), edgeIdxToTensor.at(inputs[2]),
            momentum, epsilon, training_mode != 0);
    } else {
        IT_TODO_HALT_MSG("Don't support opType " +
                         node.info().opType.toString());
    }
}

// add batchnormalization conv gemm globalaveragepool maxpool relu reshape
RefactorNodeInfoCell getNodeInfo(const Operator &obj) {
    using namespace refactor;
    auto type = obj->getOpType().underlying();
    graph::NodeInfo nodeInfo{common::OpType::Unknown};
#define CASE(T)                                                                \
    case OpType::T:                                                            \
        nodeInfo = {common::OpType::T, refactor::graph::Attributes{}};         \
        break;

    switch (type) {
    case OpType::MatMul: {
        auto matmul = dynamic_cast<const MatmulObj *>(obj.get());
        auto transA = matmul->getTransA();
        auto transB = matmul->getTransB();
        if (transA || transB) {
            nodeInfo = {common::OpType::Gemm,
                        {{"alpha", {static_cast<graph::Float>(1.0)}},
                         {"beta", {static_cast<graph::Float>(1.0)}},
                         {"transA", {static_cast<graph::Int>(transA)}},
                         {"transB", {static_cast<graph::Int>(transB)}}}};
        } else {
            nodeInfo = {common::OpType::MatMul, graph::Attributes{}};
        }
        break;
    }
    case OpType::BatchNormalization: {
        auto batchNorm = dynamic_cast<const BatchNormObj *>(obj.get());
        auto momentum = batchNorm->getMomentum();
        auto eps = batchNorm->getEps();
        auto trainingMode = batchNorm->getTrainingMode();
        nodeInfo = {
            common::OpType::BatchNormalization,
            {{"epsilon", {static_cast<graph::Float>(eps)}},
             {"momentum", {static_cast<graph::Float>(momentum)}},
             {"training_mode", {static_cast<graph::Int>(trainingMode)}}}};
        break;
    }
    case OpType::Conv: {
        auto conv = dynamic_cast<const ConvObj *>(obj.get());
        auto group = conv->getNumGroups();
        auto pads = conv->getPads();
        auto strides = conv->getStrides();
        auto dilations = conv->getDilations();
        std::transform(pads.begin(), pads.end(), pads.begin(),
                       [](int x) { return static_cast<graph::Int>(x); });
        std::transform(strides.begin(), strides.end(), strides.begin(),
                       [](int x) { return static_cast<graph::Int>(x); });
        std::transform(dilations.begin(), dilations.end(), dilations.begin(),
                       [](int x) { return static_cast<graph::Int>(x); });
        nodeInfo = {
            common::OpType::Conv,
            {{"group", {static_cast<graph::Int>(group)}},
             {"pads", {graph::Ints(pads.begin(), pads.end())}},
             {"strides", {graph::Ints(strides.begin(), strides.end())}},
             {"dilations", {graph::Ints(dilations.begin(), dilations.end())}}}};
        break;
    }
    case OpType::AveragePool: {
        auto averagePool = dynamic_cast<const AvgPoolObj *>(obj.get());
        auto pads = averagePool->getPads();
        auto strides = averagePool->getStrides();
        auto dilations = averagePool->getDilations();
        std::transform(pads.begin(), pads.end(), pads.begin(),
                       [](int x) { return static_cast<graph::Int>(x); });
        std::transform(strides.begin(), strides.end(), strides.begin(),
                       [](int x) { return static_cast<graph::Int>(x); });
        std::transform(dilations.begin(), dilations.end(), dilations.begin(),
                       [](int x) { return static_cast<graph::Int>(x); });
        nodeInfo = {
            common::OpType::AveragePool,
            {{"pads", {graph::Ints(pads.begin(), pads.end())}},
             {"strides", {graph::Ints(strides.begin(), strides.end())}},
             {"dilations", {graph::Ints(dilations.begin(), dilations.end())}}}};
        break;
    }
    case OpType::MaxPool: {
        auto maxPool = dynamic_cast<const MaxPoolObj *>(obj.get());
        auto pads = maxPool->getPads();
        auto strides = maxPool->getStrides();
        auto dilations = maxPool->getDilations();
        std::transform(pads.begin(), pads.end(), pads.begin(),
                       [](int x) { return static_cast<graph::Int>(x); });
        std::transform(strides.begin(), strides.end(), strides.begin(),
                       [](int x) { return static_cast<graph::Int>(x); });
        std::transform(dilations.begin(), dilations.end(), dilations.begin(),
                       [](int x) { return static_cast<graph::Int>(x); });
        nodeInfo = {
            common::OpType::MaxPool,
            {{"pads", {graph::Ints(pads.begin(), pads.end())}},
             {"strides", {graph::Ints(strides.begin(), strides.end())}},
             {"dilations", {graph::Ints(dilations.begin(), dilations.end())}}}};
        break;
    }
        CASE(Reshape)
        CASE(Relu)
        CASE(Add)
        CASE(Sub)
        CASE(Mul)
        CASE(Div)
        CASE(Identity)
    default:
        IT_TODO_HALT_MSG("Don't Support OpType " + obj->getOpType().toString());
    }
#undef CASE
    return {std::move(nodeInfo)};
}

void processShapeVariable(
    const Operator &obj,
    GraphTopo<RefactorNodeInfoCell, RefactorEdgeInfoCell> &graphTopo,
    std::vector<EdgeRef> &nodeInputs) {
    using namespace refactor;
    auto type = obj->getOpType().underlying();
    if (type == OpType::Reshape) {
        auto reshape = dynamic_cast<const ReshapeObj *>(obj.get());
        auto dims = reshape->getShape();
        std::vector<int64_t> shape(dims.size());
        std::transform(dims.begin(), dims.end(), shape.begin(),
                       [](int x) { return static_cast<int64_t>(x); });
        graph::EdgeInfo edge;
        edge.info = graph::ShapeVariable{shape};
        auto e = graphTopo.addEdge({std::move(edge)});
        nodeInputs.emplace_back(e);
    }
    return;
}
} // namespace infini
