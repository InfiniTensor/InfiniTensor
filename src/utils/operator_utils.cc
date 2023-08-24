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
