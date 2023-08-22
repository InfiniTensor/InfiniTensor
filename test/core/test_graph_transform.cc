#include "core/graph.h"
#include "core/runtime.h"
#include "graph/graph.h"
#include "operators/matmul.h"
#include "operators/reshape.h"
#include "operators/unary.h"
#include "test.h"

namespace infini {

TEST(Graph, transform_refactor) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
    auto matmul = g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    g->addOp<ReluObj>(o0, nullptr);
    g->print();
    using namespace refactor;
    GraphTopo<graph::NodeInfo, graph::EdgeInfo> graphTopo =
        g->transformToGraphTopo(*g);
    auto searcher = GraphTopoSearcher(std::move(graphTopo));
    auto nodeSize = searcher.nodes().size();
    auto edgeSize = searcher.edges().size();
    // build a compare graphtopo
    GraphTopo<graph::NodeInfo, graph::EdgeInfo> topo;
    {
        using NodeInfo = graph::NodeInfo;
        using EdgeInfo = graph::EdgeInfo;
        using Tensor = graph::Tensor;
        using Attributes = graph::Attributes;
        NodeInfo op1 = NodeInfo{common::OpType::MatMul,
                                Attributes{{"transA", graph::Int(false)},
                                           {"transB", graph::Int(false)}}};
        NodeInfo op2 = NodeInfo{common::OpType::Relu, Attributes{}};
        EdgeInfo i0, w0, o0, o1;
        i0.info = Tensor{common::DataType::U32, {1, 2, 3}};
        w0.info = Tensor{common::DataType::U32, {1, 3, 4}};
        o0.info = Tensor{common::DataType::U32, {1, 2, 4}};
        o1.info = Tensor{common::DataType::U32, {1, 2, 4}};
        auto i = topo.addEdge(i0);
        auto w = topo.addEdge(w0);
        auto matmul = topo.addNode(op1, {i, w}, {o0});
        auto relu = topo.addNode(op2, {matmul[0]}, {o1});
        topo.markOutput({relu[0]});
    }
    auto searcher1 = GraphTopoSearcher(std::move(topo));
    // compare nodes
    for (int32_t i = 0; i < static_cast<int32_t>(nodeSize); ++i) {
        EXPECT_EQ(searcher.nodes()[i].info(), searcher1.nodes()[i].info());
    }
    // compare edges
    for (int32_t i = 0; i < static_cast<int32_t>(edgeSize); ++i) {
        EXPECT_EQ(searcher.edges()[i].info(), searcher1.edges()[i].info());
    }
    // compare global inputs
    auto globalInput = searcher.globalInputs();
    auto globalInput1 = searcher1.globalInputs();
    EXPECT_EQ(globalInput.size(), globalInput1.size());
    for (size_t i = 0; i < globalInput.size(); ++i) {
        EXPECT_EQ(globalInput[i].info(), globalInput1[i].info());
    }
    // compare global outputs
    auto globalOutput = searcher.globalOutputs();
    auto globalOutput1 = searcher1.globalOutputs();
    EXPECT_EQ(globalOutput.size(), globalOutput1.size());
    for (size_t i = 0; i < globalOutput.size(); ++i) {
        EXPECT_EQ(globalOutput[i].info(), globalOutput1[i].info());
    }
}

TEST(Graph, transform_reshape) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
    auto op = g->addOp<ReshapeObj>(i, nullptr, Shape{3, 2, 4, 3});
    g->print();
    using namespace refactor;
    GraphTopo<graph::NodeInfo, graph::EdgeInfo> graphTopo =
        g->transformToGraphTopo(*g);
    auto searcher = GraphTopoSearcher(std::move(graphTopo));
    auto nodeSize = searcher.nodes().size();
    auto edgeSize = searcher.edges().size();
    GraphTopo<graph::NodeInfo, graph::EdgeInfo> topo;
    // build a compare graphtopo
    {
        using NodeInfo = graph::NodeInfo;
        using EdgeInfo = graph::EdgeInfo;
        using Tensor = graph::Tensor;
        using ShapeVariable = graph::ShapeVariable;
        using Attributes = graph::Attributes;
        NodeInfo op1 = NodeInfo{common::OpType::Reshape, Attributes{}};
        EdgeInfo input, shape, output;
        input.info = Tensor{common::DataType::F32, {2, 3, 3, 4}};
        shape.info = ShapeVariable{{3, 2, 4, 3}};
        output.info = Tensor{common::DataType::F32, {3, 2, 4, 3}};
        auto i = topo.addEdge(input);
        auto s = topo.addEdge(shape);
        auto reshape = topo.addNode(op1, {i, s}, {output});
        topo.markOutput({reshape[0]});
    }
    auto searcher1 = GraphTopoSearcher(std::move(topo));
    // compare nodes
    for (int32_t i = 0; i < static_cast<int32_t>(nodeSize); ++i) {
        EXPECT_EQ(searcher.nodes()[i].info(), searcher1.nodes()[i].info());
    }
    // compare edges
    for (int32_t i = 0; i < static_cast<int32_t>(edgeSize); ++i) {
        EXPECT_EQ(searcher.edges()[i].info(), searcher1.edges()[i].info());
    }
    // compare global inputs
    auto globalInput = searcher.globalInputs();
    auto globalInput1 = searcher1.globalInputs();
    EXPECT_EQ(globalInput.size(), globalInput1.size());
    for (size_t i = 0; i < globalInput.size(); ++i) {
        EXPECT_EQ(globalInput[i].info(), globalInput1[i].info());
    }
    // compare global outputs
    auto globalOutput = searcher.globalOutputs();
    auto globalOutput1 = searcher1.globalOutputs();
    EXPECT_EQ(globalOutput.size(), globalOutput1.size());
    for (size_t i = 0; i < globalOutput.size(); ++i) {
        EXPECT_EQ(globalOutput[i].info(), globalOutput1[i].info());
    }
}

} // namespace infini
