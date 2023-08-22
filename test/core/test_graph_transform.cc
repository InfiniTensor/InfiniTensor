#include "core/graph.h"
#include "core/runtime.h"
#include "graph/graph.h"
#include "operators/matmul.h"
#include "operators/unary.h"
#include "operators/conv.h"
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
	GraphTopo<graph::NodeInfo, graph::EdgeInfo> graphTopo = g->transformToGraphTopo(*g);
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
		NodeInfo op1 = NodeInfo{common::OpType::MatMul, Attributes{}};
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
		EXPECT_EQ(searcher.nodes()[i], searcher1.nodes()[i]);
	}
	// compare edges
	for (int32_t i = 0; i < static_cast<int32_t>(edgeSize); ++i) {
		EXPECT_EQ(searcher.edges()[i], searcher1.edges()[i]);
	}
	// compare global inputs
	auto globalInput = searcher.globalInputs();
	auto globalInput1 = searcher1.globalInputs();
	EXPECT_EQ(globalInput.size(), globalInput1.size());
	for (size_t i = 0; i < globalInput.size(); ++i) {
		EXPECT_EQ(globalInput[i], globalInput1[i]);
	}
	// compare global outputs
	auto globalOutput = searcher.globalOutputs();
	auto globalOutput1 = searcher1.globalOutputs();
	EXPECT_EQ(globalOutput.size(), globalOutput1.size());
	for (size_t i = 0; i < globalOutput.size(); ++i) {
		EXPECT_EQ(globalOutput[i], globalOutput1[i]);
	}
}

void cmpTensors(TensorVec tensors1, TensorVec tensors2) {
	EXPECT_EQ(tensors1.size(), tensors2.size());
	for (size_t i = 0; i < tensors1.size(); ++i) {
		EXPECT_TRUE(tensors1[i]->equalData(tensors2[i]));
	}
}

TEST(Graph, from_graph_topo) {
	// build a compare InfiniTensor graph
	Runtime runtime = NativeCpuRuntimeObj::getInstance();
	Graph infiniG = make_ref<GraphObj>(runtime);
	Tensor i0 = infiniG->addTensor({1, 3, 224, 336}, DataType::Float32);
	Tensor w0 = infiniG->addTensor({6, 3, 5, 7}, DataType::Float32);
	Tensor o0 = infiniG->addTensor({1, 6, 216, 294}, DataType::Float32);
	auto conv = infiniG->addOpWithOutputs<ConvObj>(i0, w0, o0);
	infiniG->addOp<ReluObj>(o0, nullptr);
	infiniG->topo_sort();
	infiniG->print();
	using namespace refactor;
	// build a graphtopo
	GraphTopo<graph::NodeInfo, graph::EdgeInfo> topo;
	{
		using NodeInfo = graph::NodeInfo;
		using EdgeInfo = graph::EdgeInfo;
		using Tensor = graph::Tensor;
		using Attributes = graph::Attributes;
		using Attribute = graph::Attribute;
		using Ints = graph::Ints;
		NodeInfo op1 = NodeInfo{common::OpType::Conv, Attributes{{"dilations", Attribute{Ints{2, 7}}}}};
		NodeInfo op2 = NodeInfo{common::OpType::Relu, Attributes{}};
		EdgeInfo i0, w0, o0, o1;
		i0.info = Tensor{common::DataType::F32, {1, 3, 224, 336}};
		w0.info = Tensor{common::DataType::F32, {6, 3, 5, 7}};
		o0.info = Tensor{common::DataType::F32, {1, 6, 216, 294}};
		o1.info = Tensor{common::DataType::F32, {1, 6, 216, 294}};
		auto i = topo.addEdge(i0);
		auto w = topo.addEdge(w0);
		auto conv = topo.addNode(op1, {i, w}, {o0});
		auto relu = topo.addNode(op2, {conv[0]}, {o1});
		topo.markOutput({relu[0]});
	}
	// transform topo graph to InfiniTensor graph
	graph::Graph topoG = graph::Graph(std::move(topo));
	Runtime runtime = NativeCpuRuntimeObj::getInstance();
	Graph g = make_ref<GraphObj>(runtime);
	g->transformFromGraphTopo(topoG);
	g->topo_sort();
	// compare tensor
	TensorVec tensors1 = g->getTensors();
	TensorVec tensors2 = infiniG->getTensors();
	cmpTensors(tensors1, tensors2);
	// compare op
	OpVec ops1 = g->getOperators();
	OpVec ops2 = g->getOperators();
	EXPECT_EQ(ops1.size(), ops2.size());
	for (size_t i = 0; i < ops1.size(); ++i) {
		// compare op type
		EXPECT_EQ(ops1[i]->getOpType(), ops2[i]->getOpType()); 
		// compare op inputs
		auto inputs1 = ops1[i]->getInputs();
		auto inputs2 = ops2[i]->getInputs();
		cmpTensors(inputs1, inputs2);
		// compare op outputs
		auto outputs1 = ops1[i]->getOutputs();
		auto outputs2 = ops2[i]->getOutputs();
		cmpTensors(outputs1, outputs2);
	}
}

} // namesapce infini
