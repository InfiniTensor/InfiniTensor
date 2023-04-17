#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "nnet/Visitor/MatchReshapeVisitor.h"
#include "nnet/expr.h"
#include "nnet/nmutator.h"
#include "nnet/routine.h"
#include "nnet/test.h"
#include "operators/matmul.h"
#include "operators/membound.h"
#include "test.h"
using namespace infini;
using namespace std;

TEST(nnet, MemboundOpInterpretation) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
    g->dataMalloc();
    i0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    w0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    NMutator nmutator(NMutator::Mode::ToNaiveMembound);
    auto mutations = nmutator.run(g);
    ASSERT_EQ(mutations.size(), 2u);
    Graph gNew = mutations[1];
    gNew->print();

    gNew->dataMalloc();
    runtime->run(gNew);
    // check answer
    auto ops = gNew->getOperators();
    EXPECT_EQ(ops.size(), 1u);
    auto membound = ops[0];
    EXPECT_EQ(membound->getOpType(), OpType::MemBound);
    auto ans = make_ref<TensorObj>(Shape{1, 2, 4}, DataType::UInt32, runtime);
    ans->dataMalloc();
    ans->copyin(vector<uint32_t>{38, 44, 50, 56, 83, 98, 113, 128});
    EXPECT_TRUE(membound->getOutput()->equalData(ans));
}

TEST(nnet, MemboundOp_Ansor_Codegen) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Runtime cpu = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpu);
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::Float32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::Float32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::Float32);
    g->dataMalloc();
    i0->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    w0->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    NMutator nmutator(NMutator::Mode::ToNaiveMembound);
    auto mutations = nmutator.run(g);
    ASSERT_EQ(mutations.size(), 2u);
    Graph gNew = mutations[1];
    gNew->print();
    gNew->dataMalloc();
    runtime->run(gNew, true); // tune kernels

    // check answer
    auto ops = gNew->getOperators();
    EXPECT_EQ(ops.size(), 1u);
    auto membound = ops[0];
    EXPECT_EQ(membound->getOpType(), OpType::MemBound);
    auto ans = make_ref<TensorObj>(Shape{1, 2, 4}, DataType::Float32, cpu);
    ans->dataMalloc();
    ans->copyin(vector<float>{38, 44, 50, 56, 83, 98, 113, 128});

    auto oCpu = gCpu->cloneTensor(membound->getOutput());
    oCpu->printData();
    EXPECT_TRUE(oCpu->equalData(ans));

    // Timing
    // double time = timeit([&]() { runtime->run(gNew, false); }); // tune
    // kernels std::cout << "Time (ms):" << time << std::endl;
}

pair<std::vector<nnet::Tensor>, nnet::Expr> getPReluExpr(int size) {
    using namespace nnet;
    using nnet::make_ref;
    DEFINE_VAR(i);
    auto A = make_ref<TensorNode>("A", vector{size});
    auto B = make_ref<TensorNode>("B", vector{size});
    Expr e = make_ref<FuncNode>(makeSubscript(A, {i}) - makeSubscript(B, {i}),
                                FuncType::PRelu);
    Expr ret = makeRangeOperator({{i, {0, size}}}, {}, e);
    return {{A, B}, ret};
}

TEST(nnet, PRelu_Ansor_Codegen) {
    auto cuda = make_ref<CudaRuntimeObj>();
    Runtime cpu = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(cuda);
    Tensor i0 = g->addTensor(vector{12});
    Tensor w0 = g->addTensor(vector{12});
    Tensor o0 = g->addTensor(vector{12});
    auto [nnetInputs, expr] = getPReluExpr(12);
    g->addOpWithOutputs<MemBoundObj>(vector{i0, w0}, vector{o0}, nnetInputs,
                                     expr, -1);
    g->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(ValGenerator<5>());
    cuda->run(g, true); // tune kernels

    // check answer
    auto ans = make_ref<TensorObj>(Shape{12}, DataType::Float32, cpu);
    ans->dataMalloc();
    ans->copyin(
        vector<float>{-1.25, -1., -0.75, -0.5, -0.25, 0, 1, 2, 3, 4, 5, 6});

    Graph gCpu = make_ref<GraphObj>(cpu);
    auto oCpu = gCpu->cloneTensor(o0);
    EXPECT_TRUE(oCpu->equalData(ans));
}
