#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "nnet/nmutator.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "test.h"

namespace infini {

TEST(TestCudaRuntime, CudaGraph) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);

    const int n = 2, c = 256, h = 2, w = 2, f = 448, r = 3, s = 2;
    auto i0 = g->addTensor({n, c, h, w}, DataType::Float32, TensorType::Input);
    auto w0 =
        g->addTensor({f, c, r, s}, DataType::Float32, TensorType::Initialized);
    g->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 1, 1, 1, 1);
    g->dataMalloc();
    runtime->run(g, true);
    runtime->run(g, false);
    runtime->getPerfTime(g);

    auto time = runtime->timeWithCudaGraph(g);
    EXPECT_GE(time, 0.01);
}

TEST(TestCudaRuntime, CudaGraphMembound) {
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
    runtime->run(gNew, false);
    runtime->getPerfTime(gNew);

    auto time = runtime->timeWithCudaGraph(gNew);
    EXPECT_GE(time, 0.001);
}
} // namespace infini
