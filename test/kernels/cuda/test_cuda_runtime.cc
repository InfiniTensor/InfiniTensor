#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"
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

} // namespace infini
