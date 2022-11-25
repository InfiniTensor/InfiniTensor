#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/pad.h"
#include "test.h"

namespace infini {
TEST(Pad, Cuda) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor icpu =
        make_ref<TensorObj>(Shape{1, 2, 3, 2}, DataType::Float32, cpuRuntime);
    icpu->dataMalloc();
    icpu->setData(IncrementalGenerator());

    // Build CUDA graph;
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<PadObj>(i, nullptr, vector<int>{1, 0, 1, 1},
                               vector<int>{0, 3});

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto o = op->getOutput();
    auto cpuo = o->clone(cpuRuntime);

    //  check results on CPU
    EXPECT_TRUE(cpuo->equalData(
        vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,
                      0, 1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8, 9, 0, 10, 11, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0}));
}
} // namespace infini
