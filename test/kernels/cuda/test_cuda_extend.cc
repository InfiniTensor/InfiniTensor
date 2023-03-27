#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/extend.h"

#include "test.h"

namespace infini {

TEST(CUDA_Extend, run) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor icpu =
        make_ref<TensorObj>(Shape{2, 3, 2, 2}, DataType::Float32, cpuRuntime);
    icpu->dataMalloc();
    icpu->setData(IncrementalGenerator());

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<ExtendObj>(i, nullptr, 1, 1);

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    //  check results on CPU
    EXPECT_TRUE(ocpu->equalData(vector<float>{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 0,  1,  2,  3,
        4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}));
}
} // namespace infini
