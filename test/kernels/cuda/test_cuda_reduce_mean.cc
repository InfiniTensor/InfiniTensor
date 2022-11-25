#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/reduce_mean.h"

#include "test.h"

namespace infini {

void test_reducemean(const Shape &shape, const vector<float> &data,
                     const optional<const vector<int>> &axis, bool keepDims,
                     const vector<float> &ExpectData) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor icpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    icpu->dataMalloc();
    icpu->copyin(data);

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<ReduceMeanObj>(i, nullptr, axis, keepDims);

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    //  check results on CPU
    EXPECT_TRUE(ocpu->equalData(ExpectData));
}

TEST(CUDA_ReduceMean, run) {
    test_reducemean(Shape{3, 2, 2},
                    vector<float>{5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2},
                    std::nullopt, true, vector<float>{18.25});
    test_reducemean(Shape{1, 3, 2, 2, 1},
                    vector<float>{5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2},
                    std::nullopt, false, vector<float>{18.25});

    test_reducemean(Shape{2, 3, 2, 2},
                    vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                  8,  9,  10, 11, 12, 13, 14, 15,
                                  16, 17, 18, 19, 20, 21, 22, 23},
                    vector<int>{1, 2}, false, vector<float>{5, 6, 17, 18});
    test_reducemean(Shape{2, 3, 2, 2, 1},
                    vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                  8,  9,  10, 11, 12, 13, 14, 15,
                                  16, 17, 18, 19, 20, 21, 22, 23},
                    vector<int>{1, 2}, true, vector<float>{5, 6, 17, 18});
}

} // namespace infini
