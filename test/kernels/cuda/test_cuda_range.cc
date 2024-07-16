#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/range.h"

// #include "test.h"

namespace infini {

void test_range(const float start, const float limit, const float delta,
                    const vector<float> &ExpectData) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);


    auto op = g->addOp<RangeObj>(start, limit, delta, nullptr);

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

TEST(CUDA_range, run) {
    test_range( // validate value
        1.0,
        10.0,
        2.0,
        vector<float>{1.0, 3.0, 5.0, 7.0, 9.0});
    test_range(
        3.8,
        10.0,
        2,
        vector<float>{3.8, 5.8, 7.8, 9.8});
    test_range(
        -5.0,
        10.0,
        5.0,
        vector<float>{-5., 0., 5.0});
    test_range(
        0.1,
        1.,
        0.3,
        vector<float>{0.1, 0.4, 0.7});
} // python output


} // namespace infini




