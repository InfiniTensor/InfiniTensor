#include "cmath"
#include "core/graph.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/resize.h"
#include "test.h"
namespace infini {
TEST(Resize, Mkl_downsample_sizes_nearest) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);

    auto input = gCpu->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    sizes->copyin(vector<uint32_t>{1, 1, 1, 3});

    auto runtime = make_ref<MklRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);

    auto op = g->addOp<ResizeObj>(g->cloneTensor(input), nullptr, std::nullopt,
                                  g->cloneTensor(sizes), nullptr, nullptr,
                                  ResizeObj::EKeepAspectRatioPolicy::stretch,
                                  ResizeObj::ENearestMode::ceil);
    g->dataMalloc();
    runtime->run(g);

    EXPECT_TRUE(op->getOutput(0)->equalData(vector<float>{5, 7, 8}));
}
} // namespace infini
