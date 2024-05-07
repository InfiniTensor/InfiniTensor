#include "bang/bang_runtime.h"
#include "cmath"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/resize.h"
#include "test.h"
namespace infini {
TEST(Resize, Bang_downsample_sizes_nearest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    scales->copyin(vector<float>{1, 1, 0.6, 0.6});

    auto bangRuntime = make_ref<BangRuntimeObj>();
    Graph gMlu = make_ref<GraphObj>(bangRuntime);

    auto inputMlu = gMlu->cloneTensor(input);
    auto scalesMlu = gMlu->cloneTensor(scales);
    auto op = gMlu->addOp<ResizeObj>(inputMlu, nullptr, std::nullopt, nullptr,
                                     scalesMlu, nullptr);
    gMlu->dataMalloc();
    inputMlu->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    scalesMlu->copyin(vector<float>{1, 1, 0.6, 0.6});

    bangRuntime->run(gMlu);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{5, 8}));
}

TEST(Resize, Bang_upsample_sizes_nearest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4});
    scales->copyin(vector<float>{1, 1, 2, 3});

    auto bangRuntime = make_ref<BangRuntimeObj>();
    Graph gMlu = make_ref<GraphObj>(bangRuntime);

    auto inputMlu = gMlu->cloneTensor(input);
    auto scalesMlu = gMlu->cloneTensor(scales);
    auto op = gMlu->addOp<ResizeObj>(inputMlu, nullptr, std::nullopt, nullptr,
                                     scalesMlu, nullptr);
    gMlu->dataMalloc();
    inputMlu->copyin(vector<float>{1, 2, 3, 4});
    scalesMlu->copyin(vector<float>{1, 1, 2, 3});

    bangRuntime->run(gMlu);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(
        oCpu->equalData(vector<float>{1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                                      3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4}));
}
} // namespace infini
