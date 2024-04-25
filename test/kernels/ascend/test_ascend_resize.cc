#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/resize.h"
#include "test.h"

namespace infini {

TEST(Resize, Ascend_downsample_scales_nearest) {
    aclInit(nullptr);
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    scales->copyin(vector<float>{1, 1, 0.6, 0.6});

    auto ascendRuntime = make_ref<ASCENDRuntimeObj>();
    Graph gNpu = make_ref<GraphObj>(ascendRuntime);

    auto inputNpu = gNpu->cloneTensor(input);
    auto scalesNpu = gNpu->cloneTensor(scales);
    auto op = gNpu->addOp<ResizeObj>(inputNpu, nullptr, std::nullopt, nullptr,
                                     scalesNpu, nullptr);
    gNpu->dataMalloc();
    inputNpu->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    scalesNpu->copyin(vector<float>{1, 1, 0.6, 0.6});
    ascendRuntime->run(gNpu);

    //  copy output from NPU to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{1, 3}));
    aclFinalize();
}

// TEST(Resize, Ascend_upsample_scales_nearest) {
//     Runtime runtime = NativeCpuRuntimeObj::getInstance();
//     Graph gCpu = make_ref<GraphObj>(runtime);
//
//     auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
//     auto scales = gCpu->addTensor({4}, DataType::Float32);
//     gCpu->dataMalloc();
//     input->copyin(vector<float>{1, 2, 3, 4});
//     scales->copyin(vector<float>{1, 1, 2, 3});
//
//     auto ascendRuntime = make_ref<ascendRuntimeObj>();
//     Graph gNpu = make_ref<GraphObj>(ascendRuntime);
//
//     auto inputNpu = gNpu->cloneTensor(input);
//     auto scalesNpu = gNpu->cloneTensor(scales);
//     auto op = gNpu->addOp<ResizeObj>(inputNpu, nullptr, std::nullopt,
//     nullptr,
//                                       scalesNpu, nullptr);
//     gNpu->dataMalloc();
//     inputNpu->copyin(vector<float>{1, 2, 3, 4});
//     scalesNpu->copyin(vector<float>{1, 1, 2, 3});
//     ascendRuntime->run(gNpu);
//
//     //  copy output from NPU to CPU
//     auto oCpu = gCpu->cloneTensor(op->getOutput(0));
//     EXPECT_TRUE(
//         oCpu->equalData(vector<float>{1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
//                                       3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4}));
// }
} // namespace infini
