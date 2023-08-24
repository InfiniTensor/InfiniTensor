#include "core/graph.h"
#include "core/runtime.h"
#include "operators/pad.h"
#include "test.h"
#include "xpu/xpu_kernel_without_config.h"
#include "xpu/xpu_runtime.h"

namespace infini {
TEST(xpu_Pad, run) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<XPURuntimeObj>();

    // Build input data on CPU
    Tensor icpu =
        make_ref<TensorObj>(Shape{1, 2, 3, 2}, DataType::Float32, cpuRuntime);

    // Build XPU graph;
    Graph g = make_ref<GraphObj>(xpuRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<PadObj>(i, nullptr, vector<int>{1, 0, 1, 1},
                               vector<int>{0, 3});

    // allocate XPU memory
    g->dataMalloc();
    i->setData(IncrementalGenerator());

    // Execute on XPU
    xpuRuntime->run(g);

    // clone XPU output to CPU
    auto o = op->getOutput();
    auto cpuo = o->clone(cpuRuntime);
    cpuo->printData();
    //  check results on CPU
    EXPECT_TRUE(cpuo->equalData(
        vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,
                      0, 1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8, 9, 0, 10, 11, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0}));
}
} // namespace infini
