#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/ascend_plugin_sub.h"

#include "test.h"

namespace infini {

template <class T>
void testPluginSub(const vector<float> &inputData,
                   const vector<float> &outputData,
                   const vector<float> &ExpectData, Shape shapeIn,
                   Shape shapeOut) {
    // Runtime
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();
    int kernel_size = 5;
    int stride = 1;
    Graph gCpu = make_ref<GraphObj>(runtime);
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);

    auto input = gCpu->addTensor(shapeIn, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(inputData);

    auto inputNpu = npuGraph->cloneTensor(input);
    auto npuOp = npuGraph->addOp<T>(inputNpu, nullptr, kernel_size, stride);
    npuGraph->dataMalloc();
    npuRuntime->run(npuGraph);

    auto outputCpu = gCpu->cloneTensor(npuOp->getOutput());
    EXPECT_TRUE(outputCpu->equalData(ExpectData));
}

TEST(ascend_Plugin_Sub, run1) {
    aclInit(nullptr);
    vector<float> inputData(1 * 3 * 5 * 5, 2.0f);
    vector<float> outputData(1 * 3 * 1 * 1 * 16, 2.0f);
    vector<float> ExpectData(1 * 3 * 1 * 1 * 16, 0.0f);

    testPluginSub<AscendPluginSubObj>(inputData, outputData, ExpectData,
                                      Shape{1, 3, 5, 5}, Shape{1, 3, 1, 1, 16});
    aclFinalize();
}

} // namespace infini
