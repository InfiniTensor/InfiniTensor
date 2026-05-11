#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testClip(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, 
    std::optional<float> min = std::nullopt, 
    std::optional<float> max = std::nullopt) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto npuOp = npuGraph->addOp<T>(inputNpu, nullptr, min, max); // set min and max 
    npuGraph->dataMalloc();
    inputNpu->setData(generator);
    npuRuntime->run(npuGraph);

    // Get output
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    cpuGraph->addTensor(inputCpu);
    auto cpuOp =
        cpuGraph->addOp<T>(inputCpu, nullptr, min, max);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();

    // Check result
    outputCpu->printData();
    outputNpu2Cpu->printData();
    EXPECT_TRUE(outputCpu->equalData(outputNpu2Cpu, 1e-3));
}

TEST(ascend_Clip, run) {
    aclInit(nullptr);

    // Test Clip op£¬Input is incremetal tensor
    testClip<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testClip<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3}, 1.0f, 4.0f);
    testClip<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3}, std::nullopt, 4.0f);
    testClip<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3}, 1.0f, std::nullopt);

    aclFinalize();
}

} // namespace infini
