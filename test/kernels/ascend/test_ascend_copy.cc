#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/reshape.h"
#include "operators/squeeze.h"
#include "operators/unsqueeze.h"

#include "test.h"

namespace infini {

template <class T>
void testReshape(const std::function<void(void *, size_t, DataType)> &generator,
                 const Shape &shape, const Shape &outputShape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto npuOp = npuGraph->addOp<T>(inputNpu, nullptr, outputShape);
    npuGraph->dataMalloc();
    inputNpu->setData(generator);
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

    // Check
    inputCpu->print();
    inputCpu->printData();
    outputNpu2Cpu->print();
    outputNpu2Cpu->printData();
    EXPECT_TRUE(inputCpu->equalData(outputNpu2Cpu, 1e-3));
}

TEST(ascend_Unary, run) {
    aclInit(nullptr);
    testReshape<ReshapeObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                            Shape{1, 2, 6});
    testReshape<SqueezeObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                            Shape{0});
    testReshape<UnsqueezeObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                              Shape{4});
    aclFinalize();
}

} // namespace infini
