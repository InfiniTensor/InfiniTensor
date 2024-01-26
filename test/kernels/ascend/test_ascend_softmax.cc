#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/softmax.h"

#include "test.h"

namespace infini {

template <class T>
void testSoftmax(const std::function<void(void *, size_t, DataType)> &generator,
                 const Shape &shape, int axis, vector<float> Out) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu1 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu1->dataMalloc();
    // inputCpu1->setData(generator);

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu1 = npuGraph->cloneTensor(inputCpu1);
    auto npuOp = npuGraph->addOp<T>(inputNpu1, nullptr, axis);
    npuGraph->dataMalloc();
    inputNpu1->setData(generator);
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

    // Check
    EXPECT_TRUE(outputNpu2Cpu->equalData(Out));
}

TEST(ascend_ElementWise, run) {
    aclInit(nullptr);
    testSoftmax<SoftmaxObj>(
        IncrementalGenerator(), Shape{2, 2, 2, 2}, 1,
        vector<float>{0.0179862, 0.0179862, 0.0179862, 0.0179862, 0.9820138,
                      0.9820138, 0.9820138, 0.9820138, 0.0179862, 0.0179862,
                      0.0179862, 0.0179862, 0.9820138, 0.9820138, 0.9820138,
                      0.9820138});
    testSoftmax<SoftmaxObj>(
        IncrementalGenerator(), Shape{2, 2, 2, 2}, 3,
        vector<float>{0.2689414, 0.7310586, 0.2689414, 0.7310586, 0.2689414,
                      0.7310586, 0.2689414, 0.7310586, 0.2689414, 0.7310586,
                      0.2689414, 0.7310586, 0.2689414, 0.7310586, 0.2689414,
                      0.7310586});
    aclFinalize();
}

} // namespace infini
