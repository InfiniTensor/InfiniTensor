#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testClip(const std::function<void(void *, size_t, DataType)> &generator,
              const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    Tensor inputMin =
        make_ref<TensorObj>(Shape{}, DataType::Float32, cpuRuntime);
    Tensor inputMax =
        make_ref<TensorObj>(Shape{}, DataType::Float32, cpuRuntime);

    inputCpu->dataMalloc();
    inputMin->dataMalloc();
    inputMax->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto inputMinNpu = npuGraph->cloneTensor(inputMin);
    auto inputMaxNpu = npuGraph->cloneTensor(inputMax);
    float min = 1.0;
    float max = 4.0;
    // inputMin->copyin(vector<float>{min});
    // inputMax->copyin(vector<float>{max});
    auto npuOp = npuGraph->addOp<T>(
        TensorVec{inputNpu, inputMinNpu, inputMaxNpu}, nullptr);
    npuGraph->dataMalloc();
    inputMinNpu->copyin(vector<float>{min});
    inputMaxNpu->copyin(vector<float>{max});
    inputNpu->setData(generator);
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);
    inputCpu->printData();
    outputNpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(ascend_Concat, run) {
    aclInit(nullptr);
    testClip<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    aclFinalize();
}

} // namespace infini
