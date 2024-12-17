#include "bang/bang_runtime.h"
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
    auto bangRuntime = make_ref<BangRuntimeObj>();

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
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);
    auto inputMinGpu = bangGraph->cloneTensor(inputMin);
    auto inputMaxGpu = bangGraph->cloneTensor(inputMax);
    float min = 1.0;
    float max = 4.0;
    inputMin->copyin(vector<float>{min});
    inputMax->copyin(vector<float>{max});
    auto gpuOp =
        bangGraph->addOp<T>(inputGpu, nullptr, inputMinGpu, inputMaxGpu);
    bangGraph->dataMalloc();
    inputMinGpu->copyin(vector<float>{min});
    inputMaxGpu->copyin(vector<float>{max});
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    inputCpu->printData();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Clip, run) {
    testClip<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
