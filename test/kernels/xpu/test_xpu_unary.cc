#include "xpu/xpu_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testUnary(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<XPURuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // GPU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputGpu = xpuGraph->cloneTensor(inputCpu);
    auto gpuOp = xpuGraph->addOp<T>(inputGpu, nullptr);
    xpuGraph->dataMalloc();
    inputGpu->setData(generator);
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu, 1e-6));
}

void testClip(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<XPURuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    float min = 1.0;
    float max = 5.0;

    // GPU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputGpu = xpuGraph->cloneTensor(inputCpu);
    auto gpuOp = xpuGraph->addOp<ClipObj>(inputGpu, nullptr, min, max);
    xpuGraph->dataMalloc();
    inputGpu->setData(generator);
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<ClipObj>(inputCpu, nullptr, min, max);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

void testCast(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<XPURuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // GPU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputGpu = xpuGraph->cloneTensor(inputCpu);
    auto gpuOp = xpuGraph->addOp<CastObj>(inputGpu, nullptr, CastType::Float2Int32);
    xpuGraph->dataMalloc();
    inputGpu->setData(generator);
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<CastObj>(inputCpu, nullptr, CastType::Float2Int32);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

template <LogObj::LogType T>
void testLog(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<XPURuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // GPU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputGpu = xpuGraph->cloneTensor(inputCpu);
    auto gpuOp = xpuGraph->addOp<LogObj>(inputGpu, nullptr, T);
    xpuGraph->dataMalloc();
    inputGpu->setData(generator);
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<LogObj>(inputCpu, nullptr, T);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

template <class T>
void testTrigon(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<XPURuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // GPU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputGpu = xpuGraph->cloneTensor(inputCpu);
    auto gpuOp = xpuGraph->addOp<T>(inputGpu, nullptr);
    xpuGraph->dataMalloc();
    inputGpu->setData(generator);
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu, 1e-3));
}

TEST(xdnn_Unary, run) {
    testUnary<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<TanhObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<AbsObj>(ValGenerator<-1>(), Shape{1, 2, 2, 3});
    testUnary<ATanObj>(OneGenerator(), Shape{1, 2, 2, 3});
    testLog<LogObj::Log10>(ValGenerator<2>(), Shape{1, 2, 2, 3});
    testLog<LogObj::Log2>(ValGenerator<2>(), Shape{1, 2, 2, 3});
    testLog<LogObj::LogE>(ValGenerator<2>(), Shape{1, 2, 2, 3});
    testTrigon<CosObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testTrigon<SinObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testTrigon<TanObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testTrigon<SinHObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testTrigon<CosHObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<ErfObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testTrigon<ACosObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testTrigon<ACosHObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testTrigon<ASinObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testTrigon<ASinHObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testTrigon<ATanHObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
