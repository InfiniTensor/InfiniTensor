#include "ascend/ascend_runtime.h"
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
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto npuOp = npuGraph->addOp<T>(inputNpu, nullptr);
    npuGraph->dataMalloc();
    inputNpu->setData(generator);
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputNpu2Cpu, 1e-3));
}

template <class T>
void testUnarywithExpectData(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const vector<float> &ExpectData) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    auto ExpectOutput = gCpu->addTensor(shape, DataType::Float32);
    gCpu->dataMalloc();
    ExpectOutput->copyin(ExpectData);

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto npuOp = npuGraph->addOp<T>(inputNpu, nullptr);
    npuGraph->dataMalloc();
    inputNpu->setData(generator);
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);
    // CPU
    // Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    // auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr);
    // cpuGraph->addTensor(inputCpu);
    // cpuGraph->dataMalloc();
    // inputCpu->setData(generator);
    // cpuRuntime->run(cpuGraph);
    // auto outputCpu = cpuOp->getOutput();
    // Check
    outputNpu2Cpu->printData();
    EXPECT_TRUE(outputNpu2Cpu->equalData(ExpectOutput, 1e-4));
}

void testLeakyRelu(const Shape &shape, const vector<float> &inputData,
                   const vector<float> &ExpectData, float alpha) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor(shape, DataType::Float32);

    gCpu->dataMalloc();

    input->copyin(inputData);
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    // NPU

    auto inputNpu = npuGraph->cloneTensor(input);
    auto npuOp = npuGraph->addOp<LeakyReluObj>(inputNpu, nullptr, alpha);
    npuGraph->dataMalloc();
    inputNpu->copyin(inputData);
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

    // Check
    EXPECT_TRUE(outputNpu2Cpu->equalData(ExpectData));
}

void testHardSigmoid(const Shape &shape, const vector<float> &inputData,
                   const vector<float> &ExpectData) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor(shape, DataType::Float32);

    gCpu->dataMalloc();

    input->copyin(inputData);
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    // NPU

    auto inputNpu = npuGraph->cloneTensor(input);
    auto npuOp = npuGraph->addOp<HardSigmoidObj>(inputNpu, nullptr);
    npuGraph->dataMalloc();
    inputNpu->copyin(inputData);
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

    // Check
    EXPECT_TRUE(outputNpu2Cpu->equalData(ExpectData));
}

TEST(ascend_Unary, run) {
    aclInit(nullptr);
    testHardSigmoid(Shape{1,2,2,3}, 
                    vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                    vector<float>{0.5, 0.666667, 0.833333, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    testLeakyRelu(Shape{1, 2, 2, 3},
                  vector<float>{-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6},
                  vector<float>{-0.0600, -0.0500, -0.0400, -0.0300, -0.0200,
                                -0.0100, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000,
                                6.0000},
                  0.01);
    testUnary<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<AbsObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<HardSwishObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<TanhObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SinObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<GeluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<CosObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<ACosObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<ATanObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<NegObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SqrtObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});

    testUnarywithExpectData<CeilObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                                     vector<float>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                                   6.0, 7.0, 8.0, 9.0, 10.0,
                                                   11.0});
    testUnarywithExpectData<FloorObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                                      vector<float>{0.0, 1.0, 2.0, 3.0, 4.0,
                                                    5.0, 6.0, 7.0, 8.0, 9.0,
                                                    10.0, 11.0});
    testUnarywithExpectData<ExpObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        vector<float>{1.0, 2.71828, 7.38906, 20.0855, 54.5981, 148.413, 403.429,
                      1096.63, 2980.96, 8103.08, 22026.5, 59874.1});
    testUnarywithExpectData<ReciprocalObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        vector<float>{std::numeric_limits<float>::infinity(), 1, 0.5, 0.333333,
                      0.25, 0.2, 0.166667, 0.142857, 0.125, 0.111111, 0.1,
                      0.0909091});
    testUnarywithExpectData<RoundObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                                      vector<float>{0.0, 1.0, 2.0, 3.0, 4.0,
                                                    5.0, 6.0, 7.0, 8.0, 9.0,
                                                    10.0, 11.0});
    aclFinalize();
}

} // namespace infini
