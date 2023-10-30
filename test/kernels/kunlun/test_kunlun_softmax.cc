#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/softmax.h"
#include "test.h"
#include <cmath>
namespace infini {

TEST(XDNN_Softmax, run_axis1) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{2, 4}, DataType::Float32, cpuRuntime);

    // KUNLUN XPU
    Graph kunlunGraph = make_ref<GraphObj>(kunlunRuntime);
    auto inputKunlun = kunlunGraph->cloneTensor(inputCpu);
    auto kunlunOp = kunlunGraph->addOp<SoftmaxObj>(inputKunlun, nullptr, 1);
    kunlunGraph->dataMalloc();
    inputKunlun->copyin(vector<float>{0, 1, 2, 3, 10000, 10001, 10002, 10003});
    kunlunRuntime->run(kunlunGraph);
    auto outputKunlun = kunlunOp->getOutput();
    auto outputKunlun2Cpu = outputKunlun->clone(cpuRuntime);
    
    // Check
    EXPECT_TRUE(outputKunlun2Cpu->equalData(
        vector<float>{0.032058604, 0.08714432, 0.23688284, 0.6439143,
                      0.032058604, 0.08714432, 0.23688284, 0.6439143}));
}

TEST(XDNN_Softmax, run_axis0) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{2, 4}, DataType::Float32, cpuRuntime);

    // KUNLUN XPU
    Graph kunlunGraph = make_ref<GraphObj>(kunlunRuntime);
    auto inputKunlun = kunlunGraph->cloneTensor(inputCpu);
    auto kunlunOp = kunlunGraph->addOp<SoftmaxObj>(inputKunlun, nullptr, 0);
    kunlunGraph->dataMalloc();
    inputKunlun->copyin(vector<float>{0, 1, 2, 3, 10000, 10001, 10002, 10003});
    kunlunRuntime->run(kunlunGraph);
    auto outputKunlun = kunlunOp->getOutput();
    auto outputKunlun2Cpu = outputKunlun->clone(cpuRuntime);
    
    // Check
    EXPECT_TRUE(
        outputKunlun2Cpu->equalData(vector<float>{0., 0., 0., 0., 1, 1, 1, 1}));
}

TEST(XDNN_Softmax2, run_axis1) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{2, 2, 2, 2}, DataType::Float32, cpuRuntime);

    // KUNLUN XPU
    Graph kunlunGraph = make_ref<GraphObj>(kunlunRuntime);
    auto inputKunlun = kunlunGraph->cloneTensor(inputCpu);
    auto kunlunOp = kunlunGraph->addOp<SoftmaxObj>(inputKunlun, nullptr, 1);
    kunlunGraph->dataMalloc();
    inputKunlun->setData(IncrementalGenerator());
    kunlunRuntime->run(kunlunGraph);
    auto outputKunlun = kunlunOp->getOutput();
    auto outputKunlun2Cpu = outputKunlun->clone(cpuRuntime);
    
    // Check
    EXPECT_TRUE(outputKunlun2Cpu->equalData(vector<float>{
        0.0179862, 0.0179862, 0.0179862, 0.0179862, 0.9820138, 0.9820138,
        0.9820138, 0.9820138, 0.0179862, 0.0179862, 0.0179862, 0.0179862,
        0.9820138, 0.9820138, 0.9820138, 0.9820138}));
}

TEST(XDNN_Softmax2, run_axis2) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{2, 2, 2, 2}, DataType::Float32, cpuRuntime);

    // KUNLUN XPU
    Graph kunlunGraph = make_ref<GraphObj>(kunlunRuntime);
    auto inputKunlun = kunlunGraph->cloneTensor(inputCpu);
    auto kunlunOp = kunlunGraph->addOp<SoftmaxObj>(inputKunlun, nullptr, 2);
    kunlunGraph->dataMalloc();
    inputKunlun->setData(IncrementalGenerator());
    kunlunRuntime->run(kunlunGraph);
    auto outputKunlun = kunlunOp->getOutput();
    auto outputKunlun2Cpu = outputKunlun->clone(cpuRuntime);
    
    // Check
    EXPECT_TRUE(outputKunlun2Cpu->equalData(vector<float>{
        0.1192029, 0.1192029, 0.8807971, 0.8807971, 0.1192029, 0.1192029,
        0.8807971, 0.8807971, 0.1192029, 0.1192029, 0.8807971, 0.8807971,
        0.1192029, 0.1192029, 0.8807971, 0.8807971}));
}

TEST(XDNN_Softmax2, run_axis3) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{2, 2, 2, 2}, DataType::Float32, cpuRuntime);

    // KUNLUN XPU
    Graph kunlunGraph = make_ref<GraphObj>(kunlunRuntime);
    auto inputKunlun = kunlunGraph->cloneTensor(inputCpu);
    auto kunlunOp = kunlunGraph->addOp<SoftmaxObj>(inputKunlun, nullptr, 3);
    kunlunGraph->dataMalloc();
    inputKunlun->setData(IncrementalGenerator());
    kunlunRuntime->run(kunlunGraph);
    auto outputKunlun = kunlunOp->getOutput();
    auto outputKunlun2Cpu = outputKunlun->clone(cpuRuntime);
    
    // Check
    EXPECT_TRUE(outputKunlun2Cpu->equalData(vector<float>{
        0.2689414, 0.7310586, 0.2689414, 0.7310586, 0.2689414, 0.7310586,
        0.2689414, 0.7310586, 0.2689414, 0.7310586, 0.2689414, 0.7310586,
        0.2689414, 0.7310586, 0.2689414, 0.7310586}));
}
} // namespace infini
