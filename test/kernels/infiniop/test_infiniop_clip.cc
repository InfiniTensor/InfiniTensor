#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/unary.h"

#include "test.h"

namespace infini {
void testClipCpu(
    const std::function<void(void *, size_t, DataType)> &generator_input,
    const std::optional<float> min,
    const std::optional<float> max,
    const Shape &input_shpe,
    const DataType &dataType
){
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(input_shpe, dataType);

    auto op = g->addOp<ClipObj>(input, nullptr, min, max);
    auto output = op->getOutput();
    g->dataMalloc();
    input->setData(generator_input);
    runtime->run(g);
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
void testClipCuda(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &x_shape, const DataType &dataType,
    std::optional<float> min
    std::optional<float> max) {
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpux = cpuG->addTensor(x_shape, dataType);

    auto cpuOp = cpuG->addOp<T>(cpux, nullptr, min, max);
    cpuG->dataMalloc();
    cpux->setData(generator);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudax = cudaG->addTensor(x_shape, dataType);
    auto cudaOp = cudaG->addOp<ClipObj>(cudax, nullptr, min, max);
    cudaG->dataMalloc();
    cudax->setData(generator);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Clip, Cpu) {
    testClipCpu(
        IncrementalGenerator(),
        2.0,           
        21.0,           
        {3, 3, 4},      
        DataType::Float32);
    testClipCpu(
        IncrementalGenerator(),
        2.0,           
        21.0,           
        {3, 3, 4},      
        DataType::Float16); 
    testClipCpu(
        IncrementalGenerator(),
        2.0,           
        std::nullopt,          
        {3, 3, 4},      
        DataType::Float32);   
}
#ifdef USE_CUDA
TEST(Clip, Cuda){
    testClipCuda(
        IncrementalGenerator(),
        {3, 3, 4},       
        DataType::Float32,
        2.0,           
        21.0);          
    testClipCuda(
        IncrementalGenerator(),
        {3, 3, 4},       
        DataType::Float32,
        2.0,           
        21.0);          
    testClipCuda(
        IncrementalGenerator(),
        {3, 3, 4},       
        DataType::Float32,
        2.0,           
        21.0);          
    testClipCuda(
        IncrementalGenerator(),
        {3, 3, 4},       
        DataType::Float32,
        2.0,           
        21.0);          
}
}; // namespace infini
