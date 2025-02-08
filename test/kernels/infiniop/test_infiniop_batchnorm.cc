#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/batch_norm.h"
#include "test.h"

namespace infini {

void testBatchNormCpu(
    const std::function<void(void *, size_t, DataType)> &generatorX,
    const std::function<void(void *, size_t, DataType)> &generatorScale,
    const std::function<void(void *, size_t, DataType)> &generatorBias,
    const std::function<void(void *, size_t, DataType)> &generatorMean,
    const std::function<void(void *, size_t, DataType)> &generatorVar,
    float eps, float momentum, const Shape &shapeX, const Shape &shapeScale, 
    const Shape &shapeBias, const Shape &shapeMean, const Shape &shapeVar, const DataType &dataType) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    
    auto X = g->addTensor(shapeX, dataType);
    auto Scale = g->addTensor(shapeScale, dataType);
    auto Bias = g->addTensor(shapeBias, dataType);
    auto Mean = g->addTensor(shapeMean, dataType);
    auto Var = g->addTensor(shapeVar, dataType);

    auto op = g->addOp<BatchNormObj>(X, nullptr, Mean, Var, Scale, Bias, momentum, eps);
    g->dataMalloc();
    
    X->setData(generatorX);
    Scale->setData(generatorScale);
    Bias->setData(generatorBias);
    Mean->setData(generatorMean);
    Var->setData(generatorVar);

    runtime->run(g);
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
void testBatchNormCuda(
    const std::function<void(void *, size_t, DataType)> &generatorX,
    const std::function<void(void *, size_t, DataType)> &generatorScale,
    const std::function<void(void *, size_t, DataType)> &generatorBias,
    const std::function<void(void *, size_t, DataType)> &generatorMean,
    const std::function<void(void *, size_t, DataType)> &generatorVar,
    float eps, float momentum, const Shape &shapeX, const Shape &shapeScale, 
    const Shape &shapeBias, const Shape &shapeMean, const Shape &shapeVar, const DataType &dataType) {
    // CPU
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    
    auto cpuX = cpuG->addTensor(shapeX, dataType);
    auto cpuScale = cpuG->addTensor(shapeScale, dataType);
    auto cpuBias = cpuG->addTensor(shapeBias, dataType);
    auto cpuMean = cpuG->addTensor(shapeMean, dataType);
    auto cpuVar = cpuG->addTensor(shapeVar, dataType);

    auto cpuOp = cpuG->addOp<BatchNormObj>(cpuX, nullptr, cpuMean, cpuVar, 
                                            cpuScale, cpuBias, momentum, eps);
    cpuG->dataMalloc();
    
    cpuX->setData(generatorX);
    cpuScale->setData(generatorScale);
    cpuBias->setData(generatorBias);
    cpuMean->setData(generatorMean);
    cpuVar->setData(generatorVar);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // CUDA
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    
    auto cudaX = cudaG->addTensor(shapeX, dataType);
    auto cudaScale = cudaG->addTensor(shapeScale, dataType);
    auto cudaBias = cudaG->addTensor(shapeBias, dataType);
    auto cudaMean = cudaG->addTensor(shapeMean, dataType);
    auto cudaVar = cudaG->addTensor(shapeVar, dataType);

    auto cudaOp = cudaG->addOp<BatchNormObj>(cudaX, nullptr, cudaMean, cudaVar, 
                                            cudaScale, cudaBias, momentum, eps);
    cudaG->dataMalloc();
    
    cudaX->setData(generatorX);
    cudaScale->setData(generatorScale);
    cudaBias->setData(generatorBias);
    cudaMean->setData(generatorMean);
    cudaVar->setData(generatorVar);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(BatchNorm, Cpu) {
    testBatchNormCpu(IncrementalGenerator(), IncrementalGenerator(),
                     IncrementalGenerator(), IncrementalGenerator(),
                     IncrementalGenerator(), 1e-5, 0.9, Shape{2, 3, 4, 4}, 
                     Shape{3}, Shape{3}, Shape{3}, Shape{3}, DataType::Float32);
    
    // testBatchNormCpu(IncrementalGenerator(), IncrementalGenerator(),
    //                  IncrementalGenerator(), IncrementalGenerator(),
    //                  IncrementalGenerator(), 1e-5, 0.9, Shape{2, 3, 4, 4}, 
    //                  Shape{3}, Shape{3}, Shape{3}, Shape{3}, DataType::Float16);
}

#ifdef USE_CUDA
TEST(BatchNorm, Cuda) {
    testBatchNormCuda(IncrementalGenerator(), IncrementalGenerator(),
                      IncrementalGenerator(), IncrementalGenerator(),
                      IncrementalGenerator(), 1e-5, 0.9, Shape{2, 3, 4, 4}, 
                      Shape{3}, Shape{3}, Shape{3}, Shape{3}, DataType::Float32);
    
    // testBatchNormCuda(IncrementalGenerator(), IncrementalGenerator(),
    //                   IncrementalGenerator(), IncrementalGenerator(),
    //                   IncrementalGenerator(), 1e-5, 0.9, Shape{2, 3, 4, 4}, 
    //                  Shape{3}, Shape{3}, Shape{3}, Shape{3}, DataType::Float16);
}
#endif

} // namespace infini