#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/gather.h"

#include "test.h"

namespace infini {
void testGatherCpu(
    const std::function<void(void *, size_t, DataType)> &generator_input,
    const std::function<void(void *, size_t, DataType)> &generator_indices,
    const int axis,
    const Shape &input_shape,
    const Shape &indices_shape,
    const DataType &dataType
){
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(input_shape, dataType);
    auto indices = g->addTensor(indices_shape, DataType::Int64);
    auto op = g->addOp<GatherObj>(input, indices, nullptr, axis);
    auto output = op->getOutput();
    g->dataMalloc();
    input->setData(generator_input);
    indices->setData(generator_indices);
    runtime->run(g);
    EXPECT_TRUE(1);
}
#ifdef USE_CUDA
void testGatherCuda(
    const std::function<void(void *, size_t, DataType)> &generator,
    const std::function<void(void *, size_t, DataType)> &generator_indices,
    const Shape &input_shape, const Shape &indices_shape,
    const DataType &dataType, const DataType &indicesType,
    const int axis) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuInput = cpuG->addTensor(input_shape, dataType);
    auto cpuIndex = cpuG->addTensor(indices_shape, indicesType);

    auto cpuOp = cpuG->addOp<GatherObj>(cpuInput, cpuIndex, nullptr, axis);
    cpuG->dataMalloc();
    cpuInput->setData(generator);
    cpuIndex->setData(generator_indices);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaInput = cudaG->addTensor(input_shape, dataType);
    auto cudaIndex = cudaG->addTensor(indices_shape, indicesType);
    auto cudaOp =cudaG->addOp<T>(cudaInput, cudaIndex, nullptr, axis);
    cudaG->dataMalloc();
    cudaInput->setData(generator);
    cudaIndex->setData(generator_indices);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif
TEST(Gather, Cpu) {
    testGatherCpu(
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<float*>(data);
            for (size_t i = 0; i < size; ++i){
                ptr[i] = i;
            } 
        }, 
        [](void *data, size_t size, DataType dtype) {}, 
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<int64_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<int64_t>(i % 2); 
                }
            },
        0,               
        {3, 4},          
        {2},             
        DataType::Float32
    );
    testGatherCpu(
         
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<float*>(data);
            for (size_t i = 0; i < size; ++i){
                ptr[i] = i;
            } 
        },
        
        [](void *data, size_t size, DataType dtype) {}, 
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<int64_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<int64_t>(i % 3);
                }
            },
        1,                
        {3, 4, 4},           
        {3},              
        DataType::Float32
    );
    testGatherCpu(
        IncrementalGenerator(),
        [](void *data, size_t size, DataType dtype) {}, 
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<int64_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<int64_t>(i % 2); 
                }
            },
        0,               
        {3, 4},          
        {2},             
        DataType::Float16
    );
}
#ifdef USE_CUDA
TEST(Gather, Cuda) {
    testGatherCuda(
        IncrementalGenerator(),
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<int64_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<int64_t>(i % 2); 
                }
            }, 
        {3, 4},          
        {2},             
        DataType::Float32,
        DataType::Int64,
        0
    );
    testGatherCuda(
        IncrementalGenerator(),
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<int64_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<int64_t>(i % 2); 
                }
            }, 
        {3, 4, 4},       
        {3},             
        DataType::Float32,
        DataType::Int64,
        1
    );
    testGatherCuda(
        IncrementalGenerator(),
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<int64_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<int64_t>(i % 2); 
                }
            }, 
        {3, 4},          
        {2},             
        DataType::Float16,
        DataType::Int64,
        0
    );
}
