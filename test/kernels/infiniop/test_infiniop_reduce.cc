#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/reduce.h"

#include "test.h"

namespace infini {
template<class T>
void testReduceCpu(
    const std::function<void(void *, size_t, DataType)> &generator_input,
    const std::vector<int> &axes,
    const Shape &shape_input,
    bool keepdims, const DataType &dataType
){
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape_input, dataType);

    auto op = g->addOp<T>(input, nullptr, axes, keepdims);
    auto output = op->getOutput();
    g->dataMalloc();
    input->setData(generator_input);
    runtime->run(g);
    op->getOutput()->print();
    op->getOutput()->printData();
    EXPECT_TRUE(1);
}
#ifdef USE_CUDA
template <class T>
void testReduceCuda(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &input_shape, const DataType &dataType, std::vector<int> axes,
    bool keepdims) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpux = cpuG->addTensor(input_shape, dataType);
    auto cpuOp = cpuG->addOp<T>(cpux, nullptr, axes, keepdims);

    cpuG->dataMalloc();
    cpux->setData(generator);

    cpuRuntime->run(cpuG);
    auto cpuy = cpuOp->getOutput();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudax = cudaG->addTensor(input_shape, dataType);
    auto cudaOp = cudaG->addOp<T>(cudax, nullptr, axes, keepdims);
    cudaG->dataMalloc();
    cudax->setData(generator);

    cudaRuntime->run(cudaG);
    auto cuday = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cuday->equalData(cpuy));
}
#endif
TEST(ReduceMax, Cpu) {
    testReduceCpu<ReduceMaxObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1},            
        {2, 3, 4},       
        false,         
        DataType::Float32);
    
    testReduceCpu<ReduceMaxObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},            
        {3, 3, 4},       
        true,          
        DataType::Float32);    
    testReduceCpu<ReduceMaxObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},            
        {1000, 3, 4},       
        true,         
        DataType::Float32);    
    testReduceCpu<ReduceMaxObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {1, 2},            
        {2, 3, 4},       
        false,          
        DataType::Float32);   
}
TEST(ReduceMin, Cpu) {
    testReduceCpu<ReduceMinObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1},            
        {2, 3, 4},       
        false,          
        DataType::Float32);
    
    testReduceCpu<ReduceMinObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},            
        {3, 3, 4},      
        true,          
        DataType::Float32);    
    testReduceCpu<ReduceMinObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},            
        {1000, 3, 4},       
        true,          
        DataType::Float32);    
    testReduceCpu<ReduceMinObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {1, 2},            
        {2, 3, 4},       
        false,          
        DataType::Float32);   
}
TEST(ReduceMean, Cpu) {
    testReduceCpu<ReduceMeanObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1},            
        {2, 3, 4},       
        false,          
        DataType::Float32);
    
    testReduceCpu<ReduceMeanObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},           
        {3, 3, 4},       
        true,         
        DataType::Float32);    
    testReduceCpu<ReduceMeanObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},            
        {1000, 3, 4},       
        true,          
        DataType::Float32);    
    testReduceCpu<ReduceMeanObj>(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {1, 2},            
        {2, 3, 4},       
        false,          
        DataType::Float32);   
}
#ifdef USE_CUDA
TEST(ReduceMax, Cuda) {
    testReduceCuda<ReduceMaxObj>(
        IncrementalGenerator(),
        {2, 3, 4},       
        DataType::Float32,
        {0, 1},            
        false);         
    testReduceCuda<ReduceMaxObj>(
        IncrementalGenerator(),
        {3, 3, 4},       
        DataType::Float32,
        {0, 1, 2},            
        true);          
    testReduceCuda<ReduceMaxObj>(
        IncrementalGenerator(),
        {1000, 3, 4},       
        DataType::Float32,
        {0, 1, 2},            
        true);         
    testReduceCuda<ReduceMaxObj>(
        IncrementalGenerator(),
        {2, 3, 4},       
        DataType::Float32,
        {1, 2},            
        false);         
}   
TEST(ReduceMin, Cuda) {
    testReduceCuda<ReduceMinObj>(
        IncrementalGenerator(),
        {2, 3, 4},       
        DataType::Float32,
        {0, 1},            
        false);         
    testReduceCuda<ReduceMinObj>(
        IncrementalGenerator(),
        {3, 3, 4},       
        DataType::Float32,
        {0, 1, 2},            
        true);          
    testReduceCuda<ReduceMinObj>(
        IncrementalGenerator(),
        {1000, 3, 4},       
        DataType::Float32,
        {0, 1, 2},            
        true);         
    testReduceCuda<ReduceMinObj>(
        IncrementalGenerator(),
        {2, 3, 4},       
        DataType::Float32,
        {1, 2},            
        false);         
}
TEST(ReduceMean, Cuda) {
    testReduceCuda<ReduceMeanObj>(
        IncrementalGenerator(),
        {2, 3, 4},       
        DataType::Float32,
        {0, 1},            
        false);         
    testReduceCuda<ReduceMeanObj>(
        IncrementalGenerator(),
        {3, 3, 4},       
        DataType::Float32,
        {0, 1, 2},            
        true);          
    testReduceCuda<ReduceMeanObj>(
        IncrementalGenerator(),
        {1000, 3, 4},       
        DataType::Float32,
        {0, 1, 2},            
        true);         
    testReduceCuda<ReduceMeanObj>(
        IncrementalGenerator(),
        {2, 3, 4},       
        DataType::Float32,
        {1, 2},            
        false);         
}
#endif 
}