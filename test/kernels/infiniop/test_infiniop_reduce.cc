#include "core/graph.h"
#include "core/runtime.h"

#include "operators/reduce.h"

#include "test.h"

namespace infini {
void testReduceMaxCpu(
    const std::function<void(void *, size_t, DataType)> &generator_input,
    const std::function<void(void *, size_t, DataType)> &generator_output,
    const std::vector<int> &axes,
    const Shape &shape_input,
    bool keepdims, const DataType &dataType
){
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape_input, dataType);

    auto op = g->addOp<ReduceMaxObj>(input, nullptr, axes, keepdims);
    auto output = op->getOutput();
    g->dataMalloc();
    input->setData(generator_input);
    output->setData(generator_output);
    runtime->run(g);
    op->getOutput()->print();
    op->getOutput()->printData();
    EXPECT_TRUE(1);
}
void testReduceMeanCpu(
    const std::function<void(void *, size_t, DataType)> &generator_input,
    const std::function<void(void *, size_t, DataType)> &generator_output,
    const std::vector<int> &axes,
    const Shape &shape_input,
    bool keepdims, const DataType &dataType
){
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape_input, dataType);

    auto op = g->addOp<ReduceMeanObj>(input, nullptr, axes, keepdims);
    auto output = op->getOutput();
    g->dataMalloc();
    input->setData(generator_input);
    output->setData(generator_output);
    runtime->run(g);
    op->getOutput()->print();
    op->getOutput()->printData();
    EXPECT_TRUE(1);
}
void testReduceMinCpu(
    const std::function<void(void *, size_t, DataType)> &generator_input,
    const std::function<void(void *, size_t, DataType)> &generator_output,
    const std::vector<int> &axes,
    const Shape &shape_input,
    bool keepdims, const DataType &dataType
){
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape_input, dataType);

    auto op = g->addOp<ReduceMinObj>(input, nullptr, axes, keepdims);
    auto output = op->getOutput();
    g->dataMalloc();
    input->setData(generator_input);
    output->setData(generator_output);
    runtime->run(g);
    op->getOutput()->print();
    op->getOutput()->printData();
    EXPECT_TRUE(1);
}
TEST(ReduceMax, Cpu) {
    testReduceMaxCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1},            
        {2, 3, 4},       
        false,         
        DataType::Float32);
    
    testReduceMaxCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},            
        {3, 3, 4},       
        true,          
        DataType::Float32);    
    testReduceMaxCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},            
        {1000, 3, 4},       
        true,         
        DataType::Float32);    
    testReduceMaxCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {1, 2},            
        {2, 3, 4},       
        false,          
        DataType::Float32);   
}
TEST(ReduceMin, Cpu) {
    testReduceMinCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1},            
        {2, 3, 4},       
        false,          
        DataType::Float32);
    
    testReduceMinCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},            
        {3, 3, 4},      
        true,          
        DataType::Float32);    
    testReduceMinCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},            
        {1000, 3, 4},       
        true,          
        DataType::Float32);    
    testReduceMinCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {1, 2},            
        {2, 3, 4},       
        false,          
        DataType::Float32);   
}
TEST(ReduceMean, Cpu) {
    testReduceMeanCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1},            
        {2, 3, 4},       
        false,          
        DataType::Float32);
    
    testReduceMeanCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},           
        {3, 3, 4},       
        true,         
        DataType::Float32);    
    testReduceMeanCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {0, 1, 2},            
        {1000, 3, 4},       
        true,          
        DataType::Float32);    
    testReduceMeanCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        {1, 2},            
        {2, 3, 4},       
        false,          
        DataType::Float32);   
}
    
}