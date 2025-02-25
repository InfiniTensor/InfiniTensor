#include "core/graph.h"
#include "core/runtime.h"

#include "operators/unary.h"

#include "test.h"

namespace infini {
void testClipCpu(
    const std::function<void(void *, size_t, DataType)> &generator_input,
    const std::function<void(void *, size_t, DataType)> &generator_output,
    const std::optional<float> min,
    const std::optional<float> max,
    const Shape &shape_input,
    const DataType &dataType
){
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape_input, dataType);

    auto op = g->addOp<ClipObj>(input, nullptr, min, max);
    auto output = op->getOutput();
    g->dataMalloc();
    input->setData(generator_input);
    output->setData(generator_output);
    runtime->run(g);
    EXPECT_TRUE(1);
}
TEST(Clip, Cpu) {
    testClipCpu(
        IncrementalGenerator(),
        IncrementalGenerator(),
        2.0,           
        21.0,           
        {3, 3, 4},      
        DataType::Float32);
    testClipCpu(
        [](void *data, size_t size, DataType dtype) { // 输入数据生成器
            auto ptr = static_cast<uint16_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = float_to_fp16(float(i));
            }
        },
        IncrementalGenerator(),
        2.0,           
        21.0,           
        {3, 3, 4},      
        DataType::Float16); 
    testClipCpu(
        [](void *data, size_t size, DataType dtype) { // 输入数据生成器
            auto ptr = static_cast<float *>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = i;
            }
        },
        IncrementalGenerator(),
        2.0,           
        std::nullopt,          
        {3, 3, 4},      
        DataType::Float32);   
}
}; // namespace infini
