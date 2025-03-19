#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/where.h"

#include "test.h"

namespace infini {

void testWhereCpu(
    const std::function<void(void *, size_t, DataType)> &generatorInputX,
    const std::function<void(void *, size_t, DataType)> &generatorInputY,
    const std::function<void(void *, size_t, DataType)> &generatorCon,
    const Shape &shape, const DataType &dataType) {
    
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto inputX = g->addTensor(shape, dataType);
    auto inputY = g->addTensor(shape, dataType);
    auto condition = g->addTensor(shape, DataType::UInt8);

    auto op = g->addOp<WhereObj>(inputX, inputY, condition, nullptr);
    g->dataMalloc();

    inputX->setData(generatorInputX);
    inputY->setData(generatorInputY);

    // condition 要是 u_int8 类型的
    // 由于没有直接生成 u_int8 的类，所以先创建一个 u_int32 类型的数组，然后再转换为 u_int8 类型的
    uint32_t *data_uint32 = new uint32_t[condition->size()];
    generatorCon(data_uint32, condition->size(), DataType::UInt32);
    uint8_t *data_uint8 = new uint8_t[condition->size()];
    for (size_t i = 0; i < condition->size(); ++i) {
        data_uint8[i] = data_uint32[i] > 0 ? 1 : 0;
    }
    // 将 condition 的底层数据设置为 data_uint8
    condition->setDataBlob(make_ref<BlobObj>(runtime, data_uint8));

    runtime->run(g);
    
    // 为了方便对比 condition 的输出，全部按照底层数据依次输出
    // auto res = op->getOutput();
    // if (dataType == DataType::Float16) {
    //     u_int16_t *data_inputX = inputX->getRawDataPtr<u_int16_t *>();
    //     for (size_t i = 0; i < inputX->size(); ++i) {
    //         std::cout << fp16_to_float(data_inputX[i]) << " ";
    //     }
    //     std::cout << std::endl;
    //     u_int16_t *data_inputY = inputY->getRawDataPtr<u_int16_t *>();
    //     for (size_t i = 0; i < inputY->size(); ++i) {
    //         std::cout << fp16_to_float(data_inputY[i]) << " ";
    //     }
    //     std::cout << std::endl;
    //     // uint8_t 类型的数据默认以 char 类型输出，为了方便查看，将其转换为 int 类型输出
    //     uint8_t *data_condition = condition->getRawDataPtr<uint8_t *>();
    //     for (size_t i = 0; i < condition->size(); ++i) {
    //         std::cout << int(data_condition[i]) << " ";
    //     }
    //     std::cout << std::endl;
    //     u_int16_t *data_res = res->getRawDataPtr<u_int16_t *>();
    //     for (size_t j = 0; j < res->size(); ++j) {
    //         std::cout << fp16_to_float(data_res[j]) << " ";
    //     }
    //     std::cout << std::endl << std::endl;
    // } else {
    //     float *data_inputX = inputX->getRawDataPtr<float *>();
    //     for (size_t i = 0; i < inputX->size(); ++i) {
    //         std::cout << data_inputX[i] << " ";
    //     }
    //     std::cout << std::endl;
    //     float *data_inputY = inputY->getRawDataPtr<float *>();
    //     for (size_t i = 0; i < inputY->size(); ++i) {
    //         std::cout << data_inputY[i] << " ";
    //     }
    //     std::cout << std::endl;
    //     // uint8_t 类型的数据默认以 char 类型输出，为了方便查看，将其转换为 int 类型输出
    //     u_int8_t *data_condition = condition->getRawDataPtr<u_int8_t *>();
    //     for (size_t i = 0; i < condition->size(); ++i) {
    //         std::cout << int(data_condition[i]) << " ";
    //     }
    //     std::cout << std::endl;
    //     float *data_res = res->getRawDataPtr<float *>();
    //     for (size_t j = 0; j < res->size(); ++j) {
    //         std::cout << data_res[j] << " ";
    //     }
    //     std::cout << std::endl << std::endl;
    // }
    
    delete [] data_uint32;
    delete [] data_uint8;

    EXPECT_TRUE(1);
}


TEST(Where, Cpu) {
    testWhereCpu(ValGenerator<1>(), ValGenerator<2>(), RandomGenerator(0, 1), 
                 Shape({1, 2, 3, 4}), DataType::Float16);
    testWhereCpu(ValGenerator<1>(), ValGenerator<2>(), RandomGenerator(0, 1),
                 Shape({1, 2, 3, 4}), DataType::Float32);
}

} // namespace infini
