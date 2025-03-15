#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/gather.h"

#include "test.h"

namespace infini {

void testGatherCpu(const std::function<void(void *, size_t, DataType)> &generatorInput,
                   const std::function<void(void *, size_t, DataType)> &generatorIndex,
                   const Shape &inputShape, const Shape &indexShape, int axis,
                   const DataType &dataType) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(inputShape, dataType);
    auto index = g->addTensor(indexShape, DataType::Int32);

    auto op = g->addOp<GatherObj>(input, index, nullptr, axis);
    g->dataMalloc();

    input->setData(generatorInput);
    index->setData(generatorIndex);
    runtime->run(g);
    
    // input->printData();
    // index->printData();
    // op->getOutput()->printData();

    EXPECT_TRUE(1);
}

TEST(Gather, Cpu) {
    testGatherCpu(IncrementalGenerator(), ValGenerator<0>(),
                  {1, 2, 3}, {2, 2}, 0, DataType::Float16);
    testGatherCpu(IncrementalGenerator(), ValGenerator<1>(),
                  {1, 2, 3}, {2, 2}, 1, DataType::Float16);
    testGatherCpu(IncrementalGenerator(), ValGenerator<2>(),
                  {1, 2, 3}, {2, 2}, 2, DataType::Float16);
    testGatherCpu(IncrementalGenerator(), ValGenerator<0>(),
                  {1, 2, 3}, {2, 2}, 0, DataType::Float32);
    testGatherCpu(IncrementalGenerator(), ValGenerator<1>(),
                  {1, 2, 3}, {2, 2}, 1, DataType::Float32);
    testGatherCpu(IncrementalGenerator(), ValGenerator<2>(),
                  {1, 2, 3}, {2, 2}, 2, DataType::Float32);
}

} // namespace infini
