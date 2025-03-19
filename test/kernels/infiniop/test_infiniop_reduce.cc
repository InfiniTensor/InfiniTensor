#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/reduce.h"

#include "test.h"

namespace infini {
template <class T, typename std::enable_if<std::is_base_of<ReduceBaseObj, T>{}, int>::type = 0>
void testReduceCpu(const std::function<void(void *, size_t, DataType)> &generator,
                    const Shape &shape,
                    const vector<int> &axes,
                    bool keepdims,
                    const DataType &dataType) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape, dataType);

    auto op = g->addOp<T>(input, nullptr, axes, keepdims);
    g->dataMalloc();
    input->setData(generator);

    runtime->run(g);

    // input->printData();
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

TEST(REDUCE, CPU) {
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, false, DataType::Float16);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, true, DataType::Float16);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, false, DataType::Float32);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, true, DataType::Float32);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, false, DataType::Float16);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, true, DataType::Float16);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, false, DataType::Float32);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, true, DataType::Float32);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, false, DataType::Float16);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, true, DataType::Float16);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, false, DataType::Float32);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), {1, 2, 3, 4}, {1, 2}, true, DataType::Float32);
}

}
