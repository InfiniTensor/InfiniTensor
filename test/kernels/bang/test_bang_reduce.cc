#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/reduce_mean.h"

#include "test.h"

namespace infini {

template <class T>
void testReduce(const std::function<void(void *, size_t, DataType)> &generatorA,
              const Shape &shapeA) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu1 =
        make_ref<TensorObj>(shapeA, DataType::Float32, cpuRuntime);
    inputCpu1->dataMalloc();
    inputCpu1->setData(generatorA);

    // MLU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputMlu1 = bangGraph->cloneTensor(inputCpu1);
    auto mluOp =
        bangGraph->addOp<T>(inputMlu1, nullptr, vector<int>{3}, false);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputMlu = mluOp->getOutput();
    auto outputMlu2Cpu = outputMlu->clone(cpuRuntime);
    outputMlu2Cpu->print();
    EXPECT_TRUE(true);
}

TEST(cnnl_Reduce, run) {
    testReduce<ReduceMeanObj>(IncrementalGenerator(), Shape{1, 3, 224, 224});
}

} // namespace infini
