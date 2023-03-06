#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testClip(const std::function<void(void *, size_t, DataType)> &generator,
              const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph Graph = make_ref<GraphObj>(cpuRuntime);
    float min = 1.0;
    float max = 4.0;
    auto Op = Graph->addOp<T>(inputCpu, nullptr, min, max);
    Graph->dataMalloc();
    cpuRuntime->run(Graph);
    auto output = Op->getOutput();
    inputCpu->printData();
    output->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Clip, run) {
    testClip<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
