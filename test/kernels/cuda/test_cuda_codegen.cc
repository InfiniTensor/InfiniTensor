#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/element_wise.h"
#include "operators/softmax.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

TEST(CUDA_Codegen, run) {
    // Benchmark Settings
    int warmupRounds = 100;
    int timingRounds = 200;
    auto INPUT_SHAPE = Shape{224, 768};
    auto dtype = DataType::Float32;

    // Get data size
    size_t size = 1;
    for (auto dim : INPUT_SHAPE) {
        size *= dim;
    }
    size_t sizeInBytes = size * sizeof(dtype);

    // Create runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build cpu graph
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto aCpu = gCpu->addTensor(INPUT_SHAPE, dtype);
    auto bCpu = gCpu->addTensor(INPUT_SHAPE, dtype);
    auto cCpu = gCpu->addTensor(INPUT_SHAPE, dtype);
    auto dCpu = gCpu->addTensor(INPUT_SHAPE, dtype);

    // Build input data on CPU
    gCpu->dataMalloc();
    aCpu->setData(IncrementalGenerator());
    bCpu->setData(IncrementalGenerator());
    cCpu->setData(OneGenerator());
    dCpu->setData(IncrementalGenerator());

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto a = g->cloneTensor(aCpu);
    auto b = g->cloneTensor(bCpu);
    auto c = g->cloneTensor(cCpu);
    auto d = g->cloneTensor(dCpu);
    auto add = g->addOp<AddObj>(a, b, nullptr);
    auto temp1 = add->getOutput();
    auto sub = g->addOp<SubObj>(temp1, c, nullptr);
    auto temp2 = sub->getOutput();
    auto sqrt = g->addOp<SqrtObj>(temp2, nullptr);
    auto temp3 = sqrt->getOutput();
    auto mul = g->addOp<MulObj>(d, temp3, nullptr);
    auto temp4 = mul->getOutput();
    auto softmax = g->addOp<SigmoidObj>(temp4, nullptr, 0);

    // allocate CUDA memory
    g->dataMalloc();

    double time_op = 0.0;

    // Execute on CUDA and time
    time_op +=
        timeit([&]() { cudaRuntime->run(g); }, [&]() { cudaRuntime->sync(); },
               warmupRounds, timingRounds);

    printf("Operator - Softmax:\n");
    printf("Input shape: (%d, %d)\n", INPUT_SHAPE[0], INPUT_SHAPE[1]);
    printf("Input size: %ld, dtype: %s, size in bytes: %ld\n", size,
           dtype.toString().c_str(), sizeInBytes);
    printf("Time in total: %.6lf ms\n", time_op);
}
} // namespace infini