#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/GBMML.h"
#include "test.h"

namespace infini {
using ExpectOutput = vector<float>;

TEST(GBMML, ShapeInference) {
    const int bs = 1, seqlen = 10000, w = 1000, featlen = 512, heads = 8, d = 4;
    const int hidden = featlen, hiddenPerHead = hidden / heads;
    auto cpuRuntime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto ACpu = gCpu->addTensor(Shape{bs * heads, seqlen, w * 2 + 1},
                                DataType::Float32);
    auto BCpu = gCpu->addTensor(Shape{bs * heads, seqlen, hiddenPerHead},
                                DataType::Float32);
    gCpu->dataMalloc();
    ACpu->setData(IncrementalGenerator());
    BCpu->setData(IncrementalGenerator());

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto gCuda = make_ref<GraphObj>(cudaRuntime);
    auto ACuda = gCuda->cloneTensor(ACpu);
    auto BCuda = gCuda->cloneTensor(BCpu);
    auto GBMML = gCuda->addOp<GBMMLObj>(ACuda, BCuda, nullptr, d);
    EXPECT_EQ(GBMML->getOutput()->getDims(),
              (Shape{bs * heads, seqlen, hiddenPerHead}));

    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);
}

} // namespace infini