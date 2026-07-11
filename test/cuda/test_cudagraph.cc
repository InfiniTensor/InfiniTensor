#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "operators/matmul.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

TEST(TestCudaRuntime, CudaGraph) {
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto input = gCuda->addTensor({2, 2}, DataType::Float32);
    auto weight = gCuda->addTensor({2, 2}, DataType::Float32);
    auto matmul = gCuda->addOp<MatmulObj>(input, weight, nullptr);
    auto relu = gCuda->addOp<ReluObj>(matmul->getOutput(), nullptr);
    gCuda->dataMalloc();

    input->copyin<float>({1, 0, 0, 1});
    weight->copyin<float>({-1, 2, 3, -4});

    cudaRuntime->run(gCuda);
    EXPECT_EQ(relu->getOutput()->copyout<float>(),
              (vector<float>{0, 2, 3, 0}));

    cudaRuntime->runWithCudaGraph(gCuda);
    EXPECT_EQ(relu->getOutput()->copyout<float>(),
              (vector<float>{0, 2, 3, 0}));

    input->copyin<float>({2, 0, 0, 2});
    cudaRuntime->runWithCudaGraph(gCuda);
    EXPECT_EQ(relu->getOutput()->copyout<float>(),
              (vector<float>{0, 4, 6, 0}));
}

} // namespace infini
