#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "operators/matmul_integer.h"

#include "test.h"

namespace infini {
using ExpectOutput = vector<int32_t>;

TEST(cuBLAS_MatmulInteger, ZeroPoint1) {
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto gCuda = make_ref<GraphObj>(cudaRuntime);
    auto ACuda = gCuda->addTensor({1, 4}, DataType::UInt8);
    auto BCuda = gCuda->addTensor({4, 12}, DataType::UInt8);
    auto AZeroPointCuda = gCuda->addTensor({}, DataType::UInt8);
    auto BZeroPointCuda = gCuda->addTensor({}, DataType::UInt8);
    auto op = gCuda->addOp<MatmulIntegerObj>(ACuda, BCuda, nullptr,
                                             AZeroPointCuda, BZeroPointCuda);

    // allocate CUDA memory
    gCuda->dataMalloc();
    // ACuda->copyin(vector<uint8_t>{11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});
    ACuda->copyin(vector<uint8_t>{11, 7, 3, 10});
    // BCuda->copyin(vector<uint8_t>({1, 4, 2, 5, 3, 6,}));
    BCuda->copyin(vector<uint8_t>(48, 1));
    AZeroPointCuda->copyin(vector<uint8_t>{12});
    BZeroPointCuda->copyin(vector<uint8_t>{0});
    cudaRuntime->run(gCuda);
    auto result = op->getOutput()->clone(NativeCpuRuntimeObj::getInstance());
    // ExpectOutput ans = {
    //     -38, -83, -44, -98, -50, -113, -56, -128,
    // };
    ExpectOutput ans = {-17, -17, -17, -17, -17, -17,
                        -17, -17, -17, -17, -17, -17};
    EXPECT_TRUE(result->equalData(ans));
}

TEST(cuBLAS_MatmulInteger, ZeroPoint2) {
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto gCuda = make_ref<GraphObj>(cudaRuntime);
    auto ACuda = gCuda->addTensor({2, 3, 1, 4}, DataType::UInt8);
    auto BCuda = gCuda->addTensor({2, 3, 4, 12}, DataType::UInt8);
    auto AZeroPointCuda = gCuda->addTensor({2, 3, 1, 1}, DataType::UInt8);
    auto BZeroPointCuda = gCuda->addTensor({2, 3, 1, 12}, DataType::UInt8);
    auto op = gCuda->addOp<MatmulIntegerObj>(ACuda, BCuda, nullptr,
                                             AZeroPointCuda, BZeroPointCuda);

    // allocate CUDA memory
    gCuda->dataMalloc();
    ACuda->copyin(vector<uint8_t>{11, 7, 3, 10, 11, 7, 3, 10, 11, 7, 3, 10,
                                  11, 7, 3, 10, 11, 7, 3, 10, 11, 7, 3, 10});
    BCuda->copyin(vector<uint8_t>(288, 1));
    AZeroPointCuda->copyin(vector<uint8_t>(6, 12));
    BZeroPointCuda->copyin(vector<uint8_t>(72, 0));
    cudaRuntime->run(gCuda);
    auto result = op->getOutput()->clone(NativeCpuRuntimeObj::getInstance());
    ExpectOutput ans = {-17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17,
                        -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17,
                        -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17,
                        -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17,
                        -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17,
                        -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17,
                        -17, -17, -17, -17, -17, -17};
    EXPECT_TRUE(result->equalData(ans));
}

}; // namespace infini
