#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/concat.h"

#include "test.h"

namespace infini {
/*
int inputOffset2CatOffset(int linearIndex, int dimBgNo, int dimSize,
                          int concatDim, int outputDimSize[4],
                          int outputStride[4], int nDim) {
    int offset = 0;

    for (int i = nDim - 1; i >= 1; --i) {
        int size = (i == concatDim) ? dimSize : outputDimSize[i];
        int p = linearIndex % size;
        int oP = (i == concatDim) ? (p + dimBgNo) : p;
        linearIndex = (linearIndex - p) / size;

        offset += oP * outputStride[i];
    }

    return offset + linearIndex * outputStride[0];
}

TEST(Concat, OffsetTrans) {
    int dimSize[] = {2, 3};
    int strides[] = {3, 1};
    int catDim = 1, nDim = 2;
    EXPECT_EQ(inputOffset2CatOffset(0, 0, 1, catDim, dimSize, strides, nDim),
              0);
    EXPECT_EQ(inputOffset2CatOffset(1, 0, 1, catDim, dimSize, strides, nDim),
              3);
    EXPECT_EQ(inputOffset2CatOffset(0, 1, 2, catDim, dimSize, strides, nDim),
              1);
    EXPECT_EQ(inputOffset2CatOffset(1, 1, 2, catDim, dimSize, strides, nDim),
              2);
    EXPECT_EQ(inputOffset2CatOffset(2, 1, 2, catDim, dimSize, strides, nDim),
              4);
    EXPECT_EQ(inputOffset2CatOffset(3, 1, 2, catDim, dimSize, strides, nDim),
              5);
}
*/
TEST(Concat, Cuda) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto t1 = gCpu->addTensor({2, 2, 3, 1}, DataType::Float32);
    auto t2 = gCpu->addTensor({2, 2, 1, 1}, DataType::Float32);
    auto t3 = gCpu->addTensor({2, 2, 2, 1}, DataType::Float32);
    gCpu->dataMalloc();
    t1->setData(IncrementalGenerator());
    t2->setData(OneGenerator());
    t3->setData(OneGenerator());

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ConcatObj>(TensorVec{gCuda->cloneTensor(t1),
                                                gCuda->cloneTensor(t2),
                                                gCuda->cloneTensor(t3)},
                                      nullptr, 2);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    // cudaPrintTensor(op->getOutput());
    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput());
    EXPECT_TRUE(
        oCpu->equalData(vector<float>{0, 1, 2, 1, 1, 1, 3, 4,  5,  1, 1, 1,
                                      6, 7, 8, 1, 1, 1, 9, 10, 11, 1, 1, 1}));
}

} // namespace infini
