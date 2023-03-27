#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "cuda/gather.h"
#include "operators/gather.h"

#include "test.h"
namespace infini {
/*
test1:
input = [
      [1, 2],
      [3, 4],
      [5, 6],
  ]
  indices = [
      [0, 1],
      [1, 2],
  ]
  output = [
      [
          [1, 2],
          [3, 4],
      ],
      [
          [3, 4],
          [5, 6],
      ],
  ]
  axis=0
  */

/*
test2
input = [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
  ]
  indices = [
      [0, 2],
  ]
  axis = 1,
  output = [
          [[0, 2]],
          [[3, 5]],
          [[6, 8]],
  ]
*/
/*
test3
input=[[[ 0,  1],
         [ 2,  3],
         [ 4,  5],
         [ 6,  7]],

        [[ 8,  9],
         [10, 11],
         [12, 13],
         [14, 15]]]  //(2,4,2)
indices=[[0],[3],[1]] //(3,1)
axis=1
output=

*/

int gatheredOffset2Offset(int gOffset, GatherMetaData metaData) {
    int offset = 0;
    for (int i = metaData.inNDim - 1, k = metaData.outNDim - 1; i >= 0; --i) {
        int idx = 0;
        if (i == metaData.axis) {
            int idxOffset = 0;
            for (int j = metaData.idxNDim - 1; j >= 0; --j) {
                int p = gOffset % metaData.idxDim[j];
                gOffset = gOffset / metaData.idxDim[j];
                idxOffset += p * metaData.idxStride[j];
            }

            idx = metaData.indexValue[idxOffset];
            k = k - metaData.idxNDim;

        } else {
            idx = gOffset % metaData.outDim[k];
            gOffset = gOffset / metaData.outDim[k];
            --k;
        }
        offset += idx * metaData.inStride[i];
    }
    return offset;
}

TEST(Gather, offsetTrans) {
    {
        GatherMetaData meta;
        int data[] = {0, 1, 1, 2};
        meta.indexValue = data;
        meta.axis = 0;
        meta.inNDim = 2;
        meta.outNDim = 3;
        meta.idxNDim = 2;
        int tmp[] = {2, 2, 2, 0};
        memcpy(&meta.outDim, &tmp, sizeof(tmp));
        int tmp2[] = {2, 2, 0, 0};
        memcpy(&meta.idxDim, &tmp2, sizeof(tmp));
        int tmp3[] = {2, 1, 0, 0};
        memcpy(&meta.idxStride, &tmp3, sizeof(tmp));
        memcpy(&meta.inStride, &tmp3, sizeof(tmp));

        EXPECT_EQ(gatheredOffset2Offset(0, meta), 0);
        EXPECT_EQ(gatheredOffset2Offset(1, meta), 1);
        EXPECT_EQ(gatheredOffset2Offset(2, meta), 2);
        EXPECT_EQ(gatheredOffset2Offset(3, meta), 3);
        EXPECT_EQ(gatheredOffset2Offset(4, meta), 2);
        EXPECT_EQ(gatheredOffset2Offset(5, meta), 3);
        EXPECT_EQ(gatheredOffset2Offset(6, meta), 4);
        EXPECT_EQ(gatheredOffset2Offset(7, meta), 5);
    }
    {
        GatherMetaData meta;
        int data[] = {0, 2};
        meta.indexValue = data;
        meta.axis = 1;
        meta.inNDim = 2;
        meta.outNDim = 3;
        meta.idxNDim = 2;

        int tmp[] = {3, 1, 2, 0};
        memcpy(&meta.outDim, &tmp, sizeof(tmp));
        int tmp2[] = {1, 2, 0, 0};
        memcpy(&meta.idxDim, &tmp2, sizeof(tmp2));
        int tmp3[] = {2, 1, 0, 0};
        memcpy(&meta.idxStride, &tmp3, sizeof(tmp3));
        int tmp4[] = {3, 1, 0, 0};
        memcpy(&meta.inStride, &tmp4, sizeof(tmp4));

        EXPECT_EQ(gatheredOffset2Offset(0, meta), 0);
        EXPECT_EQ(gatheredOffset2Offset(1, meta), 2);
        EXPECT_EQ(gatheredOffset2Offset(2, meta), 3);
        EXPECT_EQ(gatheredOffset2Offset(3, meta), 5);
        EXPECT_EQ(gatheredOffset2Offset(4, meta), 6);
        EXPECT_EQ(gatheredOffset2Offset(5, meta), 8);
    }
    {
        GatherMetaData meta;
        int data[] = {0, 3, 1};
        meta.indexValue = data;
        meta.axis = 1;
        meta.inNDim = 3;
        meta.outNDim = 4;
        meta.idxNDim = 2;

        int tmp[] = {2, 3, 1, 2};
        memcpy(&meta.outDim, &tmp, sizeof(tmp));
        int tmp2[] = {3, 1, 0, 0};
        memcpy(&meta.idxDim, &tmp2, sizeof(tmp2));
        int tmp3[] = {1, 1, 0, 0};
        memcpy(&meta.idxStride, &tmp3, sizeof(tmp3));
        int tmp4[] = {8, 2, 1, 0};
        memcpy(&meta.inStride, &tmp4, sizeof(tmp4));

        EXPECT_EQ(gatheredOffset2Offset(0, meta), 0);
        EXPECT_EQ(gatheredOffset2Offset(1, meta), 1);
        EXPECT_EQ(gatheredOffset2Offset(2, meta), 6);
        EXPECT_EQ(gatheredOffset2Offset(3, meta), 7);
        EXPECT_EQ(gatheredOffset2Offset(4, meta), 2);
        EXPECT_EQ(gatheredOffset2Offset(5, meta), 3);
        EXPECT_EQ(gatheredOffset2Offset(6, meta), 8);
        EXPECT_EQ(gatheredOffset2Offset(7, meta), 9);
        EXPECT_EQ(gatheredOffset2Offset(8, meta), 14);
        EXPECT_EQ(gatheredOffset2Offset(9, meta), 15);
        EXPECT_EQ(gatheredOffset2Offset(10, meta), 10);
        EXPECT_EQ(gatheredOffset2Offset(11, meta), 11);
    }
}

TEST(Gather, Cuda) {
    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph gCpu = make_ref<GraphObj>(runtime);
        auto input = gCpu->addTensor({3, 2}, DataType::Float32);
        auto index = gCpu->addTensor({2, 2}, DataType::UInt32);
        gCpu->dataMalloc();
        input->copyin(vector<float>{1, 2, 3, 4, 5, 6});
        index->copyin(vector<uint32_t>{0, 1, 1, 2});
        auto cudaRuntime = make_ref<CudaRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(cudaRuntime);

        auto op = gCuda->addOp<GatherObj>(
            gCuda->cloneTensor(input), gCuda->cloneTensor(index), nullptr, 0);
        gCuda->dataMalloc();
        cudaRuntime->run(gCuda);

        // cudaPrintTensor(op->getOutput());
        //   copy output from CUDA to CPU
        auto oCpu = gCpu->cloneTensor(op->getOutput());
        EXPECT_TRUE(oCpu->equalData(vector<float>{1, 2, 3, 4, 3, 4, 5, 6}));
    }
    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph gCpu = make_ref<GraphObj>(runtime);
        auto input = gCpu->addTensor({3, 3}, DataType::Float32);
        auto index = gCpu->addTensor({1, 2}, DataType::UInt32);
        gCpu->dataMalloc();
        input->setData(IncrementalGenerator());
        index->copyin(vector<uint32_t>{0, 2});
        auto cudaRuntime = make_ref<CudaRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(cudaRuntime);

        auto op = gCuda->addOp<GatherObj>(
            gCuda->cloneTensor(input), gCuda->cloneTensor(index), nullptr, 1);
        gCuda->dataMalloc();
        cudaRuntime->run(gCuda);

        // cudaPrintTensor(op->getOutput());
        //  copy output from CUDA to CPU
        auto oCpu = gCpu->cloneTensor(op->getOutput());
        EXPECT_TRUE(oCpu->equalData(vector<float>{0, 2, 3, 5, 6, 8}));
    }
    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph gCpu = make_ref<GraphObj>(runtime);
        auto input = gCpu->addTensor({2, 4, 2}, DataType::Float32);
        auto index = gCpu->addTensor({3, 1}, DataType::UInt32);
        gCpu->dataMalloc();
        input->setData(IncrementalGenerator());
        index->copyin(vector<uint32_t>{0, 3, 1});
        auto cudaRuntime = make_ref<CudaRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(cudaRuntime);

        auto op = gCuda->addOp<GatherObj>(
            gCuda->cloneTensor(input), gCuda->cloneTensor(index), nullptr, 1);
        gCuda->dataMalloc();
        cudaRuntime->run(gCuda);

        // cudaPrintTensor(op->getOutput());
        //  copy output from CUDA to CPU
        auto oCpu = gCpu->cloneTensor(op->getOutput());
        EXPECT_TRUE(oCpu->equalData(
            vector<float>{0, 1, 6, 7, 2, 3, 8, 9, 14, 15, 10, 11}));
    }
}

} // namespace infini
