#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
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

TEST(Gather, KUNLUN) {
    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph gCpu = make_ref<GraphObj>(runtime);
        auto input = gCpu->addTensor({3, 2}, DataType::Float32);
        auto index = gCpu->addTensor({2, 2}, DataType::Int32);
        gCpu->dataMalloc();
        input->copyin(vector<float>{1, 2, 3, 4, 5, 6});
        index->copyin(vector<int>{0, 1, 1, 2});
        auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(kunlunRuntime);

        auto inputCuda = gCuda->cloneTensor(input);
        auto indexCuda = gCuda->cloneTensor(index);
        auto op = gCuda->addOp<GatherObj>(inputCuda, indexCuda, nullptr, 0);
        gCuda->dataMalloc();
        inputCuda->copyin(vector<float>{1, 2, 3, 4, 5, 6});
        indexCuda->copyin(vector<int>{0, 1, 1, 2});
        kunlunRuntime->run(gCuda);

        // cudaPrintTensor(op->getOutput());
        //   copy output from CUDA to CPU
        auto oCpu = gCpu->cloneTensor(op->getOutput());
        EXPECT_TRUE(oCpu->equalData(vector<float>{1, 2, 3, 4, 3, 4, 5, 6}));
    }
    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph gCpu = make_ref<GraphObj>(runtime);
        auto input = gCpu->addTensor({3, 3}, DataType::Float32);
        auto index = gCpu->addTensor({1, 2}, DataType::Int32);
        gCpu->dataMalloc();
        input->setData(IncrementalGenerator());
        index->copyin(vector<int>{0, 2});
        auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(kunlunRuntime);

        auto inputCuda = gCuda->cloneTensor(input);
        auto indexCuda = gCuda->cloneTensor(index);
        auto op = gCuda->addOp<GatherObj>(inputCuda, indexCuda, nullptr, 1);
        gCuda->dataMalloc();
        inputCuda->setData(IncrementalGenerator());
        indexCuda->copyin(vector<int>{0, 2});
        kunlunRuntime->run(gCuda);

        // cudaPrintTensor(op->getOutput());
        //  copy output from CUDA to CPU
        auto oCpu = gCpu->cloneTensor(op->getOutput());
        EXPECT_TRUE(oCpu->equalData(vector<float>{0, 2, 3, 5, 6, 8}));
    }
    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph gCpu = make_ref<GraphObj>(runtime);
        auto input = gCpu->addTensor({3, 2}, DataType::Float32);
        auto index = gCpu->addTensor({2, 2}, DataType::Int32);
        gCpu->dataMalloc();
        input->copyin(std::vector<float>{1.0, 1.2, 2.3, 3.4, 4.5, 5.7});
        index->copyin(std::vector<int>{0, 1, 1, 2});
        auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(kunlunRuntime);

        auto inputCuda = gCuda->cloneTensor(input);
        auto indexCuda = gCuda->cloneTensor(index);
        auto op = gCuda->addOp<GatherObj>(inputCuda, indexCuda, nullptr, 0);
        gCuda->dataMalloc();
        inputCuda->copyin(std::vector<float>{1.0, 1.2, 2.3, 3.4, 4.5, 5.7});
        indexCuda->copyin(std::vector<int>{0, 1, 1, 2});
        kunlunRuntime->run(gCuda);

        // cudaPrintTensor(op->getOutput());
        //  copy output from CUDA to CPU
        auto oCpu = gCpu->cloneTensor(op->getOutput());
        EXPECT_TRUE(oCpu->equalData(
            std::vector<float>{1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7}));
    }
}

} // namespace infini