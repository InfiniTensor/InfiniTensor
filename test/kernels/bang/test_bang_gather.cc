#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/runtime.h"
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

TEST(Gather, Mlu) {
    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph gCpu = make_ref<GraphObj>(runtime);
        auto input = gCpu->addTensor({3, 2}, DataType::Float32);
        auto index = gCpu->addTensor({2, 2}, DataType::Int32);
        gCpu->dataMalloc();
        input->copyin(vector<float>{1, 2, 3, 4, 5, 6});
        index->copyin(vector<int>{0, 1, 1, 2});
        auto bangRuntime = make_ref<BangRuntimeObj>();
        Graph gMlu = make_ref<GraphObj>(bangRuntime);

        auto inputMlu = gMlu->cloneTensor(input);
        auto indexMlu = gMlu->cloneTensor(index);
        auto op = gMlu->addOp<GatherObj>(inputMlu, indexMlu, nullptr, 0);
        gMlu->dataMalloc();
        inputMlu->copyin(vector<float>{1, 2, 3, 4, 5, 6});
        indexMlu->copyin(vector<int>{0, 1, 1, 2});
        bangRuntime->run(gMlu);

        //   copy output from MLU to CPU
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
        auto bangRuntime = make_ref<BangRuntimeObj>();
        Graph gMlu = make_ref<GraphObj>(bangRuntime);

        auto inputMlu = gMlu->cloneTensor(input);
        auto indexMlu = gMlu->cloneTensor(index);
        auto op = gMlu->addOp<GatherObj>(inputMlu, indexMlu, nullptr, 1);
        gMlu->dataMalloc();
        inputMlu->setData(IncrementalGenerator());
        indexMlu->copyin(vector<int>{0, 2});
        bangRuntime->run(gMlu);

        //  copy output from MLU to CPU
        auto oCpu = gCpu->cloneTensor(op->getOutput());
        EXPECT_TRUE(oCpu->equalData(vector<float>{0, 2, 3, 5, 6, 8}));
    }
    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph gCpu = make_ref<GraphObj>(runtime);
        auto input = gCpu->addTensor({2, 4, 2}, DataType::Float32);
        auto index = gCpu->addTensor({3, 1}, DataType::Int32);
        gCpu->dataMalloc();
        input->setData(IncrementalGenerator());
        index->copyin(vector<int>{0, 3, 1});
        auto bangRuntime = make_ref<BangRuntimeObj>();
        Graph gMlu = make_ref<GraphObj>(bangRuntime);

        auto inputMlu = gMlu->cloneTensor(input);
        auto indexMlu = gMlu->cloneTensor(index);
        auto op = gMlu->addOp<GatherObj>(inputMlu, indexMlu, nullptr, 1);
        gMlu->dataMalloc();
        inputMlu->setData(IncrementalGenerator());
        indexMlu->copyin(vector<int>{0, 3, 1});
        bangRuntime->run(gMlu);

        //  copy output from MLU to CPU
        auto oCpu = gCpu->cloneTensor(op->getOutput());
        EXPECT_TRUE(oCpu->equalData(
            vector<float>{0, 1, 6, 7, 2, 3, 8, 9, 14, 15, 10, 11}));
    }
}

} // namespace infini
