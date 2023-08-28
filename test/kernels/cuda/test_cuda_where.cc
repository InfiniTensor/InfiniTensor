#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/where.h"

#include "test.h"

namespace infini {

void test_where(const Shape &inputxshape, const vector<float> &inputxdata,
                const Shape &inputyshape, const vector<float> &inputydata,
                const Shape &conditionshape, const vector<int> &conditiondata,
                const vector<float> &ExpectData) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);
    auto condition = gCpu->addTensor(conditionshape, DataType::Int32);
    auto inputx = gCpu->addTensor(inputxshape, DataType::Float32);
    auto inputy = gCpu->addTensor(inputyshape, DataType::Float32);

    gCpu->dataMalloc();
    condition->copyin(conditiondata); //
    inputx->copyin(inputxdata);
    inputy->copyin(inputydata); //

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto conditionGpu = gCuda->cloneTensor(condition);
    auto inputxGpu = gCuda->cloneTensor(inputx);
    auto inputyGpu = gCuda->cloneTensor(inputy);

    auto op = gCuda->addOp<WhereObj>(inputxGpu, inputyGpu, conditionGpu,
                                     nullptr); // WhereObj
    gCuda->dataMalloc();
    conditionGpu->copyin(conditiondata);
    inputxGpu->copyin(inputxdata);
    inputyGpu->copyin(inputydata);
    cudaRuntime->run(gCuda);

    auto oCpu = gCpu->cloneTensor(op->getOutput()); // move data from gpu to cpu
    oCpu->printData();                              //->printData
    EXPECT_TRUE(oCpu->equalData(ExpectData));
}

TEST(CUDA_Where, run) {
    test_where(
        Shape{2, 2, 3, 1}, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        Shape{2, 2, 3, 1}, vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        Shape{2, 2, 3, 1}, vector<int>{0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1},
        vector<float>{0., 1., 2., 0., 0., 0., 6., 7., 0., 9., 10., 11.});

    test_where(Shape{2, 1, 1, 3},                                  // inputx
               vector<float>{0, 1, 2, 3, 4, 5}, Shape{1, 2, 1, 1}, // inputy
               vector<float>{1, 1}, Shape{2, 1, 3, 1},             // condition
               vector<int>{0, 1, 1, 0, 0, 0},
               vector<float>{1., 1., 1., 0., 1., 2., 0., 1., 2., 1., 1., 1.,
                             0., 1., 2., 0., 1., 2., 1., 1., 1., 1., 1., 1.,
                             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});

} // python output

} // namespace infini
