#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/where.h"

#include "test.h"

namespace infini {

void test_where(const Shape &inputxshape, const vector<float> &inputxdata,
                const Shape &inputyshape, const vector<float> &inputydata,
                const Shape &conditionshape,
                const vector<uint8_t> &conditiondata,
                const vector<float> &ExpectData) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);
    auto condition = gCpu->addTensor(conditionshape, DataType::UInt8);
    auto inputx = gCpu->addTensor(inputxshape, DataType::Float32);
    auto inputy = gCpu->addTensor(inputyshape, DataType::Float32);

    gCpu->dataMalloc();
    condition->copyin(conditionData); //
    inputX->copyin(inputXData);
    inputY->copyin(inputYData); //

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto conditionGpu = gCuda->cloneTensor(condition);
    auto inputXGpu = gCuda->cloneTensor(inputX);
    auto inputYGpu = gCuda->cloneTensor(inputY);

    auto op = gCuda->addOp<WhereObj>(inputXGpu, inputYGpu, conditionGpu,
                                     nullptr); // WhereObj
    gCuda->dataMalloc();
    conditionGpu->copyin(conditionData);
    inputXGpu->copyin(inputXData);
    inputYGpu->copyin(inputYData);
    cudaRuntime->run(gCuda);

    auto oCpu = gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
    oCpu->printData();                              //->printData
    EXPECT_TRUE(oCpu->equalData(ExpectData));
}

TEST(CUDA_Where, run) {
    testWhere(
        Shape{2, 2, 3, 1}, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        Shape{2, 2, 3, 1}, vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        Shape{2, 2, 3, 1}, vector<uint8_t>{0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1},
        vector<float>{0., 1., 2., 0., 0., 0., 6., 7., 0., 9., 10., 11.});

<<<<<<< HEAD
    testWhere(Shape{2, 1, 1, 3}, vector<float>{0, 1, 2, 3, 4, 5}, // inputX
              Shape{1, 2, 1, 1}, vector<float>{1, 1},             // inputY
              Shape{2, 1, 3, 1}, vector<int>{0, 1, 1, 0, 0, 0},   // condition
              vector<float>{1., 1., 1., 0., 1., 2., 0., 1., 2., 1., 1., 1.,
                            0., 1., 2., 0., 1., 2., 1., 1., 1., 1., 1., 1.,
                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});
    testWhere(
        Shape{
            3,
        },
        vector<float>{0, 1, 2},                           // inputX
        Shape{2, 3, 1}, vector<float>{0, 1, 2, 3, 4, 5},  // inputY
        Shape{2, 1, 3, 1}, vector<int>{0, 1, 1, 0, 0, 0}, // condition
        vector<float>{0., 0., 0., 0., 1., 2., 0., 1., 2., 3., 3., 3.,
                      0., 1., 2., 0., 1., 2., 0., 0., 0., 1., 1., 1.,
                      2., 2., 2., 3., 3., 3., 4., 4., 4., 5., 5., 5.});
=======
    test_where(Shape{2, 1, 1, 3},                                  // inputx
               vector<float>{0, 1, 2, 3, 4, 5}, Shape{1, 2, 1, 1}, // inputy
               vector<float>{1, 1}, Shape{2, 1, 3, 1},             // condition
               vector<uint8_t>{0, 1, 1, 0, 0, 0},
               vector<float>{1., 1., 1., 0., 1., 2., 0., 1., 2., 1., 1., 1.,
                             0., 1., 2., 0., 1., 2., 1., 1., 1., 1., 1., 1.,
                             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});
    test_where(
        Shape{
            3,
        },
        vector<float>{0, 1, 2},                               // inputX
        Shape{2, 3, 1}, vector<float>{0, 1, 2, 3, 4, 5},      // inputY
        Shape{2, 1, 3, 1}, vector<uint8_t>{0, 1, 1, 0, 0, 0}, // condition
        vector<float>{0., 0., 0., 0., 1., 2., 0., 1., 2., 3., 3., 3.,
                      0., 1., 2., 0., 1., 2., 0., 0., 0., 1., 1., 1.,
                      2., 2., 2., 3., 3., 3., 4., 4., 4., 5., 5., 5.});
    test_where(
        Shape{
            3,
        },
        vector<float>{0, 1, 2},                          // inputX
        Shape{2, 3, 1}, vector<float>{0, 1, 2, 3, 4, 5}, // inputY
        Shape{2, 1, 3, 1},
        vector<uint8_t>{false, true, true, false, false, false}, // condition
        vector<float>{0., 0., 0., 0., 1., 2., 0., 1., 2., 3., 3., 3.,
                      0., 1., 2., 0., 1., 2., 0., 0., 0., 1., 1., 1.,
                      2., 2., 2., 3., 3., 3., 4., 4., 4., 5., 5., 5.});

>>>>>>> ec3adf6fa73cc6390f09a9bbd23910640d9ed000
} // python output

} // namespace infini
