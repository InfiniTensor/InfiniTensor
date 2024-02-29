#include "core/graph.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/where.h"

#include "test.h"

namespace infini {

void test_where(const Shape &inputXShape, const vector<float> &inputXData,
                const Shape &inputYShape, const vector<float> &inputYData,
                const Shape &conditionShape,
                const vector<int8_t> &conditionData,
                const vector<float> &ExpectData) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);
    auto condition = gCpu->addTensor(conditionShape, DataType::Bool);
    auto inputX = gCpu->addTensor(inputXShape, DataType::Float32);
    auto inputY = gCpu->addTensor(inputYShape, DataType::Float32);

    gCpu->dataMalloc();
    condition->copyin(conditionData); //
    inputX->copyin(inputXData);
    inputY->copyin(inputYData); //

    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(kunlunRuntime);

    auto conditionGpu = gCuda->cloneTensor(condition);
    auto inputXGpu = gCuda->cloneTensor(inputX);
    auto inputYGpu = gCuda->cloneTensor(inputY);

    auto op = gCuda->addOp<WhereObj>(inputXGpu, inputYGpu, conditionGpu,
                                     nullptr); // WhereObj
    gCuda->dataMalloc();
    conditionGpu->copyin(conditionData);
    inputXGpu->copyin(inputXData);
    inputYGpu->copyin(inputYData);
    kunlunRuntime->run(gCuda);

    auto oCpu = gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
    oCpu->printData();                              //->printData
    EXPECT_TRUE(oCpu->equalData(ExpectData));
}

TEST(KUNLUN_Where, run) {
    test_where(
        Shape{2, 2, 3, 1}, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        Shape{2, 2, 3, 1}, vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        Shape{2, 2, 3, 1}, vector<int8_t>{0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1},
        vector<float>{0., 1., 2., 0., 0., 0., 6., 7., 0., 9., 10., 11.});

    test_where(Shape{2, 2, 1, 3}, // inputx
               vector<float>{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5},
               Shape{2, 2, 1, 3}, // inputy
               vector<float>{1, 1, 3, 2, 5, 1, 5, 2, 3, 5, 6, 7},
               Shape{2, 2, 1, 3}, // condition
               vector<int8_t>{0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0},
               vector<float>{1, 1, 2, 2, 5, 1, 0, 2, 2, 3, 6, 7});

    test_where(Shape{2, 2, 1, 3},
               vector<float>{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}, // inputX
               Shape{2, 2, 1, 3},
               vector<float>{1, 1, 3, 2, 5, 1, 5, 2, 3, 5, 6, 7},   // inputY
               Shape{2, 1, 1, 3}, vector<int8_t>{1, 1, 0, 1, 1, 1}, // condition
               vector<float>{0, 1, 3, 3, 4, 1, 0, 1, 2, 3, 4, 5});  // result

    test_where(Shape{2, 2, 1, 3},
               vector<float>{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}, // inputX
               Shape{2, 2, 1, 3},
               vector<float>{1, 1, 3, 2, 5, 1, 5, 2, 3, 5, 6, 7}, // inputY
               Shape{2, 1, 1, 3},
               vector<int8_t>{1, 1, 0, 1, 1,
                              1}, // condition               } // python output
               vector<float>{0, 1, 3, 3, 4, 1, 0, 1, 2, 3, 4, 5}); // result
}
} // namespace infini
