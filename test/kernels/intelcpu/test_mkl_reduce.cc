#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/reduce_mean.h"

#include "test.h"

namespace infini {

void test_reducemean(const Shape &shape, const vector<float> &data,
                     const optional<const vector<int>> &axis, bool keepDims,
                     const vector<float> &ExpectData) {
    Runtime runtime = MklRuntimeObj::getInstance();

    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor(shape, DataType::Float32);
    auto op = g->addOp<ReduceMeanObj>(i, nullptr, axis, keepDims);

    g->dataMalloc();
    i->copyin(data);

    // Execute
    runtime->run(g);

    auto o = op->getOutput();

    //  check results
    EXPECT_TRUE(o->equalData(ExpectData));
}

TEST(MKL_ReduceMean, run) {
    test_reducemean(Shape{3, 2, 2},
                    vector<float>{5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2},
                    std::nullopt, true, vector<float>{18.25});
    test_reducemean(Shape{1, 3, 2, 2, 1},
                    vector<float>{5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2},
                    std::nullopt, false, vector<float>{18.25});

    test_reducemean(Shape{2, 3, 2, 2},
                    vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                  8,  9,  10, 11, 12, 13, 14, 15,
                                  16, 17, 18, 19, 20, 21, 22, 23},
                    vector<int>{1, 2}, false, vector<float>{5, 6, 17, 18});
    test_reducemean(Shape{2, 3, 2, 2, 1},
                    vector<float>{0,  1,  2,  3,  4,  5,  6,  7,
                                  8,  9,  10, 11, 12, 13, 14, 15,
                                  16, 17, 18, 19, 20, 21, 22, 23},
                    vector<int>{1, 2}, true, vector<float>{5, 6, 17, 18});
}

} // namespace infini
