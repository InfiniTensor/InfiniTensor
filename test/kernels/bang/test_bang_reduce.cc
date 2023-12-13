#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "bang/bang_runtime.h"
#include "operators/reduce.h"

#include "test.h"

namespace infini {

template <typename ReduceObjT>
void test_reduce(const Shape &shape, const vector<float> &data,
                 const optional<const vector<int>> &axis, bool keepDims,
                 const vector<float> &ExpectData) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor icpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // Build BANG graph
    Graph g = make_ref<GraphObj>(bangRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<ReduceObjT>(i, nullptr, axis, keepDims);

    // allocate BANG memory
    g->dataMalloc();
    i->copyin(data);

    // Execute on BANG
    bangRuntime->run(g);

    // clone BANG output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    //  check results on CPU
    EXPECT_TRUE(ocpu->equalData(ExpectData));
}

TEST(BANG_ReduceMean, run) {
    test_reduce<ReduceMeanObj>(
        Shape{3, 2, 2}, vector<float>{5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2},
        std::nullopt, true, vector<float>{18.25});
    test_reduce<ReduceMeanObj>(
        Shape{1, 3, 2, 2, 1},
        vector<float>{5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2}, std::nullopt,
        false, vector<float>{18.25});

    test_reduce<ReduceMeanObj>(
        Shape{2, 3, 2, 2},
        vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        vector<int>{1, 2}, false, vector<float>{5, 6, 17, 18});
    test_reduce<ReduceMeanObj>(
        Shape{2, 3, 2, 2, 1},
        vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        vector<int>{1, 2}, true, vector<float>{5, 6, 17, 18});
}

TEST(BANG_ReduceSum, run) {
    test_reduce<ReduceSumObj>(Shape{3, 2, 2},
                              vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                              std::nullopt, true, vector<float>{12});
    test_reduce<ReduceSumObj>(Shape{1, 3, 2, 2, 1},
                              vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                              std::nullopt, false, vector<float>{12});

    test_reduce<ReduceSumObj>(
        Shape{2, 3, 2, 2},
        vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        vector<int>{1, 2}, false, vector<float>{30, 36, 102, 108});
    test_reduce<ReduceSumObj>(
        Shape{2, 3, 2, 2, 1},
        vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        vector<int>{1, 2}, true, vector<float>{30, 36, 102, 108});
}

} // namespace infini
