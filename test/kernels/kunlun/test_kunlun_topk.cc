#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/topk.h"
#include "test.h"
#include <cmath>

namespace infini {

void kunlunTopKFp32(const Shape &inputShape, const vector<float> &inputData,
                    const Shape &KShape, int axis, int Largest, int sorted,
                    const vector<float> &ValuesData,
                    const vector<int64_t> &IndicesData) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(inputShape, DataType::Float32, cpuRuntime);

    // GPU
    Graph kunlunGraph = make_ref<GraphObj>(kunlunRuntime);
    auto inputGpu = kunlunGraph->cloneTensor(inputCpu);
    auto gpuOp = kunlunGraph->addOp<TopKObj>(inputGpu, std::nullopt, KShape,
                                             axis, Largest, sorted);
    kunlunGraph->dataMalloc();
    inputGpu->copyin(inputData);

    kunlunRuntime->run(kunlunGraph);
    auto ValuesGpu = gpuOp->getOutput(0);
    auto ValuesGpu2Cpu = ValuesGpu->clone(cpuRuntime);
    // ValuesGpu2Cpu->printData();
    //   Check
    EXPECT_TRUE(ValuesGpu2Cpu->equalData(ValuesData));

    auto IndicesGpu = gpuOp->getOutput(1);
    auto IndicesGpu2Cpu = IndicesGpu->clone(cpuRuntime);
    // IndicesGpu2Cpu->printData();
    //     Check

    EXPECT_TRUE(IndicesGpu2Cpu->equalData(IndicesData));
}
TEST(kunlunTopKFp32, run) {
    kunlunTopKFp32(Shape{2, 2, 3, 4},
                   vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
                                 8.,  9.,  10., 11., 12., 13., 14., 15.,
                                 16., 17., 18., 19., 20., 21., 22., 23.,
                                 24., 25., 26., 27., 28., 29., 30., 31.,
                                 32., 33., 34., 35., 36., 37., 38., 39.,
                                 40., 41., 42., 43., 44., 45., 46., 47.},
                   Shape{2}, 3, 1, 1,
                   vector<float>{3.,  2.,  7.,  6.,  11., 10., 15., 14.,
                                 19., 18., 23., 22., 27., 26., 31., 30.,
                                 35., 34., 39., 38., 43., 42., 47., 46.},
                   vector<int64_t>{3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
                                   3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2});
    kunlunTopKFp32(
        Shape{2, 2, 3, 4},
        vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,
                      10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
                      20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
                      30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
                      40., 41., 42., 43., 44., 45., 46., 47.},
        Shape{3}, -1, 1, 1,
        vector<float>{3.,  2.,  1.,  7.,  6.,  5.,  11., 10., 9.,
                      15., 14., 13., 19., 18., 17., 23., 22., 21.,
                      27., 26., 25., 31., 30., 29., 35., 34., 33.,
                      39., 38., 37., 43., 42., 41., 47., 46., 45.},
        vector<int64_t>{3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1,
                        3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1});
}

} // namespace infini
