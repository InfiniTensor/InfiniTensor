#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/softmax.h"
#include "test.h"
#include <cmath>
double eps = 3e-3;
namespace infini {
void cnnlSoftmaxFp32(const Shape &inputShape, const vector<float> &inputData,
                     int axis, const vector<float> &expectData) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(inputShape, DataType::Float32, cpuRuntime);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);
    auto gpuOp = bangGraph->addOp<SoftmaxObj>(inputGpu, nullptr, axis);
    bangGraph->dataMalloc();
    inputGpu->copyin(inputData);
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // Check
    EXPECT_TRUE(outputGpu2Cpu->equalData(expectData));
}
TEST(cnnlSoftmaxFp32, run) {
    cnnlSoftmaxFp32(
        Shape{2, 3, 2, 2},
        vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
                      8.,  9.,  10., 11., 12., 13., 14., 15.,
                      16., 17., 18., 19., 20., 21., 22., 23.},
        0, vector<float>{6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                         6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                         6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                         6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                         9.99993801e-01, 9.99993801e-01, 9.99993801e-01,
                         9.99993801e-01, 9.99993801e-01, 9.99993801e-01,
                         9.99993801e-01, 9.99993801e-01, 9.99993801e-01,
                         9.99993801e-01, 9.99993801e-01, 9.99993801e-01});
    cnnlSoftmaxFp32(Shape{2, 2, 2, 2},
                    vector<float>{
                        0.,
                        1.,
                        2.,
                        3.,
                        4.,
                        5.,
                        6.,
                        7.,
                        8.,
                        9.,
                        10.,
                        11.,
                        12.,
                        13.,
                        14.,
                        15.,
                    },
                    1,
                    vector<float>{0.0179862, 0.0179862, 0.0179862, 0.0179862,
                                  0.9820138, 0.9820138, 0.9820138, 0.9820138,
                                  0.0179862, 0.0179862, 0.0179862, 0.0179862,
                                  0.9820138, 0.9820138, 0.9820138, 0.9820138});
    cnnlSoftmaxFp32(Shape{2, 2, 2, 2},
                    vector<float>{
                        0.,
                        1.,
                        2.,
                        3.,
                        4.,
                        5.,
                        6.,
                        7.,
                        8.,
                        9.,
                        10.,
                        11.,
                        12.,
                        13.,
                        14.,
                        15.,
                    },
                    2,
                    vector<float>{0.1192029, 0.1192029, 0.8807971, 0.8807971,
                                  0.1192029, 0.1192029, 0.8807971, 0.8807971,
                                  0.1192029, 0.1192029, 0.8807971, 0.8807971,
                                  0.1192029, 0.1192029, 0.8807971, 0.8807971});
    cnnlSoftmaxFp32(Shape{2, 2, 2, 2},
                    vector<float>{
                        0.,
                        1.,
                        2.,
                        3.,
                        4.,
                        5.,
                        6.,
                        7.,
                        8.,
                        9.,
                        10.,
                        11.,
                        12.,
                        13.,
                        14.,
                        15.,
                    },
                    3,
                    vector<float>{0.2689414, 0.7310586, 0.2689414, 0.7310586,
                                  0.2689414, 0.7310586, 0.2689414, 0.7310586,
                                  0.2689414, 0.7310586, 0.2689414, 0.7310586,
                                  0.2689414, 0.7310586, 0.2689414, 0.7310586});
    cnnlSoftmaxFp32(Shape{2, 4},
                    vector<float>{0., 1., 2., 3., 1000, 1001, 1002, 1003}, 0,
                    vector<float>{0., 0., 0., 0., 1, 1, 1, 1});
    cnnlSoftmaxFp32(
        Shape{2, 4}, vector<float>{0., 1., 2., 3., 1000, 1001, 1002, 1003}, 1,
        vector<float>{0.032058604, 0.08714432, 0.23688284, 0.6439143,
                      0.032058604, 0.08714432, 0.23688284, 0.6439143});
}
void bangSoftmaxFp32(const Shape &inputShape, const vector<float> &inputData,
                     int axis, const vector<float> &expectData) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(inputShape, DataType::Float32, cpuRuntime);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);
    auto gpuOp = bangGraph->addOp<BangSoftmaxObj>(inputGpu, nullptr, axis);
    bangGraph->dataMalloc();
    inputGpu->copyin(inputData);
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // outputGpu2Cpu->printData();
    //  Check
    EXPECT_TRUE(outputGpu2Cpu->equalData(expectData, eps));
}
TEST(bangSoftmaxFp32, run) {
    bangSoftmaxFp32(
        Shape{2, 3, 2, 2},
        vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
                      8.,  9.,  10., 11., 12., 13., 14., 15.,
                      16., 17., 18., 19., 20., 21., 22., 23.},
        0, vector<float>{6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                         6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                         6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                         6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                         9.99993801e-01, 9.99993801e-01, 9.99993801e-01,
                         9.99993801e-01, 9.99993801e-01, 9.99993801e-01,
                         9.99993801e-01, 9.99993801e-01, 9.99993801e-01,
                         9.99993801e-01, 9.99993801e-01, 9.99993801e-01});
    bangSoftmaxFp32(Shape{2, 2, 2, 2},
                    vector<float>{
                        0.,
                        1.,
                        2.,
                        3.,
                        4.,
                        5.,
                        6.,
                        7.,
                        8.,
                        9.,
                        10.,
                        11.,
                        12.,
                        13.,
                        14.,
                        15.,
                    },
                    1,
                    vector<float>{0.0179862, 0.0179862, 0.0179862, 0.0179862,
                                  0.9820138, 0.9820138, 0.9820138, 0.9820138,
                                  0.0179862, 0.0179862, 0.0179862, 0.0179862,
                                  0.9820138, 0.9820138, 0.9820138, 0.9820138});
    bangSoftmaxFp32(Shape{2, 2, 2, 2},
                    vector<float>{
                        0.,
                        1.,
                        2.,
                        3.,
                        4.,
                        5.,
                        6.,
                        7.,
                        8.,
                        9.,
                        10.,
                        11.,
                        12.,
                        13.,
                        14.,
                        15.,
                    },
                    2,
                    vector<float>{0.1192029, 0.1192029, 0.8807971, 0.8807971,
                                  0.1192029, 0.1192029, 0.8807971, 0.8807971,
                                  0.1192029, 0.1192029, 0.8807971, 0.8807971,
                                  0.1192029, 0.1192029, 0.8807971, 0.8807971});
    bangSoftmaxFp32(Shape{2, 2, 2, 2},
                    vector<float>{
                        0.,
                        1.,
                        2.,
                        3.,
                        4.,
                        5.,
                        6.,
                        7.,
                        8.,
                        9.,
                        10.,
                        11.,
                        12.,
                        13.,
                        14.,
                        15.,
                    },
                    3,
                    vector<float>{0.2689414, 0.7310586, 0.2689414, 0.7310586,
                                  0.2689414, 0.7310586, 0.2689414, 0.7310586,
                                  0.2689414, 0.7310586, 0.2689414, 0.7310586,
                                  0.2689414, 0.7310586, 0.2689414, 0.7310586});
    bangSoftmaxFp32(Shape{2, 4},
                    vector<float>{0., 1., 2., 3., 1000, 1001, 1002, 1003}, 0,
                    vector<float>{0., 0., 0., 0., 1, 1, 1, 1});
    bangSoftmaxFp32(
        Shape{2, 4}, vector<float>{0., 1., 2., 3., 1000, 1001, 1002, 1003}, 1,
        vector<float>{0.032058604, 0.08714432, 0.23688284, 0.6439143,
                      0.032058604, 0.08714432, 0.23688284, 0.6439143});
}

} // namespace infini
