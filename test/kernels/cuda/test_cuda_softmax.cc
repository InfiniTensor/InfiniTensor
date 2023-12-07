#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/softmax.h"
#include "test.h"
#include <cmath>
namespace infini {
void test_softmax(const Shape &inputShape, const vector<float> &inputData,
                  int axis, const vector<float> &ExpectData) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor(inputShape, DataType::Float32);

    gCpu->dataMalloc();

    input->copyin(inputData);

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputGpu = gCuda->cloneTensor(input);

    auto op = gCuda->addOp<SoftmaxObj>(inputGpu, nullptr, axis);
    gCuda->dataMalloc();

    inputGpu->copyin(inputData);

    cudaRuntime->run(gCuda);

    auto oCpu = gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
    oCpu->printData();                              //->printData
    EXPECT_TRUE(oCpu->equalData(ExpectData));
}
TEST(CUDA_Softmax, run) {
    test_softmax(Shape{2, 3, 2, 2},
                 vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
                               8.,  9.,  10., 11., 12., 13., 14., 15.,
                               16., 17., 18., 19., 20., 21., 22., 23.},
                 0,
                 vector<float>{6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                               6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                               6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                               6.14417422e-06, 6.14417422e-06, 6.14417422e-06,
                               9.99993801e-01, 9.99993801e-01, 9.99993801e-01,
                               9.99993801e-01, 9.99993801e-01, 9.99993801e-01,
                               9.99993801e-01, 9.99993801e-01, 9.99993801e-01,
                               9.99993801e-01, 9.99993801e-01, 9.99993801e-01});
    test_softmax(Shape{2, 3, 2, 2},
                 vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
                               8.,  9.,  10., 11., 12., 13., 14., 15.,
                               16., 17., 18., 19., 20., 21., 22., 23.},
                 1,
                 vector<float>{3.29320435e-04, 3.29320435e-04, 3.29320435e-04,
                               3.29320435e-04, 1.79802869e-02, 1.79802869e-02,
                               1.79802869e-02, 1.79802869e-02, 9.81690347e-01,
                               9.81690347e-01, 9.81690347e-01, 9.81690347e-01,
                               3.29320435e-04, 3.29320435e-04, 3.29320435e-04,
                               3.29320435e-04, 1.79802869e-02, 1.79802869e-02,
                               1.79802869e-02, 1.79802869e-02, 9.81690347e-01,
                               9.81690347e-01, 9.81690347e-01, 9.81690347e-01});
    test_softmax(Shape{2, 3, 2, 2},
                 vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
                               8.,  9.,  10., 11., 12., 13., 14., 15.,
                               16., 17., 18., 19., 20., 21., 22., 23.},
                 2,
                 vector<float>{0.11920292, 0.11920292, 0.88079703, 0.88079703,
                               0.11920292, 0.11920292, 0.88079703, 0.88079703,
                               0.11920292, 0.11920292, 0.88079703, 0.88079703,
                               0.11920292, 0.11920292, 0.88079703, 0.88079703,
                               0.11920292, 0.11920292, 0.88079703, 0.88079703,
                               0.11920292, 0.11920292, 0.88079703, 0.88079703});
    test_softmax(Shape{2, 3, 2, 2},
                 vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
                               8.,  9.,  10., 11., 12., 13., 14., 15.,
                               16., 17., 18., 19., 20., 21., 22., 23.},
                 3,
                 vector<float>{0.26894143, 0.73105860, 0.26894143, 0.73105860,
                               0.26894143, 0.73105860, 0.26894143, 0.73105860,
                               0.26894143, 0.73105860, 0.26894143, 0.73105860,
                               0.26894143, 0.73105860, 0.26894143, 0.73105860,
                               0.26894143, 0.73105860, 0.26894143, 0.73105860,
                               0.26894143, 0.73105860, 0.26894143, 0.73105860});
} // python output

} // namespace infini
