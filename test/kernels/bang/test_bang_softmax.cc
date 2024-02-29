#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/softmax.h"
#include "test.h"
#include <cmath>
#include <sys/time.h>
namespace infini {
double eps = 3e-3;
void test_softmaxFp32(const Shape &inputShape, const vector<float> &inputData,
                      int axis, const vector<float> &expectData) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(inputShape, DataType::Float32, cpuRuntime);

    // GPU
    // cnnlSoftmax----------------
    Graph bangGraphCnnl = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraphCnnl->cloneTensor(inputCpu);
    auto gpuOp = bangGraphCnnl->addOp<SoftmaxObj>(inputGpu, nullptr, axis);
    bangGraphCnnl->dataMalloc();
    inputGpu->copyin(inputData);
    bangRuntime->run(bangGraphCnnl);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // bangSoftmax--------------
    Graph bangGraphBang = make_ref<GraphObj>(bangRuntime);
    inputGpu = bangGraphBang->cloneTensor(inputCpu);
    auto bangGpuOp =
        bangGraphBang->addOp<BangSoftmaxObj>(inputGpu, nullptr, axis);
    bangGraphBang->dataMalloc();
    inputGpu->copyin(inputData);
    bangRuntime->run(bangGraphBang);
    auto bangOutputGpu = gpuOp->getOutput();
    auto bangOutputGpu2Cpu = bangOutputGpu->clone(cpuRuntime);
    // Check
    EXPECT_TRUE(outputGpu2Cpu->equalData(expectData, eps));     // cnnlSoftmax
    EXPECT_TRUE(bangOutputGpu2Cpu->equalData(expectData, eps)); // bangSoftmax
}
double get_walltime() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
float err(float *x, float *y, const Shape &inputShape, int nDim) {
    int size = 1;
    for (int i = 0; i < nDim; i++) {
        size *= inputShape[i];
    }
    float error = 0;
    for (int i = 0; i < size; i++) {
        if (fabs(x[i] - y[i]) > error) {
            error = fabs(x[i] - y[i]);
        }
    }
    return error;
}

void test_compareSoftmaxFp32(
    int axis, const Shape &inputShape, int nDim,
    const std::function<void(void *, size_t, DataType)> &generator) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(inputShape, DataType::Float32, cpuRuntime);

    // GPU
    // cnnlSoftmax---------------------------
    Graph bangGraphCnnl = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraphCnnl->cloneTensor(inputCpu);

    auto gpuOp = bangGraphCnnl->addOp<SoftmaxObj>(inputGpu, nullptr, axis);
    inputGpu->setInput();
    bangGraphCnnl->dataMalloc();
    inputGpu->setData(generator);
    vector<float> inputData = inputGpu->copyout<float>();

    double bangst, bangela;
    bangst = get_walltime();
    bangRuntime->run(bangGraphCnnl);
    bangela = 1000 * (get_walltime() - bangst);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // bangSoftmax--------------------------------

    Graph bangGraphBang = make_ref<GraphObj>(bangRuntime);
    auto bangInputGpu = bangGraphBang->cloneTensor(inputCpu);
    auto bangGpuOp =
        bangGraphBang->addOp<BangSoftmaxObj>(bangInputGpu, nullptr, axis);
    inputGpu->setInput();
    bangInputGpu->setInput();
    bangGraphBang->dataMalloc();
    // bangInputGpu->setData(generator);
    bangInputGpu->copyin(inputData);

    double cnnlst, cnnlela;
    cnnlst = get_walltime();
    bangRuntime->run(bangGraphBang);
    cnnlela = 1000 * (get_walltime() - cnnlst);
    auto bangOutputGpu = bangGpuOp->getOutput();
    auto bangOutputGpu2Cpu = bangOutputGpu->clone(cpuRuntime);
    // Check
    float *cnnlOutput = outputGpu2Cpu->getRawDataPtr<float *>();
    float *bangOutput = bangOutputGpu2Cpu->getRawDataPtr<float *>();
    float error = err(cnnlOutput, bangOutput, inputShape, nDim);
    printf("axis:%d. bang time:%.2f ms, cnnl time:%.2f ms, err:%.8e\n", axis,
           bangela, cnnlela, error);
}
TEST(BANG_SoftmaxFp32, run) {
    test_softmaxFp32(
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
    test_softmaxFp32(
        Shape{2, 4}, vector<float>{0., 1., 2., 3., 1000, 1001, 1002, 1003}, 1,
        vector<float>{0.032058604, 0.08714432, 0.23688284, 0.6439143,
                      0.032058604, 0.08714432, 0.23688284, 0.6439143});
}
TEST(BANG_CompareSoftmaxFp32, run) {
    test_compareSoftmaxFp32(3, Shape{1, 32, 1, 5}, 4, RandomGenerator());
    test_compareSoftmaxFp32(3, Shape{1, 32, 128, 5}, 4, RandomGenerator());
    test_compareSoftmaxFp32(0, Shape{1024, 128, 64, 32}, 4, RandomGenerator());
    test_compareSoftmaxFp32(1, Shape{1024, 128, 64, 32}, 4, RandomGenerator());
    test_compareSoftmaxFp32(2, Shape{1024, 128, 64, 32}, 4, RandomGenerator());
    test_compareSoftmaxFp32(3, Shape{1024, 128, 64, 32}, 4, RandomGenerator());
    test_compareSoftmaxFp32(2, Shape{1024, 128, 64, 32}, 4,
                            IncrementalGenerator());
    test_compareSoftmaxFp32(3, Shape{1024, 128, 64, 32}, 4,
                            IncrementalGenerator());
}

} // namespace infini
