#include "core/graph.h"
#include "core/run_enigne.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"
#include "test.h"

namespace infini {

TEST(Conv, ShapeInference) {
    Runtime runtime = make_ref<RunEngine>(Device::CPU);
    // Padding modes
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv =
            g->addOp<ConvObj>(i0, w0, nullptr, ConvObj::PaddingMode::Same);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv =
            g->addOp<ConvObj>(i0, w0, nullptr, ConvObj::PaddingMode::Valid);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 2, 2}));
    }
    { // dilation & stride
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 2, 1, 1, 2);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 2, 2}));
    }
}

TEST(Conv, NaiveCPU) {
    Runtime runtime = make_ref<RunEngine>(Device::CPU);
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
    Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
    auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 2, 1, 1, 2);

    g->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());
    runtime->run(g, true, true);
    double perfTime = runtime->getPerfTime(g);
    // The example Conv takes 0.015ms with one core
    EXPECT_GT(perfTime, 0);
    EXPECT_LT(perfTime, 0.1);
    // check answer
    auto ans = make_ref<TensorObj>(Shape{1, 2, 2, 2}, DataType::UInt32);
    ans->dataMalloc(runtime);
    ans->copyData(
        vector<uint32_t>{4794, 4386, 8199, 7506, 11274, 10542, 20835, 19656});
    EXPECT_TRUE(conv->getOutput()->equalData(ans));
}

void testConvCudnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    vector<float> ansVec) {
    auto cpuRuntime = make_ref<RunEngine>(Device::CPU);
    auto cudaRuntime = make_ref<CudaRunEngine>();
    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::Float32);
    Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::Float32);
    auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 2, 1, 1, 2);

    // allocate CUDA memory
    g->dataMalloc();

    // Build input and output data on CPU
    auto cpui0 = make_ref<TensorObj>(Shape{1, 3, 4, 4}, DataType::Float32);
    cpui0->dataMalloc(cpuRuntime);
    cpui0->setData(generator);

    auto cpuw0 = make_ref<TensorObj>(Shape{2, 3, 3, 3}, DataType::Float32);
    cpuw0->dataMalloc(cpuRuntime);
    cpuw0->setData(generator);

    auto ans = make_ref<TensorObj>(Shape{1, 2, 2, 2}, DataType::Float32);
    ans->dataMalloc(cpuRuntime);
    ans->copyData(ansVec);

    // Copy inputs from CPU to CUDA
    cudaMemcpy(i0->getDataRawPtr<void *>(), cpui0->getDataRawPtr<void *>(),
               cpui0->size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w0->getDataRawPtr<void *>(), cpuw0->getDataRawPtr<void *>(),
               cpuw0->size() * sizeof(float), cudaMemcpyHostToDevice);
    // Execute on CUDA
    cudaRuntime->run(g);
    // double perfTime = cudaRuntime->getPerfTime(g);
    // // The example Conv takes 0.015ms with one core
    // EXPECT_GT(perfTime, 0);
    // EXPECT_LT(perfTime, 0.1);

    // copy CUDA output to CPU
    auto o0 = conv->getOutput();
    auto cpuo0 = make_ref<TensorObj>(Shape{1, 2, 2, 2}, DataType::Float32);
    cpuo0->dataMalloc(cpuRuntime);
    cudaMemcpy(cpuo0->getDataRawPtr<void *>(),
               conv->getOutput()->getDataRawPtr<void *>(),
               cpuo0->size() * sizeof(float), cudaMemcpyDeviceToHost);

    // check results on CPU
    EXPECT_TRUE(cpuo0->equalData<float>(ans));
}

TEST(Conv, cuDNN) {
    testConvCudnn(OneGenerator(),
                  vector<float>{12, 12, 18, 18, 12, 12, 18, 18});
    testConvCudnn(
        IncrementalGenerator(),
        vector<float>{4794, 4386, 8199, 7506, 11274, 10542, 20835, 19656});
}
} // namespace infini