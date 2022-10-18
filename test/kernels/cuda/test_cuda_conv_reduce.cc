#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv2dreduce.h"
#include "operators/matmul.h"
#include "operators/reshape.h"

#include "test.h"

namespace infini {

void testConv2dReduce(
    const std::function<void(void *, size_t, DataType)> &generator,
    vector<float> ansVec, string mode) {
    auto cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    Runtime cpu = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpu);

    const int n = 1, h = 4, w = 4, c = 3, f = 1, r = 3;

    Tensor i0Cpu = gCpu->addTensor({n, h, w, c});
    Tensor w0Cpu = gCpu->addTensor({1, c, f * r * r});
    Tensor b0Cpu = gCpu->addTensor({f});

    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);
    b0Cpu->setData(generator);

    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    Tensor b0Cuda = gCuda->cloneTensor(b0Cpu);

    auto x = gCuda->addOp<ReshapeObj>(i0Cuda, nullptr, Shape{1, n * h * w, c})
                 ->getOutput();
    x = gCuda->addOp<MatmulObj>(x, w0Cuda, nullptr)->getOutput();
    x = gCuda->addOp<ReshapeObj>(x, nullptr, Shape{n, h, w, f, r, r})
            ->getOutput();
    if (mode == "conv") {
        x = gCuda
                ->addOp<Conv2dReduce>(x, b0Cuda, nullptr, false, 0.1, r / 2,
                                      r / 2)
                ->getOutput();
    } else {
        x = gCuda
                ->addOp<Conv2dReduceTranspose>(x, b0Cuda, nullptr, false, 0.1,
                                               r / 2, r / 2, 2, 2)
                ->getOutput();
    }

    gCuda->dataMalloc();
    cuda->run(gCuda, false);

    auto o0Cpu = gCpu->cloneTensor(x);
    // o0Cpu->printData();
    EXPECT_TRUE(o0Cpu->equalData(ansVec));
}

TEST(Case, conv2dreduce) {
    testConv2dReduce(OneGenerator(),
                     vector<float>{13, 19, 19, 13, 19, 28, 28, 19, 19, 28, 28,
                                   19, 13, 19, 19, 13},
                     "conv");
    testConv2dReduce(IncrementalGenerator(),
                     vector<float>{1719, 2916, 3699, 2625, 4077, 6480, 7533,
                                   5166, 6993, 10692, 11745, 7866, 4869, 7344,
                                   7965, 5271},
                     "conv");
    testConv2dReduce(OneGenerator(),
                     vector<float>{4.,  7., 4.,  7., 4.,  7., 4.,  7., 13., 7.,
                                   13., 7., 13., 7., 4.,  7., 4.,  7., 4.,  7.,
                                   4.,  7., 13., 7., 13., 7., 13., 7., 4.,  7.,
                                   4.,  7., 4.,  7., 4.,  7., 13., 7., 13., 7.,
                                   13., 7., 4.,  7., 4.,  7., 4.,  7., 4.},
                     "convt");
    testConv2dReduce(IncrementalGenerator(),
                     vector<float>{57,   222,  174,  456,  291,  690,  408,
                                   474,  1164, 708,  1632, 942,  2100, 1176,
                                   525,  1158, 642,  1392, 759,  1626, 876,
                                   1410, 3036, 1644, 3504, 1878, 3972, 2112,
                                   993,  2094, 1110, 2328, 1227, 2562, 1344,
                                   2346, 4908, 2580, 5376, 2814, 5844, 3048,
                                   1461, 3030, 1578, 3264, 1695, 3498, 1812},
                     "convt");
}

} // namespace infini
