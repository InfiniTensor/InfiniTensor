#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/attention.h"

#include "test.h"

namespace infini {

void test_attention(const Shape &outputShape, const vector<float> &inputQData,
                const vector<float> &inputKData,
                const vector<float> &inputVData,
                const vector<float> &ExpectData) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);
    auto inputV = gCpu->addTensor(outputShape, DataType::Float32);
    auto inputQ = gCpu->addTensor(outputShape, DataType::Float32);
    auto inputK = gCpu->addTensor(outputShape, DataType::Float32);

    gCpu->dataMalloc();
    inputV->copyin(inputVData); //
    inputQ->copyin(inputQData);
    inputK->copyin(inputKData); //

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputVGpu = gCuda->cloneTensor(inputV);
    auto inputQGpu = gCuda->cloneTensor(inputQ);
    auto inputKGpu = gCuda->cloneTensor(inputK);

    auto op = gCuda->addOp<AttentionObj>(inputQGpu, inputKGpu, inputVGpu,
                                     nullptr); // AttentionObj
    gCuda->dataMalloc();
    inputVGpu->copyin(inputVData);
    inputQGpu->copyin(inputQData);
    inputKGpu->copyin(inputKData);
    cudaRuntime->run(gCuda);

    auto oCpu = gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
    oCpu->printData();                              //->printData
    EXPECT_TRUE(oCpu->equalData(ExpectData));
}

TEST(CUDA_Attention, run) {
    test_attention(
        Shape{6,5}, vector<float>{0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1.,
        2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1.},
        vector<float>{0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1.,
        2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1.},
        vector<float>{0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1.,
        2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1.},
        vector<float>{6.507058e-03, 1.001569e+00, 2.000900e+00, 2.991024e+00, 6.507058e-03,
        1.004909e+00, 1.999979e+00, 2.986577e+00, 8.536250e-03, 1.004909e+00,
        2.017291e+00, 2.945395e+00, 1.997352e-02, 1.017340e+00, 2.017291e+00,
        2.999871e+00, 3.741202e-04, 9.998805e-01, 1.999874e+00, 2.999871e+00,
        6.507058e-03, 1.001569e+00, 2.000900e+00, 2.991024e+00, 6.507058e-03,
        1.004909e+00, 1.999979e+00, 2.986577e+00, 8.536250e-03, 1.004909e+00});

    test_attention(Shape{4, 3},                                  // inputQ
               vector<float>{0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.}, // inputK
               vector<float>{0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.},             // inputV
               vector<float>{0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.},
               vector<float>{0.9640308, 1.9546683, 2.9292183, 2.9460413, 0.0886370, 1.0179861,
        1.9941283, 2.9905086, 0.0210545, 1.0006673, 1.9993325, 2.9894698});

} // python output

} // namespace infini
