#include "operators/unary.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_cumSum.h"
#include "cuda/cuda_runtime.h"

namespace infini {

void cumSum_kernel(const Tensor& input, int axis, Tensor& output, 
                    bool exclusive, bool reverse) {
    assert(axis >= 0 && axis < (int)input->getRank());

    auto dims = input->getDims();
    int rank = input->getRank();
    std::vector<int> stride(rank, 1);
    for (int i = 1; i < rank; ++i) {
        stride[i] = stride[i - 1] * dims[i - 1];
    }

    int axis_stride = stride[axis];
    int axis_size = dims[axis];

    int total_size = input->size();
    //int axis_length = dims[axis];
    auto inputData = input->getRawDataPtr<float *>();
    auto outputData = output->getRawDataPtr<float *>();
    for (int i = 0; i < total_size; i += axis_stride) {
        int sum = 0;
        if (exclusive) {
            outputData[i / axis_stride * axis_stride] = 0;
        }
        for (int j = (exclusive ? axis_stride : 0); j < axis_size * axis_stride; j += axis_stride) {
            sum += inputData[(i + j) / axis_stride];
            // 注意：setElem 的具体实现取决于 Tensor 类的定义
            outputData[(i + j) / axis_stride] = sum;
        }
    }

    // if (reverse) {
    //     Tensor reversed_output(dims, float);
    //     std::vector<int> reversed_data(total_size);
    //     output.copyData(reversed_output);
    //     std::reverse(reversed_data.begin(), reversed_data.end());
    //     reversed_output.copyData(output);
    // }
}

class CumSumCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<CumsumObj>(_op);

        auto input = op->getInputs(0);
        auto output = op->getOutput();


        auto axis = op->getAxis();
        auto exclusive = op->getExclusive();
        auto reverse = op->getReverse();

        const int dType = op->getDType().getIndex();
        cumSum_kernel(input, axis, output, exclusive, reverse);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::CumSum, CumSumCuda, "CumSum_CUDA");

} // namespace infini