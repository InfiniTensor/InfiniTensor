#include "operators/gather.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/gather.h"

namespace infini {
class GatherCuda : public CudaKernelWithoutConfig {

    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {

        auto input = op->getInputs(0);
        auto index = op->getInputs(1);

        // GatherMetaData metaData;
        // initGatherMetaData(metaData, op);
        auto input_shape = input->getDims();
        auto index_shape = index->getDims();
        int axis = as<GatherBaseObj>(op)->getAxis();
        int frontsize = 1;
        int dimsize = input_shape[axis];
        int behindsize = 1;
        int indsize = 1;
        for (int i = 0; i < (int)(index_shape.size()); i++) {
            indsize *= index_shape[i];
        }
        for (int i = 0; i < (int)(input_shape.size()); i++) {
            if (i < axis) {
                frontsize *= input_shape[i];
            } else if (i > axis) {
                behindsize *= input_shape[i];
            }
        }
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const indexData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *outputData = (op->getOutput()->getRawDataPtr<void *>());
        if (op->getDType() == DataType::Float32) {
            gather_nv_f32(inputData, indexData, outputData, frontsize, dimsize,
                          behindsize, indsize, index->getDType());
        } else if (op->getDType() == DataType::Float16) {
            gather_nv_f16(inputData, indexData, outputData, frontsize, dimsize,
                          behindsize, indsize, index->getDType());
        } else if (op->getDType() == DataType::Int8) {
            gather_nv_f8(inputData, indexData, outputData, frontsize, dimsize,
                         behindsize, indsize, index->getDType());
        } else {
            IT_ASSERT(false);
        }
        // void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        // void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        // if (op->getDType() == DataType::Float32) {
        //     gather_kernel<float>((float *)inputData, (float *)outputData,
        //                          metaData, op->getOutput()->size());
        // } else if (op->getDType() == DataType::Float16) {
        //     gather_kernel<half>((half *)inputData, (half *)outputData,
        //     metaData,
        //                         op->getOutput()->size());
        // } else if (op->getDType() == DataType::Int8) {
        //     gather_kernel<int8_t>((int8_t *)inputData, (int8_t *)outputData,
        //                           metaData, op->getOutput()->size());
        // } else {
        //     IT_ASSERT(false);
        // }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Gather, GatherCuda, "Gather_CUDA");
} // namespace infini
